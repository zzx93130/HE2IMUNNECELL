"""Label audits, blinded H&E review cards, figures, and the revised HTML report."""
from __future__ import annotations
import base64
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import openslide
from shapely.geometry import Polygon, Point, mapping
from shapely import contains_xy

from tls_methods_v2_core import ROOT,RUN,OUT,MPP,GRID,CLASSES,save_json,load_source

# Selected on the unlabelled H&E montage before examining predicted cells.
# Coordinates below are fractions of a 200 µm square around each center.
EPITHELIAL = [
 ('E1',3670,3450,[(.04,.30),(.44,.30),(.60,.55),(.55,.96),(.06,.96)]),
 ('E2',4200,3260,[(.12,.47),(.70,.43),(.94,.78),(.87,.98),(.10,.96)]),
 ('E3',3440,4900,[(.02,.60),(.52,.54),(.67,.82),(.59,.98),(.03,.98)]),
 ('E4',3940,4850,[(.08,.15),(.59,.15),(.73,.36),(.78,.78),(.46,.83),(.06,.41)]),
 ('E5',5200,4030,[(.03,.02),(.68,.02),(.76,.39),(.41,.59),(.04,.50)]),
]


def font(size=16):
    try: return ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc',size)
    except OSError: return ImageFont.load_default()


def read_roi(slide,x,y,width_um,pixels=600):
    start=np.floor((np.array([x,y])-width_um/2)/MPP).astype(int)
    # Read level zero for review cards; use pyramid level one for large regions.
    level=0 if width_um<300 else 1
    down=slide.level_downsamples[level]
    side=int(np.ceil(width_um/MPP/down))
    image=slide.read_region(tuple(start),level,(side,side)).convert('RGB').resize((pixels,pixels),Image.Resampling.LANCZOS)
    return image,start*MPP,width_um/pixels


def draw_geometry(image,geom,origin,scale,color,width=2):
    d=ImageDraw.Draw(image)
    geoms=list(geom.geoms) if geom.geom_type=='MultiPolygon' else [geom]
    for p in geoms:
        if p.geom_type!='Polygon': continue
        for ring in [p.exterior,*p.interiors]:
            coords=[((x-origin[0])/scale,(y-origin[1])/scale) for x,y in ring.coords]
            d.line(coords,fill=color,width=width)


def stratified_sample(frame,n,rng):
    """Even coverage of size/confidence quantile combinations, then random refill."""
    work=frame.copy()
    work['_a']=pd.qcut(work.area_um2.rank(method='first'),3,labels=False)
    work['_c']=pd.qcut(work.raw_confidence.rank(method='first'),3,labels=False)
    shuffled=[rng.permutation(g.index).tolist() for _,g in work.groupby(['_a','_c'])]
    selected=[]
    while len(selected)<min(n,len(work)) and any(shuffled):
        for group in shuffled:
            if group and len(selected)<n: selected.append(group.pop())
    return frame.loc[selected]


def prepare_audit(cells,features):
    raw,manifest=load_source()
    merged=cells.merge(features,on='global_instance_id',validate='one_to_one')
    samples=[];stats=[]
    rng=np.random.default_rng(569)
    candidates=pd.read_csv(RUN/'tls_analysis/ranked_candidate_regions.csv')
    for a in candidates.head(2).itertuples():
        inside=(merged.x_um-a.x_um)**2+(merged.y_um-a.y_um)**2<=150**2
        local=merged[inside]
        for label,group in local.groupby('raw_celltype'):
            stats.append({'candidate_id':a.candidate_id,'raw_celltype':label,'cells':len(group),
                'median_area_um2':group.area_um2.median(),'p90_area_um2':group.area_um2.quantile(.9),
                'median_raw_confidence':group.raw_confidence.median(),'median_TTA_confidence':group.confidence.median(),
                'crop_truncated':int(group.crop_truncated.sum()),'TTA_B':int(group.celltype.eq('B').sum()),
                'TTA_T':int(group.celltype.eq('T').sum()),'TTA_Other':int(group.celltype.eq('Other').sum()),
                'median_circularity':group.circularity.median(),'median_eccentricity':group.eccentricity.median()})
        selected=stratified_sample(local[local.raw_celltype.eq('Other')],50,rng).copy()
        selected['audit_group']=a.candidate_id+'_Other';samples.append(selected)
    rois=[];epithelial_groups=[]
    with openslide.OpenSlide(manifest['slides'][0]['slide']) as slide:
        montage=Image.new('RGB',(1500,345),'white')
        for i,(rid,x,y,points) in enumerate(EPITHELIAL):
            g=Polygon([(x-100+px*200,y-100+py*200) for px,py in points])
            rois.append({'type':'Feature','geometry':mapping(g),'properties':{'roi_id':rid,
                'selection':'Preliminary epithelial-appearing H&E polygon; selected without prediction overlays',
                'pathologist_verified':False}})
            inside=contains_xy(g,merged.x_um.to_numpy(),merged.y_um.to_numpy())
            pool=merged[inside & merged.raw_celltype.eq('T')]
            epithelial_groups.append((rid,pool))
            im,origin,scale=read_roi(slide,x,y,200,300)
            draw_geometry(im,g,origin,scale,'#00c7ed',2)
            montage.paste(im,(i*300,45))
            ImageDraw.Draw(montage).text((i*300+5,8),f'{rid}: ({x},{y}) µm',font=font(14),fill='black')
        montage.save(OUT/'epithelial_selection_HE_polygons.png')
    selected_ids=set()
    for rid,pool in epithelial_groups:
        selected=pool.sample(min(10,len(pool)),random_state=569).copy()
        selected['audit_group']='T_epithelium';selected['epithelial_ROI']=rid
        samples.append(selected);selected_ids.update(selected.global_instance_id)
    if len(selected_ids)<50:
        spare=pd.concat([g.assign(epithelial_ROI=rid) for rid,g in epithelial_groups]).drop_duplicates('global_instance_id')
        spare=spare[~spare.global_instance_id.isin(selected_ids)]
        selected=spare.sample(min(50-len(selected_ids),len(spare)),random_state=570).copy()
        selected['audit_group']='T_epithelium';samples.append(selected);selected_ids.update(selected.global_instance_id)
    macros=merged[merged.raw_celltype.eq('Macrophage')].copy();macros['audit_group']='Macrophage_called';samples.append(macros)
    near=merged[~merged.raw_celltype.eq('Macrophage')].sort_values('raw_prob_Macrophage',ascending=False).head(20).copy()
    near['audit_group']='Macrophage_near_miss';samples.append(near)
    selected=pd.concat(samples,ignore_index=True).sample(frac=1,random_state=569).reset_index(drop=True)
    selected['blind_id']=[f'C{i+1:03d}' for i in range(len(selected))]
    selected.to_csv(OUT/'audit_samples_unblinded.csv',index=False)
    columns=['blind_id','global_instance_id','x_um','y_um']
    form=selected[columns].copy()
    for c in ['morphology_category','segmentation_issue','reviewer','reviewer_confidence','comments']: form[c]=''
    if not (OUT/'expert_review_form.csv').exists():
        form.to_csv(OUT/'expert_review_form.csv',index=False)
    pd.DataFrame(stats).to_csv(OUT/'A1_A2_label_mask_diagnostics.csv',index=False)
    save_json({'type':'FeatureCollection','coordinate_units':'micrometers','features':rois},OUT/'epithelial_review_ROIs.geojson')
    save_json({'seed':569,'selected_T_calls':len(selected_ids),'T_ROI_pool_counts':{r:len(g) for r,g in epithelial_groups},
        'review_group_counts':selected.audit_group.value_counts().to_dict(),
        'sampling':'Other: round-robin sample of 3×3 area/confidence quantile bins; T: up to ten per epithelial ROI, refill from same ROIs only',
        'review_status':'Preliminary computer-assisted morphology review; expert review pending',
        'B_T_reference_labels_available':False,'Macrophage_labels_reference_verified':False},OUT/'audit_summary.json')
    return selected,merged


def make_cards(samples):
    import tifffile
    raw,manifest=load_source();source=raw.set_index('global_instance_id')
    cards=OUT/'review_cards';cards.mkdir(exist_ok=True)
    masks={}
    with openslide.OpenSlide(manifest['slides'][0]['slide']) as slide:
        for number,c in enumerate(samples.itertuples()):
            target=cards/f'{c.blind_id}.jpg'
            row=source.loc[c.global_instance_id]
            close,origin,scale=read_roi(slide,c.x_um,c.y_um,40,210)
            context,context_origin,context_scale=read_roi(slide,c.x_um,c.y_um,100,280)
            maskpath=row.patch_mask_path
            if maskpath not in masks:
                if len(masks)>8: masks.clear()
                masks[maskpath]=tifffile.imread(maskpath)
            labels=masks[maskpath]
            import cv2
            # Display all local object outlines, with target distinguished geometrically only.
            xs=int(max(0,(context_origin[0]/MPP-row.patch_x)));ys=int(max(0,(context_origin[1]/MPP-row.patch_y)))
            xe=min(labels.shape[1],int((context_origin[0]+100)/MPP-row.patch_x));ye=min(labels.shape[0],int((context_origin[1]+100)/MPP-row.patch_y))
            if xe>xs and ye>ys:
                local=labels[ys:ye,xs:xe]
                d=ImageDraw.Draw(context)
                for lid in np.unique(local):
                    if lid==0: continue
                    contours,_=cv2.findContours((local==lid).astype('uint8'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                    for contour in contours:
                        if len(contour)<2: continue
                        pts=[(((px+xs+row.patch_x)*MPP-context_origin[0])/context_scale,
                              ((py+ys+row.patch_y)*MPP-context_origin[1])/context_scale) for px,py in contour[:,0,:]]
                        d.line(pts+[pts[0]],fill='#00e5ff' if lid==row.local_label else '#d5d5d5',width=2 if lid==row.local_label else 1)
            # Thin locator brackets avoid covering the center nucleus.
            d=ImageDraw.Draw(close);middle=105
            d.line((middle-9,middle-9,middle-4,middle-9),fill='white',width=1)
            d.line((middle+4,middle+9,middle+9,middle+9),fill='white',width=1)
            card=Image.new('RGB',(510,355),'#ffffff');d=ImageDraw.Draw(card)
            d.text((10,5),f'{c.blind_id} · object {c.global_instance_id}',font=font(17),fill='black')
            card.paste(close,(10,40));card.paste(context,(225,35))
            d.text((10,258),'40 µm H&E · target at center',font=font(12),fill='black')
            d.text((10,332),'100 µm context: cyan = target mask; gray = adjacent objects',font=font(12),fill='black')
            card.save(target,quality=91)
    # Two contact-sheet types: blinded sorted cards, and close-up groups for audit.
    sheets=OUT/'audit_contact_sheets';sheets.mkdir(exist_ok=True)
    for group,rows in samples.groupby('audit_group',sort=True):
        for part,start in enumerate(range(0,len(rows),20),1):
            local=rows.iloc[start:start+20];canvas=Image.new('RGB',(1400,60+int(np.ceil(len(local)/5))*245),'white')
            d=ImageDraw.Draw(canvas);d.text((10,8),f'{group}: preliminary morphology review, sheet {part}',font=font(19),fill='black')
            for i,c in enumerate(local.itertuples()):
                card=Image.open(cards/f'{c.blind_id}.jpg')
                # Full resolution small H&E view plus ID; contextual view remains on individual card.
                crop=card.crop((10,40,220,250)).resize((240,210))
                x=(i%5)*280;y=60+(i//5)*245
                canvas.paste(crop,(x,y));d.text((x+4,y+212),f'{c.blind_id} · {c.global_instance_id}',font=font(14),fill='black')
            canvas.save(sheets/f'{group}_{part}.png')
    return cards


def artifact_diagnostics(merged):
    fields=np.load(OUT/'fields_sigma25.npz');idx=np.floor(merged[['x_um','y_um']].to_numpy()/GRID).astype(int)
    density=fields['density'][idx[:,1],idx[:,0]]
    merged=merged.copy();merged['local_density_per_mm2']=density
    probability_rows=[]
    for label in CLASSES:
        for view,prefix,types in [('identity','raw_prob_',merged.raw_celltype),('TTA','prob_',merged.celltype)]:
            values=merged[prefix+label]
            probability_rows.append({'class':label,'prediction_method':view,'argmax_calls':int(types.eq(label).sum()),
                'median_probability':values.median(),'p95_probability':values.quantile(.95),
                'p99_probability':values.quantile(.99),'mean_probability':values.mean(),
                'sum_probabilities_uncalibrated':values.sum()})
    pd.DataFrame(probability_rows).to_csv(OUT/'class_probability_distributions.csv',index=False)
    merged['density_quartile']=pd.qcut(pd.Series(density).rank(method='first'),4,labels=False).to_numpy()
    seams=[];period=512*MPP
    for axis,other in [('x_um','y_um'),('y_um','x_um')]:
        for name,offset in [('core_grid',256*MPP),('shifted_half_stride_control',0.)]:
            coordinate=merged[axis].to_numpy()
            number=np.rint((coordinate-offset)/period).astype(int)
            delta=coordinate-(number*period+offset)
            near=merged[np.abs(delta)<=10].copy()
            near['seam']=number[np.abs(delta)<=10]
            near['side']=np.where(delta[np.abs(delta)<=10]<0,'before','after')
            near['along_bin']=np.floor(near[other]/200).astype(int)
            for (seam,along),g in near.groupby(['seam','along_bin']):
                left=g[g.side.eq('before')];right=g[g.side.eq('after')]
                if len(left)<10 or len(right)<10: continue
                seams.append({'axis':axis,'grid':name,'seam_position_um':seam*period+offset,'along_bin':along,
                    'before_cells':len(left),'after_cells':len(right),
                    'B_probability_jump':right.prob_B.mean()-left.prob_B.mean(),
                    'T_probability_jump':right.prob_T.mean()-left.prob_T.mean(),
                    'B_fraction_jump':right.celltype.eq('B').mean()-left.celltype.eq('B').mean(),
                    'hematoxylin_jump':right.hematoxylin_mean.mean()-left.hematoxylin_mean.mean()})
    sf=pd.DataFrame(seams,columns=['axis','grid','seam_position_um','along_bin','before_cells','after_cells',
             'B_probability_jump','T_probability_jump','B_fraction_jump','hematoxylin_jump'])
    sf.to_csv(OUT/'patch_seam_diagnostics.csv',index=False)
    correlations=[]
    from scipy.stats import spearmanr
    for q,g in merged.groupby('density_quartile'):
        for label in ['B','T']:
            rho=spearmanr(g.hematoxylin_mean,g['prob_'+label]).statistic
            correlations.append({'density_quartile':int(q)+1,'cells':len(g),'label':label,'H_probability_spearman_rho':rho})
    pd.DataFrame(correlations).to_csv(OUT/'stain_probability_diagnostics.csv',index=False)
    summaries=[]
    for (axis,grid),g in sf.groupby(['axis','grid']):
        summaries.append({'axis':axis,'grid':grid,'paired_segments':len(g),'median_abs_B_probability_jump':g.B_probability_jump.abs().median(),
                          'median_abs_T_probability_jump':g.T_probability_jump.abs().median()})
    save_json({'seam_summary':summaries,'stain_correlations':correlations,'seam_band_half_width_um':10,
        'along_seam_segment_um':200,'minimum_cells_per_side':10,
        'interpretation':'Associations and grid contrasts only; tissue composition and segmentation may confound apparent artifacts; no causation or label accuracy inferred'},OUT/'artifact_summary.json')
    merged.to_csv(OUT/'cells_with_diagnostics.csv.gz',index=False)
    return merged,sf


def create_figures(cells,merged,components,geoms,hook_geoms,gc_geoms,seams):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    raw,manifest=load_source();candidates=pd.read_csv(RUN/'tls_analysis/ranked_candidate_regions.csv')
    fields=np.load(OUT/'fields_sigma25.npz')
    gallery=OUT/'region_gallery';gallery.mkdir(exist_ok=True)
    with openslide.OpenSlide(manifest['slides'][0]['slide']) as slide:
        thumbnail=slide.get_thumbnail((1700,1700)).convert('RGB');thumbnail.save(OUT/'whole_slide_HE.jpg',quality=92)
        w,h=slide.dimensions;extent=[0,w*MPP/1000,h*MPP/1000,0]
        fig,axes=plt.subplots(1,3,figsize=(18,7),layout='constrained')
        axes[0].imshow(thumbnail,extent=extent)
        for key,g in geoms.items():
            parts=list(g.geoms) if g.geom_type=='MultiPolygon' else [g]
            for part in parts:
                x,y=part.exterior.xy;axes[0].plot(np.array(x)/1000,np.array(y)/1000,color='#00a9b5',lw=1.5)
            axes[0].text(g.centroid.x/1000,g.centroid.y/1000,key,color='#06616d',fontsize=8)
        for g in hook_geoms.values():
            parts=list(g.geoms) if g.geom_type=='MultiPolygon' else [g]
            for part in parts:
                x,y=part.exterior.xy;axes[0].plot(np.array(x)/1000,np.array(y)/1000,color='#bd42bd',lw=1)
        for a in candidates.itertuples():
            axes[0].add_patch(Circle((a.x_um/1000,a.y_um/1000),.15,fill=False,color='#ffb000',lw=1))
            axes[0].text(a.x_um/1000,a.y_um/1000,a.candidate_id,fontsize=8)
        axes[0].set_title('Morphology cyan · HookNet magenta · original disks gold')
        valid=fields['tissue'].astype(bool)
        im=axes[1].imshow(np.ma.masked_where(~valid,fields['density']),extent=extent,cmap='magma');fig.colorbar(im,ax=axes[1],shrink=.6,label='Objects/mm²')
        axes[1].set_title('All-object density · 25 µm smoothing')
        im=axes[2].imshow(np.ma.masked_where(~valid,fields['mean_area']),extent=extent,cmap='viridis',vmin=10,vmax=35);fig.colorbar(im,ax=axes[2],shrink=.6,label='Mean mask area (µm²)')
        axes[2].set_title('Nuclear-size proxy · no phenotype input')
        for ax in axes: ax.set_xlabel('x (mm)');ax.set_ylabel('y (mm)')
        fig.savefig(OUT/'morphology_whole_slide.png',dpi=150);plt.close(fig)
        regions=[(a.candidate_id,a.x_um,a.y_um,'Original candidate') for a in candidates.itertuples()]
        regions += [(r.region_id,r.x_um,r.y_um,'Morphology component') for r in components[components.retained].head(10).itertuples()]
        peaks=pd.read_csv(RUN/'tls_analysis/all_B_density_peaks.csv')
        rejected=peaks[(peaks.x_um.between(5500,6500))&(peaks.y_um.between(1500,2500))]
        if 'mixed_aggregate_screen' in rejected: rejected=rejected[~rejected.mixed_aggregate_screen]
        rejected=rejected.sort_values('B_150um',ascending=False).head(5)
        regions += [(f'Excluded_{i+1}',r.x_um,r.y_um,'Original screen-rejected hotspot; not a verified negative') for i,r in enumerate(rejected.itertuples())]
        if rejected.empty:
            frozen_B=np.load(RUN/'tls_analysis/density_fields.npz')['B']
            positions=[]
            for ix in range(frozen_B.shape[0]):
                x=(ix+.5)*25
                if not 5500<=x<=6500: continue
                for iy in range(frozen_B.shape[1]):
                    y=(iy+.5)*25
                    if 1500<=y<=2500: positions.append((float(frozen_B[ix,iy]),x,y))
            chosen=[]
            for value,x,y in sorted(positions,reverse=True):
                if all(np.hypot(x-a,y-b)>=150 for a,b in chosen):
                    chosen.append((x,y))
                    regions.append((f'Upper_right_{len(chosen)}',x,y,'Raw B-density review-window maximum absent from original shortlist; not a verified negative'))
                if len(chosen)==3: break
        for i,r in enumerate(components[~components.retained].sort_values('area_um2',ascending=False).head(3).itertuples()):
            regions.append((f'Rejected_component_{i+1}',r.x_um,r.y_um,'Morphology component rejected by area/object count'))
        rows=[]
        for key,x,y,kind in regions:
            width=1000
            if key in geoms:
                bounds=geoms[key].bounds;width=max(width,max(bounds[2]-bounds[0],bounds[3]-bounds[1])*1.2)
            image,origin,scale=read_roi(slide,x,y,width,850)
            image.save(gallery/f'{key}_HE.jpg',quality=93)
            overlay=image.copy()
            for g in geoms.values(): draw_geometry(overlay,g,origin,scale,'#00c7e0',3)
            for g in hook_geoms.values(): draw_geometry(overlay,g,origin,scale,'#ef69df',2)
            for g in gc_geoms.values(): draw_geometry(overlay,g,origin,scale,'#ff5252',2)
            for a in candidates.itertuples(): draw_geometry(overlay,Point(a.x_um,a.y_um).buffer(150),origin,scale,'#ffca35',2)
            overlay.save(gallery/f'{key}_regions.jpg',quality=93)
            rows.append({'review_id':key,'x_um':x,'y_um':y,'field_width_um':width,'kind':kind,
                         'HE_image':f'region_gallery/{key}_HE.jpg','overlay_image':f'region_gallery/{key}_regions.jpg'})
        pd.DataFrame(rows).to_csv(OUT/'region_gallery.csv',index=False)
    fig,axes=plt.subplots(2,3,figsize=(15,9),layout='constrained')
    transition=pd.read_csv(OUT/'class_transitions.csv',index_col=0).reindex(index=CLASSES,columns=CLASSES)
    im=axes[0,0].imshow(transition.to_numpy(),cmap='Blues');axes[0,0].set_xticks(range(4),CLASSES,rotation=35);axes[0,0].set_yticks(range(4),CLASSES)
    for i in range(4):
        for j in range(4): axes[0,0].text(j,i,f'{transition.iloc[i,j]:,}',ha='center',va='center',fontsize=8,color='white' if transition.iloc[i,j]>50000 else 'black')
    axes[0,0].set(title='Raw → eight-view TTA',xlabel='TTA class',ylabel='Raw class')
    for label,color in [('B','#168d54'),('T','#e0a30a'),('Other','#7a7e86'),('Macrophage','#8a5ab9')]:
        g=merged[merged.raw_celltype.eq(label)]
        axes[0,1].hist(g.orientation_disagreement,bins=np.arange(-.0625,.94,.125),weights=np.ones(len(g))/max(1,len(g)),histtype='step',label=label,color=color)
    axes[0,1].set(title='Orientation disagreement before averaging',xlabel='1 − modal view fraction',ylabel='Fraction of raw class');axes[0,1].legend()
    diag=pd.read_csv(OUT/'A1_A2_label_mask_diagnostics.csv')
    for i,candidate in enumerate(['A1','A2']):
        local=diag[diag.candidate_id.eq(candidate)]
        axes[0,2].plot(local.raw_celltype,local.median_area_um2,'o-',label=candidate)
    axes[0,2].set(title='Full-mask size inside original aggregates',ylabel='Median object area (µm²)');axes[0,2].legend()
    for grid,color in [('core_grid','#385bba'),('shifted_half_stride_control','#dc785c')]:
        data=seams[seams.grid.eq(grid)]
        axes[1,0].hist(data.B_probability_jump.abs(),bins=25,alpha=.5,label=grid,density=True,color=color)
    axes[1,0].set(title='Paired seam segments · no permutation p-value',xlabel='Absolute jump in mean predicted B probability',ylabel='Distribution density');axes[1,0].legend(fontsize=8)
    sample=merged.sample(min(12000,len(merged)),random_state=569)
    for ax,label in zip(axes[1,1:],['B','T']):
        sc=ax.scatter(sample.hematoxylin_mean,sample['prob_'+label],c=sample.density_quartile,s=3,alpha=.22,cmap='viridis')
        ax.set(title=f'{label} probability and hematoxylin signal',xlabel='Mean H optical-density proxy',ylabel='TTA probability')
    fig.savefig(OUT/'label_artifact_diagnostics.png',dpi=150);plt.close(fig)


def table(frame):
    return frame.to_html(index=False,border=0,classes='data',na_rep='—',float_format=lambda v:f'{v:.3f}')


def render_report():
    tta=json.loads((OUT/'tta_summary.json').read_text());audit=json.loads((OUT/'audit_summary.json').read_text())
    component=json.loads((OUT/'component_summary.json').read_text());summary=pd.read_csv(OUT/'A1_A6_summary.csv')
    rows=pd.read_csv(OUT/'region_gallery.csv');samples=pd.read_csv(OUT/'audit_samples_unblinded.csv')
    morphology=pd.read_csv(OUT/'components.csv');diag=pd.read_csv(OUT/'A1_A2_label_mask_diagnostics.csv')
    tools=json.loads((OUT/'tools/summary.json').read_text())
    tool_compare=pd.read_csv(OUT/'A1_A6_tool_summary.csv')
    native_regions=pd.read_csv(OUT/'tools/regions.csv')
    icat=native_regions[native_regions.method.eq('tlsR')][['region_id','run_id','B','T','icat']]
    run_display=pd.DataFrame(tools['runs'])[['method','run_id','detections','seconds']].rename(
        columns={'method':'Tool','run_id':'Settings','detections':'Detections','seconds':'Runtime (seconds)'})
    external_path=OUT/'external/status.json'
    external=json.loads(external_path.read_text()) if external_path.exists() else {'status':'pending','reason':'Public data retrieval has not completed'}
    for key,subdir in [('rna','rna'),('region_reference','region_reference')]:
        current=OUT/'external'/subdir/'status.json'
        if current.exists(): external[key]=json.loads(current.read_text())
    preliminary_path=OUT/'preliminary_morphology_observations.json'
    preliminary=json.loads(preliminary_path.read_text()) if preliminary_path.exists() else {'status':'pending','reason':'Review cards generated; preliminary visual review not yet recorded'}
    diagnostics=json.loads((OUT/'artifact_summary.json').read_text())
    probability_summary=pd.read_csv(OUT/'class_probability_distributions.csv') if (OUT/'class_probability_distributions.csv').exists() else pd.DataFrame()
    cards=[]
    for r in samples.itertuples():
        detail=f'{r.audit_group}; raw {r.raw_celltype} → TTA {r.celltype}; area {r.area_um2:.1f} µm²; raw confidence {r.raw_confidence:.3f}; TTA {r.confidence:.3f}'
        cards.append(f'<article class="review" data-id="{r.blind_id}" data-instance-id="{r.global_instance_id}" data-x-um="{r.x_um}" data-y-um="{r.y_um}"><img loading="lazy" src="review_cards/{r.blind_id}.jpg" alt="Blinded review {r.blind_id}"><p>{r.blind_id} · object {r.global_instance_id}</p><select data-field="morphology_category"><option value="">Morphology: unreviewed</option><option>Lymphocyte-like</option><option>Epithelial-like</option><option>Macrophage-like</option><option>Other</option><option>Uncertain</option></select><select data-field="segmentation_issue"><option value="">Mask: unreviewed</option><option>No obvious issue</option><option>Possible merge</option><option>Possible missed neighbors</option><option>Possible partial object</option><option>Uncertain</option></select><input data-field="comments" placeholder="Reviewer comments"><p class="model-details">{html.escape(detail)}</p></article>')
    gallery=''.join(f'<article><h3>{r.review_id}</h3><p>{html.escape(r.kind)} · {r.field_width_um:.0f} µm field</p><div class="pair"><img loading="lazy" src="{r.HE_image}" alt="{r.review_id} H&E"><img loading="lazy" src="{r.overlay_image}" alt="{r.review_id} region contours"></div></article>' for r in rows.itertuples())
    counts=pd.DataFrame({'class':CLASSES,'raw':[tta['raw_counts'].get(c,0) for c in CLASSES],'TTA':[tta['tta_counts'].get(c,0) for c in CLASSES]})
    observed=''.join(f'<p><b>{html.escape(k)}</b>: {html.escape(str(v))}</p>' for k,v in preliminary.items() if k!='reviewed_images')
    rna=external.get('rna',{});reference=external.get('region_reference',{})
    rna_block=f'<p>RNA check: {html.escape(rna.get("status","pending"))}.</p>'
    if rna.get('status')=='complete':
        score=rna['target_scores'][0]
        score_rows=pd.DataFrame([
            {'measure':'12-chemokine PC1','LUAD percentile':score['chemokine_PC1_percentile']},
            {'measure':'CXCL13','LUAD percentile':score['CXCL13_percentile']},
            {'measure':'CD19','LUAD percentile':score['CD19_percentile']},
            {'measure':'MS4A1 / CD20 gene','LUAD percentile':score['MS4A1_percentile']},
            {'measure':'CD19 + MS4A1 mean standardized expression','LUAD percentile':score['CD19_MS4A1_percentile']}])
        rna_block=f'<p class="good">RNA signature scoring completed across {rna["cohort_patients"]} LUAD patients. The target is near the {score["chemokine_PC1_percentile"]:.0f}th percentile for the chemokine PC1 and the {score["CXCL13_percentile"]:.0f}th percentile for CXCL13. These values support an immune-context check; they do not locate or confirm TLS.</p>'+table(score_rows)
        rna_block+='<p>Public RNA sample: '+html.escape(', '.join(rna['RNA_samples']))+'. The same patient is identified; this is not the scanned diagnostic section. Xena averages aliquots belonging to a sample before its log transformation; our calculation then takes patient medians across primary-tumor samples. The cohort-specific score is not the original microarray score scale.</p>'
        rna_block+='<img loading="lazy" src="RNA_context.png" alt="Target chemokine and B-marker scores within the LUAD RNA cohort">'
    pilot=reference.get('selected_pilot',[])
    pilot_block=f'<p>Five-slide lung pilot: {html.escape(reference.get("status","pending"))}. '+html.escape(reference.get('reason','Public retrieval not yet finished'))+'</p>'
    if pilot:
        pilot_frame=pd.DataFrame(pilot)
        pilot_block+=table(pilot_frame[[c for c in ['patient_id','organ','TLS_annotations','GC_annotations','image_status'] if c in pilot_frame]])
    if reference.get('reference_metrics'): pilot_block+='<div class="scroll">'+table(pd.DataFrame(reference['reference_metrics']))+'</div>'
    for item in pilot:
        relative=f'external/region_reference/inference/{item["slide_barcode"]}/reference_HookNet_overlay.jpg'
        if (OUT/relative).exists():
            pilot_block+=f'<details><summary>{html.escape(item["patient_id"])} · reference and HookNet contours</summary><p>Green: reference TLS; blue: explicitly annotated other; magenta: HookNet prediction. Unannotated regions remain outside pixel-score evaluation.</p><img loading="lazy" src="{relative}" alt="{html.escape(item["patient_id"])} reference comparison"></details>'
    pilot_block+='<p>The downloaded release contains 375 LUSC slides and states it was used for HookNet development. It supplies no held-out split. The current LUAD case is absent. Public-pilot metrics, when available, are HookNet annotation concordance on explicit regions; independent accuracy and public validation of the new morphology screen are not claimed. Annotated thumbnails are never inference inputs.</p>'
    content=f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>TLS methods v2 · labels and morphology</title>
<style>body{{font:16px/1.5 system-ui;color:#253044;background:#f5f7fa;margin:0}}main{{max-width:1450px;margin:auto;padding:32px}}h1,h2{{color:#172743}}h2{{margin-top:40px}}.notice{{padding:20px;background:#fff1d9;border-left:5px solid #d99b35}}.good{{padding:18px;background:#e4f2f0}}article{{background:white;padding:18px;border-radius:10px;margin:15px 0}}img{{max-width:100%;height:auto}}.pair{{display:grid;grid-template-columns:1fr 1fr;gap:14px}}.data{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{padding:9px;border-bottom:1px solid #d9dee8;text-align:left}}.scroll{{overflow-x:auto}}.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,420px),1fr));gap:12px}}.model-details{{display:none;color:#6e3e4f;font-size:13px}}body.unblind .model-details{{display:block}}select,input,button{{font:inherit;padding:7px;margin:5px}}pre{{white-space:pre-wrap;background:white;padding:15px;font-size:13px}}nav{{display:flex;gap:18px;flex-wrap:wrap}}a{{color:#17608e}}</style></head><body><main>
<h1>Cell label audit and morphological aggregate regions</h1><p>TCGA-55-7725 · whole slide · methods version 2</p>
<nav><a href="#labels">Label audit</a><a href="#regions">Morphological regions</a><a href="#gallery">H&E gallery</a><a href="#review">Blinded cell review</a><a href="#external">Public references and RNA</a><a href="#files">Downloads</a></nav>
<p class="notice">These are candidate aggregates for review. TTA improves orientation consistency but does not validate cell identity. B/T ratios, cell-label accuracy, TLS status and maturity remain unvalidated. Morphological selection uses no predicted phenotype.</p>
<section id="labels"><h2>Label audit first</h2><p>All {tta['cells']:,} canonical cells were classified with eight unique joint RGB/mask rotations and reflections. Softmax probabilities were averaged before assigning labels. {tta['changed_cells']:,} calls changed. Whole-slide TTA took {tta['seconds']:.1f} seconds.</p>
{table(counts)}<p>The macrophage class remains extremely sparse. No class-prior correction or unlabeled calibration was applied. The exact four-output mapping was confirmed by the user; legacy bundled dataset code contains five classes, and the checkpoint contains no class names. The saved best_acc value is not evidence of patient-held-out biological accuracy on this slide.</p>
<p>The classifier sees 64 × 64 pixels (16.128 µm at this scan spacing), ImageNet-normalized RGB and a separate binary mask. The segmentation weights are official Cellpose-SAM, rather than verified custom Linux segmentation weights. Full masks were used for all size features: 649 classifier crops contain only part of their segmented object.</p>
<img src="label_artifact_diagnostics.png" alt="Label transitions and artifact diagnostics"><details><summary>Class-probability distributions</summary><div class="scroll">{table(probability_summary)}</div><p>Probability sums describe the model’s output mass, not calibrated biological cell counts. They do not justify class-prior adjustment on an unlabeled slide.</p></details><h3>A1/A2 full-mask diagnostic</h3><div class="scroll">{table(diag)}</div>
<p>High-confidence Other calls with lymphocyte-sized masks persist inside these aggregates. Mask size and H&E appearance are diagnostic clues, not subtype ground truth.</p><h3>Preliminary visual observations</h3>{observed}
<p>Review set: {audit['review_group_counts']}. Expert review is pending. The 50 original T calls come from epithelial-appearing H&E polygons selected without prediction overlays; local lymphocyte-like cells may still represent real infiltrating lymphocytes.</p>
<img src="epithelial_selection_HE_polygons.png" alt="Five epithelial-appearing H&E sampling polygons">
<details><summary>Patch and stain diagnostics</summary><p>Core-grid seams are compared with half-stride-shifted control seams using 10 µm bands and 200 µm segments, requiring ten objects per side. Stain associations are stratified by density quartile. Tissue composition and segmentation can confound these contrasts. No permutation p-value or artifact causation is claimed.</p><pre>{html.escape(json.dumps(diagnostics,indent=2))}</pre></details></section>
<section id="regions"><h2>Connected morphological regions</h2><p>10 µm grid; Gaussian σ=25 µm; Otsu threshold on log-density; local mean full-object area below the slide’s 40th percentile. Tissue coverage corrects density. Eight-connected components undergo 10 µm closing and fill holes up to 2,500 µm²; retained components contain ≥50 objects and ≥5,000 µm².</p>
<p>Ranking is the equal mean of within-slide percentiles for higher density, smaller mask size, greater size uniformity and solidity. These are unvalidated nuclear-size proxies. There is no B/T selection, fixed-diameter requirement, peak-based splitting, core/annulus odds ranking or slide-level B–B permutation test.</p>
<img src="morphology_whole_slide.png" alt="Whole-slide morphology, density, size fields and HookNet contours"><div class="scroll">{table(morphology.loc[morphology.retained,['region_id','area_um2','equivalent_diameter_um','all_objects','density_objects_per_mm2','median_object_area_um2','relative_area_MAD','solidity','morphology_score','tta_B','tta_T','tta_BT_fraction_all_objects']])}</div>
<h3>Smoothing checks</h3>{table(pd.DataFrame(component['settings'])[['sigma_um','density_threshold_per_mm2','area_threshold_um2','retained_components','rejected_components']])}
<h3>Every original A1–A6 region</h3><div class="scroll">{table(summary)}</div><p>Continuous disk/region area overlap and B/T membership overlap are saved for every method pair. Multiple region matches are allowed. tlsR preserves native membership; it is not given an invented contour. TLS-Finder’s disk membership is geometrically derived.</p>
<h3>Tools with TTA labels</h3>{table(run_display)}<p>HookNet’s seven TLS predictions and separate GC prediction are reused unchanged. Cell membership is recomputed using TTA labels. HookNet is comparison evidence, not ground truth. The retained GC has zero actual overlap with retained TLS polygons in the original native result; maturity is not inferred.</p><h3>A1–A6 comparison across every method</h3><div class="scroll">{table(tool_compare)}</div><p>Zero-detection runs have explicit rows. Center-in-disk matches preserve the historical rule; continuous area and membership overlaps describe irregular regions and splitting. Area values are unavailable for native tlsR memberships.</p><details><summary>Native tlsR ICAT spatial descriptors</summary>{table(icat)}<p>These are spatial descriptors, not evidence of follicle maturity. Published parameters, returned cell membership and core/expansion behavior are preserved.</p></details></section>
<section id="gallery"><h2>H&E region gallery</h2><p>Cyan: morphology; magenta: HookNet TLS; red: HookNet GC; gold: original A1–A6 disks. Rejected hotspots and components are calibration examples, not confirmed negatives.</p>{gallery}</section>
<section id="review"><h2>Blinded cell review</h2><p>Record appearance and mask issues. H&E does not reliably distinguish B from T. Entries are saved only in this browser; export them to retain your review.</p><label>Reviewer <input id="reviewer" placeholder="Reviewer name"></label><button id="export">Export review CSV</button><button id="unblind">Show model details</button><div class="cards">{''.join(cards)}</div></section>
<section id="external"><h2>Public reference and RNA checks</h2>{rna_block}<h3>Public annotation pilot</h3>{pilot_block}<p>Public TLS-region annotations cannot establish B/T accuracy. Unannotated pixels are not verified negatives. Bulk RNA scoring uses patient medians of log₂(TPM+1), cohort gene standardization and a sign-oriented 12-chemokine PC1, adapting the <a href="https://link.springer.com/article/10.1186/s13058-017-0864-z">published PCA method</a>. Reference annotations come from the <a href="https://zenodo.org/records/10614928">authors’ release</a>.</p><details><summary>Public-data provenance and retrieval details</summary><pre>{html.escape(json.dumps(external,indent=2))}</pre></details></section>
<section id="files"><h2>Downloads and reproducibility</h2><p><a href="tta_cells.csv">TTA cells</a> · <a href="class_transitions.csv">Class transitions</a> · <a href="components.geojson">Component polygons</a> · <a href="components.csv">All components</a> · <a href="component_sensitivity_pairs.csv">Region sensitivity overlaps</a> · <a href="A1_A6_all_region_pairs.csv">Candidate comparisons</a> · <a href="expert_review_form.csv">Blinded review form</a> · <a href="audit_samples_unblinded.csv">Unblinded samples</a> · <a href="external/rna/patient_scores.csv">RNA cohort scores</a> · <a href="verification.json">Verification</a> · <a href="provenance.json">Provenance</a></p><p>Repeatable entry point: deployment/revise_tls_methods.py. Original predictions and reports are preserved. <a id="original-dashboard" href="../tls_tool_comparison/TLS_tool_comparison.html">Original tool dashboard</a></p></section>
<script>const key='HE2CELL_TLS_V2_REVIEW';let saved={{}};try{{saved=JSON.parse(localStorage.getItem(key)||'{{}}')}}catch(e){{}};document.querySelectorAll('.review').forEach(card=>{{const id=card.dataset.id;card.querySelectorAll('[data-field]').forEach(el=>{{el.value=(saved[id]||{{}})[el.dataset.field]||'';el.addEventListener('change',()=>{{saved[id]={{...(saved[id]||{{}}),[el.dataset.field]:el.value,reviewer:document.getElementById('reviewer').value}};localStorage.setItem(key,JSON.stringify(saved));}})}})}});document.getElementById('unblind').onclick=()=>document.body.classList.toggle('unblind');document.getElementById('export').onclick=()=>{{const fields=['blind_id','global_instance_id','x_um','y_um','morphology_category','segmentation_issue','reviewer','comments'];const quote=x=>'"'+String(x??'').replaceAll('"','""')+'"';const lines=[fields.join(',')];document.querySelectorAll('.review').forEach(card=>{{let row=saved[card.dataset.id]||{{}};lines.push([card.dataset.id,card.dataset.instanceId,card.dataset.xUm,card.dataset.yUm,...fields.slice(4).map(f=>row[f]||'')].map(quote).join(','))}});const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([lines.join('\n')],{{type:'text/csv'}}));a.download='TLS_v2_expert_review.csv';a.click();}};</script>
</main></body></html>'''
    # Python string must leave the JavaScript newline escape intact.
    content=content.replace("lines.join('\n')", "lines.join('\\n')")
    content=content.replace('</script>',"if(location.protocol.startsWith('http')&&location.port==='8844'){document.getElementById('original-dashboard').href='/TLS_tool_comparison.html';}</script>")
    (OUT/'TLS_methods_v2_report.html').write_text(content)
    return OUT/'TLS_methods_v2_report.html'


def plot_RNA():
    path=OUT/'external/rna/patient_scores.csv'
    if not path.exists(): return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    scores=pd.read_csv(path);target=scores[scores.case_id.eq('TCGA-55-7725')].iloc[0]
    fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    for ax,column,title in zip(axes,['chemokine_PC1','CXCL13','CD19_MS4A1_mean_z'],
                              ['12-chemokine PC1','CXCL13 · log₂(TPM+1)','CD19/MS4A1 mean z']):
        ax.hist(scores[column],bins=30,color='#48799d',alpha=.8)
        ax.axvline(target[column],color='#d36148',lw=2,label='TCGA-55-7725')
        ax.set(title=title,xlabel='Cohort-specific expression descriptor',ylabel='LUAD patients');ax.legend(fontsize=8)
    fig.suptitle(f'{len(scores)} primary-tumor LUAD patients · bulk RNA context, no spatial TLS confirmation')
    fig.savefig(OUT/'RNA_context.png',dpi=150);plt.close(fig)
