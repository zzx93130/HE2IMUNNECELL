"""Frozen HE2CELL evaluation restricted to human-annotated TCGA regions.
Run audit first, then run. Source drive is read-only; resumable 64-patch batches.
"""
from pathlib import Path
import argparse, collections, hashlib, json, os, shutil, sys, tarfile, time, zipfile
import xml.etree.ElementTree as ET
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'outputs/tcga_tls_golden_20261102'
DATA = Path('/Volumes/One Touch/TCGA_TLS_golden')
sys.path[:0] = [str(ROOT/'deployment/esca_tls'), str(ROOT/'deployment/visium_tls'), str(ROOT/'script/src')]
import cv2
import numpy as np
import pandas as pd
from scipy import ndimage
from slide_io import open_slide
from visium_cells import stage, KEEP
import esca_analysis as A
PARAM_FILE = ROOT/'outputs/esca_tls_20261019/analysis_he2cell_ft_tau05/report_summary.json'
MODEL = ROOT/'models/he2cell_esca_finetuned.pth'
PARAMS = json.loads(PARAM_FILE.read_text())['chosen_parameters']
NEG = {'other', 'inflammation', 'tumor', 'lymph node'}
GRID, TARGET, STRIDE, PATCH, MARGIN = 8., .25, 768, 1024, 128

def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(obj, indent=2, allow_nan=False)); temp.replace(path)

def digest(path, kind='sha256'):
    h=hashlib.new(kind)
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024), b''): h.update(b)
    return h.hexdigest()

def annotations(z, name):
    root=ET.fromstring(z.read('data/annotations_svs_spacings/'+name))
    result=[]
    for i,a in enumerate(root.findall('.//Annotation')):
        coords=sorted(a.findall('.//Coordinate'), key=lambda c: int(c.attrib['Order']))
        points=np.array([[float(c.attrib['X']),float(c.attrib['Y'])] for c in coords])
        if len(points)<3: raise ValueError(f'Invalid polygon {name}: {i}')
        result.append((i,a.attrib['PartOfGroup'].lower(),points))
    return result

def audit():
    OUT.mkdir(parents=True,exist_ok=True)
    md5=digest(DATA/'data.zip','md5')
    if md5!='144b0f70df48eb8ea975f84c89f7dec3': raise ValueError('Annotation release checksum mismatch')
    paths=collections.defaultdict(list)
    for cohort in ('LUSC','BLCA'):
        for p in (DATA/('TCGA-'+cohort)).rglob('*.svs'): paths[p.name.split('.')[0]].append(str(p))
    members={}
    with tarfile.open(DATA/'TCGA-KIRC.tar','r:') as t:
        for m in t:
            if m.isfile() and m.name.lower().endswith('.svs'):
                barcode=Path(m.name).name.split('.')[0]
                if barcode in members: raise ValueError('Duplicate KIRC slide '+barcode)
                members[barcode]={'member':m.name,'offset':m.offset_data,'size':m.size}
    rows=[]; counts=collections.Counter()
    with zipfile.ZipFile(DATA/'data.zip') as z:
        mapping=json.loads(z.read('data/annotations.json'))
        (OUT/'annotation_readme.md').write_bytes(z.read('data/readme.md'))
        for cohort,names in mapping.items():
            for name in names:
                b=Path(name).stem; anns=annotations(z,name); c=collections.Counter(a[1] for a in anns); counts.update(c)
                p=paths.get(b,[])
                if len(p)>1: raise ValueError('Ambiguous slide '+b)
                src={'path':p[0]} if p else members.get(b,{}) if cohort=='KIRC' else {}
                rows.append(dict(cohort=cohort,barcode=b,patient=b[:12],xml=name,source=src,labels=dict(c),
                                 status='available' if src else 'missing',annotation_count=len(anns)))
    rows.sort(key=lambda r:hashlib.sha256(('tcga-golden-20261102'+r['barcode']).encode()).hexdigest())
    save(OUT/'manifest.json',rows)
    frozen={'date':'2026-11-02','reference':'https://zenodo.org/records/10614928','annotation_md5':md5,
      'model_sha256':digest(MODEL),'prior_sha256':digest(MODEL.with_suffix('.prior.json')),
      'parameters':PARAMS,'target_mpp':TARGET,'grid_um':GRID,'context_um':128,'tta':'d4',
      'scope':'All explicitly annotated regions; context-guided inference, NOT whole-slide detection.',
      'positive_label':'tls','negative_labels':sorted(NEG),'gc':'nested subtype, excluded as independent reference object',
      'region_positive_rule':'predicted TLS covers >=10% of annotation area; not one-to-one object detection',
      'pixel_metrics':'TLS recall and specificity inside explicit negative annotations; conflicts excluded',
      'empty_slides':'reported but not scored; no full-slide inference',
      'uncertainty':'2000 patient-cluster bootstrap resamples, seed 20261102, pooled and by cohort',
      'hooknet':'development dataset; no independent HookNet accuracy claim',
      'code_sha256':digest(Path(__file__))}
    save(OUT/'protocol.json',frozen)
    save(OUT/'audit.json',{'cohorts':{c:{'annotations':sum(r['cohort']==c for r in rows),
       'available':sum(r['cohort']==c and r['status']=='available' for r in rows),
       'empty':sum(r['cohort']==c and not r['annotation_count'] for r in rows)} for c in mapping},
       'labels':dict(counts),'kirc_archive_slides':len(members)})
    print((OUT/'audit.json').read_text(),flush=True)

def materialize(row,folder):
    src=row['source']
    if 'path' in src:return Path(src['path'])
    dst=folder/'source.svs'
    if dst.exists() and dst.stat().st_size==src['size']:return dst
    if shutil.disk_usage(OUT).free < src['size']+8e9:raise RuntimeError('Insufficient workspace free space')
    with open(DATA/'TCGA-KIRC.tar','rb') as f,open(dst,'wb') as g:
        f.seek(src['offset']);left=src['size']
        while left:
            b=f.read(min(left,8*1024*1024))
            if not b:raise EOFError('Truncated tar member')
            g.write(b);left-=len(b)
    return dst

def region_mask(shape,points,mpp):
    mask=np.zeros(shape,np.uint8)
    cv2.fillPoly(mask,[np.rint(points*np.array(mpp)/GRID).astype(np.int32)],1)
    return mask.astype(bool)

def process(row,device):
    folder=OUT/'slides'/row['barcode'];folder.mkdir(parents=True,exist_ok=True)
    if (folder/'summary.json').exists():return
    with zipfile.ZipFile(DATA/'data.zip') as z:anns=annotations(z,row['xml'])
    anns=[a for a in anns if a[1]=='tls' or a[1] in NEG]
    if not anns:
        save(folder/'summary.json',{'status':'no_evaluable_annotations','barcode':row['barcode']});return
    path=materialize(row,folder)
    with open_slide(path) as slide:
        mx=float(slide.properties['openslide.mpp-x']);my=float(slide.properties['openslide.mpp-y'])
        if not(.1<mx<2 and .1<my<2):raise ValueError('Invalid MPP')
        w,h=slide.dimensions;shape=(int(np.ceil(h*my/GRID)),int(np.ceil(w*mx/GRID)))
        union=np.zeros(shape,bool)
        for _,_,pts in anns:
            if (pts<0).any() or (pts[:,0]>w).any() or (pts[:,1]>h).any():raise ValueError('Annotation outside slide')
            union |= region_mask(shape,pts,(mx,my))
        expanded=cv2.dilate(union.astype(np.uint8),np.ones((33,33),np.uint8)).astype(bool)
        # Global target-resolution grid; selection depends on annotated ROI, never on predictions.
        cores=[]
        for y in range(0,int(np.ceil(h*my/TARGET)),STRIDE):
            for x in range(0,int(np.ceil(w*mx/TARGET)),STRIDE):
                if expanded[int(y*TARGET/GRID):int((y+STRIDE)*TARGET/GRID)+1,
                            int(x*TARGET/GRID):int((x+STRIDE)*TARGET/GRID)+1].any():cores.append((x,y))
        save(folder/'plan.json',{'patches':len(cores),'mpp':[mx,my],'dimensions':[w,h],'source':row['source']})
        for batch,start in enumerate(range(0,len(cores),64)):
            final=folder/f'cells_{batch:04d}.csv.gz'
            if final.exists():continue
            if shutil.disk_usage(OUT).free<8e9:raise RuntimeError('Less than 8 GB free')
            work=folder/'work';shutil.rmtree(work,ignore_errors=True);(work/'patches').mkdir(parents=True)
            patches=[]
            for j,(cx,cy) in enumerate(cores[start:start+64]):
                x,y=cx-MARGIN,cy-MARGIN
                sx,sy=int(np.floor(x*TARGET/mx)),int(np.floor(y*TARGET/my))
                size=(int(np.ceil(PATCH*TARGET/mx))+2,int(np.ceil(PATCH*TARGET/my))+2)
                rgba=np.array(slide.read_region((sx,sy),0,size));rgb=rgba[:,:,:3].copy();rgb[rgba[:,:,3]==0]=255
                matrix=np.float32([[mx/TARGET,0,(sx*mx/TARGET-x)],[0,my/TARGET,(sy*my/TARGET-y)]])
                patch=cv2.warpAffine(rgb,matrix,(PATCH,PATCH),flags=cv2.INTER_CUBIC,borderValue=(255,255,255))
                p=work/'patches'/f'{j}.png';cv2.imwrite(str(p),cv2.cvtColor(patch,cv2.COLOR_RGB2BGR))
                patches.append(dict(path=str(p),num_patch=j,sample=row['barcode'],source_tif=str(path),x=x,y=y,
                                    patch_size=PATCH,stride=STRIDE,level=0,tissue_ratio=1.))
            index=work/'patch_index.csv';pd.DataFrame(patches).to_csv(index,index=False);log=folder/'pipeline.log'
            print(row['barcode'],f'batch {batch+1}/{int(np.ceil(len(cores)/64))}',flush=True)
            stage('run_cellcrops.py','--patch_csv',index,'--out_dir',work/'cells','--model_name','cpsam_v2',
                  '--device',device,'--num_gpus',1,'--batch_size',16,'--crop_size',64,log=log)
            meta=work/'cells/mask_tables/patch_index_cellpose_mask_instances.csv'
            result=work/'classified.csv'
            stage('model_eval.py','--meta_path',meta,'--save_path',MODEL,'--output_path',result,'--device',device,
                  '--batch_size',256,'--num_workers',0,'--tta','d4',log=log)
            table=pd.read_csv(result)[KEEP];table['batch']=batch
            tmp=folder/'cells.tmp.gz';table.to_csv(tmp,index=False,compression='gzip');tmp.replace(final)
            shutil.rmtree(work)
    cells=pd.concat([pd.read_csv(p) for p in sorted(folder.glob('cells_*.csv.gz'))],ignore_index=True)
    gx=np.floor(cells.centroid_x_global.to_numpy()*TARGET/GRID).astype(int)
    gy=np.floor(cells.centroid_y_global.to_numpy()*TARGET/GRID).astype(int)
    ok=(gx>=0)&(gy>=0)&(gx<shape[1])&(gy<shape[0]);gx=gx[ok];gy=gy[ok];types=cells.celltype.to_numpy()[ok]
    counts={k:np.bincount((gy*shape[1]+gx)[types==k],minlength=np.prod(shape)).reshape(shape).astype(np.float32)
            for k in ('B','T','Other','Macrophage')}
    valid=np.zeros(shape,bool)
    for x,y in cores:valid[int(y*TARGET/GRID):int((y+STRIDE)*TARGET/GRID),int(x*TARGET/GRID):int((x+STRIDE)*TARGET/GRID)]=True
    inputs={'counts':counts,'valid':valid}
    masks={}
    for method in ['B (HE2CELL)','All nuclei (no classifier)']:
        masks[method]=A.detect(inputs,['B'] if method.startswith('B ') else None,**PARAMS[method])[0]
    # Same frozen stage-2 acceptance, calculated without reference-dependent helper features.
    labels,n=ndimage.label(masks['All nuclei (no classifier)'])
    totals=sum(counts.values());mass=np.bincount(labels.ravel(),weights=totals.ravel(),minlength=n+1)
    bmass=np.bincount(labels.ravel(),weights=counts['B'].ravel(),minlength=n+1)
    keep=np.divide(bmass,mass,out=np.zeros_like(mass),where=mass>0)>=PARAMS[A.TWO_STAGE]['b_share'];keep[0]=False
    assert not PARAMS[A.TWO_STAGE]['require_b_core']
    masks[A.TWO_STAGE]=keep[labels]
    np.savez_compressed(folder/'predictions_8um.npz',**{'method_'+str(i):m for i,m in enumerate(masks)})
    save(folder/'prediction_methods.json',list(masks))
    pos=np.zeros(shape,bool);neg=pos.copy()
    for _,kind,pts in anns:
        if kind=='tls':pos |= region_mask(shape,pts,(mx,my))
        else:neg |= region_mask(shape,pts,(mx,my))
    conflict=pos&neg;pos &= ~conflict;neg &= ~conflict
    records=[];pixel=[]
    for method,pred in masks.items():
        pixel.append(dict(cohort=row['cohort'],patient=row['patient'],barcode=row['barcode'],method=method,
                          tls_pixels=int(pos.sum()),tls_detected_pixels=int((pos&pred).sum()),
                          negative_pixels=int(neg.sum()),negative_called_pixels=int((neg&pred).sum())))
        for i,kind,pts in anns:
            mask=region_mask(shape,pts,(mx,my)) & ~conflict
            area=int(mask.sum())
            if not area:continue
            coverage=float(pred[mask].mean())
            records.append(dict(cohort=row['cohort'],patient=row['patient'],barcode=row['barcode'],
                annotation=i,label=kind,method=method,area_um2=area*GRID**2,coverage=coverage,detected=coverage>=.1))
    pd.DataFrame(records).to_csv(folder/'regions.csv',index=False);pd.DataFrame(pixel).to_csv(folder/'pixels.csv',index=False)
    save(folder/'summary.json',{'status':'complete','barcode':row['barcode'],'patches':len(cores),'cells':len(cells),
         'conflict_pixels_excluded':int(conflict.sum()),'native_mpp':[mx,my]})
    if path==folder/'source.svs':path.unlink()
    report()

def report():
    manifest=json.loads((OUT/'manifest.json').read_text());tables=list((OUT/'slides').glob('*/regions.csv'))
    summary={'status':'INTERIM','completed':len(tables),'total_manifest':len(manifest),'metrics':[],
       'scope':'Annotation-guided region evaluation; no whole-slide precision or object F1.'}
    if tables:
        df=pd.concat([pd.read_csv(p) for p in tables],ignore_index=True);df.to_csv(OUT/'regions.csv',index=False)
        for cohort in ['ALL','LUSC','BLCA','KIRC']:
            d=df if cohort=='ALL' else df[df.cohort==cohort]
            for method,g in d.groupby('method'):
                for name,subset in [('TLS_region_recall',g[g.label=='tls']),('negative_region_specificity',g[g.label!='tls'])]:
                    if subset.empty:continue
                    s=subset.assign(success=subset.detected if name.startswith('TLS') else ~subset.detected)
                    pts=s.groupby('patient').success.agg(['sum','count']).to_numpy()
                    rng=np.random.default_rng(20261102);boots=[]
                    for _ in range(2000):
                        a=pts[rng.integers(len(pts),size=len(pts))].sum(0);boots.append(a[0]/a[1])
                    summary['metrics'].append(dict(cohort=cohort,method=method,metric=name,value=float(s.success.mean()),
                        ci95=list(np.quantile(boots,[.025,.975])) if len(pts)>1 else None,patients=len(pts),regions=len(s)))
        pixels=pd.concat([pd.read_csv(p) for p in (OUT/'slides').glob('*/pixels.csv')]);pixels.to_csv(OUT/'pixels.csv',index=False)
    states=[json.loads(p.read_text())['status'] for p in (OUT/'slides').glob('*/summary.json')]
    summary['processed_including_empty']=len(states)
    summary['errors']=len(list((OUT/'slides').glob('*/error.json')))
    available=sum(r['status']=='available' for r in manifest)
    if len(states)==available:summary['status']='COMPLETE_AVAILABLE_SLIDES'
    save(OUT/'summary.json',summary)
    lines=['# TCGA gold-standard TLS evaluation',f"Status: {summary['status']}; {len(tables)} slides scored, {len(states)} processed, {available} available.",
       'Annotation-guided evaluation with frozen HE2CELL, D4 and prior correction. Region call: at least 10% of human polygon covered.',
       'This does not estimate whole-slide precision, object F1, or performance on unannotated tissue. Empty annotated slides are not scored. HookNet used this reference for development.',
       '| Cohort | Method | Metric | Estimate | 95% patient bootstrap CI | Patients | Regions |',
       '|---|---|---|---:|---|---:|---:|']
    for m in summary['metrics']:
        ci='not estimable' if m['ci95'] is None else '–'.join(f'{x:.3f}' for x in m['ci95'])
        lines.append(f"| {m['cohort']} | {m['method']} | {m['metric']} | {m['value']:.3f} | {ci} | {m['patients']} | {m['regions']} |")
    (OUT/'REPORT.md').write_text('\n\n'.join(lines[:4])+'\n\n'+'\n'.join(lines[4:])+'\n')

def run(args):
    protocol=json.loads((OUT/'protocol.json').read_text())
    if digest(MODEL)!=protocol['model_sha256'] or digest(MODEL.with_suffix('.prior.json'))!=protocol['prior_sha256'] or PARAMS!=protocol['parameters']:
        raise ValueError('Frozen model/parameters changed')
    lock=ROOT/'outputs/_gpu.lock'
    lock.mkdir();(lock/'owner').write_text(f'tcga-golden pid {os.getpid()}')
    try:
        rows=json.loads((OUT/'manifest.json').read_text())
        if args.only:rows=[r for r in rows if r['barcode'] in args.only]
        if args.limit:rows=rows[:args.limit]
        for row in rows:
            if row['status']!='available':continue
            try:
                process(row,args.device)
                err=OUT/'slides'/row['barcode']/'error.json'
                if err.exists():err.unlink()
            except Exception as exc:
                save(OUT/'slides'/row['barcode']/'error.json',{'error':repr(exc)})
                raise  # stop on infrastructure failures; never silently drop a failed patient
        report()
    finally:
        (lock/'owner').unlink();lock.rmdir()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['audit','run','report']);p.add_argument('--device',default='mps')
    p.add_argument('--limit',type=int);p.add_argument('--only',nargs='+');args=p.parse_args()
    if args.action=='audit':audit();report()
    elif args.action=='report':report()
    else:run(args)
