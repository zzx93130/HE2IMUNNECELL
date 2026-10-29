"""Five paired public LUSC images: native HookNet annotation concordance.

The release was used for model development and has no supplied fold manifest.
These are explicitly annotation-concordance metrics, not independent accuracy.
Only explicitly annotated TLS/other regions enter pixel-level calculations.
"""
from __future__ import annotations
import gc
import json
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import openslide
from shapely.geometry import Polygon,Point,shape
from shapely.affinity import scale
from shapely.ops import unary_union

from tls_methods_v2_core import ROOT,OUT,save_json,digest


def reference_polygons(path,mpp):
    root=ET.parse(path).getroot();groups={}
    for a in root.findall('./Annotations/Annotation'):
        coords=sorted(a.findall('./Coordinates/Coordinate'),key=lambda c:int(c.attrib['Order']))
        points=[(float(c.attrib['X'])*mpp,float(c.attrib['Y'])*mpp) for c in coords]
        if len(points)<3: continue
        g=Polygon(points).buffer(0)
        if not g.is_empty: groups.setdefault(a.attrib.get('PartOfGroup','other').lower(),[]).append(g)
    return groups


def concordance(truth,predicted,other):
    t=unary_union(truth);p=unary_union(predicted);o=unary_union(other)
    domain=t.union(o)
    positive=p.intersection(domain)
    intersection=positive.intersection(t).area
    dice=2*intersection/(positive.area+t.area) if positive.area+t.area else None
    iou=intersection/positive.union(t).area if positive.union(t).area else None
    recalled=sum(any(a.intersection(b).area/a.union(b).area>=.1 for b in predicted) for a in truth)
    explicit_other_predictions=sum(o.covers(g.centroid) and g.intersection(t).area==0 for g in predicted)
    return {'TLS_reference_regions':len(truth),'HookNet_TLS_regions':len(predicted),
        'explicit_annotation_domain_um2':domain.area,'annotated_TLS_area_um2':t.area,'annotated_other_area_um2':o.area,
        'TLS_area_Dice_within_annotation_domain':dice,'TLS_area_IoU_within_annotation_domain':iou,
        'annotated_TLS_area_coverage':intersection/t.area if t.area else None,
        'explicit_negative_regions_available':bool(other),
        'metric_qualification':'Explicit TLS/other domain only' if other else 'Positive annotations only: no specificity or whole-slide precision estimate',
        'reference_TLS_with_prediction_IoU_ge_0p1':recalled,
        'reference_TLS_detection_fraction':recalled/len(truth) if truth else None,
        'predicted_centers_in_explicit_other_with_no_TLS_overlap':explicit_other_predictions,
        'unannotated_pixels_evaluated_as_negative':False,'independent_performance_claim':False}


def main():
    destination=OUT/'external/region_reference'
    manifest=json.loads((destination/'pilot_manifest.json').read_text())
    if len(manifest)!=5 or len({r['patient_id'] for r in manifest})!=5:
        raise ValueError('Five unique patients required')
    import tensorflow as tf
    from run_hooknet_tls import prepare_slide,infer_and_verify,run_inference,postprocess
    from hooknet_tls_native import load_model
    tf.config.threading.set_intra_op_parallelism_threads(8)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    with tf.device('/CPU:0'): model,config,model_manifest=load_model(destination/'model_audit')
    rows=[]
    for row in manifest:
        if 'image_path' not in row: continue
        slide_path=row['image_path'];case_out=destination/'inference'/row['slide_barcode'];case_out.mkdir(parents=True,exist_ok=True)
        (case_out/'source_audit').mkdir(exist_ok=True)
        with openslide.OpenSlide(slide_path) as slide:
            mppx=float(slide.properties['openslide.mpp-x']);mppy=float(slide.properties['openslide.mpp-y'])
            if abs(mppx-mppy)>1e-6: raise ValueError('Anisotropic public slide spacing')
        truth=reference_polygons(row['annotation_path'],mppx)
        native_path=case_out/'native_polygons.geojson'
        started=time.perf_counter()
        if not native_path.exists():
            pyramid,mask,preprocessing=prepare_slide(slide_path,case_out)
            finished=case_out/'inference_complete.json'
            if finished.exists():
                timings=json.loads(finished.read_text())['timings']
                maps={name:np.load(case_out/f'{name}.npy',mmap_mode='r') for name in ['prediction_mask','TLS_confidence','GC_confidence']}
            else:
                center=truth.get('tls',truth.get('other',[Point(1000,1000)]))[0].centroid
                verify_center=[int(center.x/.5),int(center.y/.5)]
                function,device=infer_and_verify(model,pyramid[0],pyramid[2],verify_center,case_out,'auto')
                maps,timings=run_inference(model,function,device,pyramid,mask,case_out)
                save_json({'timings':timings,'complete':True,'slide':slide_path},finished)
            native_summary=postprocess(maps,config,case_out)
            save_json({'slide':slide_path,'mpp':mppx,'timings':timings,'preprocessing':preprocessing,'native_summary':native_summary},case_out/'provenance.json')
            del pyramid,mask,maps;gc.collect()
        features=json.loads(native_path.read_text())['features']
        predicted=[scale(shape(f['geometry']),xfact=.5,yfact=.5,origin=(0,0)) for f in features
                   if f['properties']['passes_published_filter'] and f['properties']['predicted_label']=='tls']
        metrics=concordance(truth.get('tls',[]),predicted,truth.get('other',[]))
        from PIL import ImageDraw
        with openslide.OpenSlide(slide_path) as slide:
            thumbnail=slide.get_thumbnail((1600,1600)).convert('RGB')
            image=thumbnail.copy();draw=ImageDraw.Draw(image)
            fx=thumbnail.width/(slide.dimensions[0]*mppx);fy=thumbnail.height/(slide.dimensions[1]*mppx)
            for group,color in [(truth.get('other',[]),'#009edb'),(truth.get('tls',[]),'#00c970'),(predicted,'#f044b6')]:
                for g in group:
                    parts=list(g.geoms) if g.geom_type=='MultiPolygon' else [g]
                    for part in parts:
                        if part.geom_type=='Polygon': draw.line([(x*fx,y*fy) for x,y in part.exterior.coords],fill=color,width=3)
            image.save(case_out/'reference_HookNet_overlay.jpg',quality=92)
        rows.append({**metrics,'slide_barcode':row['slide_barcode'],'patient_id':row['patient_id'],'organ':'LUSC',
                     'training_overlap':'Development release; exact fold unknown','seconds':time.perf_counter()-started})
        pd.DataFrame(rows).to_csv(destination/'pilot_annotation_concordance.csv',index=False)
        print('Public pilot complete:',row['slide_barcode'],flush=True)
    save_json({'model_source':model_manifest,'evaluated_methods':['HookNet-TLS'],'selected_slides':5,'completed_slides':len(rows),
        'morphology_method_public_validation':'Not claimed: frozen cell-coordinate inputs exist only for the current slide',
        'evaluation_domain':'Explicit TLS and other polygons only','training_overlap':'Unknown exact folds; development release',
        'no_cell_phenotype_validation':True},destination/'pilot_provenance.json')
    status_path=destination/'status.json'
    if status_path.exists():
        status=json.loads(status_path.read_text());status['completed_slides']=len(rows)
        status['status']='annotation_concordance_complete' if len(rows)==5 else 'partial_annotation_concordance'
        status['reference_metrics']=rows
        status['reason']='Five checksummed image/annotation pairs evaluated; development overlap prevents independent performance claims' if len(rows)==5 else 'Available image/annotation pairs evaluated; remaining transfers pending'
        status['metric_qualification']='Positive-only annotation cases do not support specificity or whole-slide precision; no aggregate performance estimate across incomplete pilot'
        save_json(status,status_path)


if __name__=='__main__': main()
