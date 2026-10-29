"""Public region reference and bulk RNA checks with explicit missing-data status.

Only public case identifiers are sent. No user images or predictions are uploaded.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import gzip
import io
import json
from pathlib import Path
import time
import urllib.parse
import urllib.request
import zipfile

import numpy as np
import pandas as pd
from tls_methods_v2_core import ROOT, RUN, OUT, save_json, digest

CHEMOKINES = ['CCL2','CCL3','CCL4','CCL5','CCL8','CCL18','CCL19','CCL21','CXCL9','CXCL10','CXCL11','CXCL13']
GENES = CHEMOKINES+['CD19','MS4A1']
EXT = OUT/'external'


def fetch(url, path, timeout=30, attempts=2, body=None):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists(): return path.read_bytes()
    errors=[]
    for i in range(attempts):
        try:
            request=urllib.request.Request(url,data=body,headers={'User-Agent':'HE2CELL-research/2','Accept':'application/json'})
            if body is not None: request.add_header('Content-Type','application/json')
            with urllib.request.urlopen(request,timeout=timeout) as response: data=response.read()
            temporary=path.with_suffix(path.suffix+'.part');temporary.write_bytes(data);temporary.replace(path)
            return data
        except Exception as error:
            errors.append(f'{type(error).__name__}: {str(error)[:240]}')
    raise RuntimeError('; '.join(errors))


def gdc_files(filters, fields, size=2000):
    query=urllib.parse.urlencode({'filters':json.dumps(filters),'fields':fields,'size':size,'format':'JSON'})
    return 'https://api.gdc.cancer.gov/files?'+query


def cohort_scores(expression):
    """Patient medians of log2 TPM, standardized within LUAD; no score imputation."""
    required={'case_id','sample_id',*GENES}
    if not required.issubset(expression): raise ValueError('Missing expression fields')
    if not np.isfinite(expression[GENES].to_numpy()).all() or (expression[GENES]<0).any().any():
        raise ValueError('Missing or invalid signature gene TPM; do not impute')
    logged=expression.copy();logged[GENES]=np.log2(logged[GENES]+1)
    patients=logged.groupby('case_id',sort=True)[GENES].median()
    if len(patients)<20: raise ValueError('At least 20 primary-tumor patients required for cohort scores')
    means=patients.mean(); sd=patients.std(ddof=1)
    if (sd<=0).any(): raise ValueError('Constant expression genes cannot be standardized')
    z=(patients-means)/sd
    matrix=z[CHEMOKINES].to_numpy()
    _,singular,vt=np.linalg.svd(matrix,full_matrices=False)
    loadings=vt[0];scores=matrix@loadings
    if np.corrcoef(scores,matrix.mean(1))[0,1]<0: loadings=-loadings;scores=-scores
    patients['chemokine_PC1']=scores
    patients['chemokine_PC1_percentile']=pd.Series(scores,index=patients.index).rank(pct=True)*100
    patients['CD19_MS4A1_mean_z']=z[['CD19','MS4A1']].mean(axis=1)
    patients['CD19_MS4A1_percentile']=patients.CD19_MS4A1_mean_z.rank(pct=True)*100
    for gene in ['CXCL13','CD19','MS4A1']:
        patients[gene+'_percentile']=patients[gene].rank(pct=True)*100
    metadata={'patients':len(patients),'genes':CHEMOKINES,'PC1_loadings':dict(zip(CHEMOKINES,loadings)),
              'PC1_explained_variance_fraction':float(singular[0]**2/(singular**2).sum()),
              'log2_TPM_gene_means':means.to_dict(),'log2_TPM_gene_sd':sd.to_dict(),
              'patient_aggregation':'Median log2(TPM+1) across available primary-tumor samples',
              'standardization':'Cohort gene-wise mean and sample SD; PCA sign follows mean chemokine z',
              'interpretation':'Cohort-specific RNA-seq adaptation; not the original microarray score scale or a TLS probability',
              'method_source':'https://link.springer.com/article/10.1186/s13058-017-0864-z'}
    return patients.reset_index(),metadata


def run_xena_rna():
    """Query only signature genes from the UCSC-hosted GDC STAR TPM mirror."""
    destination=EXT/'rna/xena';destination.mkdir(parents=True,exist_ok=True)
    host='https://gdc.xenahubs.net';dataset='TCGA-LUAD.star_tpm.tsv'
    def call(name,params):
        source_url=f'https://raw.githubusercontent.com/ucscXena/xenaPython/master/xenaPython/queries/{name}.xq'
        query_source=fetch(source_url,destination/(name+'.xq')).decode()
        def marshall(v):
            if v is None: return 'nil'
            if isinstance(v,list): return '['+' '.join(marshall(x) for x in v)+']'
            return json.dumps(v)
        query='('+query_source+' '+' '.join(marshall(p) for p in params)+')'
        path=destination/(name+'_response.json')
        if path.exists(): return json.loads(path.read_text())
        request=urllib.request.Request(host+'/data/',query.encode(),headers={'Content-Type':'text/plain','User-Agent':'HE2CELL-research/2'})
        with urllib.request.urlopen(request,timeout=35) as response: data=response.read()
        result=json.loads(data);path.write_bytes(data);return result
    metadata=call('datasetMetadata',[dataset])
    if not metadata: raise ValueError('GDC Xena STAR TPM dataset unavailable')
    item=metadata[0];text=json.loads(item.get('text','{}')) if isinstance(item.get('text'),str) else item.get('text',{})
    unit=str(text.get('unit','')).lower().replace(' ','')
    import re
    logunit=re.fullmatch(r'log2\(tpm\+([0-9.]+)\)',unit)
    if not logunit and unit!='tpm': raise ValueError('Unverified RNA unit '+unit)
    samples=call('datasetSamples',[dataset,None])
    selected=[s for s in samples if s.startswith('TCGA-') and len(s)>=15 and s[13:15]=='01']
    if not any(s.startswith('TCGA-55-7725-01') for s in selected): raise ValueError('Target primary-tumor RNA sample absent from Xena mirror')
    values=call('datasetGeneProbeAvg',[dataset,selected,GENES])
    if len(values)!=len(GENES): raise ValueError('Missing signature genes')
    matrix=[]
    for expected,row in zip(GENES,values):
        if row.get('gene')!=expected or len(row.get('position',[]))!=1:
            raise ValueError('Ambiguous gene-to-probe mapping '+expected)
        v=np.asarray(row['scores'],dtype=float).reshape(-1)
        if len(v)!=len(selected) or not np.isfinite(v).all(): raise ValueError('Missing RNA gene values '+expected)
        matrix.append(np.maximum(np.exp2(v)-float(logunit.group(1)),0) if logunit else v)
    expression=pd.DataFrame(np.array(matrix).T,columns=GENES)
    expression['sample_id']=selected;expression['case_id']=[s[:12] for s in selected]
    expression.to_csv(EXT/'rna/signature_gene_TPM.csv',index=False)
    scores,parameters=cohort_scores(expression)
    scores.to_csv(EXT/'rna/patient_scores.csv',index=False)
    parameters.update(source='UCSC Xena public GDC mirror',dataset=dataset,host=host,source_unit=unit,
        metadata=metadata,query_source_sha256={p.name:digest(p) for p in destination.glob('*.xq')})
    save_json(parameters,EXT/'rna/score_parameters.json')
    return {'status':'complete','case':'TCGA-55-7725','source':'UCSC Xena GDC STAR TPM mirror','dataset':dataset,
        'source_metadata':metadata,'source_unit':unit,'cohort_patients':len(scores),
        'RNA_samples':[s for s in selected if s.startswith('TCGA-55-7725-01')],
        'target_scores':scores.loc[scores.case_id.eq('TCGA-55-7725')].to_dict('records'),
        'specimen_relationship':'Public primary-tumor sample barcodes identify the same patient; RNA is not asserted to be this diagnostic section',
        'cell_label_validation':False,'spatial_TLS_confirmation':False}


def run_rna():
    destination=EXT/'rna';destination.mkdir(parents=True,exist_ok=True)
    status={'status':'pending','case':'TCGA-55-7725','image_sample':'TCGA-55-7725-01Z-00-DX1',
            'scope':'Public TCGA-LUAD primary-tumor STAR Counts gene-level TPM; patient-level context',
            'cell_label_validation':False,'spatial_TLS_confirmation':False}
    # Signature-only query is the default: no need to download hundreds of full
    # gene matrices when the public mirror exposes exactly the requested genes.
    try:
        status=run_xena_rna()
        direct=destination/'cohort_file_manifest.json'
        if direct.exists():
            status['verified_direct_GDC_target_metadata']=[f for f in json.loads(direct.read_text())['data']['hits']
                if any(c.get('submitter_id')=='TCGA-55-7725' for c in f.get('cases',[]))]
        save_json(status,destination/'status.json');print('RNA signature complete:',status['cohort_patients'],'patients',flush=True)
        return status
    except Exception as error:
        status['Xena_GDC_mirror_issue']=str(error)
    filters={'op':'and','content':[
        {'op':'=','content':{'field':'cases.project.project_id','value':'TCGA-LUAD'}},
        {'op':'=','content':{'field':'data_type','value':'Gene Expression Quantification'}},
        {'op':'=','content':{'field':'analysis.workflow_type','value':'STAR - Counts'}},
        {'op':'=','content':{'field':'cases.samples.sample_type','value':'Primary Tumor'}},
        {'op':'=','content':{'field':'access','value':'open'}}]}
    url=gdc_files(filters,'file_id,file_name,md5sum,file_size,cases.submitter_id,cases.samples.submitter_id,cases.samples.sample_type,analysis.workflow_type')
    status['metadata_query']=url
    try:
        record=json.loads(fetch(url,destination/'cohort_file_manifest.json'))
        hits=record['data']['hits']
        if record['data']['pagination']['total']>len(hits): raise ValueError('Truncated GDC cohort manifest')
        target=[f for f in hits if any(c.get('submitter_id')=='TCGA-55-7725' for c in f.get('cases',[]))]
        status['target_file_metadata']=target
        if not target: raise ValueError('No public primary-tumor STAR TPM file for target case in returned cohort')
        def read_file(f):
            data=fetch('https://api.gdc.cancer.gov/data/'+f['file_id'],destination/'files'/f['file_name'],timeout=45)
            import hashlib
            if hashlib.md5(data).hexdigest()!=f['md5sum']: raise ValueError('GDC checksum mismatch '+f['file_id'])
            table=pd.read_csv(io.BytesIO(data),sep='\t',comment='#')
            if not {'gene_name','tpm_unstranded'}.issubset(table): raise ValueError('No gene-symbol TPM columns')
            table=table.loc[table.gene_name.isin(GENES)]
            if table.gene_name.duplicated().any() or set(table.gene_name)!=set(GENES): raise ValueError('Missing/ambiguous signature genes')
            c=f['cases'][0];samples=[s['submitter_id'] for s in c.get('samples',[]) if s.get('sample_type')=='Primary Tumor']
            if len(samples)!=1: raise ValueError('Ambiguous primary sample provenance '+f['file_id'])
            return {'case_id':c['submitter_id'],'sample_id':samples[0],'file_id':f['file_id'],**table.set_index('gene_name').tpm_unstranded.to_dict()}
        failures=[];rows=[]
        # Verify the target first; unavailable target must not generate a substitute score.
        rows += [read_file(f) for f in target]
        remaining=[f for f in hits if f not in target]
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures={pool.submit(read_file,f):f for f in remaining}
            for number, future in enumerate(as_completed(futures)):
                try: rows.append(future.result())
                except Exception as e:
                    failures.append({'file_id':futures[future]['file_id'],'error':str(e)})
                    for pending in futures: pending.cancel()
                    break
                if number%50==0: print(f'RNA cohort: {len(rows)}/{len(hits)} files retrieved',flush=True)
        save_json(failures,destination/'retrieval_failures.json')
        expression=pd.DataFrame(rows);expression.to_csv(destination/'signature_gene_TPM.csv',index=False)
        if failures: raise ValueError(f'{len(failures)} cohort files unavailable; cohort scoring pending to avoid incomplete-cohort percentiles')
        scores,metadata=cohort_scores(expression);scores.to_csv(destination/'patient_scores.csv',index=False)
        save_json(metadata,destination/'score_parameters.json')
        target_scores=scores[scores.case_id.eq('TCGA-55-7725')]
        status.update(status='complete',cohort_patients=len(scores),target_scores=target_scores.to_dict('records'),
            specimen_relationship='Same patient and primary-tumor sample type; RNA aliquot is not asserted to be this scanned diagnostic section',
            RNA_samples=expression.loc[expression.case_id.eq('TCGA-55-7725'),'sample_id'].tolist())
    except Exception as e:
        status['reason']=str(e)
        try:
            fallback=run_xena_rna();fallback['direct_GDC_retrieval_issue']=status['reason'];status=fallback
        except Exception as mirror_error:
            status['Xena_GDC_mirror_issue']=str(mirror_error)
    save_json(status,destination/'status.json');print('RNA:',status['status'],status.get('reason',''),flush=True)
    return status


def download_slide(file,path,attempts=4):
    """Keep partial downloads; append only with a verified HTTP range response."""
    import hashlib
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    expected=int(file['file_size']);partial=path.with_suffix(path.suffix+'.part')
    def md5(p):
        h=hashlib.md5()
        with p.open('rb') as stream:
            for block in iter(lambda:stream.read(1024**2),b''): h.update(block)
        return h.hexdigest()
    if path.exists():
        if path.stat().st_size==expected and md5(path)==file['md5sum']: return path
        raise ValueError('Existing public slide checksum mismatch; preserve for inspection')
    errors=[]
    for attempt in range(attempts):
        offset=partial.stat().st_size if partial.exists() else 0
        if offset==expected:
            if md5(partial)!=file['md5sum']: raise ValueError('Partial public slide checksum mismatch')
            partial.replace(path);return path
        headers={'User-Agent':'HE2CELL-research/2','Cache-Control':'no-cache','Accept-Encoding':'identity'}
        if offset: headers['Range']=f'bytes={offset}-'
        try:
            request=urllib.request.Request('https://api.gdc.cancer.gov/data/'+file['file_id'],headers=headers)
            with urllib.request.urlopen(request,timeout=45) as response:
                if offset and response.status==206:
                    if not valid_content_range(response.headers.get('Content-Range',''),offset,expected):
                        raise ValueError('Invalid Content-Range '+str(response.headers.get('Content-Range'))+f' for requested offset {offset}; never append unverified bytes')
                    mode='ab'
                else:
                    mode='wb';offset=0
                with partial.open(mode) as output:
                    while True:
                        block=response.read(256*1024)
                        if not block: break
                        output.write(block)
            size=partial.stat().st_size
            print(file['file_name'],f'{size:,}/{expected:,} bytes',flush=True)
            if size==expected and md5(partial)==file['md5sum']:
                partial.replace(path);return path
            if size>expected: raise ValueError('Public slide exceeds declared size')
            errors.append(f'Incomplete transfer: {size}/{expected} bytes')
        except Exception as e: errors.append(str(e))
    raise RuntimeError('; '.join(errors))


def valid_content_range(header,offset,total):
    """GDC emits start-end/total without the standard 'bytes ' prefix."""
    import re
    match=re.fullmatch(r'(?:bytes\s+)?(\d+)-(\d+)/(\d+)',str(header).strip())
    return bool(match and int(match[1])==offset and offset<=int(match[2])<total and int(match[3])==total)


def run_annotations():
    destination=EXT/'region_reference';destination.mkdir(parents=True,exist_ok=True)
    status={'status':'pending','source':'https://zenodo.org/records/10614928','selection_seed':569,
       'requested_slides':5,'excluded_patient':'TCGA-55-7725','individual_cell_phenotypes_available':False,
       'evaluation_domain':'Explicitly annotated TLS/REST; unannotated pixels are not verified negatives',
       'training_overlap':'unverified; no independent performance claim'}
    try:
        metadata=json.loads(fetch('https://zenodo.org/api/records/10614928',destination/'release_metadata.json'))
        status['release_title']=metadata['metadata']['title']
        status['release_files']=[{'name':f['key'],'size':f['size'],'checksum':f.get('checksum')} for f in metadata.get('files',[])]
        annotations=[]
        for f in metadata.get('files',[]):
            if f['size']>500_000_000: continue
            if not f['key'].lower().endswith(('.zip','.json','.csv','.yaml','.yml')): continue
            print('Downloading public annotation archive:',f['key'],f['size'],'bytes',flush=True)
            data=fetch(f['links']['self'],destination/'downloads'/Path(f['key']).name,timeout=60)
            import hashlib
            expected=f.get('checksum','')
            if expected.startswith('md5:') and hashlib.md5(data).hexdigest()!=expected.split(':',1)[1]:
                raise ValueError('Annotation release checksum mismatch')
            if f['key'].endswith('.zip'):
                with zipfile.ZipFile(io.BytesIO(data)) as package:
                    # The release contains two coordinate versions. Only original SVS
                    # coordinates may be paired with the original TCGA image.
                    for name in package.namelist():
                        if name.endswith(('.xml','.json','.csv','.yaml','.yml','.md')) and not name.startswith('__MACOSX/') and package.getinfo(name).file_size<10_000_000:
                            if name.endswith('.xml') and '/annotations_svs_spacings/' not in name: continue
                            target=destination/'annotations'/Path(name).name
                            target.parent.mkdir(exist_ok=True);target.write_bytes(package.read(name));annotations.append(target)
            else: annotations.append(destination/'downloads'/Path(f['key']).name)
        names=[p.name for p in annotations]
        save_json({'files':names,'coordinate_version':'Original SVS spacing; approx 0.5 µm annotations excluded'},destination/'annotation_inventory.json')
        category=json.loads((destination/'annotations/annotations.json').read_text())
        status['organ_counts']={k:len(v) for k,v in category.items()}
        status['release_readme']=(destination/'annotations/readme.md').read_text()
        status['organ']='TCGA-LUSC, cross-cancer comparison to this LUAD case'
        status['training_overlap']='Release README explicitly states used for HookNet development; no test split supplied'
        lung=[destination/'annotations'/n for n in sorted(category['LUSC']) if 'TCGA-55-7725' not in n]
        rng=np.random.default_rng(569)
        ordered=[lung[i] for i in rng.permutation(len(lung))]
        selected=[];patients=set()
        import re
        for path in ordered:
            match=re.search(r'TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}',path.name)
            if match and match.group(0) not in patients:
                import xml.etree.ElementTree as ET
                root=ET.parse(path).getroot();counts={}
                for annotation in root.findall('./Annotations/Annotation'):
                    group=annotation.attrib.get('PartOfGroup','other');counts[group]=counts.get(group,0)+1
                selected.append({'patient_id':match.group(0),'slide_barcode':path.stem,'annotation_path':str(path),
                    'annotation_sha256':digest(path),'organ':'LUSC','training_overlap':'Used for development; exact fold unknown',
                    'TLS_annotations':counts.get('tls',0),'GC_annotations':counts.get('gc',0),'explicit_other_annotations':counts.get('other',0),
                    'selection_status':'Seeded unique-patient LUSC selection from release manifest; no test split provided'})
                patients.add(match.group(0))
            if len(selected)==5: break
        pd.DataFrame(selected).to_csv(destination/'pilot_manifest.csv',index=False)
        status['available_annotation_files']=sum(p.suffix=='.xml' for p in annotations);status['selected_pilot']=selected
        def resolve_image(row):
            filters={'op':'and','content':[{'op':'=','content':{'field':'cases.submitter_id','value':row['patient_id']}},
              {'op':'=','content':{'field':'data_type','value':'Slide Image'}},{'op':'=','content':{'field':'access','value':'open'}}]}
            url=gdc_files(filters,'file_id,file_name,md5sum,file_size,cases.submitter_id,cases.project.project_id',size=100)
            try:
                result=json.loads(fetch(url,destination/'image_metadata'/f"{row['patient_id']}.json",timeout=20))
                matches=[f for f in result['data']['hits'] if f['file_name'].startswith(row['slide_barcode']+'.') and f['file_name'].endswith('.svs')]
                if len(matches)!=1: raise ValueError('Image pairing ambiguous or unavailable')
                file=matches[0]
                if file['cases'][0]['project']['project_id']!='TCGA-LUSC': raise ValueError('Organ mismatch')
                row['image_metadata']=file
                path=destination/'slides'/file['file_name']
                download_slide(file,path)
                row['image_path']=str(path);row['image_status']='retrieved and checksummed'
            except Exception as error: row['image_status']='pending';row['image_issue']=str(error)
            return row
        with ThreadPoolExecutor(max_workers=2) as pool: selected=list(pool.map(resolve_image,selected))
        save_json(selected,destination/'pilot_manifest.json');pd.DataFrame(selected).to_csv(destination/'pilot_manifest.csv',index=False)
        status['selected_pilot']=selected
        if any('image_path' in row for row in selected):
            helper=ROOT/'deployment/tls_methods_v2_reference.py'
            import subprocess
            subprocess.run([str(ROOT/'.runtime/hooknet_tls/venv/bin/python'),str(helper)],check=True)
            complete=all('image_path' in row for row in selected)
            status['status']='annotation_concordance_complete' if complete else 'partial_annotation_concordance'
            status['reference_metrics']=pd.read_csv(destination/'pilot_annotation_concordance.csv').to_dict('records')
            status['reason']='Available image/annotation pairs evaluated; remaining downloads pending. Development overlap prevents independent performance claims' if not complete else 'Five image/annotation pairs evaluated; development overlap prevents independent performance claims'
        else:
            status['reason']='Reference archive checked and five unique LUSC cases selected; original WSI retrieval unavailable. No scores are inferred from annotated thumbnails.'
    except Exception as e: status['reason']=str(e)
    save_json(status,destination/'status.json');print('Region reference:',status['status'],status.get('reason',''),flush=True)
    return status


def main():
    EXT.mkdir(parents=True,exist_ok=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs=[pool.submit(run_rna),pool.submit(run_annotations)]
        results=[j.result() for j in jobs]
    save_json({'rna':results[0],'region_reference':results[1]},EXT/'status.json')


if __name__=='__main__': main()
