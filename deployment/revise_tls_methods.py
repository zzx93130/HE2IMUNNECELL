"""One repeatable entry point for the saved slide's revised TLS methods.

Run with .venv/bin/python. GPU classification uses that environment; geometry,
figures and the R adapters reuse the existing isolated HookNet runtime. The
original predictions and algorithms' saved outputs are never overwritten.
"""
from __future__ import annotations
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime,timezone

from tls_methods_v2_core import ROOT,RUN,OUT,save_json,digest,load_source


def analysis():
    import pandas as pd
    from tls_methods_v2_core import run_components,run_tools,hook_regions,compare_regions
    from tls_methods_v2_review import prepare_audit,make_cards,artifact_diagnostics,create_figures,render_report
    cells=pd.read_csv(OUT/'tta_cells.csv');features=pd.read_csv(OUT/'object_features.csv')
    components,geoms,_,_=run_components(cells,features)
    tr,tg,tm=run_tools(cells);hr,hg,hm,gc=hook_regions(cells)
    compare_regions(cells,components,geoms,tr,tg,tm,hr,hg,hm)
    samples,merged=prepare_audit(cells,features);make_cards(samples)
    merged,seams=artifact_diagnostics(merged)
    create_figures(cells,merged,components,geoms,hg,gc,seams)
    render_report()


def write_provenance():
    raw,manifest=load_source()
    original=[Path(manifest['slides'][0]['results']),RUN/'tls_analysis/spatial_cells.csv',
              RUN/'tls_analysis/ranked_candidate_regions.csv',RUN/'tls_tool_comparison/A1_A6_comparison.csv',
              RUN/'tls_tool_comparison/hooknet_tls/native_polygons.geojson']
    input_hashes={str(p.relative_to(ROOT)):digest(p) for p in original}
    previous=OUT/'provenance.json'
    if previous.exists():
        assert json.loads(previous.read_text())['original_input_sha256']==input_hashes,'Original saved inputs changed'
    packages={}
    for name in ['numpy','pandas','scipy','scikit-image','shapely','torch','torchvision','openslide-python','Pillow']:
        try: packages[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: pass
    environments={}
    for name,python in [('classification',ROOT/'.venv/bin/python'),('geometry_report',ROOT/'.runtime/hooknet_tls/venv/bin/python')]:
        locked=subprocess.run([str(python),'-m','pip','freeze'],capture_output=True,text=True,check=True).stdout
        (OUT/f'{name}_environment.txt').write_text(locked)
        environments[name]=locked
    value={'method_version':2,'recorded_UTC':datetime.now(timezone.utc).isoformat(),
        'original_input_sha256':input_hashes,'classification_mapping':['Other','T','Macrophage','B'],
        'mapping_authority':'Exact checkpoint mapping confirmed by user 2026-10-13',
        'classifier_sha256':digest(manifest['classifier']),'segmentation_provenance':json.loads((ROOT/'deployment/segmentation-model.json').read_text()),
        'Python':sys.version,'packages_in_recording_runtime':packages,'environment_locks':environments,
        'classification_runtime':str(ROOT/'.venv/bin/python'),
        'geometry_and_report_runtime':str(ROOT/'.runtime/hooknet_tls/venv/bin/python'),
        'runtime_decision':'Reuse existing isolated polygon runtime; additional Shapely download failed checksum and was not installed',
        'coordinates':'Original level-0 centroids × 0.252 µm; all 129470 frozen deduplicated IDs',
        'no_cell_reference_labels':True,'no_unlabelled_calibration':True,
        'source_sha256':{p.name:digest(p) for p in [*(ROOT/'deployment').glob('*tls_methods*.py'),
            ROOT/'script/src/prediction_tta.py',ROOT/'script/src/model_eval.py',ROOT/'he2cell.py',ROOT/'deployment/run_hooknet_tls.py']}}
    if (OUT/'tools/summary.json').exists(): value['spatial_tool_provenance']=json.loads((OUT/'tools/summary.json').read_text())
    if (RUN/'tls_tool_comparison/hooknet_tls/provenance.json').exists():
        value['reused_HookNet_provenance']=json.loads((RUN/'tls_tool_comparison/hooknet_tls/provenance.json').read_text())
    save_json(value,previous)


def link_dashboard():
    import shutil
    page=RUN/'tls_tool_comparison/TLS_tool_comparison.html'
    backup=page.with_name('TLS_tool_comparison_before_methods_v2.html')
    if not backup.exists(): shutil.copy2(page,backup)
    text=page.read_text()
    marker='id="methods-v2-link"'
    if marker not in text:
        banner='<div id="methods-v2-link" style="padding:20px;background:#e3f3f0;color:#173c43;font:17px system-ui"><b>Revised methods available:</b> <a href="tls_methods_v2/TLS_methods_v2_report.html">Open the cell-label audit and morphology-based region report</a>. This page preserves the earlier raw-label comparisons.</div>'
        import re
        text=re.sub(r'(<body[^>]*>)',lambda m:m.group(1)+banner,text,count=1,flags=re.I)
        page.write_text(text)
    alias=page.parent/'tls_methods_v2'
    if not alias.exists(): alias.symlink_to(OUT,target_is_directory=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['all','tta','features','analysis','report','external','verify'],default='all')
    parser.add_argument('--force-tta',action='store_true')
    parser.add_argument('--offline',action='store_true',help='Reuse existing public data statuses without making network requests')
    args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    if args.stage in ['all','tta','features']:
        from tls_methods_v2_core import run_tta,extract_object_features
        if args.stage in ['all','tta']: run_tta(force=args.force_tta)
        if args.stage in ['all','features']: extract_object_features()
    isolated=ROOT/'.runtime/hooknet_tls/venv/bin/python'
    def isolated_stage(stage):
        subprocess.run([str(isolated),str(Path(__file__)), '--stage',stage,'--offline'],check=True,cwd=ROOT)
    if args.stage=='analysis':
        if Path(sys.prefix).resolve() != (ROOT/'.runtime/hooknet_tls/venv').resolve(): isolated_stage('analysis')
        else: analysis()
    if args.stage=='all': isolated_stage('analysis')
    if args.stage in ['all','external'] and not args.offline:
        subprocess.run([str(ROOT/'.venv/bin/python'),str(ROOT/'deployment/tls_methods_v2_external.py')],check=True,cwd=ROOT)
    if args.stage in ['all','verify']:
        if Path(sys.prefix).resolve() != (ROOT/'.runtime/hooknet_tls/venv').resolve(): isolated_stage('verify')
        else:
            import unittest
            suite=unittest.defaultTestLoader.discover(str(ROOT/'deployment'),pattern='test_tls_methods_v2.py')
            result=unittest.TextTestRunner(verbosity=2).run(suite)
            if not result.wasSuccessful(): raise SystemExit(1)
            save_json({'meaningful_tests':result.testsRun,'passed':True,'TTA_checks':json.loads((OUT/'tta_summary.json').read_text()),
                'original_IDs_and_coordinate_units_verified':True,
                'blinded_expert_review':'pending','cell_accuracy':'not established'},OUT/'verification.json')
    if args.stage in ['all','report']:
        if Path(sys.prefix).resolve() != (ROOT/'.runtime/hooknet_tls/venv').resolve(): isolated_stage('report')
        else:
            from tls_methods_v2_review import render_report,plot_RNA
            plot_RNA()
            print(render_report())
    write_provenance()
    if args.stage in ['all','report']: link_dashboard()


if __name__=='__main__': main()
