"""Functional checks for averaging, physical morphology, sampling and references."""
import inspect
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np
import pandas as pd
from shapely.geometry import box

from tls_methods_v2_core import ROOT,OUT,RUN,CLASSES,load_source,field_components,label_polygon
from tls_methods_v2_review import stratified_sample
from tls_methods_v2_external import cohort_scores,GENES,CHEMOKINES,valid_content_range
from tls_methods_v2_reference import concordance,reference_polygons
from tls_tools import finder_input,r_input,load_finder


class RevisedMethodsTests(unittest.TestCase):
    def test_actual_D4_and_softmax_averaging(self):
        # Execute the actual classifier helper in its existing PyTorch runtime.
        program='''
import sys,torch,numpy as np
sys.path.insert(0,'deployment')
from tls_methods_v2_core import d4,average_views
x=torch.arange(16,dtype=torch.float32).reshape(1,1,4,4).repeat(1,4,1,1)
views=d4(x)
assert len({v.numpy().tobytes() for v in views})==8
assert all(torch.equal(v[:,0],v[:,3]) for v in views)
for v in views:
 assert {a.numpy().tobytes() for a in d4(v)}=={a.numpy().tobytes() for a in views}
class Toy(torch.nn.Module):
 def forward(self,v):
  corner=v[:,0,0,0]/5
  return torch.stack([corner,torch.zeros_like(corner),torch.zeros_like(corner),torch.zeros_like(corner)],1)
p,individual=average_views(Toy(),x)
corners=torch.tensor([0.,3.,12.,15.])/5
expected=torch.mean(torch.exp(corners)/(torch.exp(corners)+3))
assert torch.allclose(p[0,0],expected)
assert abs(float(p[0,0])-float(torch.softmax(torch.tensor([1.5,0.,0.,0.]),0)[0]))>.01
'''
        subprocess.run([str(ROOT/'.venv/bin/python'),'-c',program],cwd=ROOT,check=True,capture_output=True,text=True)

    def test_saved_TTA_complete_alignment_and_probabilities(self):
        cells=pd.read_csv(OUT/'tta_cells.csv');raw,_=load_source()
        np.testing.assert_array_equal(cells.global_instance_id,raw.global_instance_id)
        np.testing.assert_allclose(cells[['x_um','y_um']],raw[['x_um','y_um']],atol=1e-8)
        p=cells[['prob_'+c for c in CLASSES]].to_numpy()
        views=np.load(OUT/'tta_view_probabilities.npy',mmap_mode='r')
        np.testing.assert_allclose(p,views.mean(1),atol=1e-7)
        np.testing.assert_allclose(p.sum(1),1,atol=1e-6)
        np.testing.assert_array_equal(p.argmax(1),cells.prediction)
        self.assertEqual(len(cells),129470)
        saved=json.loads((OUT/'tta_summary.json').read_text())
        self.assertLess(saved['D4_invariance_max_probability_delta'],2e-5)
        self.assertLess(saved['CPU_MPS_TTA_max_probability_delta'],2e-4)
        self.assertEqual(saved['identity_repeat_label_agreement'],1)

    def fixture(self,kind):
        rng=np.random.default_rng(22)
        if kind=='compact': cluster=rng.normal([500,500],[55,55],size=(700,2))
        elif kind=='elongated': cluster=np.column_stack([rng.uniform(300,700,700),rng.normal(500,25,700)])
        elif kind=='adjacent': cluster=np.concatenate([rng.normal([455,500],25,(350,2)),rng.normal([545,500],25,(350,2))])
        else: cluster=rng.uniform(30,970,(700,2))
        background=rng.uniform(20,980,(300,2))
        xy=np.concatenate([cluster,background]);areas=np.concatenate([rng.uniform(9,13,700),rng.uniform(25,35,300)])
        return xy,areas,np.ones((100,100),dtype=bool)

    def test_compact_elongated_and_adjacent_components(self):
        for kind in ['compact','elongated','adjacent']:
            xy,areas,tissue=self.fixture(kind)
            rows,labels,density,mean_area,settings=field_components(xy,areas,tissue)
            retained=[r for r in rows if r['retained']]
            self.assertGreaterEqual(len(retained),1,kind)
            self.assertEqual(len(retained),1,kind+' should not be split around peaks')
            self.assertGreater(retained[0]['all_objects'],50)
            self.assertGreater(retained[0]['area_um2'],5000)
            self.assertTrue(np.isfinite(density).all())
            self.assertNotIn('celltype',inspect.signature(field_components).parameters)

    def test_dispersed_and_empty_results(self):
        xy,areas,tissue=self.fixture('dispersed')
        rows,*_=field_components(xy,areas,tissue)
        self.assertEqual(sum(r['retained'] for r in rows),0)
        rows,labels,density,area,settings=field_components(np.empty((0,2)),np.array([]),tissue)
        self.assertEqual(rows,[]);self.assertEqual(labels.sum(),0)

    def test_physical_polygon_area_holes_and_tissue_edges(self):
        labels=np.zeros((8,8),dtype=int);labels[1:6,1:6]=1;labels[3,3]=0
        g=label_polygon(labels,1,spacing=10)
        self.assertEqual(g.area,2400)
        self.assertFalse(g.covers(box(31,31,39,39)))
        xy,areas,tissue=self.fixture('compact');tissue[:,:50]=False
        rows,_,density,_,_=field_components(xy,areas,tissue)
        self.assertTrue(np.isfinite(density).all())

    def test_safe_tool_mapping_and_upstream_equivalence(self):
        fixture=pd.DataFrame({'global_instance_id':np.arange(100),'x_um':np.arange(100)%10,
            'y_um':np.arange(100)//10,'celltype':['B','T','Other','Macrophage']*25,'confidence':.8})
        inp=r_input(fixture)
        self.assertEqual(int(inp.phenotype.str.contains('T',case=False).sum()),25)
        self.assertEqual(int(inp.phenotype.str.contains('B',case=False).sum()),25)
        adapted,reference=load_finder();generic=finder_input(fixture)
        aliases=generic.copy();aliases.Phenotype=aliases.Phenotype.replace({'B':'CD20','T':'CD8'})
        pd.testing.assert_frame_equal(adapted(generic,160),reference(aliases,160))
        # Valid empty output; no assertion forces the new slide to have detections.
        none=generic.copy();none.Phenotype='Unclassified'
        self.assertTrue(adapted(none,160).empty)

    def test_reproducible_review_sampling_and_complete_groups(self):
        f=pd.DataFrame({'area_um2':np.arange(120)+1.,'raw_confidence':np.random.default_rng(2).random(120)})
        a=stratified_sample(f,50,np.random.default_rng(569))
        b=stratified_sample(f,50,np.random.default_rng(569))
        np.testing.assert_array_equal(a.index,b.index);self.assertEqual(len(a),50)
        selected=pd.read_csv(OUT/'audit_samples_unblinded.csv')
        counts=selected.audit_group.value_counts().to_dict()
        self.assertEqual(counts,{'Macrophage_called':63,'A1_Other':50,'T_epithelium':50,'A2_Other':50,'Macrophage_near_miss':20})
        self.assertFalse(selected.blind_id.duplicated().any())

    def test_rna_patient_aggregation_and_missing_genes(self):
        rng=np.random.default_rng(5)
        f=pd.DataFrame(rng.lognormal(2,1,(22,14)),columns=GENES)
        f['case_id']=[f'P{i:02}' for i in range(21)]+['P00'];f['sample_id']=[f'S{i:02}' for i in range(22)]
        result,metadata=cohort_scores(f)
        expected=np.median(np.log2(f.loc[f.case_id.eq('P00'),'CXCL13']+1))
        self.assertAlmostEqual(float(result.loc[result.case_id.eq('P00'),'CXCL13'].iloc[0]),expected)
        self.assertEqual(metadata['patients'],21)
        self.assertGreater(np.corrcoef(result.chemokine_PC1,result[CHEMOKINES].mean(1))[0,1],0)
        broken=f.copy();broken.loc[0,'CXCL13']=np.nan
        with self.assertRaises(ValueError): cohort_scores(broken)

    def test_reference_units_and_unknown_pixels_not_negative(self):
        xml='<ASAP_Annotations><Annotations><Annotation PartOfGroup="tls"><Coordinates>'+''.join(
            f'<Coordinate Order="{i}" X="{x}" Y="{y}"/>' for i,(x,y) in enumerate([(0,0),(40,0),(40,40),(0,40)]))+'</Coordinates></Annotation></Annotations></ASAP_Annotations>'
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'sample.xml';path.write_text(xml)
            polygons=reference_polygons(path,.25)
        self.assertAlmostEqual(polygons['tls'][0].area,100)
        score=concordance([box(0,0,10,10)],[box(0,0,10,10),box(100,100,200,200)],[])
        self.assertEqual(score['TLS_area_Dice_within_annotation_domain'],1)
        self.assertFalse(score['unannotated_pixels_evaluated_as_negative'])
        self.assertFalse(score['independent_performance_claim'])

    def test_large_slide_remap_float32_and_small_slide_equivalence(self):
        from run_hooknet_tls import physical_remap_grid
        import cv2
        for col,row in [(0,0),(70000,90000)]:
            x,y=physical_remap_grid(8,8,col,row,2.,2.,2*col,2*row)
            self.assertEqual(x.dtype,np.float32);self.assertEqual(y.dtype,np.float32)
            self.assertAlmostEqual(float(x[0,0]),.5)
            result=cv2.remap(np.zeros((20,20,3),dtype='uint8'),x,y,cv2.INTER_LINEAR)
            self.assertEqual(result.shape,(8,8,3))
        oldx,oldy=np.meshgrid((np.arange(8,dtype=np.float32)+.5)*2-.5,(np.arange(8,dtype=np.float32)+.5)*2-.5)
        x,y=physical_remap_grid(8,8,0,0,2.,2.,0,0)
        np.testing.assert_array_equal(x,oldx);np.testing.assert_array_equal(y,oldy)

    def test_GDC_range_offsets_lengths_and_integrity(self):
        self.assertTrue(valid_content_range('bytes 123-199/200',123,200))
        self.assertTrue(valid_content_range('123-199/200',123,200))
        self.assertFalse(valid_content_range('124-199/200',123,200))
        self.assertFalse(valid_content_range('123-199/201',123,200))
        self.assertFalse(valid_content_range('0-199/200',123,200))
        self.assertFalse(valid_content_range('123-200/200',123,200))


if __name__=='__main__': unittest.main()
