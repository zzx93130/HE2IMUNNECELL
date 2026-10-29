"""Storage reader checks for padding, rounding, relocation and source integrity."""
from pathlib import Path
import sys
import tempfile
import unittest
import io

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'script/src'))
import h5py
import numpy as np
from PIL import Image
from result_store import SlideResults, sha256
from run_cellcrops import crop_with_padding


class ReaderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.source = self.root/'slide.png'
        self.rgb = np.random.default_rng(1).integers(0,256,(20,24,3),dtype=np.uint8)
        Image.fromarray(self.rgb).save(self.source)
        self.patch = self.rgb[3:13,4:14]
        self.labels = np.zeros((10,10),np.uint32)
        self.labels[0:4,0:4] = 1
        self.labels[6:10,6:10] = 2
        self.centers = [(0.,0.,1,6),(9.,9.,2,5),(2.5,2.5,1,5),(3.5,3.5,1,6)]
        self.path = self.root/'test.h5'
        with h5py.File(self.path,'w') as h:
            h.attrs.update(schema='he2cell.slide-results.v1',mode='on-demand',complete=True,
                           source_slide='/old/location/slide.png',source_sha256=sha256(self.source))
            h.create_dataset('patches/labels',data=self.labels[None],compression='gzip')
            idx = h.create_group('index')
            for k,v in dict(patch_row=[0]*4,patch_x=[4]*4,patch_y=[3]*4,
                            cx=[c[0] for c in self.centers],cy=[c[1] for c in self.centers],
                            label=[c[2] for c in self.centers],size=[c[3] for c in self.centers]).items():
                idx.create_dataset(k,data=v)

    def tearDown(self):
        self.tmp.cleanup()

    def test_reconstruction_padding_rounding_relocation(self):
        with SlideResults(self.path,self.source) as results:
            for i,(cx,cy,label,size) in enumerate(self.centers):
                rgb,mask=results.read_cell(i)
                expected_rgb=crop_with_padding(self.patch,cx,cy,size,pad_value=255)
                expected_mask=crop_with_padding((self.labels==label).astype(np.uint8)*255,cx,cy,size)
                np.testing.assert_array_equal(rgb,expected_rgb)
                np.testing.assert_array_equal(mask,expected_mask)
            with self.assertRaises(IndexError):
                results.read_cell(-1)
            with self.assertRaises(IndexError):
                results.read_cell(4)

    def test_wrong_source_rejected(self):
        bad=self.root/'wrong.png'
        Image.fromarray(np.zeros_like(self.rgb)).save(bad)
        with self.assertRaisesRegex(ValueError,'checksum'):
            SlideResults(self.path,bad)

    def test_incomplete_rejected(self):
        with h5py.File(self.path,'r+') as h:
            h.attrs['complete']=False
        with self.assertRaisesRegex(ValueError,'Incomplete'):
            SlideResults(self.path,self.source)

    def test_full_archive_independent_of_source(self):
        rgb=np.stack([self.patch[:5,:5]]*4)
        masks=np.zeros((4,5,5),np.uint8)
        with h5py.File(self.path,'r+') as h:
            h.attrs['mode']='full'
            h.create_dataset('cells/rgb',data=rgb)
            h.create_dataset('cells/mask',data=masks)
        with SlideResults(self.path) as results:
            a,b=results.read_cell(0)
            np.testing.assert_array_equal(a,rgb[0])
            np.testing.assert_array_equal(b,masks[0])

    def test_encoded_archive_independent_of_source(self):
        expected = (self.patch[:5,:5], (self.labels[:5,:5] > 0).astype(np.uint8)*255)
        with h5py.File(self.path,'r+') as h:
            h.attrs['mode']='encoded'
            for name,array in zip(('rgb','mask'),expected):
                buf=io.BytesIO()
                Image.fromarray(array).save(buf,format='PNG')
                blob=buf.getvalue()
                h.create_dataset('cells/'+name+'_encoded',data=np.frombuffer(blob*4,np.uint8),compression='gzip')
                h.create_dataset('cells/'+name+'_offsets',data=np.arange(5)*len(blob))
        with SlideResults(self.path) as results:
            for i in range(4):
                for actual,wanted in zip(results.read_cell(i),expected):
                    np.testing.assert_array_equal(actual,wanted)


if __name__ == '__main__':
    unittest.main()
