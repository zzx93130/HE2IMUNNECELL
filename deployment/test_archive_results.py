"""Archive lifecycle, deletion guards, empty results and portable export tests."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'script/src'))
import h5py
import numpy as np
import pandas as pd
from PIL import Image
import tifffile
from archive_results import archive_slide,export_archive
from result_store import SlideResults
from run_cellcrops import crop_with_padding


class ArchiveTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.base=Path(self.tmp.name).resolve()
        self.root=self.base/'result'
        self.root.mkdir()
        self.source=self.base/'slide.tif'
        self.image=np.random.default_rng(12).integers(0,255,(32,32,3),dtype=np.uint8)
        Image.fromarray(self.image).save(self.source)
        self.patch=self.root/'patches/slide/folder_1/patch0_x4_y3.png'
        self.patch.parent.mkdir(parents=True)
        Image.fromarray(self.image[3:19,4:20]).save(self.patch)
        self.mask=self.root/'cells/patch_masks/slide/patch0_x4_y3_cp_mask.tif'
        self.mask.parent.mkdir(parents=True)
        self.labels=np.zeros((16,16),np.uint32)
        self.labels[4:8,5:10]=1
        tifffile.imwrite(self.mask,self.labels,compression='zlib')
        self.patch_index=self.root/'patches/tables/slide_patch_index.csv'
        self.patch_index.parent.mkdir()
        self.patches=pd.DataFrame([dict(path=str(self.patch),num_patch=0,source_tif=str(self.source),
                                        x=4,y=3,patch_size=16,level=0)])
        self.patches.to_csv(self.patch_index,index=False)
        self.rgbpath=self.root/'cells/single_cells/cell_images_8/cell.png'
        self.maskpath=self.root/'cells/single_cells/cell_masks_8/cell.png'
        self.rgbpath.parent.mkdir(parents=True)
        self.maskpath.parent.mkdir(parents=True)
        self.rgb=crop_with_padding(self.image[3:19,4:20],7,5.5,8,pad_value=255)
        self.cellmask=crop_with_padding((self.labels==1).astype(np.uint8)*255,7,5.5,8)
        Image.fromarray(self.rgb).save(self.rgbpath)
        Image.fromarray(self.cellmask).save(self.maskpath)
        self.cells=pd.DataFrame([dict(global_instance_id=42,source_tif=str(self.source),patch_id=0,
                  patch_x=4,patch_y=3,centroid_x_local=7.,centroid_y_local=5.5,local_label=1,
                  cell_crop_size=8,cell_crop_status='ok',cell_image_crop_path=str(self.rgbpath),
                  cell_mask_crop_path=str(self.maskpath),patch_path=str(self.patch),patch_mask_path=str(self.mask),
                  celltype='B',prediction=3,confidence=.7123456789)])
        self.cells.to_csv(self.root/'cell_types.csv',index=False)
        (self.root/'review.txt').write_text('review retained exactly\n')
        self.original_files=[p for p in self.root.rglob('*') if p.is_file()]

    def tearDown(self):
        self.tmp.cleanup()

    def test_archive_cleanup_and_relocated_export(self):
        summary=archive_slide(self.root,self.source)
        archive=Path(summary['archive'])
        self.assertEqual(list(self.root.iterdir()),[archive])
        self.assertTrue(summary['loose_files_removed'])
        self.assertTrue(self.source.exists())
        with SlideResults(archive) as store:
            self.assertTrue(store.h5.attrs['verified'])
            self.assertEqual(store.table().global_instance_id.tolist(),[42])
            for actual,expected in zip(store.read_cell(0),(self.rgb,self.cellmask)):
                np.testing.assert_array_equal(actual,expected)
        relocated=self.source.with_name('relocated.tif')
        self.source.rename(relocated)
        output=export_archive(archive,self.base/'export',relocated)
        table=pd.read_csv(output/'cell_types.csv')
        self.assertEqual(table.source_tif.iloc[0],str(relocated))
        for column,expected in [('cell_image_crop_path',self.rgb),('cell_mask_crop_path',self.cellmask)]:
            path=Path(table[column].iloc[0])
            self.assertTrue(path.is_relative_to(output))
            np.testing.assert_array_equal(np.asarray(Image.open(path)),expected)
        np.testing.assert_array_equal(tifffile.imread(table.patch_mask_path.iloc[0]),self.labels)
        self.assertEqual((output/'review.txt').read_text(),'review retained exactly\n')
        with self.assertRaisesRegex(ValueError,'empty'):
            export_archive(archive,output,relocated)

    def test_metadata_export_without_slide(self):
        summary=archive_slide(self.root,self.source)
        self.source.rename(self.base/'moved.tif')
        output=export_archive(summary['archive'],self.base/'tables',tables_only=True)
        self.assertEqual(pd.read_csv(output/'cell_types.csv').celltype.tolist(),['B'])
        self.assertFalse((output/'cells/single_cells').exists())

    def test_mismatch_keeps_all_loose_files(self):
        Image.fromarray(np.zeros_like(self.rgb)).save(self.rgbpath)
        with self.assertRaisesRegex(ValueError,'reconstruction mismatch'):
            archive_slide(self.root,self.source)
        self.assertTrue(all(p.is_file() for p in self.original_files))
        self.assertFalse((self.root/'results.h5').exists())
        self.assertTrue(self.source.exists())

    def test_cleanup_failure_leaves_verified_archive(self):
        with patch.object(Path,'unlink',side_effect=OSError('simulated cleanup failure')):
            with self.assertRaisesRegex(OSError,'cleanup failure'):
                archive_slide(self.root,self.source)
        self.assertTrue(all(p.exists() for p in self.original_files))
        with SlideResults(self.root/'results.h5') as store:
            self.assertTrue(store.h5.attrs['verified'])
            np.testing.assert_array_equal(store.read_cell(0)[0],self.rgb)

    def test_empty_cells_and_no_patches(self):
        self.cells.iloc[:0].to_csv(self.root/'cell_types.csv',index=False)
        self.patches.iloc[:0].to_csv(self.patch_index,index=False)
        for p in (self.patch,self.mask,self.rgbpath,self.maskpath):
            p.unlink()
        summary=archive_slide(self.root,self.source,patch_size=16)
        with SlideResults(summary['archive']) as store:
            self.assertEqual(len(store),0)
            self.assertTrue(store.table().empty)
            self.assertEqual(store.labels.shape,(0,16,16))
        export_archive(summary['archive'],self.base/'empty-export')

    def test_zero_cells_with_patch(self):
        self.cells.iloc[:0].to_csv(self.root/'cell_types.csv',index=False)
        self.rgbpath.unlink()
        self.maskpath.unlink()
        summary=archive_slide(self.root,self.source)
        with SlideResults(summary['archive']) as store:
            self.assertEqual(len(store),0)
            np.testing.assert_array_equal(store.labels[0],self.labels)

    def test_outside_reference_and_symlinks_rejected(self):
        self.cells.loc[0,'cell_image_crop_path']=str(self.source)
        self.cells.to_csv(self.root/'cell_types.csv',index=False)
        with self.assertRaisesRegex(ValueError,'outside'):
            archive_slide(self.root,self.source)
        self.assertTrue(all(p.exists() for p in self.original_files))
        (self.root/'link').symlink_to(self.source)
        with self.assertRaisesRegex(ValueError,'regular file'):
            archive_slide(self.root,self.source)

    def test_keep_files(self):
        summary=archive_slide(self.root,self.source,cleanup=False)
        self.assertFalse(summary['loose_files_removed'])
        self.assertTrue(all(p.exists() for p in self.original_files))
        with self.assertRaises(FileExistsError):
            archive_slide(self.root,self.source)

    def test_export_rewrites_alias_paths(self):
        alias=self.base/'result-alias'
        alias.symlink_to(self.root,target_is_directory=True)
        for col in self.cells.columns:
            if self.cells[col].dtype==object:
                self.cells[col]=self.cells[col].map(lambda v:v.replace(str(self.root),str(alias)))
        self.cells.to_csv(self.root/'cell_types.csv',index=False)
        self.patches['path']=self.patches['path'].str.replace(str(self.root),str(alias),regex=False)
        self.patches.to_csv(self.patch_index,index=False)
        summary=archive_slide(alias,self.source)
        output=export_archive(summary['archive'],self.base/'alias-export')
        table=pd.read_csv(output/'cell_types.csv')
        self.assertTrue(Path(table.cell_image_crop_path.iloc[0]).is_relative_to(output))
        self.assertTrue(Path(table.cell_image_crop_path.iloc[0]).is_file())


if __name__=='__main__':
    unittest.main()
