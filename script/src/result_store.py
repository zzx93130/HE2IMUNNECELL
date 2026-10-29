"""Portable, lossless per-slide HDF5 reader; no neural network dependency.

Rows use the original cell_types.csv order, not global_instance_id. The full
store is self-contained for saved results. The on-demand store needs the exact
source slide, verified once by SHA-256 (a relocated path can be supplied).
"""
from pathlib import Path
import hashlib
import io

import h5py
import numpy as np


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def window(cx, cy, size, width, height):
    x, y = int(round(float(cx))) - size // 2, int(round(float(cy))) - size // 2
    x0, y0, x1, y1 = max(0, x), max(0, y), min(width, x + size), min(height, y + size)
    return x0, y0, x1, y1, x0 - x, y0 - y


class SlideResults:
    def __init__(self, path, source_slide=None):
        self.h5 = h5py.File(path, 'r', rdcc_nbytes=32 * 1024 * 1024)
        self.slide = None
        try:
            if self.h5.attrs.get('schema') != 'he2cell.slide-results.v1':
                raise ValueError('Unsupported results schema')
            if not self.h5.attrs.get('complete', False):
                raise ValueError('Incomplete results store')
            self.mode = self.h5.attrs['mode']
            # Compact index in memory avoids dozens of tiny HDF5 reads per cell.
            self.index = {k: self.h5['index'][k][:] for k in self.h5['index']}
            self.labels = self.h5['patches/labels']
            if self.mode == 'full':
                self.rgb = self.h5['cells/rgb']
                self.masks = self.h5['cells/mask']
            elif self.mode == 'encoded':
                self.rgb = self.h5['cells/rgb_encoded']
                self.masks = self.h5['cells/mask_encoded']
                self.rgb_offsets = self.h5['cells/rgb_offsets'][:]
                self.mask_offsets = self.h5['cells/mask_offsets'][:]
            elif self.mode != 'on-demand':
                raise ValueError(f'Unsupported archive mode: {self.mode}')
            else:
                from slide_io import open_slide
                source = Path(source_slide or self.h5.attrs['source_slide'])
                if sha256(source) != self.h5.attrs['source_sha256']:
                    raise ValueError('Source slide checksum does not match this results store')
                self.slide = open_slide(source)
        except Exception:
            self.close()
            raise

    def __len__(self):
        return len(self.index['patch_row'])

    def read_cell(self, row):
        row = int(row)
        if not 0 <= row < len(self):
            raise IndexError(row)
        if self.mode == 'full':
            return self.rgb[row], self.masks[row]
        if self.mode == 'encoded':
            from PIL import Image
            arrays = []
            for payload, offsets in ((self.rgb, self.rgb_offsets), (self.masks, self.mask_offsets)):
                with Image.open(io.BytesIO(payload[int(offsets[row]):int(offsets[row+1])].tobytes())) as im:
                    arrays.append(np.asarray(im).copy())
            return tuple(arrays)
        idx = self.index
        p = int(idx['patch_row'][row])
        size = int(idx['size'][row])
        _, height, width = self.labels.shape
        x0, y0, x1, y1, dx, dy = window(idx['cx'][row], idx['cy'][row], size, width, height)
        rgb = np.full((size, size, 3), 255, np.uint8)
        mask = np.zeros((size, size), np.uint8)
        if x1 > x0 and y1 > y0:
            rgb[dy:dy+y1-y0, dx:dx+x1-x0] = np.asarray(self.slide.read_region(
                (int(idx['patch_x'][row]) + x0, int(idx['patch_y'][row]) + y0),
                0, (x1-x0, y1-y0)).convert('RGB'))
            mask[dy:dy+y1-y0, dx:dx+x1-x0] = (
                self.labels[p, y0:y1, x0:x1] == idx['label'][row]).astype(np.uint8) * 255
        return rgb, mask

    def table(self):
        """Return the original prediction CSV, including its original column types."""
        import io
        import pandas as pd
        return pd.read_csv(io.BytesIO(self.h5['files/cell_types.csv'][:].tobytes()))

    def file_bytes(self, relative_path):
        """Read an ancillary file, e.g. cell_types.csv or overlay/sample.png."""
        return self.h5['files'][relative_path][:].tobytes()

    def close(self):
        if self.slide is not None:
            self.slide.close()
            self.slide = None
        self.h5.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


if __name__ == '__main__':
    import argparse
    from PIL import Image

    parser = argparse.ArgumentParser(description='Recover a cell RGB crop and binary mask from HDF5.')
    parser.add_argument('archive', type=Path)
    parser.add_argument('--row', type=int, default=0, help='Zero-based row in the original cell_types.csv')
    parser.add_argument('--source-slide', type=Path, help='Relocated source slide for on-demand archives')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    image_path = args.output_dir / f'cell_{args.row}_rgb.png'
    mask_path = args.output_dir / f'cell_{args.row}_mask.png'
    if image_path.exists() or mask_path.exists():
        parser.error('Output already exists; choose another output directory or cell row.')
    with SlideResults(args.archive, args.source_slide) as results:
        rgb, mask = results.read_cell(args.row)
    Image.fromarray(rgb).save(image_path)
    Image.fromarray(mask).save(mask_path)
    print(image_path)
    print(mask_path)
