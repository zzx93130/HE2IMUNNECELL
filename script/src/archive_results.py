"""Verified on-demand archiving and export for native pipeline results.

The source slide is never removed. Only snapshotted regular files inside the
newly generated slide result directory are eligible for cleanup. Publication
and complete verification precede any removal; a failed check keeps loose files.
"""
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import time

import h5py
import numpy as np
import pandas as pd
from PIL import Image
import tifffile

from result_store import SlideResults, sha256

FILTERS = dict(compression='gzip', compression_opts=4, shuffle=True, fletcher32=True)


def log(message):
    print('[archive] '+message, flush=True)


def snapshot(path):
    s = path.lstat()
    if not path.is_file() or path.is_symlink():
        raise ValueError(f'Archive input must be a regular file: {path}')
    return (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns)


def image_array(path):
    with Image.open(path) as im:
        return np.asarray(im).copy()


def put_bytes(group, name, data):
    ds = group.create_dataset(name, data=np.frombuffer(data, np.uint8), **(FILTERS if data else {}))
    ds.attrs['sha256'] = hashlib.sha256(data).hexdigest()
    return ds


def put_file(group, name, path):
    size = path.stat().st_size
    ds = group.create_dataset(name, shape=(size,), dtype='u1', **(FILTERS if size else {}))
    hh = hashlib.sha256()
    with path.open('rb') as stream:
        pos = 0
        for block in iter(lambda: stream.read(4*1024*1024), b''):
            ds[pos:pos+len(block)] = np.frombuffer(block, np.uint8)
            hh.update(block)
            pos += len(block)
    ds.attrs['sha256'] = hh.hexdigest()


def dataset_hash(ds):
    hh = hashlib.sha256()
    for start in range(0,len(ds),4*1024*1024):
        hh.update(ds[start:start+4*1024*1024].tobytes())
    return hh.hexdigest()


def archive_slide(root, source, run_manifest=None, cleanup=True, patch_size=1024):
    """Directly build and validate results.h5; optionally remove archived loose files.

    Call only after classification and overlays finish. Raw RGB, white padded
    crops and level-0 patches are required and verified against every saved image.
    Existing archives are never overwritten. No full-array intermediate is built.
    """
    started = time.perf_counter()
    root_input, source_input = str(Path(root).absolute()), str(Path(source).absolute())
    root, source = Path(root).resolve(), Path(source).resolve()
    if source == root or root in source.parents:
        raise ValueError('Source slide must be outside the results directory')
    final, partial = root/'results.h5', root/'results.partial.h5'
    if final.exists() or partial.exists():
        raise FileExistsError('Archive or partial archive already exists; refusing to overwrite')
    # Refuse linked directory trees, including links that point inside the root.
    files = {}
    for base, dirs, names in os.walk(root):
        if any((Path(base)/d).is_symlink() for d in dirs):
            raise ValueError('Symlink directories cannot be archived')
        for name in names:
            path = Path(base)/name
            files[path] = snapshot(path)
    cells = pd.read_csv(root/'cell_types.csv')
    indices = list((root/'patches/tables').glob('*_patch_index.csv'))
    if len(indices) != 1:
        raise ValueError('Expected one patch index')
    patches = pd.read_csv(indices[0]).sort_values('num_patch').reset_index(drop=True)
    if not patches.empty:
        if patches.patch_size.nunique()!=1 or not patches.level.eq(0).all() or patches.num_patch.duplicated().any():
            raise ValueError('Requires unique patch IDs, a fixed patch size, and level-0 extraction')
        patch_size = int(patches.patch_size.iloc[0])
    if not cells.empty:
        if not cells.cell_crop_status.eq('ok').all() or (cells.local_label<=0).any():
            raise ValueError('Requires positive labels and successful crops')
        if not cells.cell_crop_size.eq(cells.cell_crop_size.iloc[0]).all():
            raise ValueError('Requires a fixed crop size')
    mask_candidates = {}
    for path in files:
        if path.is_relative_to(root/'cells/patch_masks') and path.name.endswith('_cp_mask.tif'):
            mask_candidates.setdefault(path.name, []).append(path)
    patch_images, masks = [], []
    for row in patches.itertuples(index=False):
        image = Path(row.path).resolve()
        candidates = mask_candidates.get(image.stem+'_cp_mask.tif', [])
        if len(candidates)!=1:
            raise ValueError(f'Expected one instance mask for patch {row.num_patch}')
        patch_images.append(image)
        masks.append(candidates[0])
    crop_pairs = [(Path(a).resolve(),Path(b).resolve()) for a,b in
                  zip(cells.get('cell_image_crop_path',[]),cells.get('cell_mask_crop_path',[]))]
    image_files = set(patch_images+masks+[p for pair in crop_pairs for p in pair])
    if not image_files.issubset(files):
        raise ValueError('Image tables reference missing files or files outside the results directory')
    # The archive includes every file, including unexpected ancillary files.
    ancillary = sorted(set(files)-image_files)
    mapping = {int(v):i for i,v in enumerate(patches.num_patch)}
    if not cells.empty and not cells.patch_id.isin(mapping).all():
        raise ValueError('Cell references an unknown patch ID')
    source_snapshot, source_hash = snapshot(source), sha256(source)
    plan = {'patch_rgb': [str(p.relative_to(root)) for p in patch_images],
            'patch_masks': [str(p.relative_to(root)) for p in masks],
            'cell_rgb': [str(p[0].relative_to(root)) for p in crop_pairs],
            'cell_masks': [str(p[1].relative_to(root)) for p in crop_pairs]}
    log(f'Writing on-demand archive: {len(cells):,} cells, {len(patches):,} patches')
    with h5py.File(partial,'x') as h:
        h.attrs.update(schema='he2cell.slide-results.v1', mode='on-demand', complete=False,
                       verified=False, source_slide=str(source), source_sha256=source_hash,
                       source_bytes=source.stat().st_size, source_results=str(root),
                       source_results_input=root_input, source_slide_input=source_input,
                       created_at=datetime.now().astimezone().isoformat(), crop_mode='raw',crop_padding='white',
                       rounding='Python round (ties to even)', h5py_version=h5py.__version__,
                       compression='gzip level 4 + shuffle + Fletcher32',loose_files_removed=False)
        idx = h.create_group('index')
        specs = {'patch_x':('patch_x','i8'), 'patch_y':('patch_y','i8'), 'cx':('centroid_x_local','f8'),
                 'cy':('centroid_y_local','f8'), 'label':('local_label','u4'),
                 'size':('cell_crop_size','i4'), 'global_instance_id':('global_instance_id','i8')}
        for key,(column,dtype) in specs.items():
            data = cells[column].to_numpy(dtype) if len(cells) else np.array([],dtype=dtype)
            idx.create_dataset(key,data=data,**(FILTERS if len(data) else {}))
        data = cells.patch_id.map(mapping).to_numpy('i4') if len(cells) else np.array([],dtype='i4')
        idx.create_dataset('patch_row',data=data,**(FILTERS if len(data) else {}))
        pg = h.create_group('patches')
        pg.create_dataset('id',data=patches.num_patch.to_numpy('i8'))
        pg.create_dataset('xy',data=patches[['x','y']].to_numpy('i8'))
        chunk=min(256,patch_size)
        opts = dict(chunks=(1,chunk,chunk),**FILTERS) if len(patches) else {}
        labels=pg.create_dataset('labels',(len(patches),patch_size,patch_size),dtype='u4',**opts)
        for i,path in enumerate(masks):
            arr=tifffile.imread(path)
            if arr.shape!=(patch_size,patch_size) or not np.issubdtype(arr.dtype,np.integer) or arr.min()<0 or arr.max()>np.iinfo(np.uint32).max:
                raise ValueError(f'Unsupported label mask: {path}')
            labels[i]=arr
        fg=h.create_group('files')
        for path in ancillary:
            put_file(fg,path.relative_to(root).as_posix(),path)
        prov=h.create_group('provenance')
        put_bytes(prov,'image_paths.json',json.dumps(plan).encode())
        if run_manifest is not None:
            put_file(prov,'run_before_archiving.json',Path(run_manifest))
        h.attrs['complete']=True
    # Read back from disk and exercise the same public reader used after cleanup.
    log('Checking every patch, crop, mask and ancillary file before cleanup')
    with SlideResults(partial,source) as store:
        for i,row in enumerate(patches.itertuples(index=False)):
            reconstructed=np.asarray(store.slide.read_region((int(row.x),int(row.y)),0,(patch_size,patch_size)).convert('RGB'))
            if not np.array_equal(reconstructed,image_array(patch_images[i])):
                raise ValueError(f'Patch RGB reconstruction mismatch: {row.num_patch}')
            if not np.array_equal(store.labels[i],tifffile.imread(masks[i])):
                raise ValueError(f'Patch label mismatch: {row.num_patch}')
        for i,pair in enumerate(crop_pairs):
            for reconstructed,path in zip(store.read_cell(i),pair):
                if not np.array_equal(reconstructed,image_array(path)):
                    raise ValueError(f'Cell reconstruction mismatch: row {i}, {path.name}')
            if (i+1)%20000==0:
                log(f'Checked {i+1:,}/{len(cells):,} cells')
        for path in ancillary:
            ds=store.h5['files'][path.relative_to(root).as_posix()]
            digest=dataset_hash(ds)
            if digest!=sha256(path) or digest!=ds.attrs['sha256']:
                raise ValueError(f'Ancillary file mismatch: {path}')
        ds=store.h5['provenance/image_paths.json']
        if dataset_hash(ds)!=ds.attrs['sha256']:
            raise ValueError('Image path index failed integrity check')
    # All original files must still be unchanged. No deletion is attempted before
    # this full preflight; the checksum-verified source must also still be intact.
    for path,stat in files.items():
        if snapshot(path)!=stat:
            raise ValueError(f'Input changed while archiving: {path}')
    if snapshot(source)!=source_snapshot or sha256(source)!=source_hash:
        raise ValueError('Source slide changed while archiving')
    with h5py.File(partial,'r+') as h:
        h.attrs['verified']=True
        h.attrs['verified_cells']=len(cells)
        h.attrs['verified_patches']=len(patches)
        h.attrs['verified_ancillary_files']=len(ancillary)
    # Close + fsync before publishing so cleanup never relies on unwritten data.
    with partial.open('rb') as stream:
        os.fsync(stream.fileno())
    if final.exists():
        raise FileExistsError('An archive appeared during verification; loose files are retained')
    partial.rename(final)
    with final.open('rb') as stream:
        os.fsync(stream.fileno())
    directory_fd=os.open(root,os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    if cleanup:
        for path,stat in files.items():
            if snapshot(path)!=stat:
                raise ValueError(f'Cleanup stopped: input changed: {path}; archive is intact at {final}')
            path.unlink()
        # Remove only now-empty directories. Never recursively remove an unknown file.
        for base,dirs,_ in os.walk(root,topdown=False):
            for name in dirs:
                path=Path(base)/name
                if not any(path.iterdir()):
                    path.rmdir()
        with h5py.File(final,'r+') as h:
            h.attrs['loose_files_removed']=True
    summary={'mode':'on-demand','status':'complete','archive':str(final),'source_slide':str(source),'source_sha256':source_hash,
             'archive_bytes':final.stat().st_size,'original_file_bytes':sum(s[2] for s in files.values()),
             'verified_cells':len(cells),'verified_patches':len(patches),'verified_ancillary_files':len(ancillary),
             'loose_files_removed':cleanup,'wall_seconds':time.perf_counter()-started}
    log(f'Archive ready: {final} ({summary["archive_bytes"]/2**20:.2f} MiB)')
    return summary


def export_archive(archive,output,source_slide=None,tables_only=False):
    """Restore loose files into an empty directory; rewrite CSV paths on relocation.

    Metadata-only export does not need the source slide. Full export also restores
    PNG patch/cell pixels and TIFF labels, without rerunning either neural model.
    """
    output=Path(output).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError('Export directory must be empty')
    output.mkdir(parents=True,exist_ok=True)
    def destination(relative):
        p=(output/relative).resolve()
        if p==output or not p.is_relative_to(output):
            raise ValueError(f'Unsafe archive path: {relative}')
        p.parent.mkdir(parents=True,exist_ok=True)
        return p
    # Validate paths before writing and check each metadata dataset's checksum.
    with h5py.File(archive,'r') as h:
        if h.attrs.get('schema')!='he2cell.slide-results.v1' or not h.attrs.get('complete'):
            raise ValueError('Unsupported or incomplete archive')
        names=[]
        h['files'].visititems(lambda name,obj:names.append(name) if isinstance(obj,h5py.Dataset) else None)
        old_root=h.attrs['source_results']
        old_source=h.attrs['source_slide']
        root_aliases={old_root,h.attrs.get('source_results_input',old_root)}
        source_aliases={old_source,h.attrs.get('source_slide_input',old_source)}
        new_source=str(Path(source_slide).expanduser().resolve()) if source_slide else old_source
        def rewrite_value(value):
            for prefix in root_aliases:
                if value.startswith(prefix+'/'):
                    return str(output)+value[len(prefix):]
            return new_source if value in source_aliases else value
        for name in names:
            destination(name)
        for name in names:
            ds=h['files'][name]
            if dataset_hash(ds)!=ds.attrs['sha256']:
                raise ValueError(f'Corrupt metadata: {name}')
            path=destination(name)
            with path.open('wb') as stream:
                for start in range(0,len(ds),4*1024*1024):
                    stream.write(ds[start:start+4*1024*1024].tobytes())
            if path.suffix.lower()=='.csv':
                table=pd.read_csv(path,dtype=str,keep_default_na=False)
                for column in table.columns:
                    table[column]=table[column].map(rewrite_value)
                table.to_csv(path,index=False)
        if tables_only:
            return output
        if 'provenance/image_paths.json' not in h:
            raise ValueError('This older benchmark archive supports metadata export only')
        ds=h['provenance/image_paths.json']
        if dataset_hash(ds)!=ds.attrs['sha256']:
            raise ValueError('Corrupt image path index')
        plan=json.loads(ds[:].tobytes())
        for paths in plan.values():
            for relative in paths:
                destination(relative)
    with SlideResults(archive,source_slide) as store:
        patch_size=store.labels.shape[1]
        xy=store.h5['patches/xy'][:]
        for i,(image_path,mask_path) in enumerate(zip(plan['patch_rgb'],plan['patch_masks'])):
            rgb=store.slide.read_region(tuple(map(int,xy[i])),0,(patch_size,patch_size)).convert('RGB')
            path=destination(image_path)
            rgb.save(path,**({'compression':'tiff_lzw'} if path.suffix.lower() in ('.tif','.tiff') else {}))
            tifffile.imwrite(destination(mask_path),store.labels[i],compression='zlib')
        for i,(image_path,mask_path) in enumerate(zip(plan['cell_rgb'],plan['cell_masks'])):
            rgb,mask=store.read_cell(i)
            Image.fromarray(rgb).save(destination(image_path))
            Image.fromarray(mask).save(destination(mask_path))
    return output
