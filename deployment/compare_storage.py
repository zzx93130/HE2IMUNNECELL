#!/usr/bin/env python3
"""Build and measure three non-destructive HDF5 archives of one completed slide.

Usage: .venv/bin/python deployment/compare_storage.py --output-dir outputs/storage-20261016
Only the selected slide's results directory is archived, plus run.json provenance.
Downstream analyses elsewhere in the run directory and model weights are excluded.
"""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'script/src'))
import h5py
import numpy as np
import pandas as pd
from PIL import Image
import tifffile
from result_store import SlideResults, sha256
from encoded_storage import create_encoded, verify_encoded

FILTERS = dict(compression='gzip', compression_opts=4, shuffle=True, fletcher32=True)


def log(message):
    print(f'{datetime.now().astimezone().isoformat(timespec="seconds")} {message}', flush=True)


def image_array(path):
    with Image.open(path) as im:
        return np.asarray(im).copy()


def original_cell(paths):
    return image_array(paths[0]), image_array(paths[1])


def category(path):
    s = path.as_posix()
    if '/cell_images_' in s:
        return 'cell_rgb'
    if '/cell_masks_' in s:
        return 'cell_masks'
    if s.startswith('cells/patch_masks/'):
        return 'patch_labels'
    if s.startswith('patches/') and '/folder_' in s:
        return 'patch_rgb'
    return 'tables_and_review_files'


def inventory(root):
    files, totals = [], defaultdict(lambda: dict(files=0, logical_bytes=0, allocated_bytes=0))
    for base, dirs, names in os.walk(root):
        for name in sorted(names):
            p = Path(base) / name
            if name == '.DS_Store':
                continue
            st = p.stat()
            rel = p.relative_to(root)
            cat = category(rel)
            totals[cat]['files'] += 1
            totals[cat]['logical_bytes'] += st.st_size
            totals[cat]['allocated_bytes'] += st.st_blocks * 512
            files.append((p, rel, cat, st.st_size))
    return files, dict(totals)


def add_file(group, name, path):
    size = path.stat().st_size
    ds = group.create_dataset(name, shape=(size,), dtype='u1', **(FILTERS if size else {}))
    h = hashlib.sha256()
    with path.open('rb') as f:
        pos = 0
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            ds[pos:pos+len(block)] = np.frombuffer(block, np.uint8)
            h.update(block)
            pos += len(block)
    ds.attrs['sha256'] = h.hexdigest()


def storage_sizes(h5):
    sizes = defaultdict(int)
    def visit(name, obj):
        if isinstance(obj, h5py.Dataset):
            key = name.split('/')[0] if name.startswith(('files/', 'index/')) else name
            sizes[key] += obj.id.get_storage_size()
    h5.visititems(visit)
    return dict(sizes)


def create_full(path, root, cells, patches, masks, files, source, run_path):
    started = time.perf_counter()
    tmp = path.with_suffix('.partial.h5')
    if path.exists() or tmp.exists():
        raise FileExistsError(path)
    n, m = len(cells), len(patches)
    size, patch_size = int(cells.cell_crop_size.iloc[0]), int(patches.patch_size.iloc[0])
    if n == 0 or cells.cell_crop_size.nunique() != 1 or patches.patch_size.nunique() != 1:
        raise ValueError('Requires nonempty cells and fixed crop/patch sizes')
    if not cells.cell_crop_status.eq('ok').all() or (cells.local_label <= 0).any():
        raise ValueError('Requires successful crops and positive instance labels')
    if not patches.level.eq(0).all():
        raise ValueError('Only original pipeline level-0 extraction is supported')
    mapping = {int(v): i for i, v in enumerate(patches.num_patch)}
    with h5py.File(tmp, 'w') as h:
        h.attrs.update(schema='he2cell.slide-results.v1', mode='full', complete=False,
                       source_slide=str(source), source_sha256=sha256(source),
                       source_bytes=source.stat().st_size, source_results=str(root),
                       created_at=datetime.now().astimezone().isoformat(),
                       crop_mode='raw', crop_padding='white', rounding='Python round (ties to even)',
                       h5py_version=h5py.__version__, hdf5_version=h5py.version.hdf5_version,
                       compression='gzip level 4 + shuffle + Fletcher32')
        idx = h.create_group('index')
        for key, data in dict(patch_row=cells.patch_id.map(mapping).to_numpy(np.int32),
                              patch_x=cells.patch_x.to_numpy(np.int64), patch_y=cells.patch_y.to_numpy(np.int64),
                              cx=cells.centroid_x_local.to_numpy(np.float64), cy=cells.centroid_y_local.to_numpy(np.float64),
                              label=cells.local_label.to_numpy(np.uint32), size=cells.cell_crop_size.to_numpy(np.int32),
                              global_instance_id=cells.global_instance_id.to_numpy(np.int64)).items():
            idx.create_dataset(key, data=data, **FILTERS)
        pg = h.create_group('patches')
        pg.create_dataset('id', data=patches.num_patch.to_numpy(np.int64), **FILTERS)
        pg.create_dataset('xy', data=patches[['x','y']].to_numpy(np.int64), **FILTERS)
        spatial_chunk = min(256, patch_size)
        labels = pg.create_dataset('labels', (m, patch_size, patch_size), dtype='u4',
                                   chunks=(1,spatial_chunk,spatial_chunk), **FILTERS)
        rgb = pg.create_dataset('rgb', (m, patch_size, patch_size,3), dtype='u1',
                                chunks=(1,spatial_chunk,spatial_chunk,3), **FILTERS)
        for j, row in enumerate(patches.itertuples(index=False)):
            a, b = image_array(row.path), tifffile.imread(masks[int(row.num_patch)])
            if a.shape != (patch_size,patch_size,3) or a.dtype != np.uint8:
                raise ValueError(f'Unsupported patch RGB format: {row.path}')
            rgb[j], labels[j] = a, b
            if j % 100 == 0:
                log(f'Packing patches {j+1}/{m}')
        cg = h.create_group('cells')
        chunk = min(32, n)
        rgb = cg.create_dataset('rgb', (n,size,size,3), dtype='u1', chunks=(chunk,size,size,3), **FILTERS)
        mask = cg.create_dataset('mask', (n,size,size), dtype='u1', chunks=(chunk,size,size), **FILTERS)
        paths = list(zip(cells.cell_image_crop_path, cells.cell_mask_crop_path))
        with ThreadPoolExecutor(max_workers=8) as pool:
            for start in range(0, n, 512):
                pairs = list(pool.map(original_cell, paths[start:start+512]))
                rgb[start:start+len(pairs)] = np.stack([x[0] for x in pairs])
                mask[start:start+len(pairs)] = np.stack([x[1] for x in pairs])
                if start % 10240 == 0:
                    log(f'Packing cells {start+len(pairs)}/{n}')
        fg = h.create_group('files')
        for p, rel, cat, _ in files:
            if cat == 'tables_and_review_files':
                add_file(fg, rel.as_posix(), p)
        add_file(h.create_group('provenance'), 'run.json', run_path)
        h.attrs['complete'] = True
    tmp.rename(path)
    return time.perf_counter() - started


def create_on_demand(full, path):
    started = time.perf_counter()
    tmp = path.with_suffix('.partial.h5')
    if path.exists() or tmp.exists():
        raise FileExistsError(path)
    with h5py.File(full, 'r') as a, h5py.File(tmp, 'w') as b:
        b.attrs.update(dict(a.attrs))
        b.attrs['mode'], b.attrs['complete'] = 'on-demand', False
        b.attrs['created_at'] = datetime.now().astimezone().isoformat()
        for key in ('index', 'files', 'provenance'):
            a.copy(key, b)
        pg = b.create_group('patches')
        for key in ('id', 'xy', 'labels'):
            a.copy('patches/' + key, pg, name=key)
        b.attrs['complete'] = True
    tmp.rename(path)
    return time.perf_counter() - started


def verify(full, demand, cells, patches, masks, files, source):
    started = time.perf_counter()
    with SlideResults(full) as a, SlideResults(demand, source) as b:
        for j, row in enumerate(patches.itertuples(index=False)):
            saved = image_array(row.path)
            reconstructed = np.asarray(b.slide.read_region((int(row.x),int(row.y)),0,
                                                            (int(row.patch_size),int(row.patch_size))).convert('RGB'))
            expected = tifffile.imread(masks[int(row.num_patch)])
            if not (np.array_equal(a.h5['patches/rgb'][j], saved) and np.array_equal(reconstructed, saved)
                    and np.array_equal(a.labels[j], expected) and np.array_equal(b.labels[j], expected)):
                raise AssertionError(f'Patch mismatch: {row.num_patch}')
            if j % 200 == 0:
                log(f'Validating patch pixels and labels {j+1}/{len(patches)}')
        paths = list(zip(cells.cell_image_crop_path, cells.cell_mask_crop_path))
        with ThreadPoolExecutor(max_workers=8) as pool:
            for start in range(0,len(cells),512):
                originals = list(pool.map(original_cell, paths[start:start+512]))
                # Validate public single-cell reader for every cell, including slide reconstruction.
                for offset, original in enumerate(originals):
                    i = start + offset
                    aa, bb = a.read_cell(i), b.read_cell(i)
                    if not all(np.array_equal(original[k], test[k]) for test in (aa,bb) for k in (0,1)):
                        raise AssertionError(f'Cell RGB/mask mismatch at row {i}')
                if start % 10240 == 0:
                    log(f'Validating cell pixels and masks {start+len(originals)}/{len(cells)}')
        ancillary = 0
        for p, rel, cat, _ in files:
            if cat != 'tables_and_review_files':
                continue
            expected = sha256(p)
            for s in (a,b):
                ds = s.h5['files'][rel.as_posix()]
                hh = hashlib.sha256()
                for start in range(0,len(ds),4*1024*1024):
                    hh.update(ds[start:start+4*1024*1024].tobytes())
                if hh.hexdigest() != expected:
                    raise AssertionError(f'Ancillary file mismatch: {rel}')
            ancillary += 1
    return dict(patches_verified=len(patches), cells_verified=len(cells), ancillary_files_verified=ancillary,
                all_rgb_pixels_exact=True, all_mask_pixels_exact=True, ancillary_sha256_exact=True,
                seconds=time.perf_counter()-started)


def benchmark(full, demand, cells, source, count=1024, encoded=None):
    rng = np.random.default_rng(20261016)
    count = min(count,len(cells))
    random_rows = rng.choice(len(cells),count,replace=False).tolist()
    start = max(0,(len(cells)-count)//2)
    sequential = list(range(start,start+count))
    paths = list(zip(cells.cell_image_crop_path,cells.cell_mask_crop_path))
    records = {}
    # Warm OS caches; do not claim cold-disk performance. Rotate method order each repeat.
    for workload, rows in [('random',random_rows),('sequential',sequential)]:
        timings = defaultdict(list)
        repetitions = 4 if encoded else 3
        for repeat in range(repetitions):
            methods = ['png','hdf5','on-demand'] + (['encoded'] if encoded else [])
            methods = methods[repeat:] + methods[:repeat]
            for method in methods:
                opened = time.perf_counter()
                store_path = {'hdf5': full, 'on-demand': demand, 'encoded': encoded}.get(method)
                store = SlideResults(store_path,source) if method != 'png' else None
                open_seconds = time.perf_counter() - opened
                reader = store.read_cell if store else lambda i: original_cell(paths[i])
                try:
                    for i in rows[:32]:
                        reader(i)
                    t = time.perf_counter()
                    checksum = 0
                    for i in rows:
                        rgb, mask = reader(i)
                        checksum += int(rgb[0,0,0]) + int(mask.sum())
                    elapsed = time.perf_counter()-t
                    timings[method].append(dict(seconds=elapsed, open_seconds=open_seconds, checksum=checksum))
                finally:
                    if store:
                        store.close()
        checksums = {r['checksum'] for runs in timings.values() for r in runs}
        if len(checksums) != 1:
            raise AssertionError('Benchmark read checksums differ')
        records[workload] = {k: dict(runs=v, median_seconds=statistics.median(x['seconds'] for x in v),
                                    median_ms_per_cell=1000*statistics.median(x['seconds'] for x in v)/count)
                             for k,v in timings.items()}
    return dict(cells_per_workload=count, repetitions=repetitions, seed=20261016, cache='warm OS cache; 32-cell warmup; rotating method order; one thread',
                includes='RGB and binary mask retrieval; excludes model inference, file open and source checksum',workloads=records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=ROOT/'outputs/whole-slide-20261012')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--slide-index', type=int, default=0)
    parser.add_argument('--resume',action='store_true',help='Reuse completed HDF5 files, then rerun verification and benchmarks')
    args = parser.parse_args()
    run_path = args.run_dir.resolve()/'run.json'
    manifest = json.loads(run_path.read_text())
    slide = manifest['slides'][args.slide_index]
    if manifest['status'] != 'complete' or slide['status'] != 'complete':
        raise ValueError('Requires a completed run')
    source, root = Path(slide['slide']),Path(slide['results']).parent
    out = args.output_dir.resolve()
    if out == root or root in out.parents:
        raise ValueError('Output must be outside the source slide directory')
    out.mkdir(parents=True,exist_ok=True)
    cells = pd.read_csv(root/'cell_types.csv')
    indices = list((root/'patches/tables').glob('*_patch_index.csv'))
    if len(indices) != 1:
        raise ValueError('Expected exactly one patch index')
    patches = pd.read_csv(indices[0]).sort_values('num_patch').reset_index(drop=True)
    mask_paths = list((root/'cells/patch_masks').rglob('*_cp_mask.tif'))
    masks = {}
    for row in patches.itertuples(index=False):
        candidates = [p for p in mask_paths if p.name == Path(row.path).stem + '_cp_mask.tif']
        if len(candidates) != 1:
            raise ValueError(f'Expected exactly one label mask for patch {row.num_patch}')
        masks[int(row.num_patch)] = candidates[0]
    log('Measuring existing files')
    files, totals = inventory(root)
    # Guard against unrecognized image paths being silently omitted from an archive.
    covered = {Path(p).resolve() for p in patches.path} | {p.resolve() for p in masks.values()}
    covered |= {Path(p).resolve() for col in ('cell_image_crop_path','cell_mask_crop_path') for p in cells[col]}
    classified = {p.resolve() for p,_,cat,_ in files if cat != 'tables_and_review_files'}
    if classified != covered:
        raise ValueError('Image inventory and result tables disagree')
    full, demand = out/'slide_full.h5',out/'slide_on_demand.h5'
    encoded = out/'slide_encoded.h5'
    packing = {}
    old_metrics = out/'packing_times.json'
    if args.resume and old_metrics.exists():
        packing.update(json.loads(old_metrics.read_text()))
    if not (args.resume and full.exists()):
        log('Building compressed full HDF5')
        packing['full_seconds'] = create_full(full,root,cells,patches,masks,files,source,run_path)
        old_metrics.write_text(json.dumps(packing,indent=2))
    if not (args.resume and demand.exists()):
        log('Building on-demand HDF5')
        packing['on_demand_from_full_seconds'] = create_on_demand(full,demand)
        old_metrics.write_text(json.dumps(packing,indent=2))
    log('Verifying every saved crop, every patch and all ancillary files')
    validation = verify(full,demand,cells,patches,masks,files,source)
    if not (args.resume and encoded.exists()):
        log('Building PNG-preserving HDF5 control')
        packing['encoded_seconds'] = create_encoded(full,encoded,cells,patches,log)
        old_metrics.write_text(json.dumps(packing,indent=2))
    encoded_validation = verify_encoded(full,encoded,log)
    log('Measuring repeated random and sequential crop reads')
    performance = benchmark(full,demand,cells,source,encoded=encoded)
    archives = {}
    for name,path in [('full',full),('on_demand',demand),('encoded',encoded)]:
        with h5py.File(path,'r') as h:
            archives[name] = dict(path=str(path),logical_bytes=path.stat().st_size,
                                 allocated_bytes=path.stat().st_blocks*512,dataset_stored_bytes=storage_sizes(h))
    metrics = dict(created_at=datetime.now().astimezone().isoformat(),source_slide=str(source),
                   source_sha256=sha256(source),source_slide_bytes=source.stat().st_size,source_results=str(root),
                   scope='All files in selected slide results directory, excluding .DS_Store; archives also contain run.json. Run-level downstream analyses and model weights excluded.',
                   cells=len(cells),patches=len(patches),original=totals,archives=archives,packing=packing,
                   validation=validation,encoded_validation=encoded_validation,retrieval=performance,
                   software=dict(python=platform.python_version(),h5py=h5py.__version__,hdf5=h5py.version.hdf5_version,
                                 numpy=np.__version__,pillow=Image.__version__))
    (out/'storage_metrics.json').write_text(json.dumps(metrics,indent=2))
    write_report(out,metrics)
    from report_storage import plot_report
    plot_report(out,metrics)
    log('Finished: '+str(out/'storage_report.md'))


def write_report(out,m):
    original = sum(v['logical_bytes'] for v in m['original'].values())
    allocated = sum(v['allocated_bytes'] for v in m['original'].values())
    nfiles = sum(v['files'] for v in m['original'].values())
    source = m['source_slide_bytes']
    lines = ['# Whole-slide storage comparison','',
             f'Measured on {m["cells"]:,} cells and {m["patches"]:,} patches. Original results are preserved.','',
             '| Storage | Results bytes (MiB) | With original slide (MiB) | Results reduction | Allocated MiB |',
             '|---|---:|---:|---:|---:|',
             f'| Original ({nfiles:,} files) | {original/2**20:,.2f} | {(original+source)/2**20:,.2f} | — | {allocated/2**20:,.2f} |']
    variants = [('full','HDF5 decoded arrays'),('encoded','HDF5 preserving PNGs'),('on_demand','On-demand HDF5')]
    for key,label in variants:
        if key not in m['archives']:
            continue
        a=m['archives'][key]; b=a['logical_bytes']
        lines.append(f'| {label} (1 file) | {b/2**20:,.2f} | {(b+source)/2**20:,.2f} | {(1-b/original)*100:.2f}% | {a["allocated_bytes"]/2**20:,.2f} |')
    lines += ['', 'The full archive contains decoded RGB crops, binary masks, patch RGB, complete patch instance labels, and byte-exact tables/review files. The on-demand archive keeps the complete labels and tables/review files, and reconstructs RGB from the checksum-verified source slide. It does not rerun segmentation or classification.', '',
              'The original slide is an external dependency for on-demand reads. The “with original slide” column charges its bytes to every option for a consistent retention policy; the full archive can read saved crops without the slide. Results reductions use logical file lengths. Allocated bytes are sums of filesystem st_blocks, excluding directory metadata; APFS sharing/compression can affect physical accounting.','',
              '## Original storage breakdown','', '| Category | Files | MiB |','|---|---:|---:|']
    for k,v in m['original'].items():
        lines.append(f'| {k} | {v["files"]:,} | {v["logical_bytes"]/2**20:,.2f} |')
    lines += ['', '## Compressed payload breakdown', '',
              '| Content | Original MiB | Array HDF5 MiB | PNG-preserving HDF5 MiB | On-demand HDF5 MiB |',
              '|---|---:|---:|---:|---:|']
    for cat, keys in [('patch_rgb',['patches/rgb','patches/rgb_encoded','patches/rgb_offsets']),
                      ('patch_labels',['patches/labels']),('cell_rgb',['cells/rgb','cells/rgb_encoded','cells/rgb_offsets']),
                      ('cell_masks',['cells/mask','cells/mask_encoded','cells/mask_offsets']),('tables_and_review_files',['files'])]:
        vals=[]
        for kind in ('full','encoded','on_demand'):
            parts=m['archives'].get(kind,{}).get('dataset_stored_bytes',{})
            vals.append(sum(parts.get(k,0) for k in keys)/2**20)
        lines.append(f'| {cat} | {m["original"][cat]["logical_bytes"]/2**20:,.2f} | '+' | '.join(f'{v:,.2f}' for v in vals)+' |')
    lines += ['', 'Payloads exclude HDF5 object/chunk indexes, the compact cell lookup index and added run provenance; the total-file table includes all overhead.', '',
              'The PNG-preserving control keeps original compressed patch and cell image bytes in indexed byte streams, with gzip level 4 on 64-KiB chunks. It retains full patch labels as lossless HDF5 arrays and the same ancillary files. This distinguishes packing already-compressed image files from replacing their encoding with raw-array compression.', '',
              '## Retrieval benchmark', '', '| Access (RGB + mask) | PNG ms/cell | Array HDF5 ms/cell | PNG-preserving HDF5 ms/cell | On-demand ms/cell |','|---|---:|---:|---:|---:|']
    for workload,results in m['retrieval']['workloads'].items():
        lines.append('| '+workload+' | '+' | '.join(f'{results[k]["median_ms_per_cell"]:.3f}' if k in results else '—' for k in ('png','hdf5','encoded','on-demand'))+' |')
    lines += ['',f'Each workload reads {m["retrieval"]["cells_per_workload"]:,} cells, with {m["retrieval"]["repetitions"]} timed repetitions and rotating method order. Warm filesystem caches and a 32-cell warmup; single-threaded retrieval. Opening stores, loading the compact index and checking the source SHA-256 are excluded from retrieval timing and recorded separately in storage_metrics.json. This is not a cold-disk benchmark or a training-throughput benchmark.', '',
              '## Integrity and scope','',
              f'Every one of {m["validation"]["cells_verified"]:,} cell RGB crops and binary masks matched pixel-for-pixel through both public readers. All {m["validation"]["patches_verified"]:,} saved patch RGB arrays and complete label masks matched; source-slide reconstruction also matched every patch. All {m["validation"]["ancillary_files_verified"]} ancillary files matched SHA-256. PNG/TIFF container encodings are not retained for converted arrays; decoded pixels are exact. Original CSV bytes, confidence values, labels and stable IDs are retained.', '',
              m['scope'], '',
              'Full HDF5 uses gzip level 4, byte shuffle and Fletcher32 checksums; RGB/cell-mask chunks contain 32 cells, while patch arrays use 256×256 tiles. These are one measured configuration, not a compression-level or chunk-size optimization sweep. Existing PNGs/TIFFs are already compressed; array compression may be worse for RGB. Repeated long paths in CSV tables compress well. On-demand savings principally remove duplicated image pixels and per-cell masks.', '',
              f'Full archive construction: {m["packing"].get("full_seconds",0)/60:.2f} min. On-demand archive copied from full: {m["packing"].get("on_demand_from_full_seconds",0):.2f} s. Full integrity validation: {m["validation"]["seconds"]/60:.2f} min. The conversion is post-processing; the current inference pipeline still writes its original files.', '',
              '## Reuse','',
              '```python','import sys',f'sys.path.insert(0, {str(ROOT/"script/src")!r})','from result_store import SlideResults','',
              f'with SlideResults({str(out/"slide_full.h5")!r}) as results:',
              '    rgb, mask = results.read_cell(0)  # zero-based original CSV row',
              '    predictions = results.table()', '',
              '# On-demand: specify source_slide="/new/location/slide.svs" if relocated.',
              f'with SlideResults({str(out/"slide_on_demand.h5")!r}) as results:',
              '    rgb, mask = results.read_cell(0)','```','',
              'The reader uses archive-relative dataset paths; paths inside the preserved CSV are provenance and need not exist for crop retrieval. Keep the source slide with the on-demand store. File writes are staged as .partial.h5 and renamed only when complete. Original files are never deleted.', '',
              'Compression/chunking reference: [h5py dataset documentation](https://docs.h5py.org/en/stable/high/dataset.html).','']
    if 'encoded_validation' in m:
        v=m['encoded_validation']
        lines += [f'PNG-preserving archive validation: all {v["cells_verified"]:,} cells and {v["patches_verified"]:,} patches matched the verified array archive pixel-for-pixel; all compressed image byte streams matched SHA-256 and ancillary files matched byte-for-byte. Construction took {m["packing"]["encoded_seconds"]/60:.2f} min; validation took {v["seconds"]/60:.2f} min.', '']
    if m.get('packing_note'):
        lines += [m['packing_note'], '']
    (out/'storage_report.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    main()
