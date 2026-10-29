"""PNG-preserving HDF5 control for the slide storage comparison."""
import hashlib
import io
from pathlib import Path
import time

import h5py
import numpy as np
from PIL import Image

from result_store import SlideResults


def pack_images(group, name, paths, log):
    ds = group.create_dataset(name+'_encoded',shape=(0,),maxshape=(None,),dtype='u1',
                              chunks=(65536,),compression='gzip',compression_opts=4,fletcher32=True)
    offsets = [0]
    digest = hashlib.sha256()
    buf = bytearray()
    pos = 0
    for i, path in enumerate(paths):
        blob = Path(path).read_bytes()
        digest.update(blob)
        offsets.append(offsets[-1]+len(blob))
        buf.extend(blob)
        if len(buf) >= 4*1024*1024:
            ds.resize((pos+len(buf),))
            ds[pos:pos+len(buf)] = np.frombuffer(buf,np.uint8)
            pos += len(buf)
            buf = bytearray()
        if i % 20000 == 0:
            log(f'Packing compressed {group.name}/{name} {i+1}/{len(paths)}')
    if buf:
        ds.resize((pos+len(buf),))
        ds[pos:pos+len(buf)] = np.frombuffer(buf,np.uint8)
    group.create_dataset(name+'_offsets',data=np.array(offsets,dtype=np.uint64),
                         compression='gzip',shuffle=True,fletcher32=True)
    ds.attrs['sha256'] = digest.hexdigest()
    ds.attrs['encoding'] = 'original PNG/TIFF file bytes; offsets index individual images'


def create_encoded(full,path,cells,patches,log):
    started=time.perf_counter()
    tmp=path.with_suffix('.partial.h5')
    if path.exists() or tmp.exists():
        raise FileExistsError(path)
    with h5py.File(full,'r') as a,h5py.File(tmp,'w') as b:
        b.attrs.update(dict(a.attrs))
        b.attrs['mode'],b.attrs['complete']='encoded',False
        for key in ('index','files','provenance'):
            a.copy(key,b)
        pg=b.create_group('patches')
        for key in ('id','xy','labels'):
            a.copy('patches/'+key,pg,name=key)
        pack_images(pg,'rgb',patches.path.tolist(),log)
        cg=b.create_group('cells')
        pack_images(cg,'rgb',cells.cell_image_crop_path.tolist(),log)
        pack_images(cg,'mask',cells.cell_mask_crop_path.tolist(),log)
        b.attrs['complete']=True
    tmp.rename(path)
    return time.perf_counter()-started


def verify_encoded(full,encoded,log):
    started=time.perf_counter()
    with SlideResults(full) as a,SlideResults(encoded) as b:
        payloads = ['cells/rgb_encoded','cells/mask_encoded','patches/rgb_encoded']
        for key in payloads:
            ds=b.h5[key]
            digest=hashlib.sha256()
            for start in range(0,len(ds),4*1024*1024):
                digest.update(ds[start:start+4*1024*1024].tobytes())
            if digest.hexdigest()!=ds.attrs['sha256']:
                raise AssertionError(f'Encoded payload mismatch: {key}')
        offsets=b.h5['patches/rgb_offsets'][:]
        blob=b.h5['patches/rgb_encoded']
        for i in range(len(offsets)-1):
            with Image.open(io.BytesIO(blob[int(offsets[i]):int(offsets[i+1])].tobytes())) as im:
                if not np.array_equal(np.asarray(im),a.h5['patches/rgb'][i]):
                    raise AssertionError(f'Encoded patch RGB mismatch: {i}')
            if not np.array_equal(b.labels[i],a.labels[i]):
                raise AssertionError(f'Encoded patch labels mismatch: {i}')
        # Ancillary datasets were copied losslessly; compare their decompressed bytes too.
        files=[]
        a.h5['files'].visititems(lambda name,obj: files.append(name) if isinstance(obj,h5py.Dataset) else None)
        for name in files:
            aa,bb=a.h5['files'][name],b.h5['files'][name]
            if aa.shape!=bb.shape:
                raise AssertionError(name)
            for start in range(0,len(aa),4*1024*1024):
                if not np.array_equal(aa[start:start+4*1024*1024],bb[start:start+4*1024*1024]):
                    raise AssertionError(name)
        for i in range(len(a)):
            aa,bb=a.read_cell(i),b.read_cell(i)
            if not all(np.array_equal(x,y) for x,y in zip(aa,bb)):
                raise AssertionError(f'Encoded cell mismatch: {i}')
            if i % 20000 == 0:
                log(f'Validating PNG-preserving HDF5 cells {i+1}/{len(a)}')
        result=dict(cells_verified=len(a),patches_verified=len(offsets)-1,ancillary_files_verified=len(files),
                    encoded_payload_sha256_exact=True,all_pixels_exact=True)
    result['seconds']=time.perf_counter()-started
    return result
