"""Membrane enhancement, at the source grid before any segmentation sampling."""
from pathlib import Path
import hashlib
import time
import os
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import numpy as np
from r3_fast import balanced


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def preprocess(raw, destination, normalization='auto', progress=lambda *a: None,
               check=lambda: None, workers=8):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + '.partial')
    started = time.time()
    output = None
    scales = [None]*len(raw)
    workers=max(1,min(int(workers),os.cpu_count() or 1,len(raw),32))
    # Bound concurrent scratch volumes as well as queued results.
    try:
        import ctypes
        class Memory(ctypes.Structure):
            _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(name,ctypes.c_ulonglong) for name in ('total','available','total_page','avail_page','total_virtual','avail_virtual','avail_extended')]
        if os.name=='nt':
            memory=Memory();memory.length=ctypes.sizeof(memory)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory)):
                workers=min(workers,max(1,int(memory.available*.5/(raw.shape[1]*raw.shape[2]*256))))
    except (AttributeError,OSError):pass
    def plane(z):
        indices = np.clip(np.arange(z - 2, z + 3), 0, len(raw) - 1)
        source = np.asarray(raw[indices])
        maximum = float(source.max())
        scale = (1 if source.dtype.kind == 'f' and maximum <= 1.5 else
                 255 if maximum < 256 else 65535) if normalization == 'auto' else float(normalization)
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError('Invalid preprocessing intensity scale')
        stack = source.astype(np.float32) / scale
        if not np.isfinite(stack).all():
            raise ValueError('Preprocessing input contains NaN/Inf')
        return z,scale,balanced(stack)
    try:
        output = np.lib.format.open_memmap(partial, mode='w+', dtype=np.float32, shape=raw.shape)
        progress(0, 'R3 preprocessing', f'0 / {len(raw)}')
        with ThreadPoolExecutor(max_workers=workers,thread_name_prefix='R3') as pool:
            pending=set();next_z=0;completed=0
            try:
                while next_z<len(raw) or pending:
                    check()
                    while next_z<len(raw) and len(pending)<workers:
                        pending.add(pool.submit(plane,next_z));next_z+=1
                    done,pending=wait(pending,timeout=.1,return_when=FIRST_COMPLETED)
                    for future in done:
                        z,scale,result=future.result()
                        output[z]=result;scales[z]=scale;completed+=1
                        check()
                        progress(round(100*completed/len(raw)),'R3 preprocessing',f'{completed} / {len(raw)}')
            finally:
                for future in pending:future.cancel()
        output.flush()
        output._mmap.close()
        output = None
        partial.replace(destination)
    finally:
        if output is not None:
            output._mmap.close()
        partial.unlink(missing_ok=True)
    digest = hashlib.sha256()
    with destination.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            check()
            digest.update(chunk)
    return dict(enabled=True, method='Membrane enhancement', file=destination.name,
                sha256=digest.hexdigest(), source_grid=True, shape=list(raw.shape),
                normalization=normalization, scale_per_z=scales,
                workers=workers,execution='balanced-only kernel; parallel Z threads',
                execution_implementation_sha256=file_hash(Path(__file__).with_name('r3_fast.py')),
                implementation_sha256=hashlib.sha256(Path(__file__).with_name('membrane_r3.py').read_bytes()).hexdigest(),
                seconds=round(time.time() - started, 3))


if __name__ == '__main__':
    import sys, json, traceback
    from engine import read_volume
    try:
        request = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
        def report(percent, stage, detail):
            print(json.dumps(dict(percent=percent, stage=stage, detail=detail)), flush=True)
        raw = read_volume(request['input'], request.get('dataset') or None)
        info = preprocess(raw, request['output'], request.get('normalization', 'auto'), report,workers=request.get('workers',8))
        info.update(input=str(Path(request['input']).resolve()),
                    input_sha256=file_hash(request['input']),dataset=request.get('dataset') or '')
        Path(request['output']).with_suffix('.json').write_text(json.dumps(info,indent=2),encoding='utf-8')
    except Exception:
        traceback.print_exc()
        sys.exit(1)
