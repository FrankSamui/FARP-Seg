"""Prediction-only desktop inference. No reference or experimental cache reads."""
from __future__ import annotations

import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_name, '4')
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from scipy import ndimage as ndi
from skimage.segmentation import watershed
from skimage.measure import label
import historical_core as old

ROOT = Path(__file__).resolve().parent

def json_value(value):
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,np.ndarray):return value.tolist()
    raise TypeError(f'Unsupported metadata value: {type(value).__name__}')

DEFAULTS = dict(xy_step=2, fold=0, threshold=0.942, marker_z=6, marker_xy=24,
                cleanup=True, width=6, fraction=.60, window=5, radius=3,
                flip_y=False, normalization='auto', threads=4,
                f_enabled=True, f_neck=.85, f_enclosure=.50,
                r_enabled=True, r_budget=32, stitch_enabled=True,
                post_enabled=True, pf_threshold=.10,input_is_crop=False,
                preprocessing_enabled=False,preprocessing_workers=8)


def read_volume(path, dataset=None):
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == '.npy':
        arr = np.load(path, mmap_mode='r', allow_pickle=False)
    elif suffix == '.npz':
        with np.load(path, allow_pickle=False) as f:
            key = dataset or ('prediction' if 'prediction' in f else f.files[0])
            arr = np.array(f[key])
    elif suffix in ('.h5', '.hdf5'):
        import h5py
        with h5py.File(path, 'r') as f:
            keys = []
            f.visititems(lambda name, obj: keys.append(name) if isinstance(obj, h5py.Dataset) and obj.ndim in (2, 3) else None)
            if dataset is None:
                preferred = [k for k in keys if k.split('/')[-1].lower() in ('prediction', 'predictions', 'exported_data')]
                if len(preferred) == 1:
                    dataset = preferred[0]
                elif len(keys) == 1:
                    dataset = keys[0]
                else:
                    raise ValueError('HDF5 含多个数据集，请填写数据集名称：' + ', '.join(keys[:15]))
            arr = np.asarray(f[dataset])
    elif suffix in ('.tif', '.tiff'):
        import tifffile
        with tifffile.TiffFile(path) as tf:
            axes = tf.series[0].axes
            if 'S' in axes or 'C' in axes:
                raise ValueError('TIFF 含颜色/通道轴，请先选择单个 prediction 通道后导入。')
        try:
            arr = tifffile.memmap(path)
        except ValueError:
            arr = tifffile.imread(path)
    else:
        raise ValueError('支持 TIFF、NPY、NPZ、HDF5 文件。')
    if arr.ndim == 2:
        arr = arr[None]
    if arr.ndim != 3 or arr.dtype.kind not in 'uifb':
        raise ValueError(f'需要灰度 Z×Y×X 体积，实际 shape={arr.shape}, dtype={arr.dtype}。多通道数据请先选择通道。')
    if min(arr.shape) == 0 or min(arr.shape[1:]) < 2:
        raise ValueError('输入为空或 XY 尺寸小于 2。')
    return arr


def clean_exterior(atoms, mapping, cfg, progress, check, result=None):
    """Exterior cleanup using fixed contours, local bands, and thin components.

    Applied after the full PF_PA current-cell cleanup endpoint.
    """
    n = len(mapping); nz = len(atoms)
    area = np.zeros((nz, n), np.float64)
    band = np.zeros_like(area)
    width, frac = cfg['width'], cfg['fraction']
    for z, aa in enumerate(atoms):
        check()
        mask = mapping[aa] > 0
        area[z] = np.bincount(aa[mask], minlength=n)
        filled = ndi.binary_fill_holes(mask)
        if not filled.all():
            dist = ndi.distance_transform_edt(np.pad(filled, 1, constant_values=True))[1:-1, 1:-1]
            band[z] = np.bincount(aa[mask & (dist <= width)], minlength=n)
        if z % 16 == 0:
            progress(84+int(4*(z+1)/nz),'P · 固定初始外轮廓',f'统计条带 {z+1} / {nz}')
    total = area.sum(axis=0)
    global_cut = (total > 0) & (band.sum(axis=0) >= frac * total)
    # constant padding is the truncated Z±2 window used in the sparse reference code.
    kernel=np.ones(cfg['window'])
    smooth_area = ndi.convolve1d(area, kernel, axis=0, mode='constant')
    smooth_band = ndi.convolve1d(band, kernel, axis=0, mode='constant')
    local_cut = (smooth_area > 0) & (smooth_band >= frac * smooth_area)
    if result is None:result = np.empty(atoms.shape, np.uint32)
    for z, aa in enumerate(atoms):
        check()
        original = mapping[aa]
        current = original.copy()
        current[global_cut[aa] | local_cut[z, aa]] = 0
        mask = original > 0
        filled = ndi.binary_fill_holes(mask)
        if not filled.all():
            comp = label(np.where(mask, aa, 0), connectivity=1, background=0)
            count = int(comp.max()) + 1
            size = np.bincount(comp.ravel(), minlength=count)
            interior = comp > 0
            interior[1:] &= comp[1:] == comp[:-1]
            interior[:-1] &= comp[:-1] == comp[1:]
            interior[:, 1:] &= comp[:, 1:] == comp[:, :-1]
            interior[:, :-1] &= comp[:, :-1] == comp[:, 1:]
            radius = ndi.maximum(ndi.distance_transform_edt(interior), labels=comp, index=np.arange(count))
            thin = (radius <= cfg['radius']) & (size > 0)
            thin[0] = False
            thin[np.unique(np.r_[comp[0], comp[-1], comp[:, 0], comp[:, -1]])] = False
            dist = ndi.distance_transform_edt(np.pad(filled, 1, constant_values=True))[1:-1, 1:-1]
            inside = np.bincount(comp[mask & (dist <= width)], minlength=count)
            current[(thin & (inside >= frac * size))[comp]] = 0
        result[z] = current
        if z % 8 == 0:
            progress(88 + int(7 * (z + 1) / nz), 'P · 外缘三步清理', f'切片 {z + 1} / {nz}')
    return result


def infer(path, output, config=None, progress=lambda *a: None, check=lambda: None):
    cfg = dict(DEFAULTS, **(config or {}))
    if cfg['xy_step'] not in (1, 2, 4) or cfg['window'] < 1 or cfg['window'] % 2 == 0:
        raise ValueError('XY 采样应为 1/2/4，Z 窗口应为正奇数。')
    if cfg['r_budget'] not in (8,16,32):
        raise ValueError('重划分区域预算应为 8、16 或 32。')
    import full_pipeline as full
    assets=full.Assets(cfg['fold'],cfg['threads']) if any(cfg[k] for k in ('r_enabled','stitch_enabled','post_enabled')) else None
    stage_stats={};stage_maps={};stage_times={}
    def mark(name,start):stage_times[name]=round(time.time()-start,3)
    started = time.time()
    progress(3, '读取 prediction', Path(path).name)
    raw = read_volume(path, cfg.get('dataset') or None)
    original_shape = tuple(raw.shape)
    step = cfg['xy_step']
    for z in range(raw.shape[0]):
        check()
        if not np.isfinite(raw[z]).all():
            raise ValueError('prediction 含 NaN/Inf，请先清理无效值。')
    scale = old.infer_prediction_scale_from_canonical(raw) if cfg['normalization'] == 'auto' else float(cfg['normalization'])
    preprocessing = dict(enabled=False)
    if cfg['preprocessing_enabled']:
        from preprocessing import file_hash
        import shutil
        source = Path(cfg.get('preprocessing_input') or '')
        preprocessing = dict(cfg.get('preprocessing_metadata') or {})
        if not source.is_file() or not preprocessing.get('enabled'):
            raise ValueError('Run preprocessing before using its result.')
        if (preprocessing.get('normalization') != cfg['normalization'] or
                preprocessing.get('dataset','') != (cfg.get('dataset') or '') or
                preprocessing.get('input_sha256') != file_hash(path)):
            raise ValueError('Input settings changed. Run preprocessing again.')
        if preprocessing.get('sha256') != file_hash(source):
            raise ValueError('Preprocessing result checksum mismatch.')
        target = Path(output).parent / 'preprocessed.npy'
        target.parent.mkdir(parents=True,exist_ok=True)
        if source.resolve() != target.resolve():
            temporary = target.with_name(target.name+'.partial')
            temporary.unlink(missing_ok=True)
            try:os.link(source,temporary)
            except OSError:shutil.copyfile(source,temporary)
            temporary.replace(target)
        preprocessing['file'] = target.name
        processed = np.load(target, mmap_mode='r', allow_pickle=False)
        if processed.shape != raw.shape or processed.dtype != np.float32:
            raise ValueError('Preprocessing shape or dtype mismatch.')
        pred = old.normalize_prediction(np.asarray(processed[:, ::step, ::step]), 1)
    else:
        pred = old.normalize_prediction(np.asarray(raw[:, ::step, ::step]), scale)
    if not np.isfinite(pred).all():
        raise ValueError('prediction 含 NaN/Inf，请先清理无效值。')
    if cfg['flip_y']:
        pred = pred[:, ::-1].copy()
    check(); progress(10, '提取组织区域', '沿用项目的自适应 support mask')
    mask, support_threshold, _ = old.build_prediction_mask(pred)
    box = old.bbox_from_mask(mask)
    roi = np.ascontiguousarray(pred[box])
    support = np.ascontiguousarray(mask[box])
    old.MARKER_BLOCK_Z = cfg['marker_z']; old.MARKER_BLOCK_YX = cfg['marker_xy']
    markers, n_markers, bz, bxy, _ = old.build_dense_markers(roi, support)
    if not n_markers:
        raise ValueError('未找到可用分水岭种子。')
    check(); progress(25, '三维分水岭', f'{n_markers:,} 个种子 · {tuple(roi.shape)}')
    atoms = watershed(roi, markers=markers, mask=support, connectivity=1).astype(np.int32)
    del markers, support, mask
    source_sides=dict(top=True,bottom=True,left=True,right=True)
    sidecar=Path(path).with_suffix('.json')
    if sidecar.exists():
        source_sides.update(json.loads(sidecar.read_text(encoding='utf-8')).get('frame_sides',{}))
    if cfg['input_is_crop']:source_sides={k:False for k in source_sides}
    if cfg['flip_y']:source_sides['top'],source_sides['bottom']=source_sides['bottom'],source_sides['top']
    na=int(atoms.max());frame_sides=dict(top=box[1].start==0 and source_sides['top'],bottom=box[1].stop==pred.shape[1] and source_sides['bottom'],left=box[2].start==0 and source_sides['left'],right=box[2].stop==pred.shape[2] and source_sides['right'])
    tick=time.time()
    if cfg.get('capture_stages') and cfg['f_enabled']:
        progress(26,'保存论文阶段','独立计算无 F、重分割前的合并结果')
        from paper_stages import a_only_mapping
        stage_maps['A_NO_F']=a_only_mapping(atoms,roi,cfg)
        mark('A_NO_F',tick);tick=time.time()
    if cfg['f_enabled']:
        atoms,f_features,f_stats=full.f_atoms(atoms,frame_sides,cfg,progress,check)
    else:f_features=np.empty((0,7));f_stats=dict(n_A_atoms=na,n_F_atoms=0,disabled=True)
    if cfg.get('capture_stages'):
        snapshot_folder=Path(output).parent;snapshot_folder.mkdir(parents=True,exist_ok=True)
        np.save(snapshot_folder/'viewer_atoms.npy',atoms,allow_pickle=False)
    mark('F',tick);stage_stats['F']=f_stats
    check(); progress(40, '计算 A14 特征', '边界响应、接触面积与三维几何')
    counts, centers = old.compute_node_geometry(atoms)
    mapping = np.arange(int(atoms.max()) + 1, dtype=np.int32)
    edge_count = 0
    try:
        edges, features, raw_edges = old.build_rag_edge_features(atoms, roi, counts, centers)
    except RuntimeError as e:
        if 'no adjacency edges' not in str(e):
            raise
        edges=np.empty((0,2),np.int32);features=np.empty((0,14),np.float32);raw_edges=np.empty((0,8),np.float32);prob=np.empty(0)
    else:
        import lightgbm as lgb
        check(); progress(50, 'A14 模型合并', f'{len(edges):,} 条邻接边')
        model_path = ROOT / 'resources' / 'models' / f'a14_fold{cfg["fold"]}.txt'
        manifest = json.loads((model_path.parent / 'manifest.json').read_text(encoding='utf-8'))
        expected = manifest['fold_artifacts'][str(cfg['fold'])]['model_sha256']
        if hashlib.sha256(model_path.read_bytes()).hexdigest() != expected:
            raise ValueError('A14 模型校验失败，请重新复制原始模型。')
        model = lgb.Booster(model_file=str(model_path))
        prob = model.predict(features, num_threads=cfg['threads'])
        mapping = old.decode_merge_edges(int(atoms.max()), edges, prob >= cfg['threshold'])
        edge_count = len(edges)
    stage_maps['A14']=mapping.copy()
    if cfg.get('capture_stages') and 'A_NO_F' not in stage_maps:stage_maps['A_NO_F']=mapping.copy()
    a=dict(mapping=mapping.copy(),prob=prob,edges=edges,mass=counts,base=features,faces=raw_edges[:,0].astype(float))
    occupied=set();tick=time.time()
    if cfg['r_enabled'] and len(edges):
        progress(54,'准备重划分几何','统计逐层原子形状与三维表面')
        slabs=full.node_slabs(atoms,check)
        mapping,occupied,rstats=full.risk_repartition(a,slabs,len(atoms),assets,cfg,progress,check)
        del slabs
        stage_stats['R']=rstats
    else:stage_stats['R']=dict(actions=0,disabled=not cfg['r_enabled'],no_edges=not len(edges))
    mark('R',tick);stage_maps['R']=mapping.copy();check();tick=time.time()
    if cfg['stitch_enabled'] and len(edges):
        progress(71,'残余碎片连接','物理体积加权收益 · 单轮互不冲突合并')
        mapping,sstats=full.stitch(a,mapping,occupied,assets);stage_stats['stitch']=sstats
    else:stage_stats['stitch']=dict(actions=0,disabled=not cfg['stitch_enabled'],no_edges=not len(edges))
    mark('stitch',tick);stage_maps['R_PLUS_ABS']=mapping.copy();check();tick=time.time()
    if cfg['post_enabled']:
        progress(77,'P-F / P-A 清理','训练折形态库与当前细胞接触图')
        mapping,pstats=full.postprocess(atoms,mapping,a,na,f_features,assets,cfg);stage_stats['PF_PA']=pstats
    else:stage_stats['PF_PA']=dict(disabled=True)
    mark('PF_PA',tick);stage_maps['PF_PA']=mapping.copy()
    check(); progress(83, '生成 segment', '固定初始外轮廓，保留真实图像裁剪边')
    tick=time.time()
    if cfg['cleanup'] and cfg['width'] > 0:
        segment = clean_exterior(atoms, mapping, cfg, progress, check)
    else:
        segment = mapping[atoms].astype(np.uint32)
    mark('P143',tick)
    result = np.zeros(pred.shape, np.uint32)
    result[box] = segment
    if cfg['flip_y']:
        result = result[:, ::-1].copy()
    # Preserve label IDs; restore the exact source shape with nearest-neighbor replication.
    if step > 1:
        result = np.repeat(np.repeat(result, step, axis=1), step, axis=2)[:, :original_shape[1], :original_shape[2]]
    ids = np.unique(result); n_cells = int(np.count_nonzero(ids))
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    progress(97, '保存结果', f'{n_cells:,} 个实例')
    partial = output.with_name(output.name + '.partial')
    with partial.open('wb') as f:
        np.save(f, result, allow_pickle=False)
    partial.replace(output)
    np.savez_compressed(output.parent/'stage_cluster_maps.npz',**stage_maps)
    optional={'Membrane enhancement':cfg['preprocessing_enabled'],'Region filling':cfg['f_enabled'],'risk-aware repartition':cfg['r_enabled'],'residual fragment stitching':cfg['stitch_enabled'],'PF_PA':cfg['post_enabled'],'P143 exterior cleanup':cfg['cleanup'] and cfg['width']>0}
    meta = dict(pipeline='FARP-Seg',
                included=['prediction support','3D watershed','A14 model merge']+[k for k,v in optional.items() if v],
                not_included=[k for k,v in optional.items() if not v],stage_statistics=stage_stats,stage_seconds=stage_times,
                input=str(Path(path).resolve()), shape=list(result.shape), dtype='uint32',
                axes='ZYX', background=0, parameters=cfg, prediction_scale=scale,
                preprocessing=preprocessing,
                support_threshold=support_threshold, atoms=int(atoms.max()), original_atoms=na,edges=edge_count,
                instances=n_cells, seconds=round(time.time()-started, 3),
                bbox_processing_grid=[[s.start, s.stop] for s in box], marker_block=[bz,bxy,bxy],
                frame_sides=frame_sides,seed=83131)
    encoded=json.dumps(meta,ensure_ascii=False,indent=2,default=json_value)
    meta=json.loads(encoded)
    output.with_suffix('.json').write_text(encoded, encoding='utf-8')
    if cfg.get('capture_stages'):
        from paper_stages import write_manifest
        write_manifest(output,meta)
    progress(100, '分割完成', f'{n_cells:,} 个实例 · {meta["seconds"]:.1f} 秒')
    return meta


def export_volume(segment, destination, metadata):
    destination = Path(destination)
    partial = destination.with_name(destination.name + '.partial')
    suffix = destination.suffix.lower()
    try:
        if suffix == '.npy':
            with partial.open('wb') as f:
                np.save(f, segment, allow_pickle=False)
        elif suffix in ('.tif', '.tiff'):
            import tifffile
            tifffile.imwrite(partial, segment, photometric='minisblack', metadata={'axes':'ZYX'}, bigtiff=segment.nbytes >= 4_000_000_000)
        elif suffix in ('.h5', '.hdf5'):
            import h5py
            with h5py.File(partial, 'w') as f:
                ds = f.create_dataset('segment', data=segment, compression='gzip', compression_opts=1)
                ds.attrs['axes'] = 'ZYX'; ds.attrs['background'] = 0
        else:
            raise ValueError('导出格式应为 TIFF、NPY 或 HDF5。')
        partial.replace(destination)
        destination.with_suffix(destination.suffix+'.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2,default=json_value), encoding='utf-8')
    finally:
        partial.unlink(missing_ok=True)


if __name__ == '__main__':
    import sys, traceback
    def report(percent, stage, detail):
        print(json.dumps(dict(percent=percent, stage=stage, detail=detail), ensure_ascii=True), flush=True)
    try:
        request = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
        infer(request['input'], request['output'], request['config'], report)
    except Exception:
        traceback.print_exc()
        sys.exit(1)
