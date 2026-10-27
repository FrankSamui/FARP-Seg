# FARP-Seg segmentation kernels.

from __future__ import annotations

import math, gc

import numpy as np

from scipy import ndimage as ndi
from scipy import sparse

MARKER_BLOCK_Z = 6

MARKER_BLOCK_YX = 24

MARKER_MAX_BOUNDARY = 0.45

PREDICTION_SUPPORT_THRESHOLDS = (0.10, 0.15, 0.20, 0.25, 0.30)

SUPPORT_MAX_MASK_FRACTION = 0.90

SUPPORT_DILATION_XY = 2

SUPPORT_DILATION_Z = 1

SUPPORT_CLOSING_YX = 5

SUPPORT_MIN_COMPONENT_VOXELS = 32

RAG_Z_BLOCK = 6

TARGET_ATOMS_PER_VOLUME = 160_000

MAX_ATOMS_PER_VOLUME = 220_000

MORPH_HIGH_THRESHOLD = 0.50

MORPH_SMALL_WINDOW = 3

MORPH_LARGE_WINDOW = 9

MORPH_COHERENCE_WINDOW = 5

def _remove_small_2d_components(mask: np.ndarray, min_voxels: int) -> np.ndarray:
    labels, n = ndi.label(mask)
    if n == 0:
        return np.zeros(mask.shape, dtype=bool)
    sizes = np.bincount(labels.ravel())
    keep = sizes >= int(min_voxels)
    keep[0] = False
    return keep[labels]

def _prediction_mask_at_threshold(pred: np.ndarray, threshold: float) -> np.ndarray:
    mask = np.zeros(pred.shape, dtype=bool)
    structure_xy = ndi.generate_binary_structure(2, 1)
    closing_structure = np.ones((SUPPORT_CLOSING_YX, SUPPORT_CLOSING_YX), dtype=bool)
    for z in range(pred.shape[0]):
        m = pred[z] >= float(threshold)
        m = _remove_small_2d_components(m, SUPPORT_MIN_COMPONENT_VOXELS)
        if not np.any(m):
            continue
        if SUPPORT_DILATION_XY > 0:
            m = ndi.binary_dilation(m, structure=structure_xy, iterations=SUPPORT_DILATION_XY)
        m = ndi.binary_closing(m, structure=closing_structure)
        m = ndi.binary_fill_holes(m)
        m = _remove_small_2d_components(m, SUPPORT_MIN_COMPONENT_VOXELS)
        mask[z] = m
    if SUPPORT_DILATION_Z > 0 and np.any(mask):
        mask = ndi.binary_dilation(
            mask,
            structure=np.ones((3, 1, 1), dtype=bool),
            iterations=SUPPORT_DILATION_Z,
        )
    return mask

def build_prediction_mask(pred: np.ndarray) -> tuple[np.ndarray, float, float]:
    best_mask = None
    best_threshold = float(PREDICTION_SUPPORT_THRESHOLDS[-1])
    best_fraction = math.inf
    for threshold in PREDICTION_SUPPORT_THRESHOLDS:
        mask = _prediction_mask_at_threshold(pred, float(threshold))
        fraction = float(np.mean(mask))
        if fraction < best_fraction:
            if best_mask is not None:
                del best_mask
            best_mask = mask
            best_threshold = float(threshold)
            best_fraction = fraction
        elif mask is not best_mask:
            candidate = mask
            del candidate
        if np.any(mask) and fraction <= SUPPORT_MAX_MASK_FRACTION:
            return mask, float(threshold), fraction
        if mask is not best_mask:
            del mask
        gc.collect()
    if best_mask is None or not np.any(best_mask):
        raise RuntimeError("Prediction-only support mask is empty at all configured thresholds")
    return best_mask, best_threshold, best_fraction

def bbox_from_mask(mask: np.ndarray) -> tuple[slice, slice, slice]:
    z_any = np.any(mask, axis=(1, 2))
    y_any = np.any(mask, axis=(0, 2))
    x_any = np.any(mask, axis=(0, 1))
    if not np.any(z_any) or not np.any(y_any) or not np.any(x_any):
        return slice(0, mask.shape[0]), slice(0, mask.shape[1]), slice(0, mask.shape[2])
    z_idx = np.flatnonzero(z_any)
    y_idx = np.flatnonzero(y_any)
    x_idx = np.flatnonzero(x_any)
    return (
        slice(max(0, int(z_idx[0]) - 1), min(mask.shape[0], int(z_idx[-1]) + 2)),
        slice(max(0, int(y_idx[0]) - 8), min(mask.shape[1], int(y_idx[-1]) + 9)),
        slice(max(0, int(x_idx[0]) - 8), min(mask.shape[2], int(x_idx[-1]) + 9)),
    )

def _marker_grid_capacity(shape: tuple[int, int, int], block_z: int, block_yx: int) -> int:
    nz, ny, nx = shape
    return math.ceil(nz / block_z) * math.ceil(ny / block_yx) * math.ceil(nx / block_yx)

def choose_marker_block_sizes(shape: tuple[int, int, int]) -> tuple[int, int, int]:
    block_z = int(MARKER_BLOCK_Z)
    block_yx = int(MARKER_BLOCK_YX)
    capacity = _marker_grid_capacity(shape, block_z, block_yx)
    if capacity <= TARGET_ATOMS_PER_VOLUME:
        return block_z, block_yx, capacity
    scale = (capacity / float(TARGET_ATOMS_PER_VOLUME)) ** (1.0 / 3.0)
    block_z = max(MARKER_BLOCK_Z, int(math.ceil(MARKER_BLOCK_Z * scale)))
    block_yx = max(MARKER_BLOCK_YX, int(math.ceil(MARKER_BLOCK_YX * scale)))
    capacity = _marker_grid_capacity(shape, block_z, block_yx)
    while capacity > TARGET_ATOMS_PER_VOLUME:
        block_yx += 1
        capacity = _marker_grid_capacity(shape, block_z, block_yx)
    return block_z, block_yx, capacity

def build_dense_markers(pred: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, int, int, int, int]:
    block_z, block_yx, grid_capacity = choose_marker_block_sizes(pred.shape)
    markers = np.zeros(pred.shape, dtype=np.int32)
    marker_id = 0
    nz, ny, nx = pred.shape
    for z0 in range(0, nz, block_z):
        z1 = min(nz, z0 + block_z)
        for y0 in range(0, ny, block_yx):
            y1 = min(ny, y0 + block_yx)
            for x0 in range(0, nx, block_yx):
                x1 = min(nx, x0 + block_yx)
                local_mask = mask[z0:z1, y0:y1, x0:x1]
                if not np.any(local_mask):
                    continue
                local_pred = pred[z0:z1, y0:y1, x0:x1]
                safe = local_mask & (local_pred <= MARKER_MAX_BOUNDARY)
                score_mask = safe if np.any(safe) else local_mask
                score = np.where(score_mask, local_pred, np.inf)
                flat_index = int(np.argmin(score))
                lz, ly, lx = np.unravel_index(flat_index, score.shape)
                marker_id += 1
                markers[z0 + lz, y0 + ly, x0 + lx] = marker_id
    return markers, marker_id, block_z, block_yx, grid_capacity

def infer_prediction_scale_from_canonical(pred_raw: np.ndarray) -> float:
    z_count = pred_raw.shape[0]
    z_indices = sorted(set(int(v) for v in np.linspace(0, z_count - 1, min(12, z_count), dtype=int)))
    sampled = np.stack([np.asarray(pred_raw[z, ::8, ::8]) for z in z_indices], axis=0)
    finite = sampled[np.isfinite(sampled)]
    if finite.size == 0:
        return 1.0
    q999 = float(np.percentile(finite, 99.9))
    vmax = float(np.max(finite))
    dtype = pred_raw.dtype
    if np.issubdtype(dtype, np.floating):
        if vmax <= 1.5:
            return 1.0
        if vmax <= 255.5:
            return 255.0
        if vmax <= 4095.5:
            return 4095.0
        return max(vmax, 1.0)
    if q999 <= 255.0 and vmax <= 255.0:
        return 255.0
    if q999 <= 4095.0 and vmax <= 4095.0:
        return 4095.0
    return 65535.0

def normalize_prediction(pred: np.ndarray, scale: float) -> np.ndarray:
    out = pred.astype(np.float32, copy=False) / max(float(scale), 1e-12)
    return np.clip(out, 0.0, 1.0)

def compute_node_geometry(atoms: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n_atoms = int(atoms.max())
    counts = np.zeros(n_atoms + 1, dtype=np.int64)
    sum_z = np.zeros(n_atoms + 1, dtype=np.float64)
    sum_y = np.zeros(n_atoms + 1, dtype=np.float64)
    sum_x = np.zeros(n_atoms + 1, dtype=np.float64)
    for z in range(atoms.shape[0]):
        frame = np.asarray(atoms[z])
        yy, xx = np.nonzero(frame > 0)
        if yy.size == 0:
            continue
        ids = frame[yy, xx].astype(np.int64, copy=False)
        counts += np.bincount(ids, minlength=n_atoms + 1)
        sum_z += np.bincount(ids, weights=np.full(ids.shape, z, dtype=np.float64), minlength=n_atoms + 1)
        sum_y += np.bincount(ids, weights=yy.astype(np.float64, copy=False), minlength=n_atoms + 1)
        sum_x += np.bincount(ids, weights=xx.astype(np.float64, copy=False), minlength=n_atoms + 1)
    centroids = np.zeros((n_atoms + 1, 3), dtype=np.float32)
    valid = counts > 0
    centroids[valid, 0] = (sum_z[valid] / counts[valid]).astype(np.float32)
    centroids[valid, 1] = (sum_y[valid] / counts[valid]).astype(np.float32)
    centroids[valid, 2] = (sum_x[valid] / counts[valid]).astype(np.float32)
    return counts, centroids

def morphology_fields(pred_block: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    high = (pred_block >= MORPH_HIGH_THRESHOLD).astype(np.float32)
    occ_small = ndi.uniform_filter(high, size=(1, MORPH_SMALL_WINDOW, MORPH_SMALL_WINDOW), mode="nearest")
    occ_large = ndi.uniform_filter(high, size=(1, MORPH_LARGE_WINDOW, MORPH_LARGE_WINDOW), mode="nearest")
    gy = np.gradient(pred_block, axis=1)
    gx = np.gradient(pred_block, axis=2)
    jxx = ndi.uniform_filter(gx * gx, size=(1, MORPH_COHERENCE_WINDOW, MORPH_COHERENCE_WINDOW), mode="nearest")
    jyy = ndi.uniform_filter(gy * gy, size=(1, MORPH_COHERENCE_WINDOW, MORPH_COHERENCE_WINDOW), mode="nearest")
    jxy = ndi.uniform_filter(gx * gy, size=(1, MORPH_COHERENCE_WINDOW, MORPH_COHERENCE_WINDOW), mode="nearest")
    coherence = np.sqrt((jxx - jyy) ** 2 + 4.0 * jxy * jxy) / (jxx + jyy + 1e-6)
    return occ_small.astype(np.float32), occ_large.astype(np.float32), coherence.astype(np.float32)

def aggregate_block_edges(atom_a, atom_b, pred_a, pred_b, small_a, small_b, large_a, large_b, coh_a, coh_b):
    valid = (atom_a > 0) & (atom_b > 0) & (atom_a != atom_b)
    if not np.any(valid):
        return np.empty(0, dtype=np.int64), np.empty((0, 8), dtype=np.float64)
    a = atom_a[valid].astype(np.int64, copy=False)
    b = atom_b[valid].astype(np.int64, copy=False)
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    keys = (lo << 32) | hi
    boundary = 0.5 * (pred_a[valid].astype(np.float32) + pred_b[valid].astype(np.float32))
    occ_small = 0.5 * (small_a[valid] + small_b[valid])
    occ_large = 0.5 * (large_a[valid] + large_b[valid])
    coherence = 0.5 * (coh_a[valid] + coh_b[valid])
    order = np.argsort(keys, kind="mergesort")
    keys = keys[order]
    boundary = boundary[order]
    occ_small = occ_small[order]
    occ_large = occ_large[order]
    coherence = coherence[order]
    unique_keys, starts, counts = np.unique(keys, return_index=True, return_counts=True)
    stats = np.column_stack(
        [
            counts.astype(np.float64),
            np.add.reduceat(boundary, starts),
            np.add.reduceat(boundary * boundary, starts),
            np.maximum.reduceat(boundary, starts),
            np.add.reduceat((boundary >= MORPH_HIGH_THRESHOLD).astype(np.float32), starts),
            np.add.reduceat(occ_small, starts),
            np.add.reduceat(occ_large, starts),
            np.add.reduceat(coherence, starts),
        ]
    ).astype(np.float64)
    return unique_keys.astype(np.int64), stats

def build_rag_edge_features(atoms, pred, node_counts, centroids):
    global_acc: dict[int, np.ndarray] = {}
    nz = atoms.shape[0]
    for z0 in range(0, nz, RAG_Z_BLOCK):
        z1 = min(nz, z0 + RAG_Z_BLOCK)
        read_z1 = min(nz, z1 + 1)
        a = np.asarray(atoms[z0:read_z1])
        p = pred[z0:read_z1]
        small, large, coherence = morphology_fields(p)
        core = z1 - z0
        contributions = [
            aggregate_block_edges(
                a[:core, :, :-1], a[:core, :, 1:], p[:core, :, :-1], p[:core, :, 1:],
                small[:core, :, :-1], small[:core, :, 1:], large[:core, :, :-1], large[:core, :, 1:],
                coherence[:core, :, :-1], coherence[:core, :, 1:]
            ),
            aggregate_block_edges(
                a[:core, :-1, :], a[:core, 1:, :], p[:core, :-1, :], p[:core, 1:, :],
                small[:core, :-1, :], small[:core, 1:, :], large[:core, :-1, :], large[:core, 1:, :],
                coherence[:core, :-1, :], coherence[:core, 1:, :]
            ),
        ]
        z_faces = min(core, a.shape[0] - 1)
        if z_faces > 0:
            contributions.append(
                aggregate_block_edges(
                    a[:z_faces], a[1:z_faces + 1], p[:z_faces], p[1:z_faces + 1],
                    small[:z_faces], small[1:z_faces + 1], large[:z_faces], large[1:z_faces + 1],
                    coherence[:z_faces], coherence[1:z_faces + 1]
                )
            )
        for keys, stats in contributions:
            for key, values in zip(keys.tolist(), stats):
                existing = global_acc.get(key)
                if existing is None:
                    global_acc[key] = values.copy()
                else:
                    existing[0] += values[0]
                    existing[1] += values[1]
                    existing[2] += values[2]
                    existing[3] = max(existing[3], values[3])
                    existing[4:] += values[4:]
        del a, p, small, large, coherence, contributions
        gc.collect()

    if not global_acc:
        raise RuntimeError("RAG contains no adjacency edges")
    keys = np.asarray(sorted(global_acc.keys()), dtype=np.int64)
    raw = np.vstack([global_acc[int(key)] for key in keys]).astype(np.float64)
    u = (keys >> 32).astype(np.int32)
    v = (keys & np.int64(0xFFFFFFFF)).astype(np.int32)
    edge_index = np.column_stack([u, v]).astype(np.int32)

    count = raw[:, 0]
    b_mean = raw[:, 1] / np.maximum(count, 1.0)
    b_var = raw[:, 2] / np.maximum(count, 1.0) - b_mean * b_mean
    b_std = np.sqrt(np.maximum(b_var, 0.0))
    b_max = raw[:, 3]
    b_high_fraction = raw[:, 4] / np.maximum(count, 1.0)
    occ_small = raw[:, 5] / np.maximum(count, 1.0)
    occ_large = raw[:, 6] / np.maximum(count, 1.0)
    coherence = raw[:, 7] / np.maximum(count, 1.0)

    vol_u = node_counts[u].astype(np.float64)
    vol_v = node_counts[v].astype(np.float64)
    log_shared = np.log1p(count)
    log_vol_u = np.log1p(vol_u)
    log_vol_v = np.log1p(vol_v)
    log_volume_ratio = np.abs(np.log((vol_u + 1.0) / (vol_v + 1.0)))
    delta = centroids[u] - centroids[v]
    scale = np.asarray(atoms.shape, dtype=np.float32)
    centroid_distance = np.linalg.norm(delta / np.maximum(scale[None, :], 1.0), axis=1)
    blob_score = b_high_fraction * occ_large
    thin_ridge_score = b_mean * (1.0 - occ_large) * coherence
    features = np.column_stack(
        [
            b_mean, b_std, b_max, b_high_fraction,
            log_shared, log_vol_u, log_vol_v, log_volume_ratio, centroid_distance,
            occ_small, occ_large, coherence, blob_score, thin_ridge_score,
        ]
    ).astype(np.float32)
    return edge_index, features, raw.astype(np.float32)

class UnionFind:
    def __init__(self, n: int):
        self.parent = np.arange(n, dtype=np.int32)
        self.rank = np.zeros(n, dtype=np.uint8)

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = int(self.parent[x])
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(int(a)), self.find(int(b))
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1

    def labels(self) -> np.ndarray:
        result = np.zeros(len(self.parent), dtype=np.int32)
        mapping: dict[int, int] = {}
        next_label = 1
        for i in range(1, len(self.parent)):
            root = self.find(i)
            if root not in mapping:
                mapping[root] = next_label
                next_label += 1
            result[i] = mapping[root]
        return result

def decode_merge_edges(n_atoms: int, edge_index: np.ndarray, merge_state: np.ndarray) -> np.ndarray:
    """Decode MERGE edges with SciPy's compiled connected-components routine.

    Atom 0 is kept as an isolated sentinel. The returned component IDs are
    arbitrary labels, which is sufficient for contingency-based segmentation
    metrics and subsequent region grouping.
    """
    merge_mask = np.asarray(merge_state, dtype=bool)
    if not np.any(merge_mask):
        labels = np.arange(n_atoms + 1, dtype=np.int32)
        labels[0] = 0
        return labels

    selected = np.asarray(edge_index[merge_mask], dtype=np.int32)
    data = np.ones(len(selected), dtype=np.uint8)
    adjacency = sparse.csr_matrix(
        (data, (selected[:, 0], selected[:, 1])),
        shape=(n_atoms + 1, n_atoms + 1),
        dtype=np.uint8,
    )
    _, labels = sparse.csgraph.connected_components(
        adjacency,
        directed=False,
        return_labels=True,
    )
    labels = labels.astype(np.int32, copy=False)
    # Node 0 is isolated and is visited first by connected_components, so its
    # component is 0. Keep this invariant explicit for downstream code.
    if labels[0] != 0:
        zero_component = int(labels[0])
        zero_mask = labels == zero_component
        labels = labels.copy()
        labels[zero_mask] = 0
        conflicting = (~zero_mask) & (labels == 0)
        if np.any(conflicting):
            labels[conflicting] = int(labels.max()) + 1
    return labels
