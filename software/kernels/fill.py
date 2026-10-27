# FARP-Seg segmentation kernels.

from __future__ import annotations
import math
from dataclasses import dataclass
import numpy as np
from scipy import ndimage as ndi
from skimage.segmentation import watershed
from . import rays as E22

ENCLOSURE_THRESHOLD = 0.50

RAY_DIRECTIONS = 64

PEAK_NMS_RADIUS = 2

MIN_PEAK_DEPTH = 1.0

MIN_CHAMBER_AREA = 4

MIN_SEGMENT_RAY_HITS = 1

@dataclass(frozen=True)
class BasinAdjacency:
    a: int
    b: int
    saddle_depth: float
    peak_min: float
    neck_ratio: float

@dataclass
class ChamberObservation:
    chamber_id: int
    center_y: int
    center_x: int
    peak_depth: float
    area: int
    initial_basin_count: int
    enclosure: float
    segment_fraction: float
    frame_fraction: float
    open_fraction: float
    segment_hits: int
    frame_hits: int
    open_hits: int
    touches_true_frame: bool
    accepted: bool

@dataclass
class ConfigRun:
    neck_ratio: float
    initial_basin_count: int
    consolidated_chamber_count: int
    chamber_labels: np.ndarray
    chamber_boundary_mask: np.ndarray
    observations: list[ChamberObservation]
    repair_labels: np.ndarray
    repair_mask: np.ndarray

def true_frame_distance(shape: tuple[int, int], frame_sides: dict[str, bool]) -> np.ndarray:
    h, w = int(shape[0]), int(shape[1])
    yy = np.arange(h, dtype=np.float32)[:, None]
    xx = np.arange(w, dtype=np.float32)[None, :]
    out = np.full((h, w), np.inf, dtype=np.float32)
    # Pixel-center distance to the virtual wall immediately outside the image.
    if frame_sides["top"]:
        out = np.minimum(out, yy + 0.5)
    if frame_sides["bottom"]:
        out = np.minimum(out, (h - 0.5) - yy)
    if frame_sides["left"]:
        out = np.minimum(out, xx + 0.5)
    if frame_sides["right"]:
        out = np.minimum(out, (w - 0.5) - xx)
    return out

def background_clearance(
    foreground: np.ndarray,
    frame_sides: dict[str, bool],
) -> np.ndarray:
    """Clearance to the nearest existing segment or true image frame.

    Existing segment pixels are hard walls.  The physical image frame is also a
    hard wall at half a pixel outside the array, so edge cells receive the same
    geometric treatment as cells bounded by an observed segment.  Ordinary crop
    edges do not receive this privilege.
    """
    fg = np.asarray(foreground, dtype=bool)
    if np.any(fg):
        dist_segment = ndi.distance_transform_edt(~fg).astype(np.float32, copy=False)
    else:
        dist_segment = np.full(fg.shape, np.inf, dtype=np.float32)
    dist_frame = true_frame_distance(fg.shape, frame_sides)
    depth = np.minimum(dist_segment, dist_frame)
    finite = np.isfinite(depth)
    if not np.all(finite):
        fallback = float(math.hypot(*fg.shape)) + 1.0
        depth = np.where(finite, depth, fallback).astype(np.float32, copy=False)
    depth[fg] = 0.0
    return depth

def initial_watershed_basins(
    foreground: np.ndarray,
    frame_sides: dict[str, bool],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    fg = np.asarray(foreground, dtype=bool)
    bg = ~fg
    depth = background_clearance(fg, frame_sides)
    size = 2 * int(PEAK_NMS_RADIUS) + 1
    local_max = depth == ndi.maximum_filter(depth, size=size, mode="nearest")
    maxima = bg & local_max & (depth >= float(MIN_PEAK_DEPTH))

    # A crop edge that is not a physical image frame has incomplete context.  It
    # may belong to a chamber, but it should not create a watershed marker itself.
    if not frame_sides["top"]:
        maxima[:2, :] = False
    if not frame_sides["bottom"]:
        maxima[-2:, :] = False
    if not frame_sides["left"]:
        maxima[:, :2] = False
    if not frame_sides["right"]:
        maxima[:, -2:] = False

    markers, n_markers = ndi.label(maxima, structure=ndi.generate_binary_structure(2, 2))
    if n_markers == 0 and np.any(bg):
        # Degenerate slices still receive one deterministic background marker so
        # the external-background control can be explicitly evaluated and rejected
        # later if it has no segment support.
        y, x = np.unravel_index(int(np.argmax(depth * bg)), depth.shape)
        markers = np.zeros(depth.shape, dtype=np.int32)
        markers[int(y), int(x)] = 1
        n_markers = 1

    basins = watershed(
        -depth,
        markers=np.asarray(markers, dtype=np.int32),
        mask=bg,
        connectivity=1,
        watershed_line=False,
    ).astype(np.int32, copy=False)
    return basins, depth, markers, int(n_markers)

def basin_peak_depths(labels: np.ndarray, depth: np.ndarray) -> np.ndarray:
    n = int(np.max(labels))
    peaks = np.zeros(n + 1, dtype=np.float64)
    if n == 0:
        return peaks
    values = ndi.maximum(depth, labels=labels, index=np.arange(1, n + 1))
    peaks[1:] = np.asarray(values, dtype=np.float64)
    return peaks

def basin_adjacencies(labels: np.ndarray, depth: np.ndarray) -> list[BasinAdjacency]:
    labels = np.asarray(labels, dtype=np.int32)
    depth = np.asarray(depth, dtype=np.float32)
    n = int(np.max(labels))
    if n <= 1:
        return []

    pair_keys: list[np.ndarray] = []
    saddle_values: list[np.ndarray] = []
    for a, b, da, db in (
        (labels[:, :-1], labels[:, 1:], depth[:, :-1], depth[:, 1:]),
        (labels[:-1, :], labels[1:, :], depth[:-1, :], depth[1:, :]),
    ):
        valid = (a > 0) & (b > 0) & (a != b)
        if not np.any(valid):
            continue
        av = a[valid].astype(np.int64, copy=False)
        bv = b[valid].astype(np.int64, copy=False)
        lo = np.minimum(av, bv)
        hi = np.maximum(av, bv)
        pair_keys.append(lo * (n + 1) + hi)
        saddle_values.append(np.minimum(da[valid], db[valid]).astype(np.float64, copy=False))

    if not pair_keys:
        return []

    keys = np.concatenate(pair_keys)
    saddles = np.concatenate(saddle_values)
    order = np.argsort(keys, kind="mergesort")
    keys = keys[order]
    saddles = saddles[order]
    starts = np.r_[0, np.flatnonzero(keys[1:] != keys[:-1]) + 1]
    unique_keys = keys[starts]
    max_saddles = np.maximum.reduceat(saddles, starts)
    peaks = basin_peak_depths(labels, depth)

    edges: list[BasinAdjacency] = []
    for key, saddle in zip(unique_keys.tolist(), max_saddles.tolist()):
        a = int(key // (n + 1))
        b = int(key % (n + 1))
        peak_min = float(min(peaks[a], peaks[b]))
        ratio = float(saddle / peak_min) if peak_min > 0.0 else 0.0
        ratio = float(np.clip(ratio, 0.0, 1.0))
        edges.append(
            BasinAdjacency(
                a=a,
                b=b,
                saddle_depth=float(saddle),
                peak_min=peak_min,
                neck_ratio=ratio,
            )
        )
    edges.sort(key=lambda e: (-e.neck_ratio, e.a, e.b))
    return edges

class UnionFind:
    def __init__(self, n: int):
        self.parent = np.arange(n + 1, dtype=np.int32)
        self.rank = np.zeros(n + 1, dtype=np.int8)

    def find(self, x: int) -> int:
        x = int(x)
        root = x
        while int(self.parent[root]) != root:
            root = int(self.parent[root])
        while int(self.parent[x]) != x:
            nxt = int(self.parent[x])
            self.parent[x] = root
            x = nxt
        return root

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1

def consolidate_basins(
    initial_labels: np.ndarray,
    depth: np.ndarray,
    edges: list[BasinAdjacency],
    merge_ratio: float,
) -> tuple[np.ndarray, np.ndarray]:
    n = int(np.max(initial_labels))
    if n == 0:
        return np.zeros_like(initial_labels, dtype=np.int32), np.zeros(1, dtype=np.int32)

    # Dynamic agglomeration is essential.  A shallow distance-ridge plateau in a
    # narrow corridor can itself become a tiny watershed basin.  Static union of
    # all pairwise high-ratio edges would let that corridor transitively collapse
    # two real chambers.  After each merge, therefore, the component peak is
    # updated before the next edge is tested.
    peaks = basin_peak_depths(initial_labels, depth)
    component_peak = peaks.copy()
    uf = UnionFind(n)
    ordered_edges = sorted(
        edges,
        key=lambda e: (
            -float(e.saddle_depth),
            -float(max(peaks[e.a], peaks[e.b])),
            int(e.a),
            int(e.b),
        ),
    )
    for edge in ordered_edges:
        ra, rb = uf.find(edge.a), uf.find(edge.b)
        if ra == rb:
            continue
        peak_min = float(min(component_peak[ra], component_peak[rb]))
        ratio = float(edge.saddle_depth / peak_min) if peak_min > 0.0 else 0.0
        if ratio < float(merge_ratio):
            continue
        merged_peak = float(max(component_peak[ra], component_peak[rb]))
        uf.union(ra, rb)
        root = uf.find(ra)
        component_peak[root] = merged_peak

    roots = np.zeros(n + 1, dtype=np.int32)
    for basin_id in range(1, n + 1):
        roots[basin_id] = uf.find(basin_id)
    unique_roots = sorted(set(int(v) for v in roots[1:].tolist()))
    root_to_new = {root: i + 1 for i, root in enumerate(unique_roots)}
    basin_to_chamber = np.zeros(n + 1, dtype=np.int32)
    for basin_id in range(1, n + 1):
        basin_to_chamber[basin_id] = int(root_to_new[int(roots[basin_id])])
    consolidated = basin_to_chamber[np.asarray(initial_labels, dtype=np.int32)]
    return consolidated.astype(np.int32, copy=False), basin_to_chamber

def chamber_boundary_mask(chambers: np.ndarray) -> np.ndarray:
    labels = np.asarray(chambers, dtype=np.int32)
    out = np.zeros(labels.shape, dtype=bool)
    a, b = labels[:, :-1], labels[:, 1:]
    diff = (a > 0) & (b > 0) & (a != b)
    out[:, :-1] |= diff
    out[:, 1:] |= diff
    a, b = labels[:-1, :], labels[1:, :]
    diff = (a > 0) & (b > 0) & (a != b)
    out[:-1, :] |= diff
    out[1:, :] |= diff
    return out

def chamber_centers_and_stats(
    chambers: np.ndarray,
    depth: np.ndarray,
    basin_to_chamber: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = int(np.max(chambers))
    if n == 0:
        empty = np.empty(0, dtype=np.int64)
        return (
            np.empty((0, 2), dtype=np.int32),
            np.empty(0, dtype=np.float64),
            empty,
            empty,
            empty,
        )

    flat_labels = chambers.ravel()
    areas = np.bincount(flat_labels, minlength=n + 1).astype(np.int64)
    peak_depths = np.zeros(n + 1, dtype=np.float64)
    centers = np.zeros((n + 1, 2), dtype=np.int32)
    flat_depth = depth.ravel()
    for chamber_id in range(1, n + 1):
        idx = np.flatnonzero(flat_labels == chamber_id)
        if len(idx) == 0:
            continue
        best = int(idx[int(np.argmax(flat_depth[idx]))])
        y, x = np.unravel_index(best, chambers.shape)
        centers[chamber_id] = (int(y), int(x))
        peak_depths[chamber_id] = float(depth[y, x])

    initial_counts = np.bincount(
        basin_to_chamber[1:], minlength=n + 1
    ).astype(np.int64)
    return centers[1:], peak_depths[1:], areas[1:], initial_counts[1:], np.arange(1, n + 1, dtype=np.int64)

def observe_chamber_ray(
    foreground: np.ndarray,
    chambers: np.ndarray,
    chamber_id: int,
    seed_y: int,
    seed_x: int,
    path: E22.RayPath,
    frame_sides: dict[str, bool],
) -> str:
    h, w = foreground.shape
    ys = int(seed_y) + path.dy
    xs = int(seed_x) + path.dx
    valid = (ys >= 0) & (ys < h) & (xs >= 0) & (xs < w)
    invalid_idx = np.flatnonzero(~valid)
    first_invalid = int(invalid_idx[0]) if len(invalid_idx) else len(path.dy)

    if first_invalid > 0:
        vy = ys[:first_invalid]
        vx = xs[:first_invalid]
        fg_event = np.flatnonzero(foreground[vy, vx])
        other_event = np.flatnonzero((~foreground[vy, vx]) & (chambers[vy, vx] != int(chamber_id)))
        first_fg = int(fg_event[0]) if len(fg_event) else first_invalid + 1
        first_other = int(other_event[0]) if len(other_event) else first_invalid + 1
        if first_fg < first_other:
            return "segment"
        if first_other < first_fg:
            return "open"
        if first_fg <= first_invalid:
            return "segment"

    if first_invalid < len(path.dy):
        raw_y = int(ys[first_invalid])
        raw_x = int(xs[first_invalid])
        if E22.exits_through_true_frame(raw_y, raw_x, foreground.shape, frame_sides):
            return "frame"
        return "open"
    return "open"

def chamber_enclosure(
    foreground: np.ndarray,
    chambers: np.ndarray,
    chamber_id: int,
    center_y: int,
    center_x: int,
    paths: list[E22.RayPath],
    frame_sides: dict[str, bool],
) -> tuple[float, int, int, int]:
    kinds = [
        observe_chamber_ray(
            foreground,
            chambers,
            chamber_id,
            center_y,
            center_x,
            path,
            frame_sides,
        )
        for path in paths
    ]
    segment_hits = int(sum(kind == "segment" for kind in kinds))
    frame_hits = int(sum(kind == "frame" for kind in kinds))
    open_hits = int(sum(kind == "open" for kind in kinds))
    enclosure = float((segment_hits + frame_hits) / max(1, len(kinds)))
    return enclosure, segment_hits, frame_hits, open_hits

def chamber_touches_true_frame(
    mask: np.ndarray,
    frame_sides: dict[str, bool],
) -> bool:
    if frame_sides["top"] and np.any(mask[0, :]):
        return True
    if frame_sides["bottom"] and np.any(mask[-1, :]):
        return True
    if frame_sides["left"] and np.any(mask[:, 0]):
        return True
    if frame_sides["right"] and np.any(mask[:, -1]):
        return True
    return False

def run_configuration(
    seg2d: np.ndarray,
    frame_sides: dict[str, bool],
    ray_paths: list[E22.RayPath],
    initial_basins: np.ndarray,
    depth: np.ndarray,
    edges: list[BasinAdjacency],
    merge_ratio: float,
) -> ConfigRun:
    foreground = np.asarray(seg2d) > 0
    chambers, basin_to_chamber = consolidate_basins(initial_basins, depth, edges, merge_ratio)
    boundaries = chamber_boundary_mask(chambers)
    centers, peak_depths, areas, initial_counts, chamber_ids = chamber_centers_and_stats(
        chambers, depth, basin_to_chamber
    )

    repair_labels = np.zeros(chambers.shape, dtype=np.int32)
    observations: list[ChamberObservation] = []
    next_repair_id = 1

    for idx, chamber_id in enumerate(chamber_ids.tolist()):
        area = int(areas[idx])
        cy, cx = [int(v) for v in centers[idx]]
        enclosure, segment_hits, frame_hits, open_hits = chamber_enclosure(
            foreground,
            chambers,
            int(chamber_id),
            cy,
            cx,
            ray_paths,
            frame_sides,
        )
        mask = chambers == int(chamber_id)
        touches_frame = chamber_touches_true_frame(mask, frame_sides)
        # Frame alone must never turn a completely empty/external slice into a
        # cell.  At least one ray must terminate on an observed segment; after
        # that, frame hits are fully equivalent to segment hits in the 0.50 rule.
        accepted = bool(
            area >= int(MIN_CHAMBER_AREA)
            and enclosure >= float(ENCLOSURE_THRESHOLD)
            and segment_hits >= int(MIN_SEGMENT_RAY_HITS)
        )
        n_rays = max(1, int(RAY_DIRECTIONS))
        observations.append(
            ChamberObservation(
                chamber_id=int(chamber_id),
                center_y=cy,
                center_x=cx,
                peak_depth=float(peak_depths[idx]),
                area=area,
                initial_basin_count=int(initial_counts[idx]),
                enclosure=float(enclosure),
                segment_fraction=float(segment_hits / n_rays),
                frame_fraction=float(frame_hits / n_rays),
                open_fraction=float(open_hits / n_rays),
                segment_hits=segment_hits,
                frame_hits=frame_hits,
                open_hits=open_hits,
                touches_true_frame=bool(touches_frame),
                accepted=accepted,
            )
        )
        if accepted:
            repair_labels[mask] = int(next_repair_id)
            next_repair_id += 1

    return ConfigRun(
        neck_ratio=float(merge_ratio),
        initial_basin_count=int(np.max(initial_basins)),
        consolidated_chamber_count=int(np.max(chambers)),
        chamber_labels=chambers,
        chamber_boundary_mask=boundaries,
        observations=observations,
        repair_labels=repair_labels,
        repair_mask=repair_labels > 0,
    )
