# FARP-Seg segmentation kernels.

from __future__ import annotations
import math
from collections import defaultdict
import numpy as np

def overlap_pairs(
    previous: np.ndarray,
    current: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    previous = np.asarray(previous, dtype=np.uint32)
    current = np.asarray(current, dtype=np.uint32)
    if previous.shape != current.shape:
        raise RuntimeError(
            f"Adjacent fill slices have different shape: {previous.shape} vs {current.shape}"
        )

    mask = (previous > 0) & (current > 0)
    if not np.any(mask):
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
        )

    left = previous[mask].astype(np.uint64)
    right = current[mask].astype(np.uint64)
    packed = (left << np.uint64(32)) | right
    unique, counts = np.unique(
        packed,
        return_counts=True,
    )

    prev_label = (unique >> np.uint64(32)).astype(np.int64)
    curr_label = (
        unique & np.uint64(0xFFFFFFFF)
    ).astype(np.int64)

    return (
        prev_label,
        curr_label,
        counts.astype(np.int64),
    )

def _label_geometry(
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return per-label area and centroid arrays indexed by integer label.

    Label zero is kept as an unused row so candidate labels can index directly.
    """
    labels = np.asarray(labels, dtype=np.uint32)
    max_label = int(labels.max(initial=0))

    area = np.zeros(max_label + 1, dtype=np.int64)
    cy = np.zeros(max_label + 1, dtype=np.float64)
    cx = np.zeros(max_label + 1, dtype=np.float64)

    if max_label <= 0:
        return area, cy, cx

    y, x = np.nonzero(labels)
    if len(y) == 0:
        return area, cy, cx

    lab = labels[y, x].astype(np.int64)
    area = np.bincount(
        lab,
        minlength=max_label + 1,
    ).astype(np.int64)

    sy = np.bincount(
        lab,
        weights=y.astype(np.float64),
        minlength=max_label + 1,
    )
    sx = np.bincount(
        lab,
        weights=x.astype(np.float64),
        minlength=max_label + 1,
    )

    positive = area > 0
    cy[positive] = sy[positive] / area[positive]
    cx[positive] = sx[positive] / area[positive]
    return area, cy, cx

def _unique_best_indices(
    endpoint_labels: np.ndarray,
    coverage: np.ndarray,
    overlap: np.ndarray,
    iou: np.ndarray,
    centroid_distance2: np.ndarray,
) -> tuple[dict[int, int], dict[int, float], int, int]:
    """
    Choose one unique best candidate for each endpoint.

    Returns:
      best_index_by_endpoint
      best_to_second_primary_ratio
      n_multi_candidate_endpoints
      n_exact_full_key_ties

    The competition ratio is diagnostic only and never gates a link.
    """
    grouped: dict[int, list[int]] = defaultdict(list)
    for idx, endpoint in enumerate(endpoint_labels.tolist()):
        grouped[int(endpoint)].append(int(idx))

    best: dict[int, int] = {}
    ratio: dict[int, float] = {}
    multi = 0
    exact_ties = 0

    for endpoint, indices in grouped.items():
        if len(indices) > 1:
            multi += 1

        # Main ranking key.  Floating quantities are deterministic functions of
        # integer masks.  The final exact-key tie abstains rather than picking
        # by arbitrary label ID.
        keys = [
            (
                float(coverage[idx]),
                int(overlap[idx]),
                float(iou[idx]),
                -float(centroid_distance2[idx]),
            )
            for idx in indices
        ]
        best_key = max(keys)
        winners = [
            idx
            for idx, key in zip(indices, keys)
            if key == best_key
        ]

        primary_values = sorted(
            (float(coverage[idx]) for idx in indices),
            reverse=True,
        )
        if len(primary_values) == 1:
            ratio[endpoint] = math.inf
        else:
            second = max(primary_values[1], 1e-12)
            ratio[endpoint] = float(primary_values[0] / second)

        if len(winners) == 1:
            best[endpoint] = int(winners[0])
        else:
            exact_ties += 1

    return best, ratio, int(multi), int(exact_ties)

def reciprocal_best_links(
    previous: np.ndarray,
    current: np.ndarray,
) -> tuple[
    list[tuple[int, int, int, float, float, float, float, float]],
    dict[str, int],
]:
    """
    Reciprocal-best adjacent-Z linkage.

    Each accepted tuple is:
      (
        prev_label,
        curr_label,
        overlap_voxels,
        prev_directional_coverage,
        curr_directional_coverage,
        prev_best_second_ratio,
        curr_best_second_ratio,
        competition_ratio,
      )

    competition_ratio = min(prev_ratio, curr_ratio).  It is recorded only for
    diagnostics and never acts as a threshold.
    """
    prev_label, curr_label, overlap = overlap_pairs(
        previous,
        current,
    )

    if len(overlap) == 0:
        return [], {
            "candidate_pairs": 0,
            "accepted_links": 0,
            "ambiguous_prev_nodes": 0,
            "ambiguous_curr_nodes": 0,
            "candidate_prev_nodes": 0,
            "candidate_curr_nodes": 0,
            "exact_tie_prev_nodes": 0,
            "exact_tie_curr_nodes": 0,
        }

    prev_area, prev_cy, prev_cx = _label_geometry(previous)
    curr_area, curr_cy, curr_cx = _label_geometry(current)

    pa = prev_area[prev_label].astype(np.float64)
    ca = curr_area[curr_label].astype(np.float64)
    ov = overlap.astype(np.float64)

    if np.any(pa <= 0) or np.any(ca <= 0):
        raise RuntimeError("Positive-overlap pair references an empty patch")

    prev_coverage = ov / pa
    curr_coverage = ov / ca
    union = pa + ca - ov
    iou = ov / np.maximum(union, 1.0)

    dy = prev_cy[prev_label] - curr_cy[curr_label]
    dx = prev_cx[prev_label] - curr_cx[curr_label]
    distance2 = dy * dy + dx * dx

    (
        prev_best,
        prev_ratio,
        multi_prev,
        tie_prev,
    ) = _unique_best_indices(
        prev_label,
        prev_coverage,
        overlap,
        iou,
        distance2,
    )
    (
        curr_best,
        curr_ratio,
        multi_curr,
        tie_curr,
    ) = _unique_best_indices(
        curr_label,
        curr_coverage,
        overlap,
        iou,
        distance2,
    )

    accepted = []

    for idx, (a, b, count) in enumerate(
        zip(
            prev_label.tolist(),
            curr_label.tolist(),
            overlap.tolist(),
        )
    ):
        if (
            prev_best.get(int(a)) == int(idx)
            and curr_best.get(int(b)) == int(idx)
        ):
            pr = float(prev_ratio[int(a)])
            cr = float(curr_ratio[int(b)])
            accepted.append(
                (
                    int(a),
                    int(b),
                    int(count),
                    float(prev_coverage[idx]),
                    float(curr_coverage[idx]),
                    pr,
                    cr,
                    float(min(pr, cr)),
                )
            )

    return accepted, {
        "candidate_pairs": int(len(overlap)),
        "accepted_links": int(len(accepted)),
        # Keep the old names for viewer/report compatibility.  In 4-1-2 these
        # mean "had >1 positive-overlap candidate", not "was vetoed".
        "ambiguous_prev_nodes": int(multi_prev),
        "ambiguous_curr_nodes": int(multi_curr),
        "candidate_prev_nodes": int(len(set(prev_label.tolist()))),
        "candidate_curr_nodes": int(len(set(curr_label.tolist()))),
        "exact_tie_prev_nodes": int(tie_prev),
        "exact_tie_curr_nodes": int(tie_curr),
    }
