# FARP-Seg segmentation kernels.

from __future__ import annotations
import math
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class RayPath:
    dy: np.ndarray
    dx: np.ndarray
    radius: np.ndarray

def build_ray_paths(shape: tuple[int, int], n_directions: int) -> list[RayPath]:
    h, w = (int(shape[0]), int(shape[1]))
    max_step = int(math.ceil(math.hypot(h, w))) + 3
    raw_steps = np.arange(1, max_step + 1, dtype=np.float64)
    paths: list[RayPath] = []
    for direction_id in range(int(n_directions)):
        angle = 2.0 * math.pi * direction_id / float(n_directions)
        dy = np.rint(np.sin(angle) * raw_steps).astype(np.int32)
        dx = np.rint(np.cos(angle) * raw_steps).astype(np.int32)
        keep = np.ones(len(dy), dtype=bool)
        if len(dy) > 1:
            keep[1:] = (dy[1:] != dy[:-1]) | (dx[1:] != dx[:-1])
        dy = dy[keep]
        dx = dx[keep]
        radius = np.sqrt(dy.astype(np.float64) ** 2 + dx.astype(np.float64) ** 2)
        paths.append(RayPath(dy=dy, dx=dx, radius=radius))
    return paths

def exits_through_true_frame(
    raw_y: int,
    raw_x: int,
    shape: tuple[int, int],
    frame_sides: dict[str, bool],
) -> bool:
    h, w = shape
    hit = False
    if raw_y < 0:
        hit = hit or bool(frame_sides["top"])
    if raw_y >= h:
        hit = hit or bool(frame_sides["bottom"])
    if raw_x < 0:
        hit = hit or bool(frame_sides["left"])
    if raw_x >= w:
        hit = hit or bool(frame_sides["right"])
    return bool(hit)
