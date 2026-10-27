# FARP-Seg segmentation kernels.

from __future__ import annotations
import heapq
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components,dijkstra,minimum_spanning_tree
from types import SimpleNamespace

GEODESIC_MAX_CLUSTER_ATOMS=8000

def normalize(x): return np.unique(x,return_inverse=True)[1].astype(np.int32)+1

def logit(p): p=np.clip(p,1e-5,1-1e-5);return np.log(p/(1-p))

BASE=SimpleNamespace(normalize_labels=normalize)

def gaec_partition(
    atom_ids: np.ndarray,
    edge_pairs: np.ndarray,
    merge_score: np.ndarray,
    bias: float = 0.0,
    initial_labels: np.ndarray | None = None,
) -> np.ndarray:
    atom_ids = np.asarray(atom_ids, dtype=np.int64)
    n = len(atom_ids)
    if n == 0:
        return np.zeros(0, dtype=np.int32)

    if initial_labels is None:
        labels0 = np.arange(n, dtype=np.int64)
    else:
        labels0 = BASE.normalize_labels(initial_labels).astype(np.int64) - 1
    unique = np.unique(labels0)
    remap = {int(v): i for i, v in enumerate(unique.tolist())}
    comp_of_atom = np.asarray([remap[int(v)] for v in labels0], dtype=np.int64)
    m = len(unique)

    weights = logit(merge_score) + float(bias)
    iu = np.searchsorted(atom_ids, edge_pairs[:, 0])
    iv = np.searchsorted(atom_ids, edge_pairs[:, 1])

    adj: list[dict[int, float]] = [dict() for _ in range(m)]
    for a, b, w in zip(comp_of_atom[iu].tolist(), comp_of_atom[iv].tolist(), weights.tolist()):
        if a == b:
            continue
        if a > b:
            a, b = b, a
        adj[a][b] = adj[a].get(b, 0.0) + float(w)
        adj[b][a] = adj[b].get(a, 0.0) + float(w)

    parent = np.arange(m, dtype=np.int64)
    size = np.ones(m, dtype=np.int64)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = int(parent[x])
        return x

    heap: list[tuple[float, int, int]] = []
    for a in range(m):
        for b, w in adj[a].items():
            if a < b and w > 0:
                heapq.heappush(heap, (-float(w), a, int(b)))

    while heap:
        neg_w, a0, b0 = heapq.heappop(heap)
        a, b = find(int(a0)), find(int(b0))
        if a == b:
            continue
        current = adj[a].get(b)
        if current is None:
            current = adj[b].get(a)
        if current is None or current <= 0:
            continue
        if abs(float(current) + float(neg_w)) > 1e-8:
            heapq.heappush(heap, (-float(current), a, b))
            continue

        if size[a] < size[b]:
            a, b = b, a
        parent[b] = a
        size[a] += size[b]

        neighbors = set(adj[a]) | set(adj[b])
        neighbors.discard(a)
        neighbors.discard(b)
        adj[a].pop(b, None)
        adj[b].pop(a, None)
        for nb0 in list(neighbors):
            nb = find(int(nb0))
            if nb in (a, b):
                continue
            wa = adj[a].get(nb, 0.0)
            wb = adj[b].get(nb, 0.0)
            new_w = float(wa + wb)
            adj[a][nb] = new_w
            adj[nb][a] = new_w
            adj[nb].pop(b, None)
            if new_w > 0:
                heapq.heappush(heap, (-new_w, a, nb))
        adj[b].clear()

    roots = np.asarray([find(int(c)) for c in comp_of_atom], dtype=np.int64)
    return BASE.normalize_labels(roots)

def mst_cut_partition(
    atom_ids: np.ndarray,
    edge_pairs: np.ndarray,
    merge_score: np.ndarray,
    n_cuts: int,
) -> np.ndarray | None:
    n = len(atom_ids)
    if n < 2 or len(edge_pairs) == 0:
        return None
    iu = np.searchsorted(atom_ids, edge_pairs[:, 0])
    iv = np.searchsorted(atom_ids, edge_pairs[:, 1])
    cost = 1.0 - np.clip(merge_score, 0.0, 1.0) + 1e-6
    graph = sparse.csr_matrix(
        (np.r_[cost, cost], (np.r_[iu, iv], np.r_[iv, iu])),
        shape=(n, n),
    )
    tree = minimum_spanning_tree(graph).tocoo()
    if tree.nnz == 0:
        return None
    order = np.argsort(tree.data)[::-1]
    remove = set(order[: min(int(n_cuts), len(order))].tolist())
    keep = np.asarray([i not in remove for i in range(len(tree.data))], dtype=bool)
    rows = tree.row[keep]
    cols = tree.col[keep]
    data = np.ones(len(rows) * 2, dtype=np.uint8)
    forest = sparse.csr_matrix(
        (data, (np.r_[rows, cols], np.r_[cols, rows])),
        shape=(n, n),
    )
    _, labels = connected_components(forest, directed=False, return_labels=True)
    return labels.astype(np.int32) + 1

def seeded_geodesic_split(
    region_atom_ids: np.ndarray,
    current_map: np.ndarray,
    edge_pairs: np.ndarray,
    merge_score: np.ndarray,
    seed_edge: tuple[int, int],
) -> np.ndarray | None:
    a_seed, b_seed = int(seed_edge[0]), int(seed_edge[1])
    seed_cluster = int(current_map[a_seed])
    if seed_cluster <= 0 or int(current_map[b_seed]) != seed_cluster:
        return None

    cluster_atoms = region_atom_ids[current_map[region_atom_ids] == seed_cluster]
    if len(cluster_atoms) < 2 or len(cluster_atoms) > GEODESIC_MAX_CLUSTER_ATOMS:
        return None

    in_cluster = (
        (current_map[edge_pairs[:, 0]] == seed_cluster)
        & (current_map[edge_pairs[:, 1]] == seed_cluster)
    )
    pairs = edge_pairs[in_cluster]
    score = merge_score[in_cluster]
    if len(pairs) == 0:
        return None

    iu = np.searchsorted(cluster_atoms, pairs[:, 0])
    iv = np.searchsorted(cluster_atoms, pairs[:, 1])
    cost = -np.log(np.clip(score, 1e-5, 1.0)) + 1e-3
    graph = sparse.csr_matrix(
        (np.r_[cost, cost], (np.r_[iu, iv], np.r_[iv, iu])),
        shape=(len(cluster_atoms), len(cluster_atoms)),
    )

    sa = int(np.searchsorted(cluster_atoms, a_seed))
    sb = int(np.searchsorted(cluster_atoms, b_seed))
    if sa >= len(cluster_atoms) or sb >= len(cluster_atoms):
        return None
    dist = dijkstra(graph, directed=False, indices=[sa, sb])
    if dist.shape != (2, len(cluster_atoms)):
        return None
    assignment = np.argmin(dist, axis=0).astype(np.int32) + 1
    if len(np.unique(assignment)) < 2:
        return None

    labels = BASE.normalize_labels(current_map[region_atom_ids])
    seed_mask = current_map[region_atom_ids] == seed_cluster
    max_label = int(np.max(labels))
    labels = labels.astype(np.int32)
    labels[seed_mask] = assignment + max_label
    return BASE.normalize_labels(labels)
