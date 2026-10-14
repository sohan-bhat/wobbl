"""Surface helpers for scenes: the outer skin of a tet mesh and a decimation that keeps original vertices.

Decimation clusters vertices on a grid and lets one real vertex stand for each cluster, so every vertex left is a
node of the solver's mesh and carries its values exactly. Vertices are also split by which way they face, so the two
skins of a thin wall never merge into one.
"""
from __future__ import annotations

import numpy as np

# corner triples of a tet's four faces, outward for CalculiX and VTK node order
TET_FACES = ((0, 1, 3), (1, 2, 3), (2, 0, 3), (0, 2, 1))


def boundary(X: np.ndarray, tets: np.ndarray) -> np.ndarray:
    """Outward boundary triangles (corner nodes) of a tetrahedral mesh."""
    t = np.asarray(tets)[:, :4]
    tris = np.concatenate([t[:, list(f)] for f in TET_FACES])
    opp = np.concatenate([t[:, ({0, 1, 2, 3} - set(f)).pop()] for f in TET_FACES])
    key = np.sort(tris, axis=1)
    _, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    once = cnt[inv.ravel()] == 1
    tris, opp = tris[once], opp[once]
    a, b, c = X[tris[:, 0]], X[tris[:, 1]], X[tris[:, 2]]
    inward = np.einsum("ij,ij->i", np.cross(b - a, c - a), X[opp] - a) > 0
    tris[inward] = tris[inward][:, [0, 2, 1]]
    return tris


def split_quadratic(tris: np.ndarray) -> np.ndarray:
    """Six node triangles (a, b, c, ab, bc, ca) as the four flat triangles VTK draws, so midside nodes show too."""
    t = np.asarray(tris)
    if t.shape[1] < 6:
        return t[:, :3]
    a, b, c, ab, bc, ca = (t[:, k] for k in range(6))
    return np.concatenate([np.stack(x, 1) for x in ((a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca))])


def vertex_normals(X: np.ndarray, T: np.ndarray) -> np.ndarray:
    n = np.cross(X[T[:, 1]] - X[T[:, 0]], X[T[:, 2]] - X[T[:, 0]])
    out = np.zeros_like(X, dtype=float)
    for k in range(3):
        np.add.at(out, T[:, k], n)
    return out / np.maximum(np.linalg.norm(out, axis=1), 1e-30)[:, None]


def compact(T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The vertices the triangles use and the triangles renumbered onto them."""
    used, inv = np.unique(T.ravel(), return_inverse=True)
    return used, inv.reshape(T.shape)


def decimate(X: np.ndarray, T: np.ndarray, max_vertices: int, keep=()) -> tuple[np.ndarray, np.ndarray]:
    """(original index of each vertex kept, triangles on them). ``keep`` lists vertices that must survive, such as
    the node where the peak is, so the drawn field still reaches its true extreme."""
    T = np.asarray(T, np.int64)
    used, Tc = compact(T)
    if len(used) <= max_vertices:
        return used, Tc
    P = np.asarray(X, float)[used]
    nrm = vertex_normals(P, Tc)
    ax = np.argmax(np.abs(nrm), axis=1)
    bucket = ax * 2 + (nrm[np.arange(len(P)), ax] < 0)
    lo, span = P.min(0), np.ptp(P, 0).max() or 1.0

    def clusters(h):
        cell = np.floor((P - lo) / h).astype(np.int64)
        key = np.column_stack([cell, bucket])
        _, lab = np.unique(key, axis=0, return_inverse=True)
        return lab.ravel()

    a, b = np.log(span / 4000), np.log(span / 4)
    lab = clusters(np.exp(b))
    for _ in range(18):
        m = (a + b) / 2
        trial = clusters(np.exp(m))
        if trial.max() + 1 > max_vertices:
            a = m
        else:
            b, lab = m, trial
    k = lab.max() + 1
    cnt = np.bincount(lab, minlength=k)
    mean = np.stack([np.bincount(lab, P[:, i], k) for i in range(3)], 1) / cnt[:, None]
    d = np.linalg.norm(P - mean[lab], axis=1)
    if len(keep):
        must = np.isin(used, np.asarray(list(keep)))
        d[must] = -1.0
    order = np.lexsort((d, lab))
    first = np.ones(len(order), bool)
    first[1:] = lab[order][1:] != lab[order][:-1]
    rep = np.empty(k, np.int64)
    rep[lab[order][first]] = order[first]
    T2 = rep[lab][Tc]
    ok = (T2[:, 0] != T2[:, 1]) & (T2[:, 1] != T2[:, 2]) & (T2[:, 0] != T2[:, 2])
    T2 = T2[ok]
    _, first_row = np.unique(np.sort(T2, axis=1), axis=0, return_index=True)
    T2 = T2[np.sort(first_row)]
    kept, T3 = compact(T2)
    return used[kept], T3
