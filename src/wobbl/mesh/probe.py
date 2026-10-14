"""Read nodal fields at any points inside a tetrahedral mesh."""
from __future__ import annotations

import numpy as np

# midside nodes 4 to 9 of a 10 node tetrahedron sit on these edges, CalculiX's C3D10 order
TET_EDGES = np.array([(0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3)])


def locate(X: np.ndarray, tets: np.ndarray, pts, candidates: int = 32, tol: float = 1e-3):
    """The element holding each point, its barycentric coordinates there, and whether the point is in the mesh.

    Candidates are the elements with the nearest centroids. A point counts as inside when no barycentric
    coordinate is below -tol, the same margin VTK's probe allows, so points on the boundary are found from either
    side.
    """
    from scipy.spatial import cKDTree

    pts = np.atleast_2d(np.asarray(pts, float))
    c = np.asarray(tets)[:, :4]
    k = min(candidates, len(c))
    _, cand = cKDTree(X[c].mean(1)).query(pts, k=k)
    cand = cand.reshape(len(pts), k)
    P0 = X[c[cand, 0]]
    E = X[c[cand, 1:]] - P0[:, :, None]                          # (points, candidates, 3 edges, xyz)
    with np.errstate(all="ignore"):
        l123 = np.linalg.solve(np.swapaxes(E, -1, -2), (pts[:, None] - P0)[..., None])[..., 0]
    lam = np.concatenate([1.0 - l123.sum(-1, keepdims=True), l123], axis=-1)
    score = np.nan_to_num(lam.min(-1), nan=-np.inf)
    best = np.argmax(score, axis=1)
    rows = np.arange(len(pts))
    return cand[rows, best], lam[rows, best], score[rows, best] >= -tol


def shape(lam: np.ndarray, nodes: int) -> np.ndarray:
    """Shape function values at barycentric coordinates, for 4 or 10 node tetrahedra."""
    if nodes == 4:
        return lam
    corners = lam * (2.0 * lam - 1.0)
    return np.concatenate([corners, 4.0 * lam[:, TET_EDGES[:, 0]] * lam[:, TET_EDGES[:, 1]]], axis=1)


def sample(X: np.ndarray, tets: np.ndarray, values: np.ndarray, pts) -> tuple[np.ndarray, np.ndarray]:
    """A nodal field, scalar or with components, at points inside the mesh, interpolated with the shape functions of
    the element holding each point, and which points were inside (the others read zero)."""
    t, lam, inside = locate(X, tets, pts)
    N = shape(lam, tets.shape[1])
    out = np.einsum("pj,pj...->p...", N, np.asarray(values, float)[np.asarray(tets)[t]])
    out[~inside] = 0.0
    return out, inside
