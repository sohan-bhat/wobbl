"""Cut a tetrahedral mesh open with a plane and keep one side, as a triangle surface that fields can be carried onto.

The surface of the kept half is the part's own boundary clipped by the plane, plus the cut face. Every point on it
is a mesh node or a point on a mesh edge, kept as the edge's two nodes a and b and how far along it lies, t, so a
nodal field is carried onto it exactly as the elements interpolate along their edges.
"""
from __future__ import annotations

import numpy as np

from ..verify.beam import cut_tets


def split_quadratic(tris: np.ndarray) -> np.ndarray:
    """(m, 6) quadratic triangles as (4m, 3) flat ones through their midside nodes, each ordered like its parent."""
    if tris.shape[1] == 3:
        return tris
    a, b, c, ab, bc, ca = tris.T
    return np.vstack([np.column_stack(x) for x in ((a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca))])


def boundary(X: np.ndarray, tets: np.ndarray) -> np.ndarray:
    """The outer faces of the mesh as flat triangles facing out."""
    from .mesher import _boundary

    return split_quadratic(_boundary(tets, X, 2 if tets.shape[1] == 10 else 1)[0])


def clip_triangles(tris: np.ndarray, d: np.ndarray):
    """The parts of flat triangles where d <= 0, as corners (a, b, t), each (k, 3), facing as the triangles did."""
    inside = d[tris] <= 0
    n_in = inside.sum(1)
    whole = tris[n_in == 3]
    A, B, T = [whole], [whole], [np.zeros(whole.shape)]

    def rolled(sel, first):
        """The selected triangles turned so the corner at position first comes first, same facing."""
        r = (first[:, None] + np.arange(3)) % 3
        return np.take_along_axis(tris[sel], r, axis=1)

    def at(p, q):
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.clip(np.nan_to_num(d[p] / (d[p] - d[q])), 0.0, 1.0)

    one = n_in == 1                               # one corner kept: a triangle to the two crossings
    p, q, r = rolled(one, np.argmax(inside[one], axis=1)).T
    A.append(np.column_stack([p, p, p]))
    B.append(np.column_stack([p, q, r]))
    T.append(np.column_stack([np.zeros(len(p)), at(p, q), at(p, r)]))
    two = n_in == 2                               # two kept: the quadrilateral up to the crossings, in two
    p, q, r = rolled(two, (np.argmin(inside[two], axis=1) + 1) % 3).T
    tq, tp = at(q, r), at(p, r)
    A += [np.column_stack([p, q, q]), np.column_stack([p, q, p])]
    B += [np.column_stack([p, q, r]), np.column_stack([p, r, r])]
    T += [np.column_stack([np.zeros(len(p)), np.zeros(len(p)), tq]),
          np.column_stack([np.zeros(len(p)), tq, tp])]
    return np.vstack(A), np.vstack(B), np.vstack(T)


def clip_surface(X: np.ndarray, tets: np.ndarray, origin=None, normal=None):
    """The surface of the part, or with a plane (origin, normal) of the half the normal points away from, closed by
    the cut face. Returns the vertices as edges (a, b, t) and the triangles, all facing out, with points that lie
    at the same place merged so the surface is one closed piece."""
    X, tets = np.asarray(X, float), np.asarray(tets)
    tris = boundary(X, tets)
    if origin is None:
        A, B, T = tris, tris, np.zeros(tris.shape)
    else:
        n = np.asarray(normal, float) / np.linalg.norm(normal)
        d = (X - np.asarray(origin, float)) @ n
        A, B, T = clip_triangles(tris, d)
        pts, faces, ca, cb, ct = cut_tets(X, tets, -n, origin, quadratic=True)
        ca, cb, ct = ca[faces], cb[faces], ct[faces]
        P = X[ca] + ct[..., None] * (X[cb] - X[ca])
        back = np.einsum("ij,j->i", np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]), n) < 0
        for v in (ca, cb, ct):
            v[back] = v[back][:, ::-1]
        A, B, T = np.vstack([A, ca]), np.vstack([B, cb]), np.vstack([T, ct])
    a, b, t = A.ravel(), B.ravel(), T.ravel()
    # a point at a node is that node; a point inside an edge is keyed by the edge, whichever way it was found
    node = np.where(t <= 0, a, np.where(t >= 1, b, -1))
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    key = np.where(node >= 0, node.astype(np.int64) * (len(X) + 1), lo.astype(np.int64) * (len(X) + 1) + hi + 1)
    _, first, inv = np.unique(key, return_index=True, return_inverse=True)
    faces = inv.reshape(-1, 3)
    ok = (faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 0] != faces[:, 2])
    return (a[first], b[first], t[first]), faces[ok]


def carry(values: np.ndarray, a: np.ndarray, b: np.ndarray, t: np.ndarray) -> np.ndarray:
    """A nodal field, scalar or with components, at points on mesh edges. A point at a node takes the node's value
    even when the other end of its edge has none."""
    v = np.asarray(values, float)
    w = t.reshape(-1, *([1] * (v.ndim - 1)))
    out = v[a] + w * (v[b] - v[a])
    out[t <= 0] = v[a[t <= 0]]
    out[t >= 1] = v[b[t >= 1]]
    return out
