"""Carry loads and supports picked on the solid's faces over to its midsurface.

A support holds every shell node that lies on the picked faces: within half a wall of a face that runs along
the wall (the outside of a lid), or right on a face that cuts across it (a rim, a hole). A load is first turned
into forces at the solid's surface nodes, exactly as the solid study spreads it, and each of those forces then
moves straight onto the nearest point of the midsurface, shared among that element's six nodes by its shape
functions. The total force and its line of action stay the same.
"""
from __future__ import annotations

import numpy as np

from ..mesh.mesher import SurfaceMesh
from .mesh import ShellMesh, shape_s6

ALONG_COS = 0.7      # a picked face whose normal is this close to a node's normal runs along the wall there


class SolidSurface:
    """The solid's oriented boundary triangles with the interface the load builders expect of a mesh."""

    def __init__(self, sm: SurfaceMesh):
        self.X, self.tris, self.tri_surf = sm.X, sm.tris[:, :3], sm.tri_surf
        self.surface_types, self.surface_curves = sm.surface_types, sm.surface_curves

    n_nodes = property(lambda self: len(self.X))
    tri_geometry = SurfaceMesh.tri_geometry
    surface_mask = SurfaceMesh.surface_mask
    surface_nodes = SurfaceMesh.surface_nodes
    surfaces = SurfaceMesh.surfaces


def point_triangle_distance(P: np.ndarray, A: np.ndarray, B: np.ndarray, C: np.ndarray) -> tuple[np.ndarray, ...]:
    """Distance from each point to its triangle and the closest point's area coordinates (u on B, v on C)."""
    ab, ac = B - A, C - A
    n = np.cross(ab, ac)
    nn = np.maximum(np.einsum("ij,ij->i", n, n), 1e-300)
    ap = P - A
    # area coordinates of the projection, then clamped onto the triangle through its edges
    v = np.einsum("ij,ij->i", np.cross(ab, ap), n) / nn
    u = np.einsum("ij,ij->i", np.cross(ap, ac), n) / nn
    inside = (u >= 0) & (v >= 0) & (u + v <= 1)
    best_u, best_v = np.where(inside, u, 0.0), np.where(inside, v, 0.0)
    q = A + best_u[:, None] * ab + best_v[:, None] * ac
    d = np.linalg.norm(P - q, axis=1)
    d = np.where(inside, d, np.inf)
    for (p0, uv0), (p1, uv1) in (((A, (0, 0)), (B, (1, 0))), ((B, (1, 0)), (C, (0, 1))), ((C, (0, 1)), (A, (0, 0)))):
        e = p1 - p0
        s = np.clip(np.einsum("ij,ij->i", P - p0, e) / np.maximum(np.einsum("ij,ij->i", e, e), 1e-300), 0, 1)
        de = np.linalg.norm(P - (p0 + s[:, None] * e), axis=1)
        better = de < d
        d = np.where(better, de, d)
        best_u = np.where(better, uv0[0] + s * (uv1[0] - uv0[0]), best_u)
        best_v = np.where(better, uv0[1] + s * (uv1[1] - uv0[1]), best_v)
    return d, best_u, best_v


def nodes_on(shell: ShellMesh, solid: SolidSurface, mask: np.ndarray) -> np.ndarray:
    """Shell nodes that lie on the masked solid triangles (see the module notes for what on means)."""
    from scipy.spatial import cKDTree

    idx = np.flatnonzero(mask)
    if not len(idx):
        return np.zeros(0, dtype=np.int64)
    tn, _, tc = solid.tri_geometry(idx)
    T = solid.tris[idx]
    A, B, C = (solid.X[T[:, k]] for k in range(3))
    rad = float(np.linalg.norm(np.stack([A, B, C]) - tc[None], axis=2).max())
    k = min(12, len(idx))
    dist, near = cKDTree(tc).query(shell.X, k=k, distance_upper_bound=0.6 * shell.thickness.max() + rad)
    dist, near = dist.reshape(len(shell.X), k), near.reshape(len(shell.X), k)
    out = np.zeros(shell.n_nodes, bool)
    for j in range(k):
        ok = near[:, j] < len(idx)
        if not ok.any():
            continue
        p = np.flatnonzero(ok)
        t = near[p, j]
        d, _, _ = point_triangle_distance(shell.X[p], A[t], B[t], C[t])
        along = np.abs(np.einsum("ij,ij->i", tn[t], shell.normals[p])) >= ALONG_COS
        tol = np.where(along, 0.55 * shell.thickness[p], 0.3 * shell.thickness[p]) + 1e-3 * shell.h
        out[p[d <= tol]] = True
    return np.flatnonzero(out)


def transfer_forces(shell: ShellMesh, points: np.ndarray, forces: np.ndarray) -> np.ndarray:
    """Move point forces onto the nearest point of the midsurface and share each among that element's six nodes
    by its shape functions. Returns (n,3) nodal forces."""
    from scipy.spatial import cKDTree

    out = np.zeros((shell.n_nodes, 3))
    keep = np.abs(forces).max(1) > 0
    points, forces = points[keep], forces[keep]
    if not len(points):
        return out
    _, _, cen = shell.element_geometry()
    k = min(8, shell.n_elems)
    _, near = cKDTree(cen).query(points, k=k)
    near = near.reshape(len(points), k)
    A, B, C = (shell.X[shell.tris[:, i]] for i in range(3))
    best = np.full(len(points), np.inf)
    eb, ub, vb = np.zeros(len(points), dtype=np.int64), np.zeros(len(points)), np.zeros(len(points))
    for j in range(k):
        e = near[:, j]
        d, u, v = point_triangle_distance(points, A[e], B[e], C[e])
        better = d < best
        best = np.where(better, d, best)
        eb, ub, vb = np.where(better, e, eb), np.where(better, u, ub), np.where(better, v, vb)
    N = shape_s6(ub, vb)
    for c in range(6):
        np.add.at(out, shell.tris[eb, c], N[:, c, None] * forces)
    return out


__all__ = ["SolidSurface", "nodes_on", "transfer_forces", "point_triangle_distance"]
