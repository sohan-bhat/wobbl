"""The hand calculation for a collapse load: a fully plastic section from beam theory.

Cut the part through the plastic hinge, the way the elastic beam check cuts it through the peak. When every fixture
lies on one side and every load on the other, the cut carries the whole load as an axial force and a bending moment.
Fully plastic, every point of the section sits at yield in tension or compression on either side of the plastic
neutral axis, which halves the area. The plastic moment is the yield stress times Z, the sum of each area's distance
from that axis, and the axial force interacts with it as (M / Mp) + (N / Np)^2 = 1, exact for a rectangle and close
for compact sections. The load factor that reaches it is the beam theory collapse load.
"""
from __future__ import annotations

import warnings

import numpy as np

from ..verify.beam import beam_estimate


def _subdivide(tri: np.ndarray, n: int = 6) -> tuple[np.ndarray, np.ndarray]:
    """Centroids and areas of the n x n equal sub triangles of each triangle, for integrating over a section."""
    bary = []
    for i in range(n):
        for j in range(n - i):
            bary.append(((i + 1 / 3) / n, (j + 1 / 3) / n))
            if i + j < n - 1:
                bary.append(((i + 2 / 3) / n, (j + 2 / 3) / n))
    b = np.array(bary)
    p0, p1, p2 = tri[:, 0], tri[:, 1], tri[:, 2]
    pts = (p0[:, None] * (1 - b[:, 0] - b[:, 1])[None, :, None] + p1[:, None] * b[None, :, 0, None]
           + p2[:, None] * b[None, :, 1, None])
    area = 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=1)
    return pts.reshape(-1, 3), np.repeat(area / (n * n), len(b))


def plastic_section(mesh, at, model, fy: float, fix_nodes=None) -> tuple[dict | None, str]:
    """The beam theory collapse load factor through a section at ``at``, or None with the reason no section applies.
    ``fix_nodes`` are the supports that carry the load (all of them when not given)."""
    import pyvista as pv

    F, M = model.applied_total()
    if fix_nodes is None:
        held = [c.nodes for c in model.constraints] + [c.nodes for c in model.normal_constraints]
        fix_nodes = np.unique(np.concatenate(held)) if held else np.zeros(0, int)
    load_nodes = np.nonzero(np.abs(model.nodal_forces()).sum(1) > 0)[0]
    extra = [np.asarray(r.ref_point) for r in model.rigid]
    if model.gravity is not None and np.linalg.norm(model.gravity) > 0:
        return None, "self weight acts everywhere, so no single section carries a known load"
    est, why = beam_estimate(mesh, np.asarray(at, float), F, M, load_nodes, fix_nodes, extra)
    if est is None:
        return None, why
    axis, G = np.asarray(est["axis"]), np.asarray(est["centroid"])
    corners = mesh.tets[:, :4]
    cells = np.hstack([np.full((len(corners), 1), 4), corners]).ravel()
    grid = pv.UnstructuredGrid(cells, np.full(len(corners), pv.CellType.TETRA, dtype=np.uint8), mesh.X)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sl = grid.slice(normal=axis, origin=np.asarray(at, float)).triangulate().extract_surface().triangulate()
    tris = np.asarray(sl.points)[sl.faces.reshape(-1, 4)[:, 1:]]
    q, a = _subdivide(tris)
    A = float(a.sum())
    u = np.cross(axis, [1.0, 0, 0] if abs(axis[0]) < 0.9 else [0, 1.0, 0])
    u /= np.linalg.norm(u)
    B = np.array([u, np.cross(axis, u)])
    r = (q - G) @ B.T
    M_G = M - np.cross(G, F)
    N, T = float(F @ axis), float(M_G @ axis)
    Mb = M_G - T * axis
    mb = float(np.linalg.norm(Mb))
    if abs(T) > 0.3 * max(mb, 1e-12):
        return None, "the section is twisted as much as it is bent, which this beam check does not cover"
    if mb < 1e-9 * max(abs(N), 1.0) * np.sqrt(A):
        dv = np.zeros(1)
        Z = 0.0
    else:
        # bending stress varies across the section along the elastic stress gradient; the plastic neutral axis runs
        # across it where the areas on either side are equal
        rr = r - (r * a[:, None]).sum(0) / A
        I2 = (rr[:, :, None] * rr[:, None, :] * a[:, None, None]).sum(0)
        k = B @ np.cross(Mb, axis)
        g = np.linalg.solve(I2, k)
        g /= np.linalg.norm(g)
        d = rr @ g
        dv = ((tris.reshape(-1, 3) - G) @ B.T - (r * a[:, None]).sum(0) / A) @ g   # the section's own corners
        order = np.argsort(d)
        c = d[order][np.searchsorted(np.cumsum(a[order]), 0.5 * A)]
        Z = float((a * np.abs(d - c)).sum())
    Mp, Np = fy * Z, fy * A
    bend, pull = mb / Mp if Mp > 0 else 0.0, abs(N) / Np
    if bend == 0 and pull == 0:
        return None, "no net load passes through the section"
    # (lam M / Mp) + (lam N / Np)^2 = 1
    lam = 1.0 / bend if pull == 0 else (-bend + np.sqrt(bend ** 2 + 4 * pull ** 2)) / (2 * pull ** 2)
    return {"factor": float(lam), "point": np.asarray(at, float).tolist(), "axis": axis.tolist(), "area_mm2": A,
            "Z_mm3": Z, "Mp_Nmm": Mp, "moment_Nmm": mb, "axial_N": N, "depth_mm": float(dv.max() - dv.min()),
            "bending_share": float(lam * bend)}, ""
