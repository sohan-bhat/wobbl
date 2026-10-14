"""The hand calculation for a collapse load: a fully plastic section from beam theory.

Cut the part through the plastic hinge, the way the elastic beam check cuts it through the peak. When every fixture
lies on one side and every load on the other, the cut carries the whole load as an axial force and a bending moment.
Fully plastic, every point of the section sits at yield in tension or compression on either side of the plastic
neutral axis, which halves the area. The plastic moment is the yield stress times Z, the sum of each area's distance
from that axis, and the axial force interacts with it as (M / Mp) + (N / Np)^2 = 1, exact for a rectangle and close
for compact sections. The load factor that reaches it is the beam theory collapse load.
"""
from __future__ import annotations

import numpy as np

from ..verify.beam import beam_estimate, slice_tets


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


def _piece(pts: np.ndarray, faces: np.ndarray, at: np.ndarray) -> np.ndarray:
    """The triangles of the one piece of a cut that holds the point: a plane through a hook's shank also crosses
    its bend, which is another section."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    scale = float(np.linalg.norm(np.ptp(pts, axis=0))) or 1.0
    _, ids = np.unique(np.round(pts / (1e-7 * scale)).astype(np.int64), axis=0, return_inverse=True)
    f = ids.ravel()[faces]
    rows = np.concatenate([f[:, 0], f[:, 1], f[:, 2]])
    cols = np.concatenate([f[:, 1], f[:, 2], f[:, 0]])
    n = int(ids.max()) + 1
    _, label = connected_components(coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n)), directed=False)
    tri = pts[faces]
    near = int(np.argmin(np.linalg.norm(tri.mean(1) - at, axis=1)))
    return tri[label[f[:, 0]] == label[f[near, 0]]]


def member_axis(mesh, at: np.ndarray, guess: np.ndarray, area: float, fix_nodes, load_points) -> np.ndarray:
    """The direction the member runs at a point: the longest principal direction of the material around it, square
    to the axes when within 20 degrees of one. It is used when the supports and the loads fall on opposite sides of
    a cut square to it; otherwise the guess (from the point toward the load) stands. On a hook or a crank the line to
    the load leaves the member at a steep angle, and a cut along it is not a cross section."""
    size = np.sqrt(area) if area > 0 else 0.05 * float(np.linalg.norm(np.ptp(mesh.X, axis=0)))
    near = mesh.X[np.linalg.norm(mesh.X - at, axis=1) <= 1.5 * size]
    if len(near) < 20:
        return guess
    w, v = np.linalg.eigh(np.cov((near - near.mean(0)).T))
    if w[-1] < 1.5 * w[-2]:           # no clear long direction: a plate or a block, not a member
        return guess
    ax = v[:, -1]
    k = int(np.argmax(np.abs(ax)))
    if abs(ax[k]) > np.cos(np.radians(20)):
        ax = np.eye(3)[k] * np.sign(ax[k])
    s_fix = (mesh.X[fix_nodes] - at) @ ax
    s_load = (np.asarray(load_points, float) - at) @ ax
    if len(s_fix) and len(s_load) and ((s_fix.max() < 0 < s_load.min()) or (s_load.max() < 0 < s_fix.min())):
        return ax
    return guess


def plastic_section(mesh, at, model, fy: float, fix_nodes=None) -> tuple[dict | None, str]:
    """The beam theory collapse load factor through a section at ``at``, or None with the reason no section applies.
    ``fix_nodes`` are the supports that carry the load (all of them when not given)."""
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
    at = np.asarray(at, float)
    axis = member_axis(mesh, at, np.asarray(est["axis"], float), float(est.get("area_mm2", 0.0)), fix_nodes,
                       np.vstack([mesh.X[load_nodes], *[e[None] for e in extra]]) if len(load_nodes) or extra
                       else mesh.X[:0])
    pts, faces = slice_tets(mesh.X, mesh.tets, axis, at)
    if not len(faces):
        return None, "the section plane missed the part"
    tris = _piece(pts, faces, at)
    q, a = _subdivide(tris)
    A = float(a.sum())
    G = (q * a[:, None]).sum(0) / A
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
