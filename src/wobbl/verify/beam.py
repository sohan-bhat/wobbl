"""A hand calculation Wobbl does on its own: beam theory through the critical section.

Cut the part with a plane through the governing point, perpendicular to the
line from the fixtures to the load. If every fixture is on one side and every
load on the other, the cut section carries the whole load like a beam:
axial force N, bending moment M and torsion T. Its area and second moments
give the nominal stress sigma = N/A + M c/I. The FEA peak divided by that
number is the stress concentration factor at that spot, which must be
physically sensible (about 0.8 to 4). Far outside that range means the load
path, the units or the setup deserve a second look.
"""
from __future__ import annotations

import numpy as np


def section_properties(tri_pts: np.ndarray, normal: np.ndarray):
    """Area, centroid and in-plane second moment tensor of a triangulated planar section."""
    p0, p1, p2 = tri_pts[:, 0], tri_pts[:, 1], tri_pts[:, 2]
    cr = np.cross(p1 - p0, p2 - p0)
    a = 0.5 * np.abs(cr @ normal)
    A = a.sum()
    if A <= 0:
        return None
    cen = ((p0 + p1 + p2) / 3.0 * a[:, None]).sum(0) / A
    # second moments with the 3 point (midside) rule, exact for quadratics on triangles
    I = np.zeros((3, 3))
    for q0, q1 in ((p0, p1), (p1, p2), (p2, p0)):
        m = 0.5 * (q0 + q1) - cen
        I += np.einsum("i,ij,ik->jk", a / 3.0, m, m)
    P = np.eye(3) - np.outer(normal, normal)
    I = P @ I @ P
    return A, cen, I


_EDGES = np.array([(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)])


def slice_tets(X: np.ndarray, tets: np.ndarray, normal, origin) -> tuple[np.ndarray, np.ndarray]:
    """Cut the tetrahedra (their four corners) with a plane: the points and triangles of the cut.

    Each cut tetrahedron gives a triangle or a quadrilateral split in two, with a corner on the plane counted as in
    front of it, as VTK's cutter does.
    """
    n = np.asarray(normal, float) / np.linalg.norm(normal)
    d = (X - np.asarray(origin, float)) @ n
    c = np.asarray(tets)[:, :4]
    front = d[c] >= 0
    c = c[(front.sum(1) > 0) & (front.sum(1) < 4)]
    a, b = c[:, _EDGES[:, 0]], c[:, _EDGES[:, 1]]
    cross = (d[a] >= 0) != (d[b] >= 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(cross, d[a] / (d[a] - d[b]), 0.0)
    P = X[a] + t[..., None] * (X[b] - X[a])
    tris = []
    for k in (3, 4):
        sel = cross.sum(1) == k
        pts = P[sel][cross[sel]].reshape(-1, k, 3)
        if k == 4:   # order the quadrilateral's corners around its center, then split it
            u = np.cross(n, [1.0, 0, 0] if abs(n[0]) < 0.9 else [0, 1.0, 0])
            r = pts - pts.mean(1, keepdims=True)
            order = np.argsort(np.arctan2(r @ np.cross(n, u), r @ u), axis=1)
            pts = np.take_along_axis(pts, order[..., None], axis=1)
            pts = np.concatenate([pts[:, [0, 1, 2]], pts[:, [0, 2, 3]]])
        tris.append(pts)
    pts = np.concatenate(tris).reshape(-1, 3)
    return pts, np.arange(len(pts)).reshape(-1, 3)


def beam_estimate(mesh, measure_at: np.ndarray, loads_force: np.ndarray, loads_moment: np.ndarray,
                  load_nodes: np.ndarray, fixture_nodes: np.ndarray, extra_load_points: list[np.ndarray]):
    """Return a dict with the nominal beam stress at the governing point, or None with a reason string."""
    X = mesh.X
    F = np.asarray(loads_force, float)
    Mo = np.asarray(loads_moment, float)
    if np.linalg.norm(F) < 1e-9 and np.linalg.norm(Mo) < 1e-9:
        return None, "no net load"
    if len(fixture_nodes) == 0 or (len(load_nodes) == 0 and not extra_load_points):
        return None, "no clear fixture to load path"
    fc = X[fixture_nodes].mean(0)
    lpts = [X[load_nodes]] if len(load_nodes) else []
    lpts += [np.atleast_2d(p) for p in extra_load_points]
    lp = np.vstack(lpts)
    lc = lp.mean(0)
    H = np.asarray(measure_at, float)
    # the local beam axis runs from the critical point toward the load
    axis = lc - H
    if np.linalg.norm(axis) < 1e-6 * np.linalg.norm(X.max(0) - X.min(0)):
        axis = lc - fc
    if np.linalg.norm(axis) < 1e-9:
        return None, "loads and fixtures coincide"
    axis /= np.linalg.norm(axis)
    k = int(np.argmax(np.abs(axis)))
    if abs(axis[k]) > np.cos(np.radians(20)):  # parts are usually modeled square to the axes: cut square
        axis = np.eye(3)[k] * np.sign(axis[k])
    s_fix = (X[fixture_nodes] - H) @ axis
    s_load = (lp - H) @ axis
    if not ((s_fix < 0).all() and (s_load > 0).all()):
        return None, "the governing point is not between the fixtures and the load, so no single section carries it"
    # the plane separates every fixture from every load, so all material it cuts carries the load:
    # the whole cut is the beam section (two chords of a truss, both walls of a channel)
    pts, faces = slice_tets(X, mesh.tets, axis, H)
    if len(faces) == 0:
        return None, "the section plane missed the part"
    props = section_properties(pts[faces], axis)
    if props is None:
        return None, "empty section"
    A, G, I = props
    if np.linalg.norm((G - H) - ((G - H) @ axis) * axis) > 3.0 * np.sqrt(A):
        return None, "the cut through the critical point is not a single compact section"
    # resultant of the loads about the section centroid; force F with moment Mo about the origin
    M_G = Mo - np.cross(G, F)
    N = float(F @ axis)
    T = float(M_G @ axis)
    Mb = M_G - T * axis
    u = np.cross(axis, [1.0, 0, 0] if abs(axis[0]) < 0.9 else [0, 1.0, 0])
    u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    B = np.array([u, v])
    I2 = B @ I @ B.T
    try:
        I2inv = np.linalg.inv(I2)
    except np.linalg.LinAlgError:
        return None, "degenerate section"
    k = B @ np.cross(Mb, axis)
    r = (pts - G) @ B.T
    sig = N / A + r @ I2inv @ k
    rH = (H - G) @ B.T
    sig_H = N / A + rH @ I2inv @ k
    # torsion, thin or compact section estimate tau = T r / J with J = Ixx + Iyy (approximate for non circular)
    J = float(np.trace(I2))
    rmax = float(np.linalg.norm(r, axis=1).max())
    tau = abs(T) * rmax / J if J > 0 else 0.0
    nominal = float(np.sqrt(np.abs(sig).max() ** 2 + 3 * tau ** 2))
    return {
        "section_point": H.tolist(), "axis": axis.tolist(), "area_mm2": float(A), "centroid": G.tolist(),
        "axial_N": N, "bending_Nmm": float(np.linalg.norm(Mb)), "torsion_Nmm": T,
        "nominal_peak_MPa": nominal, "nominal_at_point_MPa": float(abs(sig_H)),
        "shear_from_torsion_MPa": tau, "I_principal_mm4": np.linalg.eigvalsh(I2).tolist(),
    }, ""
