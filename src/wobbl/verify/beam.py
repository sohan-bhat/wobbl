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


# a 10 node tetrahedron as eight linear ones: a corner tetrahedron at each corner, and the octahedron between them
# cut along one of its three diagonals (pairs of opposite midside nodes)
_CORNER_TETS = np.array([(0, 4, 6, 7), (4, 1, 5, 8), (6, 5, 2, 9), (7, 8, 9, 3)])
_DIAGONALS = np.array([(4, 9), (5, 7), (6, 8)])
_OCTAHEDRON = np.array([[(4, 9, 5, 6), (4, 9, 6, 7), (4, 9, 7, 8), (4, 9, 8, 5)],
                        [(5, 7, 4, 6), (5, 7, 6, 9), (5, 7, 9, 8), (5, 7, 8, 4)],
                        [(6, 8, 4, 5), (6, 8, 5, 9), (6, 8, 9, 7), (6, 8, 7, 4)]])


def cut_tets(X: np.ndarray, tets: np.ndarray, normal, origin, quadratic: bool = False):
    """Cut the tetrahedra (their four corners) with a plane.

    Returns the cut as triangles, three points each (a tetrahedron gives a triangle or a quadrilateral split in
    two), and for every point the mesh edge it lies on, a to b, and how far along it, so that a nodal field f is
    f[a] + t (f[b] - f[a]) there. A corner on the plane counts as in front of it, as VTK's cutter counts it. With
    quadratic=True a 10 node tetrahedron is first split into eight linear ones, so its midside nodes' values count,
    along the diagonal whose ends lie nearest the same distance from the plane, which is how VTK cuts one.
    """
    n = np.asarray(normal, float) / np.linalg.norm(normal)
    d = (X - np.asarray(origin, float)) @ n
    tets = np.asarray(tets)
    if quadratic and tets.shape[1] == 10:
        pick = np.argmin(np.abs(d[tets[:, _DIAGONALS[:, 0]]] - d[tets[:, _DIAGONALS[:, 1]]]), axis=1)
        inner = tets[np.arange(len(tets))[:, None, None], _OCTAHEDRON[pick]]
        tets = np.concatenate([tets[:, _CORNER_TETS].reshape(-1, 4), inner.reshape(-1, 4)])
    c = tets[:, :4]
    front = d[c] >= 0
    c = c[(front.sum(1) > 0) & (front.sum(1) < 4)]
    a, b = c[:, _EDGES[:, 0]], c[:, _EDGES[:, 1]]
    cross = (d[a] >= 0) != (d[b] >= 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(cross, d[a] / (d[a] - d[b]), 0.0)
    out = []
    for k in (3, 4):
        sel = cross.sum(1) == k
        ea, eb, et = (v[sel][cross[sel]].reshape(-1, k) for v in (a, b, t))
        if k == 4:   # order the quadrilateral's corners around its center, then split it
            P = X[ea] + et[..., None] * (X[eb] - X[ea])
            u = np.cross(n, [1.0, 0, 0] if abs(n[0]) < 0.9 else [0, 1.0, 0])
            r = P - P.mean(1, keepdims=True)
            order = np.argsort(np.arctan2(r @ np.cross(n, u), r @ u), axis=1)
            ea, eb, et = (np.take_along_axis(v, order, axis=1)[:, [0, 1, 2, 0, 2, 3]].reshape(-1, 3)
                          for v in (ea, eb, et))
        out.append((ea, eb, et))
    ea, eb, et = (np.concatenate([o[i] for o in out]).ravel() for i in range(3))
    pts = X[ea] + et[:, None] * (X[eb] - X[ea])
    return pts, np.arange(len(pts)).reshape(-1, 3), ea, eb, et


def slice_tets(X: np.ndarray, tets: np.ndarray, normal, origin) -> tuple[np.ndarray, np.ndarray]:
    """The points and triangles where a plane cuts the tetrahedra."""
    return cut_tets(X, tets, normal, origin)[:2]


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
