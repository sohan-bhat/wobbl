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


def beam_estimate(mesh, measure_at: np.ndarray, loads_force: np.ndarray, loads_moment: np.ndarray,
                  load_nodes: np.ndarray, fixture_nodes: np.ndarray, extra_load_points: list[np.ndarray]):
    """Return a dict with the nominal beam stress at the governing point, or None with a reason string."""
    import pyvista as pv

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
    axis = lc - fc
    if np.linalg.norm(axis) < 1e-9:
        return None, "loads and fixtures coincide"
    axis /= np.linalg.norm(axis)
    H = np.asarray(measure_at, float)
    s_fix = (X[fixture_nodes] - H) @ axis
    s_load = (lp - H) @ axis
    if not ((s_fix < 0).all() and (s_load > 0).all()):
        return None, "the governing point is not between the fixtures and the load, so no single section carries it"
    corners = mesh.tets[:, :4]
    cells = np.hstack([np.full((len(corners), 1), 4), corners]).ravel()
    grid = pv.UnstructuredGrid(cells, np.full(len(corners), pv.CellType.TETRA, dtype=np.uint8), X)
    sl = grid.slice(normal=axis, origin=H)
    if sl.n_points == 0:
        return None, "the section plane missed the part"
    sl = sl.triangulate()
    try:
        conn = sl.connectivity("closest", closest_point=H)
    except TypeError:
        conn = sl.connectivity(extraction_mode="closest", closest_point=H)
    conn = conn.extract_surface().triangulate()
    faces = conn.faces.reshape(-1, 4)[:, 1:]
    pts = np.asarray(conn.points)
    props = section_properties(pts[faces], axis)
    if props is None:
        return None, "empty section"
    A, G, I = props
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
