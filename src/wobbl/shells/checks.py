"""Checks shell answers share: force balance, and a beam hand calculation for slender shell parts."""
from __future__ import annotations

import numpy as np

from ..units import fmt
from ..verify.checks import Check
from .ccx import ShellModel, ShellResult
from .render import split_tris


def force_balance(model: ShellModel, res: ShellResult) -> Check:
    """Reactions against the applied loads. Shell supports also hold rotations, whose reaction moments
    CalculiX does not report, so only the forces are compared."""
    applied, _ = model.applied_total()
    fa = float(np.linalg.norm(applied))
    if fa < 1e-12:
        return Check("force balance", "skip", "no net force to balance")
    # a load that lands right on a held node goes straight into the support, and CalculiX's reaction there
    # counts only what the elements pass on, so that share is set aside on both sides
    direct = np.zeros(3)
    if model.forces is not None:
        held = np.zeros((model.mesh.n_nodes, 3), bool)
        for _, nodes, dofs in model.constraints:
            shell = nodes[nodes < model.mesh.n_nodes]
            for d in dofs:
                if d <= 3:
                    held[shell, d - 1] = True
        direct = np.where(held, model.forces, 0.0).sum(0)
    err = float(np.linalg.norm(applied - direct + res.reaction)) / fa
    s = (f"reactions balance the loads to {100 * err:.2g}% (forces; shell supports also hold rotations, whose "
         "reaction moments are not compared)")
    status = "pass" if err <= 0.005 else ("warn" if err <= 0.02 else "fail")
    return Check("force balance", status, s + ("" if status == "pass" else "; the solution is not in equilibrium"),
                 {"applied_N": applied, "reaction_N": res.reaction, "force_error": err})


def slender_axis(X: np.ndarray) -> tuple[int, float] | None:
    """The long axis of a part at least three times longer than it is wide, and that length."""
    size = np.ptp(X, axis=0)
    a = int(np.argmax(size))
    rest = np.delete(size, a)
    if size[a] < 3.0 * rest.max():
        return None
    return a, float(size[a])


def section(X: np.ndarray, tris: np.ndarray, thick: np.ndarray, axis: int, at: float) -> dict | None:
    """Cut the midsurface with the plane x[axis] = at: the wall segments of the cross section with their
    thickness and wall normal, the centroid and the area. Each flat sub triangle crossing the plane gives one
    segment."""
    t3 = split_tris(tris)
    d = X[t3, axis] - at
    segs, ts, nrm = [], [], []
    for k in range(len(t3)):
        dk = d[k]
        if dk.min() >= 0 or dk.max() <= 0:
            continue
        pts, tk = [], []
        for i, j in ((0, 1), (1, 2), (2, 0)):
            if (dk[i] < 0) != (dk[j] < 0):
                w = dk[i] / (dk[i] - dk[j])
                pts.append(X[t3[k, i]] + w * (X[t3[k, j]] - X[t3[k, i]]))
                tk.append(thick[t3[k, i]] + w * (thick[t3[k, j]] - thick[t3[k, i]]))
        if len(pts) == 2:
            P = X[t3[k]]
            n = np.cross(P[1] - P[0], P[2] - P[0])
            segs.append(pts)
            ts.append(np.mean(tk))
            nrm.append(n / max(np.linalg.norm(n), 1e-300))
    if not segs:
        return None
    S = np.asarray(segs)
    t = np.asarray(ts)
    L = np.linalg.norm(S[:, 1] - S[:, 0], axis=1)
    mid = S.mean(1)
    A = float((t * L).sum())
    return {"segments": S, "t": t, "normal": np.asarray(nrm), "length": L, "area": A,
            "centroid": (mid * (t * L)[:, None]).sum(0) / A}


def face_points(sec: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The ends of every section segment on both faces of the wall: (points on the midsurface, offsets along the
    wall normal, the face points themselves), each flattened."""
    S, t, n = sec["segments"], sec["t"], sec["normal"]
    mids = np.repeat(S[:, :, None, :], 2, axis=2)                          # (k, 2 ends, 2 faces, 3)
    off = np.stack([-0.5 * t, 0.5 * t], axis=1)[:, None, :].repeat(2, axis=1)   # (k, 2, 2)
    faces = mids + off[..., None] * n[:, None, None, :]
    return mids.reshape(-1, 3), off.ravel(), faces.reshape(-1, 3)


def bending_strain(sec: dict, axis: int, force: np.ndarray, moment: np.ndarray, E: float,
                   towards: float = 1.0, Ef: float | None = None) -> np.ndarray:
    """Axial strain on both faces at the ends of each section segment (in face_points order) from beam theory.

    The axial force over EA plus the bending curvature that the moment about the centroid gives through the
    section's rigidity: E times the second moment of the walls' areas about the centroid, plus the flexural
    modulus Ef times each wall's own t^3 / 12 term, which is what carries a flat plate's bending. force and
    moment are what the part beyond the section carries, and towards is +1 when that part lies toward +axis.
    """
    Ef = E if Ef is None else Ef
    a = np.eye(3)[axis]
    S, t, c, n = sec["segments"], sec["t"], sec["centroid"], sec["normal"]
    uv = [i for i in range(3) if i != axis]
    r = S - c                               # (k,2,3)
    y, z = r[..., uv[0]], r[..., uv[1]]
    tl = t * sec["length"]
    own = tl * t ** 2 / 12.0
    wy, wz = n[:, uv[0]], n[:, uv[1]]
    # exact second moments of straight segments, L (y1^2 + y1 y2 + y2^2) / 3, plus each wall's own thickness
    Iyy = E * float((tl * (z[:, 0] ** 2 + z[:, 0] * z[:, 1] + z[:, 1] ** 2) / 3).sum()) + Ef * float((own * wz ** 2).sum())
    Izz = E * float((tl * (y[:, 0] ** 2 + y[:, 0] * y[:, 1] + y[:, 1] ** 2) / 3).sum()) + Ef * float((own * wy ** 2).sum())
    Iyz = E * float((tl * (2 * y[:, 0] * z[:, 0] + y[:, 0] * z[:, 1] + y[:, 1] * z[:, 0]
                           + 2 * y[:, 1] * z[:, 1]) / 6).sum()) + Ef * float((own * wy * wz).sum())
    My, Mz = float(moment[uv[0]]), float(moment[uv[1]])
    # strain e = e0 + ky z - kz y with the moments M_y = ky EIyy - kz EIyz and M_z = kz EIzz - ky EIyz
    det = Iyy * Izz - Iyz ** 2
    ky = (My * Izz + Mz * Iyz) / det
    kz = (Mz * Iyy + My * Iyz) / det
    e0 = float(force @ a) / (E * float(tl.sum()))
    _, _, P = face_points(sec)
    rp = P - c
    # the second section axis can make a left handed pair with the beam axis, which turns the moments round
    hand = 1.0 if (np.cross(np.eye(3)[uv[0]], np.eye(3)[uv[1]]) @ a) > 0 else -1.0
    return towards * (e0 + hand * (ky * rp[:, uv[1]] - kz * rp[:, uv[0]]))


def beam_hand_check(model: ShellModel, axial_strain_at, zones: dict, E: float, what: str,
                    Ef: float | None = None) -> Check:
    """Compare the shell's axial strain with beam theory at a section 30% of the way from the supports.

    Holds for a slender part held at one end: everything past the section, loads and rigid attachments, makes a
    force and a moment about the section's centroid, which beam theory turns into a strain on each face of the
    walls. ``axial_strain_at(points, offsets)`` gives the shell's axial strain at midsurface points, offset
    through the wall along its normal; E is the walls' axial modulus and Ef their flexural one.
    """
    mesh = model.mesh
    sl = slender_axis(mesh.X)
    if sl is None:
        return Check("hand calculation", "skip", "no simple beam model applies (the part is not slender)")
    ax, length = sl
    held = np.concatenate([n for _, n, _ in model.constraints]) if model.constraints else np.zeros(0, int)
    xs = mesh.X[:, ax]
    lo, hi = xs.min(), xs.max()
    h = mesh.X[held, ax]
    if h.max() <= lo + 0.15 * length:
        root, sgn = lo, 1.0
    elif h.min() >= hi - 0.15 * length:
        root, sgn = hi, -1.0
    else:
        return Check("hand calculation", "skip", "no simple beam model applies (the supports are not at one end)")
    at = root + sgn * 0.3 * length
    sec = section(mesh.X, mesh.tris, model.section_thickness(), ax, at)
    if sec is None:
        return Check("hand calculation", "skip", "no simple beam model applies (no section cuts the part)")
    beyond = (xs - at) * sgn > 0
    F = model.forces if model.forces is not None else np.zeros((mesh.n_nodes, 3))
    c = sec["centroid"]
    force = F[beyond].sum(0)
    moment = np.cross(mesh.X[beyond] - c, F[beyond]).sum(0)
    for r in model.rigid:
        if (np.asarray(r.ref_point)[ax] - at) * sgn > 0:
            force = force + r.force
            moment = moment + np.cross(np.asarray(r.ref_point) - c, r.force) + r.moment
    if model.gravity is not None and np.linalg.norm(model.gravity) > 0:
        return Check("hand calculation", "skip", "beam check skipped: self weight acts everywhere")
    e_beam = bending_strain(sec, ax, force, moment, E, sgn, Ef)
    pts, off, _ = face_points(sec)
    e_fea = axial_strain_at(pts, off)
    nb, nf = float(np.abs(e_beam).max()), float(np.abs(e_fea).max())
    if nb <= 0:
        return Check("hand calculation", "skip", "no bending or axial load reaches the section")
    ratio = nf / nb
    s = (f"beam theory with {what} through a section {fmt(0.3 * length)} mm from the supports gives a peak axial "
         f"strain of {fmt(1e6 * nb)} microstrain; the shell model gives {fmt(1e6 * nf)} ({ratio:.3g}x)")
    status = "pass" if 0.9 <= ratio <= 1.1 else ("warn" if 0.75 <= ratio <= 1.33 else "fail")
    if status != "pass":
        s += "; check the layup angles, the loads and how the supports hold the part"
    return Check("hand calculation", status, s, {"beam_strain": nb, "shell_strain": nf, "ratio": ratio,
                                                 "section_at_mm": at, "axis": "xyz"[ax], "E_MPa": E})


__all__ = ["force_balance", "beam_hand_check", "section", "face_points", "bending_strain", "slender_axis"]
