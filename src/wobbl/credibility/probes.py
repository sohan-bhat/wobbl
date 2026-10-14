"""Where a gauge would read the answer: surface points, the stress state there and the strain a gauge would show.

A strain gauge sits on the surface and reads the strain along its axis, averaged over its grid. At a surface the
stress normal to it is zero, so the gauge axis worth using is the larger principal direction in the surface
plane. Next to a fillet the stress falls off within a millimeter or two, and a gauge averages over that fall, so
each point also says how much the stress changes across a gauge length.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

GAUGE_MM = 3.0       # a common strain gauge grid length


def load_fields(result_dir) -> dict | None:
    """The converged fields of a run (strength in work/, heat at the top), or None if they were not kept."""
    d = Path(result_dir)
    for p in (d / "work" / "fields.npz", d / "fields.npz"):
        if p.exists():
            with np.load(p) as f:
                return {k: f[k] for k in f.files}
    return None


def tensor(s6: np.ndarray) -> np.ndarray:
    """(n, 6) stress xx, yy, zz, xy, yz, zx to (n, 3, 3)."""
    s6 = np.atleast_2d(s6)
    T = np.empty((len(s6), 3, 3))
    T[:, 0, 0], T[:, 1, 1], T[:, 2, 2] = s6[:, 0], s6[:, 1], s6[:, 2]
    T[:, 0, 1] = T[:, 1, 0] = s6[:, 3]
    T[:, 1, 2] = T[:, 2, 1] = s6[:, 4]
    T[:, 0, 2] = T[:, 2, 0] = s6[:, 5]
    return T


def strain_along(s6: np.ndarray, E: float, nu: float, d: np.ndarray) -> np.ndarray:
    """Normal strain along unit direction d from Hooke's law, eps_dd = ((1 + nu) s_dd - nu tr s) / E."""
    T = tensor(s6)
    d = np.asarray(d, float) / np.linalg.norm(d)
    sdd = np.einsum("i,nij,j->n", d, T, d)
    return ((1 + nu) * sdd - nu * np.trace(T, axis1=1, axis2=2)) / E


def surface(fields: dict) -> tuple[np.ndarray, np.ndarray]:
    """Nodes a gauge can sit on and their outward unit normals, area weighted over the faces around each node.

    Midside nodes count, since the governing point is often one. A node on an edge between two CAD faces has no
    single normal and no flat spot for a gauge, so nodes that touch more than one CAD face are left out."""
    X, tris = fields["X"], fields["tris"]
    p0, p1, p2 = X[tris[:, 0]], X[tris[:, 1]], X[tris[:, 2]]
    c = np.cross(p1 - p0, p2 - p0)
    n = np.zeros_like(X)
    for k in range(tris.shape[1]):
        np.add.at(n, tris[:, k], c)
    ids = np.unique(tris)
    if "tri_surf" in fields:
        node = tris.ravel()
        face = np.repeat(fields["tri_surf"], tris.shape[1])
        lo = np.full(len(X), np.iinfo(np.int64).max)
        hi = np.full(len(X), np.iinfo(np.int64).min)
        np.minimum.at(lo, node, face)
        np.maximum.at(hi, node, face)
        inside = ids[lo[ids] == hi[ids]]
        ids = inside if len(inside) else ids
    nn = n[ids] / np.maximum(np.linalg.norm(n[ids], axis=1), 1e-300)[:, None]
    return ids, nn


def gauge(fields: dict, xyz, E: float, nu: float, direction=None, length: float = GAUGE_MM, surf=None) -> dict:
    """What a strain gauge centred at the surface point nearest xyz would read for this run's load.

    With no direction the gauge is laid along the larger principal stress in the surface plane. The reading is
    the mean strain over surface nodes within half the gauge length of the centre, and the spread says how much
    the stress changes under the grid.
    """
    ids, nn = surf if surf is not None else surface(fields)
    X, S = fields["X"], fields["S"]
    j = int(np.argmin(np.linalg.norm(X[ids] - np.asarray(xyz, float), axis=1)))
    node, nrm = int(ids[j]), nn[j]
    if direction is None:
        P = np.eye(3) - np.outer(nrm, nrm)
        w, v = np.linalg.eigh(P @ tensor(S[node])[0] @ P)
        k = int(np.argmax(np.abs(w)))
        direction = v[:, k]
    d = np.asarray(direction, float) / np.linalg.norm(direction)
    near = ids[np.linalg.norm(X[ids] - X[node], axis=1) <= 0.5 * length]
    if node not in near:
        near = np.append(near, node)
    eps = strain_along(S[near], E, nu, d)
    meas = fields.get("measure")
    return {"node": node, "xyz": X[node].tolist(), "normal": nrm.tolist(), "direction": d.tolist(),
            "strain": float(eps.mean()), "strain_at_centre": float(strain_along(S[node], E, nu, d)[0]),
            "strain_range": [float(eps.min()), float(eps.max())],
            "measure_range": [float(meas[near].min()), float(meas[near].max())] if meas is not None else None,
            "excluded": bool(fields["excluded"][node]) if "excluded" in fields else False}


def candidates(result_dir, outcome: dict, E: float, nu: float) -> list[dict]:
    """Gauge spots worth considering for a strength run: the governing point, a calm spot on the main load path
    and the point of largest deflection. Each carries the reading for the run's load, so a test plan can scale it to
    any test load."""
    f = load_fields(result_dir)
    if f is None or "S" not in f:
        return []
    surf = surface(f)
    out = []
    peak = gauge(f, outcome["governing_xyz"], E, nu, surf=surf)
    out.append({"kind": "peak", "why": "the governing point, where the answer is decided", **peak})
    ids, _ = surf
    m, gov = f["measure"], outcome["governing_MPa"]
    ok = ids[~f["excluded"][ids]] if "excluded" in f else ids
    band = ok[(m[ok] > 0.3 * gov) & (m[ok] < 0.7 * gov)]
    if len(band):
        far = np.linalg.norm(f["X"][band] - np.asarray(peak["xyz"]), axis=1) > 4 * GAUGE_MM
        band = band[far] if far.any() else band
        # the calmest of the busier spots: score by how little the stress changes under a gauge
        pick = band[np.argsort(-m[band])[:200]]
        best, score = None, None
        for node in pick:
            g = gauge(f, f["X"][node], E, nu, surf=surf)
            lo, hi = g["measure_range"]
            s = (hi - lo) / max(hi, 1e-12)
            if score is None or s < score:
                best, score = g, s
        if best is not None:
            out.append({"kind": "far field", "why": "a calm spot on the main load path, where a gauge checks the "
                                                    "overall stiffness without fighting a steep gradient", **best})
    U = f["U"]
    i = int(np.argmax(np.linalg.norm(U, axis=1)))
    u = U[i]
    out.append({"kind": "deflection", "why": "the point that moves most, for a dial gauge or a displacement sensor",
                "node": i, "xyz": f["X"][i].tolist(), "direction": (u / max(np.linalg.norm(u), 1e-300)).tolist(),
                "displacement": float(np.linalg.norm(u))})
    return out
