"""A solid motor run as a 3D cutaway on a static test stand: casing, grains, nozzle and plume at their real sizes.

Everything comes from the run's playback.json. Each grain's port grows and its burning ends move back frame by
frame as the run's regression says, for round cores, tubes, end burners and any port outline (star, finocyl,
moon), whose outline is read along rays from the axis. A quarter of the casing, grains and nozzle is cut away so
the port is in sight. The casing is colored by its hoop stress, the chamber pressure times the bore radius over
the wall, against the yield of its material, and the plume follows the thrust.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .. import props as P
from .. import props_flight as F
from ..scene import Scene
from .common import read, verdict

PHI = np.radians(np.linspace(215, 485, 25))      # the solid part of every cutaway, open toward the viewer


def arc_solid(r_in, z_in, r_out, z_out, phi=PHI) -> tuple[np.ndarray, np.ndarray]:
    """A body of revolution over the angles phi, between an inner and an outer profile of the same number of
    points (radius, height). r_in may also give one radius per angle at each profile point, for a port that is not
    round. Walls, end rings and the two cut faces are all closed, and the vertex order depends only on the sizes,
    so a body rebuilt with new radii lines up vertex for vertex."""
    m, n = len(z_in), len(phi)
    ri = np.broadcast_to(np.asarray(r_in, float).reshape(m, -1), (m, n))
    ro = np.broadcast_to(np.asarray(r_out, float).reshape(m, -1), (m, n))
    c, s = np.cos(phi), np.sin(phi)
    inner = np.stack([ri * c, ri * s, np.broadcast_to(np.asarray(z_in, float)[:, None], (m, n))], -1)
    outer = np.stack([ro * c, ro * s, np.broadcast_to(np.asarray(z_out, float)[:, None], (m, n))], -1)
    X = np.concatenate([inner.reshape(-1, 3), outer.reshape(-1, 3)])
    I = lambda i, j: i * n + j               # noqa: E731
    O = lambda i, j: m * n + i * n + j       # noqa: E731
    T = []
    for i in range(m - 1):
        for j in range(n - 1):
            T += [(O(i, j), O(i, j + 1), O(i + 1, j + 1)), (O(i, j), O(i + 1, j + 1), O(i + 1, j)),
                  (I(i, j), I(i + 1, j + 1), I(i, j + 1)), (I(i, j), I(i + 1, j), I(i + 1, j + 1))]
    for i, flip in ((0, False), (m - 1, True)):            # the end rings
        for j in range(n - 1):
            a = [(I(i, j), O(i, j + 1), O(i, j)), (I(i, j), I(i, j + 1), O(i, j + 1))]
            T += [(x, z, y) for x, y, z in a] if flip else a
    for j, flip in ((0, False), (n - 1, True)):            # the cut faces
        for i in range(m - 1):
            a = [(I(i, j), O(i, j), O(i + 1, j)), (I(i, j), O(i + 1, j), I(i + 1, j))]
            T += [(x, z, y) for x, y, z in a] if flip else a
    return X, np.asarray(T)


def port_radius(outline: list, phi=PHI, fallback: float = 0.0) -> np.ndarray:
    """The port's radius along each angle, from its outline polylines: where a ray from the axis first leaves it."""
    if not outline:
        return np.full(len(phi), fallback)
    out = np.full(len(phi), np.inf)
    d = np.c_[np.cos(phi), np.sin(phi)]
    for ln in outline:
        p = np.asarray(ln, float)
        a, b = p[:-1], p[1:]
        e = b - a
        for k, (dx, dy) in enumerate(d):
            den = dx * e[:, 1] - dy * e[:, 0]
            ok = np.abs(den) > 1e-12
            with np.errstate(divide="ignore", invalid="ignore"):
                t = (a[:, 0] * e[:, 1] - a[:, 1] * e[:, 0]) / den
                u = (a[:, 0] * dy - a[:, 1] * dx) / den
            hit = ok & (t > 0) & (u >= 0) & (u <= 1)
            if hit.any():
                out[k] = min(out[k], float(t[hit].min()))
    return np.where(np.isfinite(out), out, fallback)


def build(out: Path, pb: dict, o: dict | None = None) -> Path | None:
    o = o or read(out, "outcome.json") or {}
    m, fr = pb["motor"], pb["frames"]
    t = np.array([f["t"] for f in fr], float)
    Pm = np.array([f["P_MPa"] for f in fr], float)
    Fn = np.array([f["F_N"] for f in fr], float)
    n = len(t)
    s = Scene(f"{m.get('name', 'Motor')} firing on a test stand, cut away", domain="motor",
              question=o.get("question", ""), length="mm",
              source="Each grain's port grows and its ends burn back as the run's regression says. The casing's "
                     "hoop stress is the chamber pressure times the bore radius over the wall.")
    s.timeline(t, unit="s", label="time", seconds=float(np.clip(4 * t[-1], 6, 14)), between="interpolated")
    case = m.get("casing") or {}
    grains, types = pb["grains"], pb["grain_types"]
    Ro = (case.get("outer_mm") or max(g["diameter_mm"] for g in types) + 4) / 2
    Ri = (case.get("inner_mm") or Ro * 2 - 3) / 2
    stack = max(sl["z_mm"] + sl["length_mm"] for sl in grains)
    L = max(case.get("length_mm") or 0, stack)
    off = (L - stack) / 2
    # the motor lies along +x with its nozzle that way; its own y is up, so the cut opens toward the viewer
    q = P.quat([0, 1, 0], [0, 0, 1], [1, 0, 0])
    lift = Ro + 30
    at = np.array([-L / 2, 0, lift])

    def place(mesh):
        mesh.move(at, q)
        return mesh

    # the casing, colored by hoop stress against the yield of its material
    wall = max(Ro - Ri, 0.1)
    hoop = Pm * Ri / wall
    py = case.get("yield_pressure_MPa")
    sy = py * Ri / wall if py else None
    allow = case.get("allowable_MPa")
    allow_hoop = allow * Ri / wall if allow else (sy / (case.get("target_sf") or 1) if sy else None)
    Xc, Tc = arc_solid([Ri, Ri], [-6, L + 6], [Ro, Ro], [-6, L + 6])
    shell = place(s.mesh(Xc, Tc, name="casing", color="#b9bdc4", smooth=False))
    shell.color_by("hoop stress", np.outer(hoop, np.ones(len(Xc))), unit="MPa", colormap="turbo",
                   range=(0, sy or max(hoop.max(), 1e-9)), limit=sy, limit_label="yield")
    # forward closure
    Xf, Tf = arc_solid([0, 0], [-6 - wall * 4, -6], [Ri, Ri], [-6 - wall * 4, -6])
    place(s.mesh(Xf, Tf, name="forward closure", color="#8d939b", smooth=False))

    # grains: every vertex recomputed per frame from the run's port and length
    for k, sl in enumerate(grains):
        g = types[sl["type"]]
        z0, L0, R = sl["z_mm"] + off, sl["length_mm"], g["diameter_mm"] / 2
        prof = g.get("profile", "round")
        now = np.asarray(g.get("length_now_mm") or [L0] * n, float)
        ends = int(g.get("ends_burning", 2))
        frames_X, glow_X = [], []
        for j in range(n):
            if prof == "outline":
                port = g.get("port") or []
                rin = port_radius(port[j] if j < len(port) else [], fallback=R)
            elif prof in ("round", "tube"):
                rin = np.full(len(PHI), (g.get("core_mm") or [0] * n)[j] / 2)
            else:
                rin = np.zeros(len(PHI))
            rin = np.minimum(np.maximum(rin, 0.3), R)
            rout = (g.get("outer_mm") or [R * 2] * n)[j] / 2 if prof == "tube" else R
            gone = L0 - now[j]
            a = gone / 2 if ends == 2 else 0.0
            b = gone - a if ends >= 1 else 0.0
            za, zb = z0 + a, z0 + L0 - b
            if now[j] <= 0:
                rin, rout, za, zb = np.full(len(PHI), R * 0.999), R, z0 + L0 / 2, z0 + L0 / 2 + 1e-3
            X, T = arc_solid(np.stack([rin, rin]), [za, zb], [rout, rout], [za, zb])
            frames_X.append(X)
            Xg, Tg = arc_solid(np.stack([rin * 0.985, rin * 0.985]), [za, zb], np.stack([rin, rin]), [za, zb])
            glow_X.append(Xg)
        frames_X, glow_X = np.array(frames_X), np.array(glow_X)
        gm = place(s.mesh(frames_X[0], T, name="grain" if k == 0 else f"grain {k + 1}", color="#d8c08a",
                          smooth=False, max_vertices=20000))
        gm.deform(frames_X - frames_X[0])
        if prof != "end":
            pm = place(s.mesh(glow_X[0], Tg, name="burning port" if k == 0 else f"burning port {k + 1}",
                              color="#ff7a1a", smooth=False, max_vertices=20000))
            pm.deform(glow_X - glow_X[0])
            pm.glow = np.clip(Pm / max(Pm.max(), 1e-9), 0, 1)
    s.notes = [x for x in s.notes if not x.startswith("Deformation")]

    # the nozzle: converging to the throat, then out at its half angle to the exit
    noz = m.get("nozzle", {})
    rt, re_ = noz.get("throat_mm", Ri * 0.4) / 2, noz.get("exit_mm", Ri) / 2
    half = np.radians(noz.get("half_angle_deg", 12) or 12)
    zc, zt = L + 6, L + 6 + (Ri - rt) * 0.9
    ze = zt + max(re_ - rt, 0.5) / np.tan(half)
    zi = np.array([zc, zc + (zt - zc) * 0.5, zt, zt + (ze - zt) * 0.5, ze])
    ri = np.array([Ri, (Ri + rt) / 2, rt, (rt + re_) / 2, re_])
    ro = np.array([Ro, Ro, max(rt + wall * 2.5, Ro * 0.6), max((rt + re_) / 2 + wall * 1.5, Ro * 0.6),
                   re_ + wall * 1.2])
    Xn, Tn = arc_solid(ri, zi, ro, zi)
    place(s.mesh(Xn, Tn, name="nozzle", color="#8d939b", smooth=False))

    # the plume, as long and bright as the thrust
    frac = np.clip(Fn / max(Fn.max(), 1e-9), 0, 1)
    Xp, Tp = F.plume(re_, L * 1.1)
    Xp = Xp * [1, 1, -1] + [0, 0, ze]
    pl = place(s.mesh(Xp, Tp, name="exhaust", color="#ffa040", opacity=0.7, max_vertices=600))
    pl.deform(np.c_[np.zeros(len(Xp)), np.zeros(len(Xp)), Xp[:, 2] - ze], scale=frac - 1)
    pl.glow = frac
    s.notes = [x for x in s.notes if not x.startswith("Deformation")]

    # the stand: two saddles under the casing and a thrust plate against the forward closure
    for x in (-L * 0.3, L * 0.3):
        s.body("box", [12, Ro * 0.9, (lift - Ro) / 2 + 2], pos=[x, 0, (lift - Ro) / 2 + 2], color=P.STEEL,
               name="stand saddle", prop=True)
    s.body("box", [6, Ro * 1.6, (lift + Ro) / 2], pos=[-L / 2 - 6 - wall * 4 - 6, 0, (lift + Ro) / 2],
           color=P.DARK, name="thrust plate", prop=True)
    s.ground(0, size=L * 4, grid=50, center=(L * 0.4, 0))

    s.readout("thrust", Fn, unit="N", digits=4)
    s.readout("chamber pressure", Pm, unit="MPa", digits=3, limit=allow)
    s.readout("hoop stress", hoop, unit="MPa", digits=3, limit=allow_hoop)
    s.readout("propellant left", [f["propellant_kg"] for f in fr], unit="kg", digits=3)
    if "kn" in fr[0]:
        s.readout("Kn", [f["kn"] for f in fr], digits=3)
    k = int(np.argmax(Pm))
    s.event(float(t[k]), "peak pressure", f"{Pm[k]:.3g} MPa")
    peak_at = at + np.array([Ri * 0, -Ro, Ro]) + [L * 0.5, 0, 0]
    s.label("hoop stress {hoop stress}", peak_at, kind="peak")
    on = np.nonzero(Fn > 0.05 * Fn.max())[0]
    if len(on):
        s.event(float(t[on[-1]]), "burnout")
    verdict(s, o)
    s.camera(target=[L * 0.15, 0, lift], direction=(-0.25, -1.0, 0.45), distance=L * 2.3, fov=30,
             presets=[{"name": "Into the port", "target": [-L * 0.1, 0, lift], "dir": (-0.05, -1.0, 0.35),
                       "dist": L * 1.2},
                      {"name": "Nozzle", "target": [L / 2 + 20, 0, lift], "dir": (0.6, -1.0, 0.3), "dist": L * 1.1},
                      {"name": "Whole stand", "target": [L * 0.4, 0, lift], "dir": (-0.25, -1.0, 0.45),
                       "dist": L * 3.6}])
    return s.save(out)
