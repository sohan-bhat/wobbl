"""A solid motor run as a 3D cutaway on a static test stand, swept over the design input a slider controls.

Everything comes from the run's playback.json. Its sweep holds one full burn per value of the input the question
turns on (the nozzle throat by default, see wobbl.chem.motor_sweep), each kept at its peak pressure, the hardest
moment for the casing. Every slider value shows that burn's grains at their regression then, for round cores,
tubes, end burners and any port outline (star, finocyl, moon), read along rays from the axis, with the nozzle at
its throat and the plume as long as the thrust. A quarter of the casing, grains and nozzle is cut away so the port
is in sight. The casing is colored by its thick wall hoop stress through the wall against the allowable, its yield
over the target safety factor. A run written before sweeps shows its own design at its peak pressure.
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


def _at_peak(pb: dict) -> dict:
    """A one value sweep from a run's time frames, for a run written before sweeps: its design at peak pressure."""
    m, fr = pb["motor"], pb["frames"]
    k = int(np.argmax([f["P_MPa"] for f in fr]))
    f, case = fr[k], pb["motor"].get("casing") or {}
    geo = []
    for sl in pb["grains"]:
        g = pb["grain_types"][sl["type"]]
        geo.append({key: val[k] if isinstance(val, list) else val for key, val in g.items()
                    if key in ("profile", "length_now_mm", "core_mm", "outer_mm", "port")})
    th = m.get("nozzle", {}).get("throat_mm", 10.0)
    pt = {"value": th, "P_MPa": f["P_MPa"], "F_N": f["F_N"], "kn": f.get("kn"), "impulse_Ns": m.get("impulse_Ns"),
          "safety_factor": f.get("casing_sf"), "throat_mm": th, "inner_mm": case.get("inner_mm"),
          "outer_mm": case.get("outer_mm"), "grains": geo}
    return {"input": "throat", "label": "nozzle throat", "unit": "mm", "design": th, "points": [pt]}


def _cross(v, y, level) -> float | None:
    """The input value where y first crosses level, between two swept points."""
    y = np.asarray(y, float)
    for i in range(len(y) - 1):
        a, b = y[i] - level, y[i + 1] - level
        if np.isfinite(a) and np.isfinite(b) and a * b < 0:
            return float(v[i] + (v[i + 1] - v[i]) * a / (a - b))
    return None


def _grain_frame(g: dict, z0: float, L0: float, R: float, ends: int):
    """One grain's solid and its burning port surface at the regression its point gives."""
    prof = g.get("profile", "round")
    if prof == "outline":
        rin = port_radius(g.get("port") or [], fallback=R)
    elif prof in ("round", "tube"):
        rin = np.full(len(PHI), (g.get("core_mm") or 0) / 2)
    else:
        rin = np.zeros(len(PHI))
    rin = np.minimum(np.maximum(rin, 0.3), R)
    rout = (g.get("outer_mm") or R * 2) / 2 if prof == "tube" else R
    now = g.get("length_now_mm")
    now = L0 if now is None else now
    gone = L0 - now
    a = gone / 2 if ends == 2 else 0.0
    b = gone - a if ends >= 1 else 0.0
    za, zb = z0 + a, z0 + L0 - b
    if now <= 0:
        rin, rout, za, zb = np.full(len(PHI), R * 0.999), R, z0 + L0 / 2, z0 + L0 / 2 + 1e-3
    X, T = arc_solid(np.stack([rin, rin]), [za, zb], [rout, rout], [za, zb])
    Xg, Tg = arc_solid(np.stack([rin * 0.985, rin * 0.985]), [za, zb], np.stack([rin, rin]), [za, zb])
    return X, T, Xg, Tg


def _nozzle(Ri, Ro, rt, re_, half, L, wall):
    zc, zt = L + 6, L + 6 + (Ri - rt) * 0.9
    ze = zt + max(re_ - rt, 0.5) / np.tan(half)
    zi = np.array([zc, zc + (zt - zc) * 0.5, zt, zt + (ze - zt) * 0.5, ze])
    ri = np.array([Ri, (Ri + rt) / 2, rt, (rt + re_) / 2, re_])
    ro = np.array([Ro, Ro, max(rt + wall * 2.5, Ro * 0.6), max((rt + re_) / 2 + wall * 1.5, Ro * 0.6),
                   re_ + wall * 1.2])
    X, T = arc_solid(ri, zi, ro, zi)
    return X, T, ze


def build(out: Path, pb: dict, o: dict | None = None) -> Path | None:
    o = o or read(out, "outcome.json") or {}
    m = pb["motor"]
    sw = pb.get("sweep") or _at_peak(pb)
    pts = sw["points"]
    if not pts:
        return None
    v = np.array([p["value"] for p in pts], float)
    n = len(v)
    num = lambda key: np.array([np.nan if p.get(key) is None else p[key] for p in pts], float)  # noqa: E731
    Pm, Fn = num("P_MPa"), num("F_N")
    s = Scene(f"{m.get('name', 'Motor')} at its peak pressure, cut away", domain="motor",
              question=o.get("question", ""), length="mm",
              source=f"Each {sw['label']} on the slider is its own burn, shown at its peak pressure with every "
                     "grain burned back as far as it had by then. The casing's hoop stress is the thick wall "
                     "stress through the wall at that pressure.")
    s.timeline(v, unit=sw.get("unit", "mm"), label=sw["label"], kind="input", design=sw.get("design", v[0]),
               between="step", note=sw.get("note", ""))
    case = m.get("casing") or {}
    grains, types = pb["grains"], pb["grain_types"]
    Ro0 = (case.get("outer_mm") or max(g["diameter_mm"] for g in types) + 4) / 2
    Ri0 = (case.get("inner_mm") or Ro0 * 2 - 3) / 2
    Ri = np.array([(p.get("inner_mm") or Ri0 * 2) / 2 for p in pts])
    Ro = np.array([(p.get("outer_mm") or Ro0 * 2) / 2 for p in pts])
    stack = max(sl["z_mm"] + sl["length_mm"] for sl in grains)
    L = max(case.get("length_mm") or 0, stack)
    off = (L - stack) / 2
    # the motor lies along +x with its nozzle that way; its own y is up, so the cut opens toward the viewer
    q = P.quat([0, 1, 0], [0, 0, 1], [1, 0, 0])
    lift = Ro.max() + 30
    at = np.array([-L / 2, 0, lift])

    def place(mesh):
        mesh.move(at, q)
        return mesh

    # the casing, colored by the thick wall hoop stress at each vertex's radius against the allowable
    py = case.get("yield_pressure_MPa")
    sy = py * (Ro0 ** 2 + Ri0 ** 2) / (Ro0 ** 2 - Ri0 ** 2) if py else None
    tsf = case.get("target_sf") or 1.0
    allow_hoop = sy / tsf if sy else None
    allow_p = (sy * (Ro ** 2 - Ri ** 2) / (Ro ** 2 + Ri ** 2) / tsf) if sy else None
    if allow_p is not None and np.ptp(allow_p) < 1e-9 * allow_p.max():
        allow_p = float(allow_p[0])                      # one wall over the whole sweep, one allowable
    Xc = np.array([arc_solid([a, a], [-6, L + 6], [b, b], [-6, L + 6])[0] for a, b in zip(Ri, Ro)])
    Tc = arc_solid([Ri[0]] * 2, [-6, L + 6], [Ro[0]] * 2, [-6, L + 6])[1]
    r = np.hypot(Xc[..., 0], Xc[..., 1])
    lame = (Pm * Ri ** 2 / (Ro ** 2 - Ri ** 2))[:, None] * (1 + (Ro ** 2)[:, None] / r ** 2)
    hoop = np.nan_to_num(lame[:, : Xc.shape[1] // 2].max(axis=1))
    shell = place(s.mesh(Xc[0], Tc, name="casing", color="#b9bdc4", smooth=False))
    if n > 1:
        shell.deform(Xc - Xc[0])
    # as the run judges the casing: past yield it fails, past yield over the target safety factor it is marginal
    shell.color_by("hoop stress", np.nan_to_num(lame), unit="MPa", colormap="turbo",
                   range=(0, sy or max(hoop.max(), 1e-9)), limit=sy, limit_label="yield" if sy else "",
                   target=allow_hoop if sy and tsf > 1 else None)
    # forward closure
    wall = float(Ro0 - Ri0)
    Xf, Tf = arc_solid([0, 0], [-6 - wall * 4, -6], [Ri0, Ri0], [-6 - wall * 4, -6])
    place(s.mesh(Xf, Tf, name="forward closure", color="#8d939b", smooth=False))

    # grains, each at the regression of its burn's peak pressure
    glow = np.clip(Pm / max(np.nanmax(Pm), 1e-9), 0, 1)
    for k, sl in enumerate(grains):
        g = types[sl["type"]]
        z0, L0, R = sl["z_mm"] + off, sl["length_mm"], g["diameter_mm"] / 2
        ends = int(g.get("ends_burning", 2))
        made = [_grain_frame(p["grains"][k], z0, L0, R, ends) for p in pts]
        GX, T = np.array([x[0] for x in made]), made[0][1]
        PX, Tg = np.array([x[2] for x in made]), made[0][3]
        gm = place(s.mesh(GX[0], T, name="grain" if k == 0 else f"grain {k + 1}", color="#d8c08a",
                          smooth=False, max_vertices=20000))
        if n > 1:
            gm.deform(GX - GX[0])
        if g.get("profile", "round") != "end":
            pm = place(s.mesh(PX[0], Tg, name="burning port" if k == 0 else f"burning port {k + 1}",
                              color="#ff7a1a", smooth=False, max_vertices=20000))
            if n > 1:
                pm.deform(PX - PX[0])
            pm.glow = glow

    # the nozzle at each value's throat, out at its half angle to the exit
    noz = m.get("nozzle", {})
    re_ = noz.get("exit_mm", Ri0) / 2
    half = np.radians(noz.get("half_angle_deg", 12) or 12)
    th = np.array([(p.get("throat_mm") or noz.get("throat_mm", Ri0 * 0.8)) / 2 for p in pts])
    made = [_nozzle(Ri0, Ro0, rt, max(re_, rt * 1.05), half, L, wall) for rt in th]
    NX, Tn = np.array([x[0] for x in made]), made[0][1]
    ze = np.array([x[2] for x in made])
    nm = place(s.mesh(NX[0], Tn, name="nozzle", color="#8d939b", smooth=False))
    if n > 1:
        nm.deform(NX - NX[0])

    # the plume at the peak thrust of each value's burn, as long and bright as that thrust
    frac = np.clip(Fn / max(np.nanmax(Fn), 1e-9), 0.05, 1)
    Xp, Tp = F.plume(re_, L * 1.1)
    Xp = Xp * [1, 1, -1] + [0, 0, ze[0]]
    pl = place(s.mesh(Xp, Tp, name="exhaust", color="#ffa040", opacity=0.7, max_vertices=600))
    stretch = np.c_[np.zeros(len(Xp)), np.zeros(len(Xp)), Xp[:, 2] - ze[0]]
    pl.deform(stretch[None] * (frac - 1)[:, None, None] + (ze - ze[0])[:, None, None] * [0, 0, 1])
    pl.glow = frac
    s.notes = [x for x in s.notes if not x.startswith("Deformation")]

    # the stand: two saddles under the casing and a thrust plate against the forward closure
    for x in (-L * 0.3, L * 0.3):
        s.body("box", [12, Ro0 * 0.9, (lift - Ro0) / 2 + 2], pos=[x, 0, (lift - Ro0) / 2 + 2], color=P.STEEL,
               name="stand saddle", prop=True)
    s.body("box", [6, Ro0 * 1.6, (lift + Ro0) / 2], pos=[-L / 2 - 6 - wall * 4 - 6, 0, (lift + Ro0) / 2],
           color=P.DARK, name="thrust plate", prop=True)
    s.ground(0, size=L * 4, grid=50, center=(L * 0.4, 0))

    sf = num("safety_factor")
    p_yield = None if allow_p is None else allow_p * tsf
    s.readout("peak pressure", Pm, unit="MPa", digits=3, limit=p_yield,
              target=allow_p if tsf > 1 and np.ndim(allow_p) == 0 else None)
    if np.isfinite(sf).any():
        s.readout("casing safety factor", sf, digits=3)
    s.readout("hoop stress", hoop, unit="MPa", digits=3, limit=sy, target=allow_hoop if tsf > 1 else None)
    if np.isfinite(num("impulse_Ns")).all():
        s.readout("total impulse", num("impulse_Ns"), unit="N s", digits=4)
    s.readout("peak thrust", Fn, unit="N", digits=4)
    if np.isfinite(num("kn")).all():
        s.readout("Kn", num("kn"), digits=3)
    if np.isfinite(num("port_throat")).all():
        s.readout("port to throat", num("port_throat"), digits=3)
    if np.isfinite(num("mass_flux")).any():
        from ...chem.motor import MASS_FLUX_LIMIT
        # past the usual erosive burning limit the run calls the motor marginal, as the slider does
        s.readout("core mass flux", num("mass_flux"), unit="kg/(m2 s)", digits=4, target=MASS_FLUX_LIMIT)

    # where on the slider the casing and the ports change their verdict
    if n > 1 and np.isfinite(sf).any():
        for level, name, kind in ((tsf, f"safety factor {tsf:g}", "warn"), (1.0, "casing yields", "fail")):
            x = _cross(v, sf, level)
            if x is not None:
                s.event(x, name, f"at {x:.3g} {sw.get('unit', 'mm')}", kind="fail" if kind == "fail" else "info")
    if n > 1 and np.isfinite(num("port_throat")).all():
        x = _cross(v, num("port_throat"), 1.0)
        if x is not None:
            s.event(x, "port chokes", "the core is narrower than the throat", kind="fail")
    s.label("hoop stress {hoop stress}", at + np.array([L * 0.5, -Ro0, Ro0]), kind="peak")
    s.label("{peak pressure} in the chamber", at + np.array([L * 0.2, -Ro0, -Ro0 * 1.6]), kind="note")
    verdict(s, o)
    s.camera(target=[L * 0.15, 0, lift], direction=(-0.25, -1.0, 0.45), distance=L * 2.3, fov=30,
             presets=[{"name": "Into the port", "target": [-L * 0.1, 0, lift], "dir": (-0.05, -1.0, 0.35),
                       "dist": L * 1.2},
                      {"name": "Nozzle", "target": [L / 2 + 20, 0, lift], "dir": (0.6, -1.0, 0.3), "dist": L * 1.1},
                      {"name": "Whole stand", "target": [L * 0.4, 0, lift], "dir": (-0.25, -1.0, 0.45),
                       "dist": L * 3.6}])
    return s.save(out)
