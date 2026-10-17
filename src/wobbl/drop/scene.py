"""A drop run as a 3D model you set the drop height of: the part on the floor at the instant of its peak stress.

The part is linear elastic and the floor pushes back in proportion to how far the part sinks into it, so the whole
impact scales with the landing speed, which goes as the square root of the height. Every height's peak state is
the solved one scaled by that, exactly up to gravity's pull during the millisecond of contact. The slider moves
the height, and the stress, the elastic shape, the accelerometer and the verdict follow.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

FLOOR_COLOR = {"concrete": "#8a8d91", "steel": "#9aa3ad", "tile": "#d9d4c7", "stone": "#7d7f80", "wood": "#9a6b3f",
               "vinyl": "#b7b2a8", "carpet": "#6b5d73", "foam": "#e8e8e0", "grass": "#4f7a3a"}
G = 9810.0


def rotation_to_down(d: np.ndarray) -> np.ndarray:
    """The rotation that turns the landing direction d into -z, the way the part falls in the world."""
    a = np.asarray(d, float) / np.linalg.norm(d)
    b = np.array([0.0, 0.0, -1.0])
    v = np.cross(a, b)
    c = float(a @ b)
    if np.linalg.norm(v) < 1e-12:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * (1.0 / (1.0 + c))


def heights(H: float, n: int = 41) -> np.ndarray:
    """Drop heights from a tenth of the asked one to twice it, with the asked one among them."""
    return np.unique(np.r_[np.linspace(0.1 * H, 2.0 * H, n), H])


def height_sweep(s, X, tris, S_peak, el_peak, H: float, allow: float, *, target: float | None = None,
                 g_peak: float = 0.0, force_peak: float = 0.0, node: int | None = None, watch_at=None,
                 measure: str = "von Mises", floor_key: str = "", landing: str = "", zone=None):
    """Fill scene s with the part resting on the floor in its peak state, scaled to each drop height. X is the
    part already turned so it lands along -z with its first touch at z = 0, S_peak the stress at each vertex at
    the peak and el_peak the elastic displacement then (without the part's travel). zone marks the contact patch,
    whose stress is the local crush of an idealized landing: it is left uncolored and out of the peak."""
    h = heights(H)
    k = np.sqrt(h / H)
    S_peak, el_peak = np.array(S_peak, float), np.asarray(el_peak, float)
    if zone is not None and np.any(zone) and not np.all(zone):
        S_peak[np.asarray(zone, bool)] = np.nan
    gov = float(np.nanmax(S_peak))
    s.timeline(h / 1000, unit="m", label="drop height", between="interpolated", kind="input", design=H / 1000)
    mesh = s.mesh(X, tris, name="part", keep=[] if node is None else [int(node)], max_vertices=12000)
    mesh.color_by(measure, S_peak, scale=k, unit="MPa", range=(0.0, allow), limit=allow, limit_label="strength")
    if target and target > 1:
        s.fields[-1]["target"] = float(allow / target)
    span = float(np.ptp(X, axis=0).max())
    emax = float(np.abs(el_peak).max() * k.max()) if el_peak.size else 0.0
    ex = float(f"{max(1.0, 0.03 * span / emax):.2g}") if emax > 0 else 1.0
    mesh.deform(el_peak, scale=k, exaggerate=ex)
    s.ground(0.0, size=4 * span, grid=span / 4, color=FLOOR_COLOR.get(floor_key, "#8a8d91"))
    s.readout("peak stress", gov * k, unit="MPa", limit=allow)
    s.readout("safety factor", allow / np.maximum(gov * k, 1e-30), digits=3)
    v = np.sqrt(2 * G * h) / 1000
    s.readout("landing speed", v, unit="m/s", digits=3)
    if g_peak > 0:
        s.readout("accelerometer", g_peak * k, unit="g", digits=4)
    if force_peak > 0:
        s.readout("floor force", force_peak * k, unit="N", digits=4)
    if gov > 0 and allow > 0:
        h_fail = H * (allow / gov) ** 2
        if h[0] < h_fail < h[-1]:
            s.event(h_fail / 1000, "reaches its strength", f"from {h_fail / 1000:.3g} m", kind="fail")
    if node is not None:
        s.label("peak {peak stress}", X[int(node)], kind="peak")
    if zone is not None and np.any(zone) and not np.all(zone):
        s.label("contact patch, a local dent left out", X[np.asarray(zone, bool)].mean(0), kind="note")
    if watch_at is not None:
        s.label("accelerometer {accelerometer}" if g_peak > 0 else "watched spot", watch_at, kind="note")
    if landing:
        s.note(f"It lands {landing.replace(' down', '')} first, its worst landing, and each height shows the "
               "instant of its peak stress.")
    s.note("The impact of a linear elastic part scales with its landing speed, the square root of the height, so "
           "each height is the solved impact scaled exactly, up to gravity during the millisecond of contact.")
    return mesh


def _quat(Rm: np.ndarray) -> np.ndarray:
    """A rotation matrix as a unit quaternion (w, x, y, z)."""
    w = np.sqrt(max(0.0, 1.0 + Rm[0, 0] + Rm[1, 1] + Rm[2, 2])) / 2
    x = np.sqrt(max(0.0, 1.0 + Rm[0, 0] - Rm[1, 1] - Rm[2, 2])) / 2
    y = np.sqrt(max(0.0, 1.0 - Rm[0, 0] + Rm[1, 1] - Rm[2, 2])) / 2
    z = np.sqrt(max(0.0, 1.0 - Rm[0, 0] - Rm[1, 1] + Rm[2, 2])) / 2
    x = np.copysign(x, Rm[2, 1] - Rm[1, 2])
    y = np.copysign(y, Rm[0, 2] - Rm[2, 0])
    z = np.copysign(z, Rm[1, 0] - Rm[0, 1])
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def _rigid(X: np.ndarray, Y: np.ndarray, m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The rotation and translation that best take points X to Y, weighted by mass (Kabsch): Y ~ X @ Rm.T + p."""
    w = m / m.sum()
    cx, cy = w @ X, w @ Y
    Hm = ((X - cx) * w[:, None]).T @ (Y - cy)
    U, _, Vt = np.linalg.svd(Hm)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
    Rm = Vt.T @ D @ U.T
    return Rm, cy - Rm @ cx


def _rotvec(Rm: np.ndarray) -> np.ndarray:
    """A rotation matrix as its axis times its angle."""
    a = np.arccos(np.clip((np.trace(Rm) - 1) / 2, -1, 1))
    if a < 1e-12:
        return np.zeros(3)
    return a / (2 * np.sin(a)) * np.array([Rm[2, 1] - Rm[1, 2], Rm[0, 2] - Rm[2, 0], Rm[1, 0] - Rm[0, 1]])


def _rot(v: np.ndarray) -> np.ndarray:
    a = float(np.linalg.norm(v))
    if a < 1e-15:
        return np.eye(3)
    k = v / a
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K


def ramped_play(spans: list[tuple[float, float]], ramp: float = 0.5):
    """The playback second of each physical time over stretches played at their own slowing. spans is a list of
    (physical length, how many times slower). Where the slowing changes, the rate eases over ramp seconds of
    playback on a log scale, always inside the faster stretch, so a slow motion stretch plays slowed all through
    and a 0.5 s fall in real time slows into a millisecond impact without a jump. Returns a function from physical
    time (from the first stretch's start) to playback seconds."""
    lens = [float(a) for a, _ in spans]
    rates = [1.0 / float(b) for _, b in spans]          # physical seconds per playback second
    # for each stretch, the ramp at its start and its end: (physical length it covers, playback it takes, lambda)
    head, tail = [None] * len(spans), [None] * len(spans)
    for i in range(len(spans) - 1):
        f, sl = (i, i + 1) if rates[i] > rates[i + 1] else (i + 1, i)
        rf, rs = rates[f], rates[sl]
        if rf <= rs * (1 + 1e-9):
            continue
        P = ramp
        lam = np.log(rf / rs) / P
        D = (rf - rs) / lam
        if D > 0.5 * lens[f]:                          # a short fast stretch eases over what it has
            D = 0.5 * lens[f]
            # solve lam for D: D = (rf - rs) P / ln(rf/rs) with P shortened
            P = D * np.log(rf / rs) / (rf - rs)
            lam = np.log(rf / rs) / P
        ease = (D, P, lam, rf, rs)
        if f == i:
            tail[i] = ease                              # fast then slow: the fast stretch's end eases down
        else:
            head[i + 1] = ease                          # slow then fast: the fast stretch's start eases up

    def one(i, x):
        """Playback seconds to physical x into stretch i."""
        L, r = lens[i], rates[i]
        p = 0.0
        h, t = head[i], tail[i]
        a = h[0] if h else 0.0
        b = L - (t[0] if t else 0.0)
        if h:
            D, P, lam, rf, rs = h
            y = min(x, D)
            p += np.log1p(lam * y / rs) / lam
            if x <= D:
                return p
        p += (min(x, b) - a) / r
        if x <= b or not t:
            return p
        D, P, lam, rf, rs = t
        y = min(x - b, D)
        return p - np.log1p(-min(lam * y / rf, 1 - 1e-12)) / lam

    edges = np.concatenate([[0.0], np.cumsum(lens)])
    starts = [0.0]
    for i in range(len(spans)):
        starts.append(starts[-1] + one(i, lens[i]))

    def clock(t):
        t = np.atleast_1d(np.asarray(t, float))
        out = np.empty_like(t)
        for j, x in enumerate(t):
            i = int(np.clip(np.searchsorted(edges, x, side="right") - 1, 0, len(spans) - 1))
            out[j] = starts[i] + one(i, min(max(x - edges[i], 0.0), lens[i]))
        return out

    return clock


def drop_motion(s, mesh, X: np.ndarray, model, hist, ev: dict, R: np.ndarray, H: float, v0: float, el_ex: float,
                rest_s: float = 0.0):
    """The drop as it happens at the asked height: the part falls from rest H above its first touch, hits the floor
    as the explicit solve found frame by frame (its travel, its turn, its elastic shape and its stress at every
    saved instant), then leaves the floor with the speed and spin it had at the end of the solve and flies up to the
    top of its bounce under gravity. The fall and the bounce are rigid body flight, exact without drag; the impact
    is the solve's own. The second landing is not solved, so the motion ends at the top of the bounce."""
    m = model.mass
    n = len(X)
    t0 = float(hist.t_contact[0])
    ft = np.asarray(hist.frames_t, float) - t0
    keep = ft >= -1e-12
    U = np.array([np.asarray(u, float) @ R.T for u, k in zip(hist.frames_u, keep) if k])
    ft = ft[keep]
    S = np.asarray(ev["S"], float)[keep]
    zone = ev.get("zone")
    if zone is not None and np.any(zone) and not np.all(zone):
        S = S.copy()
        S[:, np.asarray(zone, bool)] = np.nan
    # each solved instant as a rigid move about the part's center of mass and the elastic shape left over
    poses, elastic = [], []
    for u in U:
        Rm, p = _rigid(X, X + u, m)
        poses.append((Rm, p))
        elastic.append((X + u - p) @ Rm - X)          # in the part's own frame, so it turns with it
    elastic = np.array(elastic)
    tf = float(np.sqrt(2 * H / G))                    # the fall's time, s
    # the flight after the last solved instant: its center of mass velocity and spin, then gravity alone
    vL = np.asarray(hist.com_v[-1], float) @ R.T          # mm/s
    (R1, p1), (R2, p2) = poses[-2], poses[-1]
    dt_l = float(ft[-1] - ft[-2]) or 1e-9
    w = _rotvec(R2 @ R1.T) / dt_l                     # rad/s, world frame
    com = (m / m.sum()) @ X
    cL = R2 @ com + p2
    up = max(float(vL[2]), 0.0)
    t_up = up / G
    # spans for the clock: the fall in real time, the impact slowed so it plays over about 2.4 s, the bounce
    T_imp = float(ft[-1])
    slow = float(f"{max(1.0, 2.4 / max(T_imp, 1e-9)):.1g}")
    clock = ramped_play([(tf, 1.0), (T_imp, slow), (max(t_up, 1e-3), 1.0)])
    t_fall = np.unique(np.r_[np.linspace(0, tf, 28)[:-1], tf - np.geomspace(1e-5, 0.06, 10)[::-1]])
    t_fall = t_fall[t_fall < tf - 1e-12]
    t_imp = tf + ft
    # the viewer blends poses frame to frame, so the bounce keeps each frame's turn under 6 degrees of its spin
    T_up = max(t_up, 1e-3)
    step = min(0.03, np.radians(6.0) / max(float(np.linalg.norm(w)), 1e-9), T_up / 14)
    t_b = np.unique(np.r_[np.geomspace(2e-5, step, 8), np.arange(step, T_up, step), T_up])
    t_bounce = tf + T_imp + t_b[t_b <= T_up + 1e-12]
    tau = np.r_[t_fall, t_imp, t_bounce]
    M = len(tau)
    pos, quat = np.zeros((M, 3)), np.zeros((M, 4))
    fld = np.zeros((M, n))
    dfm = np.zeros((M, n, 3))
    i = 0
    for t in t_fall:                                    # free fall: rigid, unloaded
        pos[i] = [0.0, 0.0, 0.5 * G * (tf * tf - t * t)]
        quat[i] = [1, 0, 0, 0]
        i += 1
    for f in range(len(ft)):                            # the solved impact
        Rm, p = poses[f]
        pos[i], quat[i] = p, _quat(Rm)
        fld[i], dfm[i] = S[f], elastic[f]
        i += 1
    for t in t_bounce:                                  # the bounce: rigid flight with the last solved spin
        dt = t - tf - T_imp
        Rb = _rot(w * dt) @ R2
        c = cL + vL * dt + np.array([0, 0, -0.5 * G * dt * dt])
        pos[i], quat[i] = c - Rb @ com, _quat(Rb)
        # it keeps the shape and stress of its last solved instant: the solve does not follow its ringing in the air
        fld[i], dfm[i] = S[-1], elastic[-1]
        i += 1
    s.motion(tau, clock(tau) + rest_s, label="time", unit="s",
             note="The fall and the bounce are rigid flight under gravity; the impact is the explicit solve's own, "
                  "slowed down. After it leaves the floor the part keeps its last solved stress, since the solve "
                  "ends there.")
    mesh.track(pos=pos, quat=quat, field=fld, deform=dfm)
    peak = np.nanmax(np.where(np.isfinite(fld), fld, -np.inf), axis=1)
    return {"tau": tau, "peak": np.maximum(peak, 0.0), "slow": slow, "tf": tf, "t_up": t_up}


def write_scene(out: Path, model, hist, ev: dict, d, H: float, v0: float, floor, name: str, watch, gov: float,
                allow: float, verdict: str, headline: str, question: str = "", part: str = "",
                measure: str = "von Mises", target: float | None = None) -> Path:
    from ..viz import Scene

    R = rotation_to_down(d)
    X = model.X @ R.T
    shift = np.array([0.0, 0.0, -X[:, 2].min()])
    X = X + shift - np.array([*X[:, :2].mean(0), 0.0])
    f = int(ev["frame"])
    U = hist.frames_u[f].astype(float) @ R.T
    m = model.mass
    el = U - (U * m[:, None]).sum(0) / m.sum()
    t = hist.t - hist.t_contact[0]
    title = f"{(Path(part).stem or 'part').replace('_', ' ').replace('-', ' ').capitalize()} dropped on " \
            f"{floor.name.lower()}"
    s = Scene(title, domain="drop", question=question,
              source="The solved impact of the part on the floor by explicit time integration, at the instant of "
                     "peak stress, scaled to each drop height by the landing speed.")
    mesh = height_sweep(s, X, model.tris, ev["S"][f], el, H, allow, target=target,
                        g_peak=float(ev["g_watch"]), force_peak=float(np.max(hist.force)) if len(t) else 0.0,
                        node=int(ev["node"]), watch_at=X[np.asarray(watch)].mean(0), measure=measure,
                        floor_key=floor.key, landing=name, zone=ev.get("zone"))
    # the drop itself at the asked height: the fall, the solved impact and the bounce, on the motion's clock
    mo = drop_motion(s, mesh, X, model, hist, ev, R, H, v0, mesh.dfm[2] if mesh.dfm else 1.0)
    s.readout_motion("peak stress", mo["peak"])
    tc = mo["tau"] - mo["tf"]                       # time from first touch
    inside = (tc >= t[0]) & (tc <= t[-1])
    have = {r["name"] for r in s.readouts}
    if len(t) and "acc_f" in ev and "accelerometer" in have:
        s.readout_motion("accelerometer", np.where(inside, np.interp(tc, t, np.asarray(ev["acc_f"]) / G), 0.0))
    if len(t) and "floor force" in have:
        s.readout_motion("floor force", np.where(inside, np.interp(tc, t, hist.force), 0.0))
    tone = {"holds": "pass", "marginal": "warn", "fails": "fail"}.get(verdict, "info")
    s.verdict({"holds": "SURVIVES", "marginal": "MARGINAL", "fails": "FAILS"}.get(verdict, verdict.upper()),
              headline.split(". ")[0] + ".", tone)
    span = float(np.ptp(X, axis=0).max())
    s.camera(target=(0.0, 0.0, 0.5 * float(X[:, 2].max())), direction=(1.0, -1.4, 0.6), distance=3.4 * span, fov=30,
             presets=[{"name": "contact", "target": X[int(ev["node"])], "direction": (1.0, -1.0, 0.35),
                       "distance": 1.4 * span}])
    return s.save(out)


__all__ = ["write_scene", "rotation_to_down"]
