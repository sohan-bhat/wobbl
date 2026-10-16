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
    height_sweep(s, X, model.tris, ev["S"][f], el, H, allow, target=target,
                 g_peak=float(ev["g_watch"]), force_peak=float(np.max(hist.force)) if len(t) else 0.0,
                 node=int(ev["node"]), watch_at=X[np.asarray(watch)].mean(0), measure=measure,
                 floor_key=floor.key, landing=name, zone=ev.get("zone"))
    tone = {"holds": "pass", "marginal": "warn", "fails": "fail"}.get(verdict, "info")
    s.verdict({"holds": "SURVIVES", "marginal": "MARGINAL", "fails": "FAILS"}.get(verdict, verdict.upper()),
              headline.split(". ")[0] + ".", tone)
    span = float(np.ptp(X, axis=0).max())
    s.camera(target=(0.0, 0.0, 0.5 * float(X[:, 2].max())), direction=(1.0, -1.4, 0.6), distance=3.4 * span, fov=30,
             presets=[{"name": "contact", "target": X[int(ev["node"])], "direction": (1.0, -1.0, 0.35),
                       "distance": 1.4 * span}])
    return s.save(out)


__all__ = ["write_scene", "rotation_to_down"]
