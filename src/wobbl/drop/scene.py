"""A drop run as a 3D scene: the part falls, lands on its worst side and the stress flashes through it.

Frames before the impact are the free fall from the drop height, exact for a body without drag. Frames after it
are the solved motion: the part's travel plus its elastic shape, drawn larger so it shows, with the stress the
study compares with the strength. The playback spends most of its time on the impact, which lasts about a
millisecond, so the timeline's labels carry the real times.
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


def write_scene(out: Path, model, hist, ev: dict, d, H: float, v0: float, floor, name: str, watch, gov: float,
                allow: float, verdict: str, headline: str, question: str = "", part: str = "",
                measure: str = "von Mises", fall_frames: int = 20, impact_frames: int = 60) -> Path:
    from ..viz import Scene

    R = rotation_to_down(d)
    X = model.X @ R.T
    shift = np.array([0.0, 0.0, -X[:, 2].min()])
    X = X + shift - np.array([*X[:, :2].mean(0), 0.0])
    t_fall = float(np.sqrt(2 * H / G))
    tf = np.linspace(-t_fall, 0.0, fall_frames, endpoint=False)
    rise = v0 * (-tf) - 0.5 * G * tf ** 2                     # height above first touch during the fall
    pick = np.unique(np.linspace(0, len(hist.frames_t) - 1, min(impact_frames, len(hist.frames_t))).round()
                     .astype(int))
    pick = np.union1d(pick, [int(ev["frame"])])                # always the frame of the peak
    ti = np.asarray(hist.frames_t)[pick] - hist.t_contact[0]
    t = np.concatenate([tf, ti])
    U = np.array([hist.frames_u[i].astype(float) @ R.T for i in pick])     # (fi,n,3) world frame
    m = model.mass
    trans = (U * m[None, :, None]).sum(1) / m.sum()
    elastic = U - trans[:, None, :]
    pos = np.concatenate([np.column_stack([np.zeros(fall_frames), np.zeros(fall_frames), rise]), trans])
    el = np.concatenate([np.zeros((fall_frames, *elastic.shape[1:])), elastic])
    S = np.concatenate([np.zeros((fall_frames, len(X))), ev["S"][pick]])
    title = f"{(Path(part).stem or 'part').replace('_', ' ').replace('-', ' ').capitalize()} dropped {H / 1000:g} m"
    s = Scene(title, domain="drop", question=question,
              source="Free fall before first touch, then the solved impact: explicit time integration of the part "
                     "on the floor. The impact lasts about a millisecond and is slowed down to be seen.")
    s.timeline(t, unit="s", label="time", seconds=8, between="interpolated",
               note="The fall plays fast and the impact slowly; the times are real.")
    mesh = s.mesh(X, model.tris, name="part", keep=[int(ev["node"])], max_vertices=12000)
    mesh.color_by(measure, S, unit="MPa",
                  range=(0.0, max(allow, gov)), limit=allow, limit_label="strength")
    span = float(np.ptp(X, axis=0).max())
    emax = float(np.abs(elastic).max()) if elastic.size else 0.0
    ex = float(f"{max(1.0, 0.03 * span / emax):.2g}") if emax > 0 else 1.0
    mesh.deform(el, exaggerate=ex)
    mesh.move(pos)
    s.ground(0.0, size=4 * span, grid=span / 4, color=FLOOR_COLOR.get(floor.key, "#8a8d91"))
    acc = np.interp(ti, hist.t - hist.t_contact[0], ev["acc_f"]) / G
    force = np.interp(ti, hist.t - hist.t_contact[0], hist.force)
    s.readout("accelerometer", np.concatenate([np.zeros(fall_frames), acc]), unit="g", digits=4)
    s.readout("floor force", np.concatenate([np.zeros(fall_frames), force]), unit="N", digits=4)
    s.readout("peak stress", S.max(1), unit="MPa", limit=allow)
    vi = hist.com_v[np.clip(np.searchsorted(hist.t, np.asarray(hist.frames_t)[pick]), 0, len(hist.t) - 1)]
    s.readout("speed", np.concatenate([G * (tf + t_fall), np.linalg.norm(vi, axis=1)]) / 1000, unit="m/s", digits=3)
    s.event(0.0, "impact", f"lands {name.replace(' down', '')} first at {v0 / 1000:.3g} m/s", kind="info")
    t_peak = float(hist.frames_t[ev["frame"]] - hist.t_contact[0])
    s.event(t_peak, "peak stress", f"{gov:.3g} MPa", kind="fail" if gov > allow else "info")
    t_off = hist.t_contact[1] - hist.t_contact[0]
    if t_off > 0:
        s.event(float(t_off), "leaves the floor", f"after {1e3 * t_off:.3g} ms", kind="info")
    g = X[int(ev["node"])]
    s.label("peak {peak stress}", g, kind="peak", t0=0.0)
    w = X[np.asarray(watch)].mean(0)
    s.label("accelerometer {accelerometer}", w, kind="note", t0=0.0)
    tone = {"holds": "pass", "marginal": "warn", "fails": "fail"}.get(verdict, "info")
    s.verdict({"holds": "SURVIVES", "marginal": "MARGINAL", "fails": "FAILS"}.get(verdict, verdict.upper()),
              headline.split(". ")[0] + ".", tone)
    s.camera(target=(0.0, 0.0, 0.3 * span), direction=(1.0, -1.4, 0.6), distance=2.6 * span, fov=30)
    return s.save(out)


__all__ = ["write_scene", "rotation_to_down"]
