"""Scenes for the runs that write playback.json in the typed array encoding of life/playback.py: vibration modes,
buckling, drops, plastic runs (any mesh with framed displacement and fields) and fatigue.

Every vertex is an original mesh node, so the fields sit exactly where the run solved them. Mode shapes and
buckling shapes are normalized, so their motion is drawn larger and slower than life, and the view says so.
"""
from __future__ import annotations

import base64
from pathlib import Path

import numpy as np

from .. import props as P
from ..scene import Scene
from .common import plain, read, verdict



def arr(d: dict) -> np.ndarray:
    a = np.frombuffer(base64.b64decode(d["data"]), dtype="<" + np.dtype(d["dtype"]).str[1:]).reshape(d["shape"])
    return d.get("offset", 0) + np.asarray(d.get("scale", 1.0)) * a.astype(float)


def mesh_of(pb: dict) -> tuple[np.ndarray, np.ndarray]:
    return arr(pb["mesh"]["vertices"]), arr(pb["mesh"]["triangles"]).astype(np.int64)


def build(out: Path, pb: dict) -> Path | None:
    kind = pb.get("kind", "")
    o = read(out, "outcome.json") or {}
    fn = {"vibration": vibration, "buckling": buckling, "drop": drop, "fatigue": fatigue}.get(kind)
    if fn is None and "mesh" in pb and isinstance(pb.get("frames"), list) and pb["frames"] and \
            "displacement" in pb["frames"][0]:
        fn = framed
    return fn(out, pb, o) if fn else None


def vibration(out, pb, o):
    """The steady response as the excitation speeds up through its range, from the stored modes (see
    wobbl.life.scene.speed_sweep)."""
    from ...life.scene import speed_sweep

    X, T = mesh_of(pb)
    modes = pb.get("modes", [])
    if not modes:
        return None
    v = o.get("values", {})
    bands = [(float(b["lo_Hz"]), float(b["hi_Hz"]), b.get("label", "excitation")) for b in pb.get("excitations", [])]
    s = Scene(o.get("question") or "Vibration", domain="vibration", question=o.get("question", ""),
              source="The run's mode shapes, each driven at the excitation's frequency with its damping.")
    speed_sweep(s, X, T, [arr(m["shape"]) for m in modes], [m["frequency_Hz"] for m in modes],
                [m["mode"] for m in modes], bands, zeta=float(v.get("damping") or 0.03),
                margin=float(v.get("margin") or 0.2),
                quiet=[m["mode"] for m in modes if m.get("status") == "barely excited"])
    verdict(s, o)
    return s.save(out)


def buckling(out, pb, o):
    """The part under each load up to 95% of its buckling load, from the stored shape (see
    wobbl.life.scene.load_sweep). Without a stored stress the part is colored by its bow."""
    from ...life.scene import load_sweep

    X, T = mesh_of(pb)
    fr = pb["frames"]
    lam = float(pb.get("buckling_factor") or 0)
    if lam <= 0:
        return None
    v = o.get("values", {})
    e0 = float(fr["bow_mm"][0])
    yf = pb.get("yield_factor")
    s = Scene(o.get("question") or "Buckling", domain="buckling", question=o.get("question", ""),
              source="The first buckling shape from the run, grown from a starting crookedness with the load.")
    load_sweep(s, X, T, arr(pb["shape"]), lam, float(pb.get("design_factor") or lam),
               float(v.get("applied_load_N") or 1000.0), e0,
               measure=arr(pb["stress_MPa"]) if "stress_MPa" in pb else None,
               yield_factor=float(yf) if yf else float("inf"),
               target=float(v["target_sf"]) if v.get("target_sf") else None)
    verdict(s, o)
    return s.save(out)


def drop(out, pb, o):
    """The part on the floor at its peak stress, scaled to each drop height by the landing speed (see
    wobbl.drop.scene). The contact patch is the part's stress above the governing peak, which the run leaves out."""
    from ...drop.scene import height_sweep

    X, T = mesh_of(pb)
    f = pb["fall"]
    d = np.asarray(f["down"], float)
    u, v, w = P.basis(-d)
    R = np.vstack([u, v, w])                  # turns the run's down onto -z
    Xr = X @ R.T
    base = Xr - [*Xr[:, :2].mean(0), Xr[:, 2].min()]
    H = float(f["height_mm"])
    fr = pb["frames"]
    gov = pb.get("governing", {})
    t_hit = np.asarray(fr["time_s"], float)
    stress = arr(fr["stress"])
    k = int(np.argmin(np.abs(t_hit - float(gov.get("time_s", t_hit[int(np.argmax(stress.max(1)))])))))
    el = np.einsum("k,kvc->vc", arr(fr["coeff"])[k], arr(pb["shapes"])) @ R.T
    S = stress[k]
    peak = float(gov.get("stress_MPa") or S.max())
    zone = S > peak * 1.02
    v_ = o.get("values", {})
    allow = float(v_.get("allowable_MPa") or 0) or peak
    target = next((float(m[1]) for m in o.get("metrics", []) if m[0] == "target"
                   and str(m[1]).replace(".", "", 1).isdigit()), None)
    at = None
    if gov.get("at_mm"):
        at_r = np.asarray(gov["at_mm"], float) @ R.T - [*Xr[:, :2].mean(0), Xr[:, 2].min()]
        at = int(np.argmin(np.linalg.norm(base - at_r, axis=1)))
    hist = pb.get("history", {})
    s = Scene(f"Drop onto {f.get('floor_name', 'the floor').lower()}", domain="drop", question=o.get("question", ""),
              source="The run's impact at the instant of peak stress, scaled to each drop height by the landing "
                     "speed.")
    watch = pb.get("watch_mm")
    height_sweep(s, base, T, S, el, H, allow, target=target, g_peak=max(hist.get("g_watch", [0]) or [0]),
                 force_peak=max(hist.get("force_N", [0]) or [0]), node=at,
                 watch_at=None if watch is None else np.asarray(watch, float) @ R.T - [*Xr[:, :2].mean(0),
                                                                                      Xr[:, 2].min()],
                 floor_key=f.get("floor", ""), landing=f.get("landing", ""), zone=zone)
    verdict(s, o)
    span = float(np.ptp(base, 0).max())
    s.camera(target=[0, 0, 0.5 * float(base[:, 2].max())], direction=(1.0, -1.4, 0.6), distance=span * 3.4)
    return s.save(out)


def framed(out, pb, o):
    X, T = mesh_of(pb)
    frames = pb["frames"]
    t = np.asarray([f["t"] for f in frames], float)
    U = np.stack([arr(f["displacement"]) for f in frames])
    span = float(np.ptp(X, 0).max())
    umax = float(np.linalg.norm(U, axis=2).max())
    ex = float(f"{max(0.06 * span / umax, 1.0):.2g}") if umax > 0 else 1.0
    title = pb.get("title") or o.get("domain", "run").replace("-", " ").capitalize()
    s = Scene(title, domain=o.get("domain", pb.get("kind", "")), question=o.get("question", ""),
              source=pb.get("about", "Each frame is a saved increment of the run."))
    s.timeline(t, unit="", label=pb.get("time_label", "step"), seconds=max(4, min(len(t), 10)), between="interpolated")
    m = s.mesh(X, T, name="part", max_vertices=8000)
    fields = pb.get("fields", {})
    name = next((k for k in fields if k in frames[0]), None)
    if name:
        spec = fields[name]
        m.color_by(name.replace("_", " "), np.stack([arr(f[name]) for f in frames]), unit=spec.get("unit", ""),
                   range=(float(spec["min"]), float(spec["max"])), limit=spec.get("limit"),
                   limit_label=spec.get("limit_label", ""))
    m.deform(U, exaggerate=ex)
    keys = [k for k, v in frames[0].get("values", {}).items() if isinstance(v, (int, float))][:4]
    for k in keys:
        s.readout(k.replace("_", " "), [f["values"].get(k, np.nan) for f in frames], digits=3)
    for f in frames:
        if f.get("label"):
            s.event(f["t"], f["label"])
    verdict(s, o)
    return s.save(out)


def _source(out: Path, source: str) -> Path | None:
    """The strength run a fatigue study read, wherever its path was written from."""
    p = Path(source)
    for c in [p, Path(out) / p, *[q / p for q in Path(out).resolve().parents]]:
        if c.is_dir():
            return c.resolve()
    return None


def fatigue(out, pb, o):
    """The damage over the required duty as the duty's loads are scaled, on the strength run it came from (see
    wobbl.life.scene.duty_sweep). A run from before the stored sweep reads its S-N slope instead, which leaves
    out the knee and the mean stress correction's own curve."""
    from ...life.scene import duty_sweep, source_surface

    st = read(out, "study.json") or {}
    src = _source(out, st.get("source", "")) if st.get("source") else None
    v = o.get("values", {})
    where = pb.get("where_mm") or v.get("xyz")
    got = source_surface(src, where) if src else None
    if got is None:
        return None
    X, T, ratio, cam = got
    sw = pb.get("sweep")
    sf = float(v.get("sf") or pb.get("safety_factor") or float("inf"))
    if not sw:
        b = float((v.get("curve") or {}).get("b") or -0.1)
        d0 = float(v.get("damage") or np.asarray(pb["frames"]["damage"], float)[-1])
        g = np.linspace(0.0, 2.0 * max(1.5, 1.3 * sf) if np.isfinite(sf) else 3.0, 81)[1:]
        sw = {"stress_scale": g.tolist(), "damage": (d0 * g ** (-1.0 / b)).tolist(),
              "required": v.get("required", pb.get("required", 1.0)), "unit": "repeats"}
    s = Scene(o.get("question") or "Fatigue damage", domain="fatigue", question=o.get("question", ""),
              source="The strength run's stress scaled with the duty's loads, and the damage at the judged peak "
                     "by Miner's rule.")
    duty_sweep(s, X, T, ratio, sw, sf=sf, target=float(v["target_sf"]) if v.get("target_sf") else sw.get("target"),
               required=float(sw.get("required") or 1.0), unit=sw.get("unit", "repeats"), where=where)
    verdict(s, o, plain(o.get("headline", "")))
    if cam:
        s.camera(target=cam.get("target"), direction=cam.get("dir"), distance=cam.get("dist"), fov=cam.get("fov", 30))
    return s.save(out)
