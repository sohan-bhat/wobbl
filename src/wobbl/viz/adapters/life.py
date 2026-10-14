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
from ..scene import FILE, Scene, array, load
from .common import plain, read, verdict

G = 9810.0


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
    X, T = mesh_of(pb)
    amp = float(pb.get("amplitude_mm", 0.06 * np.linalg.norm(np.ptp(X, 0))))
    modes = pb.get("modes", [])
    if not modes:
        return None
    K = 24                                    # frames a cycle, two cycles a mode
    n = 2 * K * len(modes)
    t = np.arange(n, dtype=float)
    s = Scene("Mode shapes", domain="vibration", question=o.get("question", ""), source=pb.get("about", ""))
    s.timeline(t / K, unit="cycles", label="playback", seconds=3.0 * len(modes), between="interpolated")
    freq = np.zeros(n)
    for i, m in enumerate(modes):
        phi = arr(m["shape"])
        win = (t >= i * 2 * K) & (t < (i + 1) * 2 * K)
        sc = np.where(win, np.sin(2 * np.pi * (t - i * 2 * K) / K), 0.0)
        mesh = s.mesh(X, T, name=f"mode {m['mode']}", max_vertices=6000)
        mesh.color_by("relative motion", np.linalg.norm(phi, axis=1), scale=np.abs(sc), colormap="viridis",
                      range=(0, 1))
        mesh.deform(phi * amp, scale=sc)
        mesh.show(i * 2, (i + 1) * 2 - 1 / K)
        freq[win] = m["frequency_Hz"]
        bad = m.get("status") in ("resonant", "inside")
        s.event(i * 2, f"mode {m['mode']} at {m['frequency_Hz']:.4g} Hz", m.get("status", ""),
                kind="fail" if bad else "info")
        s.label(f"mode {m['mode']}, {m['frequency_Hz']:.4g} Hz, {m.get('status', '')}".rstrip(", "),
                X[int(np.argmax(np.linalg.norm(phi, axis=1)))] + [0, 0, amp], t0=i * 2, t1=(i + 1) * 2 - 1 / K,
                kind="fail" if bad else "note")
    s.notes = [n_ for n_ in s.notes if not n_.startswith("Deformation")]
    s.note(f"Each mode swings to {amp:.3g} mm and is slowed down so the eye can follow. The real motion is far "
           "smaller and faster.")
    s.readout("frequency", freq, unit="Hz", digits=4)
    verdict(s, o)
    return s.save(out)


def buckling(out, pb, o):
    X, T = mesh_of(pb)
    phi = arr(pb["shape"])
    fr = pb["frames"]
    lam = np.asarray(fr["load_factor"], float)
    bow = np.asarray(fr["bow_mm"], float)
    L = float(pb.get("length_mm") or np.ptp(X, 0).max())
    ex = float(f"{max(0.06 * L / max(bow.max(), 1e-9), 1.0):.2g}")
    s = Scene("Buckling as the load grows", domain="buckling", question=o.get("question", ""),
              source=pb.get("about", ""))
    s.timeline(lam, unit="x load", label="load factor", seconds=7, between="interpolated")
    m = s.mesh(X, T, name="part", max_vertices=8000)
    m.color_by("bow", np.linalg.norm(phi, axis=1), scale=bow, unit="mm", colormap="viridis")
    m.deform(phi, scale=bow, exaggerate=ex)
    s.readout("load factor", lam, unit="x", digits=3)
    s.readout("bow", bow, unit="mm", digits=3)
    df = pb.get("design_factor")
    if df and df <= lam[-1]:
        s.event(df, "design factor", "with the knockdown for imperfections", kind="info")
    s.event(lam[-1], "nearing the buckling load", f"buckles at {pb.get('buckling_factor', 0):.3g} times the load",
            kind="fail")
    lo, hi = X.min(0), X.max(0)
    ax = int(np.argmax(hi - lo))
    for end in (lo[ax], hi[ax]):
        c = (lo + hi) / 2
        c[ax] = end
        n = np.zeros(3)
        n[ax] = 1 if end == lo[ax] else -1
        P.slab(s, c, n, L * 0.12, L * 0.12, L * 0.03, color=P.STEEL, name="end plate")
    verdict(s, o)
    return s.save(out)


def drop(out, pb, o):
    X, T = mesh_of(pb)
    f = pb["fall"]
    d = np.asarray(f["down"], float)
    u, v, w = P.basis(-d)
    R = np.vstack([u, v, w])                  # turns the run's down onto -z
    Xr = X @ R.T
    floor = float(f["floor_level_mm"])
    H = float(f["height_mm"])
    fr = pb["frames"]
    t_hit = np.asarray(fr["time_s"], float)
    trans = arr(fr["translation"]) @ R.T
    coeff, shapes = arr(fr["coeff"]), arr(pb["shapes"])
    el = np.einsum("fk,kvc->fvc", coeff, shapes) @ R.T
    stress = arr(fr["stress"])
    t_fall = np.sqrt(2 * H / G)
    k = 30
    tf = np.linspace(-t_fall, 0, k, endpoint=False)
    zf = 0.5 * G * tf ** 2                    # height above first touch
    lowest = float((Xr[:, 2]).min())
    base = Xr - [0, 0, lowest]                # first touch at z = 0
    n = len(base)
    U = np.concatenate([np.repeat((np.c_[np.zeros(k), np.zeros(k), zf])[:, None, :], n, 1),
                        trans[:, None, :] + el], 0)
    ex = 1.0
    emax = float(np.abs(el).max())
    span = float(np.ptp(base, 0).max())
    if emax > 0:
        ex = float(f"{max(0.04 * span / emax, 1.0):.2g}")
        U[k:] = trans[:, None, :] + el * ex
    S = np.concatenate([np.zeros((k, n)), stress], 0)
    t = np.concatenate([tf, t_hit])
    s = Scene(f"Drop from {H / 1000:.3g} m onto {f.get('floor_name', 'the floor')}", domain="drop",
              question=o.get("question", ""), source="Free fall from the height, then the run's own frames from "
                                                     "first touch.")
    s.timeline(t * 1000, unit="ms", label="time", seconds=6, between="interpolated")
    m = s.mesh(base, T, name="part", max_vertices=8000)
    gov = pb.get("governing", {})
    m.color_by("stress", S, unit="MPa", range=(0, max(float(stress.max()), 1e-9)))
    m.deform(U, exaggerate=1.0)
    if ex > 1:
        s.note(f"The part's own bending is drawn {ex:.3g} times larger so it shows, while its fall and rebound "
               "are true to scale.")
    s.ground(0.0, size=span * 8, grid=float(10 ** np.floor(np.log10(span / 2))), color="#2a2c30")
    s.readout("height", np.r_[zf, np.zeros(len(t_hit))], unit="mm", digits=3)
    s.readout("peak stress", S.max(1), unit="MPa", digits=3)
    s.event(0.0, "impact", f"at {f['impact_speed_mm_s'] / 1000:.3g} m/s", kind="fail" if o.get("verdict") == "fails"
            else "info")
    if gov:
        at = (np.asarray(gov["at_mm"], float) @ R.T) - [0, 0, lowest]
        s.label("peak {peak stress}", at, t0=float(gov.get("time_s", 0)) * 1000, kind="peak")
    verdict(s, o)
    s.camera(target=[0, 0, span * 0.6], direction=(0.6, -1, 0.35), distance=span * 3.2)
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


def fatigue(out, pb, o):
    """Damage over the duty cycle, drawn on the strength run it came from when that run has a scene."""
    st = read(out, "study.json") or {}
    src = (Path(out) / st.get("source", "")).resolve() if st.get("source") else None
    if src is None or not (src / FILE).exists():
        return None
    h, blob = load(src / FILE)
    part = next((ob for ob in h["objects"] if ob["kind"] == "mesh" and ob.get("name") == "part"), None)
    if part is None:
        return None
    X, T = array(h, blob, part["pos"]), array(h, blob, part["tri"])
    fr = pb["frames"]
    cyc, dmg = np.asarray(fr["cycles"], float), np.asarray(fr["damage"], float)
    s = Scene("Fatigue damage over the duty cycle", domain="fatigue", question=o.get("question", ""),
              source=pb.get("about", ""))
    s.timeline(cyc, unit="cycles", label="cycles", seconds=8, between="interpolated")
    m = s.mesh(X, T, name="part", max_vertices=len(X) + 1)
    if part.get("field"):
        f = h["fields"][part["field"]["f"]]
        m.color_by(f"{f['name']} at full load", array(h, blob, part["field"]["values"]), unit=f.get("unit", ""),
                   range=f.get("range"), limit=f.get("limit"), limit_label=f.get("limit_label", ""))
    s.readout("damage", dmg, digits=3, limit=1.0)
    s.readout("cycles", cyc, digits=3)
    where = pb.get("where_mm")
    if where:
        s.label("damage {damage}", where, kind="peak")
    hit = np.nonzero(dmg >= 1)[0]
    if len(hit):
        s.event(float(cyc[hit[0]]), "a crack starts", "damage reaches one", kind="fail")
    s.note("The color is the stress of the strength run this fatigue study reads, and the damage at its critical "
           "spot builds up by Miner's rule.")
    verdict(s, o, plain(o.get("headline", "")))
    if h.get("camera"):
        c = h["camera"]
        s.camera(target=c.get("target"), direction=c.get("dir"), distance=c.get("dist"), fov=c.get("fov", 30))
    return s.save(out)
