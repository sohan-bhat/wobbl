"""A crack growth run as a 3D model you set the load of: the crack at the end of its solved path, colored by K.

The stress intensity is linear in the load, so every front's K at a scaled load is exactly the solved one scaled.
The lives come from the run's own surrogate and sampled parts at each load (the playback's sweep), so the slider
reads the life of the one part in a hundred and the share failing before the target as the load changes. The
color's limit is the K at which one part in ten fails before the target, and its target the K at which one in a
hundred does, the run's own lines for fails and marginal.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from ..scene import Scene
from .common import read

FORMAT = "wobbl-crack-playback"
# the start distance, in half diagonals of the part, from the crack region: near enough to see the crack
CRACK_VIEW = 3.6


def _resample(p: np.ndarray, n: int) -> np.ndarray:
    d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
    if d[-1] <= 0:
        return np.repeat(p[:1], n, 0)
    s = np.linspace(0, d[-1], n)
    return np.c_[[np.interp(s, d, p[:, k]) for k in range(3)]].T


def _crossing(f: np.ndarray, life: np.ndarray, target: float) -> float | None:
    """The load factor at which a life falls to the target, by straight lines in log life against log load, carried
    past the ends of the table when it does not cross inside it."""
    ok = np.isfinite(life) & (life > 0)
    if ok.sum() < 2:
        return None
    lf, ll = np.log(f[ok]), np.log(life[ok])
    y = np.log(target)
    for k in range(len(lf) - 1):
        if (ll[k] - y) * (ll[k + 1] - y) <= 0 and ll[k] != ll[k + 1]:
            return float(np.exp(lf[k] + (y - ll[k]) * (lf[k + 1] - lf[k]) / (ll[k + 1] - ll[k])))
    a, b = (0, 1) if y > ll[0] else (-2, -1)
    if ll[a] == ll[b]:
        return None
    return float(np.exp(lf[a] + (y - ll[a]) * (lf[b] - lf[a]) / (ll[b] - ll[a])))


def _num(v) -> np.ndarray:
    return np.array([np.nan if x is None else x for x in v], float)


def build(out: Path, pb: dict, run=None, outcome: dict | None = None) -> Path | None:
    """The scene from a crack playback. With the run's uncracked mesh, loads and supports (``run`` as the strength
    adapter reads it) the loads and supports are drawn at their real faces."""
    if pb.get("manifest", {}).get("format") != FORMAT:
        return None
    o = outcome or read(out, "outcome.json") or {}
    part, crack = pb["part"], pb["crack"]
    X = np.asarray(part["vertices"], float)
    T = np.asarray(part["triangles"], int).reshape(-1, 3)
    steps = [st for st in crack["steps"] if st["fronts"]]
    if not steps:
        return None
    sw = pb.get("sweep") or {}
    law = (o.get("values") or {}).get("law") or {}
    kc = float(sw.get("kc") or law.get("kc") or 0)
    load0 = float(sw.get("load_N") or 0)
    if not load0 and run is not None:
        load0 = sum(float(ld.magnitude) for ld in run.study.loads if ld.kind in ("force", "remote", "bolt"))
    f = np.asarray(sw["factor"], float) if sw.get("factor") else np.unique(np.r_[np.linspace(0.5, 1.5, 41), 1.0])
    x, unit = (f * load0, "N") if load0 > 0 else (100 * f, "% of the load")
    s = Scene("The crack at the end of its path", domain="crack", question=o.get("question", "") or
              pb["summary"].get("headline", ""), source="Every front is a solved crack. K scales exactly with the "
                                                        "load, and the lives are the run's own at each load.")
    s.timeline(x, unit=unit, label="load", between="interpolated", kind="input", design=float(x[np.argmin(
        np.abs(f - 1))]))
    s.mesh(X, T, name="part", color="#9aa0a6", opacity=0.35)
    if part.get("edges"):
        ev = np.asarray(part["edge_vertices"], float)
        s.lines([ev[list(e)] for e in part["edges"]], color="#c3c7cd", width=1, name="edges")
    n = 24
    rings, kvals = [], []
    starts = [np.asarray(fl, float) for fl in crack.get("starting_flaw", []) if len(fl) > 1]
    for st in steps:
        fr = max(st["fronts"], key=lambda q: len(q["points"]))
        pts = np.asarray(fr["points"], float)
        k = np.asarray(fr["k"], float)
        k1 = np.sqrt((k[:, :2] ** 2).sum(1)) if k.ndim == 2 else k
        rings.append(_resample(pts, n))
        kvals.append(np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(k1)), k1))
    origin = np.asarray(crack.get("origin", rings[0].mean(0)), float)
    first = _resample(starts[0], n) if starts else np.repeat(origin[None], n, 0)
    R = np.vstack([first] + rings)
    K = np.r_[kvals[0], np.concatenate(kvals)]
    kmax = np.array([st["k_max"] for st in steps], float)
    tri = []
    for i in range(len(rings)):
        for j in range(n - 1):
            a, b = i * n + j, (i + 1) * n + j
            tri += [(a, b, b + 1), (a, b + 1, a + 1)]
    face = s.mesh(R, np.array(tri), name="crack face", color="#ff4d4f", max_vertices=len(R) + 1, smooth=False)
    target = sw.get("target") or (pb.get("life") or {}).get("target_cycles")
    b1, b10 = _num(sw.get("B1", [])), _num(sw.get("B10", []))
    kpk = float(K.max())
    lim, tgt, word = (kc or None), None, "toughness"
    if target and len(b10) == len(f):
        s10, s1 = _crossing(f, b10, float(target)), _crossing(f, b1, float(target))
        if s10:
            lim, word = kpk * s10, "one in ten fails before the target"
            tgt = kpk * s1 if s1 and s1 < s10 else None
    face.color_by("stress intensity K", K, scale=f, unit="MPa sqrt(m)", colormap="inferno",
                  range=[0, max(lim or 0, kpk * float(f.max()))], limit=lim, limit_label=word)
    if tgt:
        s.fields[-1]["target"] = float(tgt)
    s.lines(rings, color="#ffd400", width=2.5, name="fronts")
    # the view starts on the crack and keeps most of the part in frame, rather than the whole part at its far fit
    r = 0.5 * float(np.linalg.norm(np.ptp(X, axis=0)))
    s.camera(target=rings[-1].mean(0), distance=CRACK_VIEW * r)
    big = 1e9                                   # a crack that stops below threshold never fails: shown as 1e9
    if len(b1) == len(f):
        s.readout("B1 life", np.nan_to_num(b1, nan=big), unit="cycles", digits=3)
        s.readout("B10 life", np.nan_to_num(b10, nan=big), unit="cycles", digits=3)
        if target and sw.get("pof"):
            allowed = 100 * float(sw.get("max_pof") or 0.01)
            s.readout(f"failing before {float(target):,.0f}", 100 * np.nan_to_num(_num(sw["pof"])), unit="%",
                      digits=3, limit=allowed)
        s.readout("typical life", np.nan_to_num(_num(sw["nominal"]), nan=big), unit="cycles", digits=3)
        if np.isnan(b1).any() or np.isnan(_num(sw["nominal"])).any():
            s.note("A life of 1e9 cycles means the crack stops below its growth threshold at that load.")
    s.readout("largest K", float(kmax[-1]) * f, unit="MPa sqrt(m)", digits=3, limit=kc or None)
    s.readout("crack size", np.full(len(f), float(steps[-1]["size_mm"])), unit="mm", digits=3)
    if lim and word != "toughness":
        s.event(float(np.interp(lim / kpk, f, x)), "one in ten fails early", f"before {float(target):,.0f} cycles",
                kind="fail")
    if tgt:
        s.event(float(np.interp(tgt / kpk, f, x)), "one in a hundred fails early",
                f"before {float(target):,.0f} cycles", kind="warn")
    s.label("largest K {largest K}", rings[-1][int(np.argmax(kvals[-1]))], kind="peak")
    if run is not None:
        _setting(s, run, X, o, f)
        s.readout("load", x, unit=unit, digits=4)      # the load arrows read "{load}"
    word_v = pb["summary"].get("verdict", "")
    from .common import WORDS, plain
    w, tone = WORDS.get(word_v, (word_v.upper() or "DONE", "info"))
    s.verdict(w, plain(re.sub(r"\s+at\s*\([^)]*\)", "", o.get("headline") or pb["summary"].get("headline", ""))),
              tone)
    return s.save(out)


def _setting(s: Scene, run, X: np.ndarray, o: dict, f: np.ndarray) -> None:
    """Supports and loads on the part's own faces, as a strength scene draws them, with the arrows scaled to the
    slider's load."""
    from .strength import load_props, supports

    size = float(np.ptp(X, axis=0).max())
    words = set(re.findall(r"[a-z]+", (o.get("question") or "").lower()))
    supports(s, run, run.mesh.X, size, words)
    load_props(s, run, run.mesh.X, size, f, words)
