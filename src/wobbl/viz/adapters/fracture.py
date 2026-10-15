"""A crack growth run as a scene: the crack face spreading through the part front by solved front, colored by K.

The fronts are the run's own, each at the cycle count when the nominal part's crack reached it. Between two solved
fronts the face is blended, and the view says so.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from ..scene import Scene
from .common import read

FORMAT = "wobbl-crack-playback"


def _resample(p: np.ndarray, n: int) -> np.ndarray:
    d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
    if d[-1] <= 0:
        return np.repeat(p[:1], n, 0)
    s = np.linspace(0, d[-1], n)
    return np.c_[[np.interp(s, d, p[:, k]) for k in range(3)]].T


def _timeline(pb: dict, solved: np.ndarray) -> np.ndarray:
    """Cycle counts the frames sit at: the nominal growth curve from the start to the last of it when the run has
    one, so the crack size ticks smoothly, else just the solved fronts."""
    g = pb.get("growth", {})
    gc = np.array([np.nan if v is None else v for v in g.get("cycles", [])], float)
    gc = gc[np.isfinite(gc)]
    if len(gc) > 1 and gc.max() > gc.min():
        t = np.unique(np.r_[np.linspace(gc.min(), gc.max(), 90), solved])
        return t[(t >= gc.min()) & (t <= gc.max())]
    return solved


def build(out: Path, pb: dict, run=None, outcome: dict | None = None) -> Path | None:
    """The scene from a crack playback. With the run's uncracked mesh, loads and supports (``run`` as the strength
    adapter reads it) the loads and supports are drawn at their real faces."""
    if pb.get("manifest", {}).get("format") != FORMAT:
        return None
    o = outcome or read(out, "outcome.json") or {}
    part, crack = pb["part"], pb["crack"]
    X = np.asarray(part["vertices"], float)
    T = np.asarray(part["triangles"], int).reshape(-1, 3)
    steps = [st for st in crack["steps"] if st["cycles"] is not None and st["fronts"]]
    if not steps:
        return None
    solved = np.array([st["cycles"] for st in steps], float)
    cyc = _timeline(pb, solved)
    s = Scene("The crack growing through the part", domain="crack", question=o.get("question", "") or
              pb["summary"].get("headline", ""), source="Every front is a solved crack, placed at the cycle count "
                                                        "the nominal part's crack reached it. The size and K "
                                                        "readouts follow the nominal growth between them.")
    s.timeline(cyc, unit="cycles", label="cycles", seconds=8, between="interpolated")
    s.mesh(X, T, name="part", color="#9aa0a6", opacity=0.35)
    if part.get("edges"):
        ev = np.asarray(part["edge_vertices"], float)
        s.lines([ev[list(e)] for e in part["edges"]], color="#c3c7cd", width=1, name="edges")
    n = 24
    rings, kvals, when = [], [], []
    starts = [np.asarray(f, float) for f in crack.get("starting_flaw", []) if len(f) > 1]
    for st in steps:
        f = max(st["fronts"], key=lambda f: len(f["points"]))
        pts = np.asarray(f["points"], float)
        k = np.asarray(f["k"], float)
        k1 = k[:, 0] if k.ndim == 2 else k
        rings.append(_resample(pts, n))
        kvals.append(np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(k1)), k1))
        when.append(st["cycles"])
    origin = np.asarray(crack.get("origin", rings[0].mean(0)), float)
    first = _resample(starts[0], n) if starts else np.repeat(origin[None], n, 0)
    R = np.vstack([first] + rings)
    W = np.r_[np.full(n, cyc[0]), np.repeat(when, n)]
    sizes = np.array([st["size_mm"] for st in steps], float)
    kmax = np.array([st["k_max"] for st in steps], float)
    K = np.r_[kvals[0], np.concatenate(kvals)]
    tri = []
    for i in range(len(rings)):
        for j in range(n - 1):
            a, b = i * n + j, (i + 1) * n + j
            tri += [(a, b, b + 1), (a, b + 1, a + 1)]
    face = s.mesh(R, np.array(tri), name="crack face", color="#ff4d4f", max_vertices=len(R) + 1, smooth=False)
    face.color_by("stress intensity K I", K, unit="MPa sqrt(m)", colormap="inferno")
    face.reveal(W, cyc)
    s.lines(rings, values=kvals, field="stress intensity K I", unit="MPa sqrt(m)", width=2.5, at=when, fade=True,
            name="fronts")
    if run is not None:
        _setting(s, run, X, o, len(cyc))
    g = pb.get("growth", {})
    gc = np.array([np.nan if v is None else v for v in g.get("cycles", [])], float)
    gs = np.array([np.nan if v is None else v for v in g.get("size_mm", [])], float)
    ok = np.isfinite(gc) & np.isfinite(gs)
    size_t = np.interp(cyc, gc[ok], gs[ok]) if ok.sum() > 1 else np.interp(cyc, solved, sizes)
    s.readout("crack size", size_t, unit="mm", digits=3)
    s.readout("largest K", np.interp(size_t, sizes, kmax), unit="MPa sqrt(m)", digits=3)
    s.readout("cycles", cyc, digits=3)
    end = pb["summary"].get("path_end", "")
    s.event(float(solved[-1]), "last solved front", end, kind="fail" if pb["summary"].get("verdict") == "fails" else "info")
    if ok.sum() > 1:
        s.event(float(cyc[-1]), "end of the nominal growth", end, kind="info" if "threshold" in end else "fail")
    word = pb["summary"].get("verdict", "")
    from .common import WORDS, plain
    w, tone = WORDS.get(word, (word.upper() or "DONE", "info"))
    s.verdict(w, plain(re.sub(r"\s+at\s*\([^)]*\)", "", o.get("headline") or pb["summary"].get("headline", ""))), tone)
    return s.save(out)


def _setting(s: Scene, run, X: np.ndarray, o: dict, n: int) -> None:
    """Supports and loads on the part's own faces, as a strength scene draws them. The loads are the peak ones and
    stay constant, since the cycles repeat them."""
    from .strength import load_props, supports

    size = float(np.ptp(X, axis=0).max())
    words = set(re.findall(r"[a-z]+", (o.get("question") or "").lower()))
    total = sum(float(ld.magnitude) for ld in run.study.loads if ld.kind in ("force", "remote", "bolt"))
    supports(s, run, run.mesh.X, size, words)
    load_props(s, run, run.mesh.X, size, np.ones(n), words)
    if total > 0:
        s.readout("peak load", np.full(n, total), unit="N", digits=4)
