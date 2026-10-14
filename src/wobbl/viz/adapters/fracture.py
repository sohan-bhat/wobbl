"""A crack growth run as a scene: the crack face spreading through the part front by solved front, colored by K.

The fronts are the run's own, each at the cycle count when the nominal part's crack reached it. Between two solved
fronts the face is blended, and the view says so.
"""
from __future__ import annotations

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


def build(out: Path, pb: dict) -> Path | None:
    if pb.get("manifest", {}).get("format") != FORMAT:
        return None
    o = read(out, "outcome.json") or {}
    part, crack = pb["part"], pb["crack"]
    X = np.asarray(part["vertices"], float)
    T = np.asarray(part["triangles"], int).reshape(-1, 3)
    steps = [st for st in crack["steps"] if st["cycles"] is not None and st["fronts"]]
    if not steps:
        return None
    cyc = np.array([st["cycles"] for st in steps], float)
    s = Scene("The crack growing through the part", domain="crack", question=o.get("question", "") or
              pb["summary"].get("headline", ""), source="Every front is a solved crack, placed at the cycle count "
                                                        "the nominal part's crack reached it.")
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
    s.readout("crack size", [st["size_mm"] for st in steps], unit="mm", digits=3)
    s.readout("largest K", [st["k_max"] for st in steps], unit="MPa sqrt(m)", digits=3)
    s.readout("cycles", cyc, digits=3)
    end = pb["summary"].get("path_end", "")
    s.event(float(cyc[-1]), "last solved front", end, kind="fail" if pb["summary"].get("verdict") == "fails" else "info")
    word = pb["summary"].get("verdict", "")
    from .common import WORDS, plain
    w, tone = WORDS.get(word, (word.upper() or "DONE", "info"))
    s.verdict(w, plain(o.get("headline") or pb["summary"].get("headline", "")), tone)
    return s.save(out)
