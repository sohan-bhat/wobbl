"""A compact record of a crack growth run that a viewer can play back: the part, each solved crack front with its
K and the cycle count when the nominal part's crack reached it, the growth curve and how lives spread.

Everything comes from the run itself and is rounded so the file stays small (well under 300 KB) and the same for
the same input. Coordinates are in mm in the part's own frame.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

FORMAT = "wobbl-crack-playback"
VERSION = 1


def _r(a, digits: int):
    """Rounded nested lists, with None where a value is not finite (JSON has no NaN)."""
    a = np.round(np.asarray(a, float), digits)
    return a.tolist() if np.isfinite(a).all() else _finite(a.tolist())


def _finite(x):
    if isinstance(x, list):
        return [_finite(v) for v in x]
    return x if np.isfinite(x) else None


def _sharp_edges(pts: np.ndarray, faces: np.ndarray, angle: float = 35.0) -> np.ndarray:
    """Vertex pairs along the edges where two triangles meet at more than angle degrees."""
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    key = np.sort(e, axis=1)
    order = np.lexsort((key[:, 1], key[:, 0]))
    same = np.all(key[order][1:] == key[order][:-1], axis=1)
    f, g = order[:-1][same] % len(faces), order[1:][same] % len(faces)
    n = np.cross(pts[faces[:, 1]] - pts[faces[:, 0]], pts[faces[:, 2]] - pts[faces[:, 0]])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-300)
    return key[order][:-1][same][np.einsum("ij,ij->i", n[f], n[g]) < np.cos(np.radians(angle))]


def part_surface(X: np.ndarray, tris: np.ndarray, target: int = 3000) -> dict:
    """The part's outer surface as a triangle mesh, decimated to about ``target`` triangles, and its sharp edges."""
    from ..life.playback import decimate

    nodes, inv = np.unique(np.asarray(tris)[:, :3].ravel(), return_inverse=True)
    faces, pts = inv.reshape(-1, 3), np.asarray(X, float)[nodes]
    if len(faces) > target:
        used, faces = np.unique(decimate(pts, faces, target // 2), return_inverse=True)
        faces, pts = faces.reshape(-1, 3), pts[used]
    sharp = _sharp_edges(pts, faces)
    ev, lines = np.unique(sharp, return_inverse=True)
    return {"vertices": _r(pts, 2), "triangles": faces.astype(int).tolist(),
            "edge_vertices": _r(pts[ev], 2), "edges": lines.reshape(-1, 2).astype(int).tolist()}


def write(path: Path, *, part: dict, origin, normal, steps: list[dict], starts: list, growth: tuple,
          lives: np.ndarray,
          percentiles: dict, target: float | None, pof: float | None, inspection: dict, verdict: str,
          headline: str, material: str, spectrum: str, end: str) -> str:
    """Write playback.json. ``steps`` hold, per solved crack, its size, the cycle count when the nominal crack
    reached it, the largest K and the fronts (points and K_I, K_II, K_III along each)."""
    size, cycles = growth
    finite = np.sort(lives[np.isfinite(lives)])
    q = np.linspace(0.001, 0.999, 61)
    cdf = {"fraction_failed": _r(q, 4), "cycles": _r(np.quantile(finite, q), 0) if len(finite) else []}
    keep = np.unique(np.linspace(0, len(size) - 1, min(len(size), 120)).astype(int)) if len(size) else []
    doc = {
        "manifest": {"format": FORMAT, "version": VERSION, "made_by": "wobbl crack",
                     "units": {"length": "mm", "stress_intensity": "MPa sqrt(m)", "cycles": "load cycles",
                               "fraction": "0 to 1"},
                     "contents": {"part": "outer surface (vertices, triangles) and sharp edges, decimated",
                                  "crack": "the line each side of the starting flaw grows from, then one entry per "
                                           "solved crack: size, nominal cycles counted from the first solved crack, "
                                           "largest K and each front as points with K_I, K_II, K_III",
                                  "growth": "crack size against cycles for the nominal part",
                                  "life": "share of parts failed against cycles from the Monte Carlo, with "
                                          "percentiles"}},
        "summary": {"verdict": verdict, "headline": headline, "material": material, "loading": spectrum,
                    "path_end": end},
        "part": part,
        "crack": {"origin": _r(origin, 3), "normal": _r(normal, 4),
                  "starting_flaw": [_r(s, 3) for s in starts], "steps": [
            {"step": s["step"], "size_mm": _n(s["size_mm"], 4), "cycles": _n(s["cycles"], 1),
             "k_max": _n(s["k_lead"], 4),
             "fronts": [{"points": _r(f["xyz"], 3), "k": _r(f["k"], 3)} for f in s["fronts"]]}
            for s in steps]},
        "growth": {"cycles": _r(np.asarray(cycles)[keep], 1), "size_mm": _r(np.asarray(size)[keep], 4)},
        "life": {"samples": int(lives.size), "percentiles": {k: _n(v, 0) for k, v in percentiles.items()},
                 "target_cycles": target, "failed_before_target": pof, "cdf": cdf,
                 "inspection": {k: _n(v, 3) for k, v in inspection.items()}},
    }
    Path(path).write_text(json.dumps(doc, separators=(",", ":"), allow_nan=False))
    return str(path)


def _n(x, digits: int):
    """A rounded number, or None when it is not finite (a crack that never fails)."""
    return round(float(x), digits) if x is not None and np.isfinite(float(x)) else None
