"""Playback files: the numbers a viewer needs to replay a field run, kept small and written by the run itself.

Every field study writes output/playback.json next to its report. It holds a
manifest (what each entry is, its units and shape) and the data: fields
sampled on a regular grid over a 2D slice, packed as integers 0 to 250 on a
stated range so a frame costs a few kilobytes, plus the curves and numbers the
report quotes. Grid values are row major from the low y edge, -1 marks points
outside the model, and a value v unpacks as min + v / 250 * (max - min).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

LEVELS = 250
LIMIT = 300_000                   # bytes


def grid(X: np.ndarray, tris: np.ndarray, window: tuple, n: tuple[int, int], node_vals: np.ndarray | None = None,
         tri_vals: np.ndarray | None = None) -> np.ndarray:
    """Sample a mesh field on an nx by ny grid over window (x0, x1, y0, y1): node values are interpolated
    linearly, triangle values taken as they are. Points outside the mesh are NaN."""
    from matplotlib.tri import LinearTriInterpolator, Triangulation

    x0, x1, y0, y1 = window
    nx, ny = n
    gx, gy = np.meshgrid(np.linspace(x0, x1, nx), np.linspace(y0, y1, ny))
    tr = Triangulation(X[:, 0], X[:, 1], tris)
    if node_vals is not None:
        v = LinearTriInterpolator(tr, np.nan_to_num(node_vals))(gx, gy)
        out = np.ma.filled(v, np.nan)
    else:
        idx = tr.get_trifinder()(gx, gy)
        out = np.where(idx >= 0, np.asarray(tri_vals)[np.maximum(idx, 0)], np.nan)
    return np.asarray(out, float)


def pack(a: np.ndarray, lo: float | None = None, hi: float | None = None, clip: float = 100.0) -> dict:
    """Integers 0 to 250 on [min, max], -1 where there is no value. clip, a percentile, keeps a few singular
    points (a sharp corner, the wire surface) from flattening the rest of the picture."""
    a = np.asarray(a, float)
    ok = np.isfinite(a)
    if not ok.any():
        return {"min": 0.0, "max": 0.0, "shape": list(a.shape), "q": [-1] * a.size}
    vals = a[ok]
    lo = float(np.min(vals)) if lo is None else lo
    hi = float(np.percentile(vals, clip)) if hi is None else hi
    if hi <= lo:
        hi = lo + 1e-30
    q = np.full(a.shape, -1, dtype=int)
    q[ok] = np.clip(np.rint((a[ok] - lo) / (hi - lo) * LEVELS), 0, LEVELS).astype(int)
    return {"min": _r(lo), "max": _r(hi), "shape": list(a.shape), "q": q.ravel().tolist()}


def pack_signed(a: np.ndarray, clip: float = 100.0) -> dict:
    """Like pack, on a range symmetric about zero, for waves and mode shapes."""
    a = np.asarray(a, float)
    ok = np.isfinite(a)
    m = float(np.percentile(np.abs(a[ok]), clip)) if ok.any() else 1.0
    return pack(a, -m, m)


def shapes_mm(shapes) -> list[dict]:
    """The model's parts as drawn shapes in mm: rectangles (x0, y0, x1, y1), disks (cx, cy, r) and polygons."""
    out = []
    for s in shapes:
        d = np.round(np.asarray(s.data, float) * 1e3, 4).tolist()
        out.append({"name": s.name, "kind": s.kind, "data": d, "hole": bool(s.hole)})
    return out


def series(x, ys: dict, n: int = 600) -> dict:
    """Curves on a shared x, thinned to at most n points."""
    x = np.asarray(x, float)
    step = max(1, int(np.ceil(len(x) / n)))
    return {"x": [_r(v) for v in x[::step]], **{k: [_r(v) for v in np.asarray(y, float)[::step]] for k, y in ys.items()}}


def _r(v: float, sig: int = 5):
    v = float(v)
    if not np.isfinite(v):
        return None
    return float(f"{v:.{sig}g}")


def static_field(out_dir, kind: str, msh, window: tuple, e_tri: np.ndarray, v_node: np.ndarray, shapes,
                 numbers: dict, mirror: bool, unit_e: str = "V/m", nx: int = 64) -> str:
    """One electrostatic frame: |E| and the potential on a grid over the window, with the parts and the numbers."""
    x0, x1, y0, y1 = window
    ny = int(min(100, max(20, round(nx * (y1 - y0) / (x1 - x0)))))
    e = grid(msh.X, msh.tris, window, (nx, ny), tri_vals=np.where(msh.mask("shell"), np.nan, e_tri))
    v = grid(msh.X, msh.tris, window, (nx, ny), node_vals=v_node)
    manifest = {"e": f"electric field magnitude on the grid, {unit_e}, packed",
                "v": "electric potential on the grid, V, packed; its contours are equipotentials",
                "parts": "the model's parts in mm; holes are conductors",
                "grid": "x0, x1, y0, y1 in mm and nx by ny points" + (", axisymmetric about x = 0" if mirror else ""),
                "numbers": "the headline results with their units in the key names"}
    return write(out_dir, kind, manifest,
                 {"grid": {"x0": _r(x0 * 1e3), "x1": _r(x1 * 1e3), "y0": _r(y0 * 1e3), "y1": _r(y1 * 1e3),
                           "nx": nx, "ny": ny, "mirror": mirror},
                  "e": pack(e, 0.0, clip=99.5), "v": pack(v), "parts": shapes_mm(shapes),
                  "numbers": {k: _r(v) if isinstance(v, (int, float)) else v for k, v in numbers.items()}})


def write(out_dir: str | Path, kind: str, manifest: dict, data: dict) -> str:
    """Write playback.json. The manifest maps each data key to a one line description with units."""
    doc = {"format": "wobbl-playback", "version": 1, "kind": kind, "manifest": manifest, **data}
    path = Path(out_dir) / "playback.json"
    text = json.dumps(doc, separators=(",", ":"))
    if len(text) > LIMIT:
        raise ValueError(f"playback.json would take {len(text) / 1e3:.0f} kB, over the {LIMIT / 1e3:.0f} kB budget")
    path.write_text(text)
    return str(path)
