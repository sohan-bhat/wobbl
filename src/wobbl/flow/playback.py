"""playback.json: what a flow run computed, small enough for a web page to replay it as a moving simulation.

Every flow study writes one next to its report. Fields are resampled from the solver's cells onto a regular
grid and quantized to one byte per value, base64 encoded: a value v is stored as round(254 (v - lo) / (hi - lo)),
and 255 marks a point with no fluid (inside a body or outside the domain). A page reads the grid, decodes each
component, moves particles through the velocity and colours them by any scalar. Curves and single numbers come
as plain arrays with their units. The file stays under 400 kB, and the same run writes the same file.

    {"format": "wobbl-playback", "version": 1, "kind": "channel", "title": "...", "manifest": "...",
     "fields": [{"name": "air", "grid": {"x0", "x1", "y0", "y1", "nx", "ny", "unit"},
                 "components": {"u": {"lo", "hi", "unit", "data"}, ...}}],
     "frames": [{"t": 0.0, "fields": [...]}],          unsteady runs only
     "series": {"name": {"x": [...], "y": [...], "x_unit", "y_unit"}},
     "scalars": {"name": {"value", "unit"}}}
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np

LIMIT = 400_000


def grid_field(name: str, x, y, comps: dict, units: dict, nx: int = 160, ny: int = 48, unit: str = "m",
               bounds: tuple | None = None, solid=None) -> dict:
    """Resample scattered cell values onto an nx by ny grid. comps maps a component name to values per cell,
    units maps it to its unit, and solid(x, y) is True where there is no fluid."""
    from scipy.interpolate import griddata

    x, y = np.asarray(x, float), np.asarray(y, float)
    x0, x1, y0, y1 = bounds or (float(x.min()), float(x.max()), float(y.min()), float(y.max()))
    gx, gy = np.meshgrid(np.linspace(x0, x1, nx), np.linspace(y0, y1, ny))
    out = {"name": name, "grid": {"x0": _r(x0), "x1": _r(x1), "y0": _r(y0), "y1": _r(y1), "nx": nx, "ny": ny,
                                  "unit": unit, "order": "row by row from y0, x fastest"}, "components": {}}
    holes = solid(gx, gy) if solid is not None else np.zeros_like(gx, bool)
    for key, v in comps.items():
        g = griddata((x, y), np.asarray(v, float), (gx, gy), method="linear")
        near = griddata((x, y), np.asarray(v, float), (gx, gy), method="nearest")
        g = np.where(np.isnan(g), near, g)
        out["components"][key] = quantize(np.where(holes, np.nan, g), units.get(key, ""))
    return out


def quantize(values: np.ndarray, unit: str = "") -> dict:
    v = np.asarray(values, float)
    ok = np.isfinite(v)
    lo, hi = (float(v[ok].min()), float(v[ok].max())) if ok.any() else (0.0, 1.0)
    span = hi - lo if hi > lo else 1.0
    q = np.full(v.shape, 255, np.uint8)
    q[ok] = np.round(254 * (v[ok] - lo) / span).astype(np.uint8)
    return {"lo": _r(lo), "hi": _r(hi), "unit": unit, "data": base64.b64encode(q.tobytes()).decode()}


def series(x, y, x_unit: str, y_unit: str) -> dict:
    return {"x": [_r(v) for v in x], "y": [_r(v) for v in y], "x_unit": x_unit, "y_unit": y_unit}


def write(out_dir: str | Path, kind: str, title: str, manifest: str, fields: list | None = None,
          frames: list | None = None, curves: dict | None = None, scalars: dict | None = None) -> str:
    doc = {"format": "wobbl-playback", "version": 1, "kind": kind, "title": title, "manifest": manifest,
           "fields": fields or [], "frames": frames or [], "series": curves or {},
           "scalars": {k: {"value": _r(v[0]), "unit": v[1]} for k, v in (scalars or {}).items()}}
    text = json.dumps(doc, separators=(",", ":"))
    if len(text) > LIMIT:
        raise ValueError(f"playback.json would be {len(text) / 1000:.0f} kB, over the {LIMIT / 1000:.0f} kB limit")
    path = Path(out_dir) / "playback.json"
    path.write_text(text)
    return str(path)


def _r(v) -> float:
    return float(f"{float(v):.6g}")
