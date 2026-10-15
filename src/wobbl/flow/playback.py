"""playback.json: what a flow run computed, small enough for a web page to replay it as a moving simulation.

Every flow study writes one next to its report, in the encoding the other domains use (life/playback.py): arrays
are base64 of little endian typed data with their dtype and shape, and quantized ones carry the scale and offset
that turn the stored integers back into values, value = offset + scale * stored. Flow fields are resampled from
the solver's cells onto a regular grid in a plane, one byte per value, with 255 marking a point with no fluid
(inside the part or outside the domain), so a page can move particles through the velocity and colour them by any
scalar. A part's surface comes as the shared mesh block with a value at each vertex. Curves and single numbers are
plain arrays with their units. The file stays under 400 kB, and the same run writes the same file.

    {"format": "wobbl-playback", "version": 1, "kind": "...", "about": "...", "units": {...},
     "mesh": {"vertices", "triangles", "units"},                 a part's surface, when there is one
     "surface": {"name": encoded uint8 per vertex, ...},
     "fields": [{"name", "plane": {"origin", "u", "v"}, "grid": {"x0", "x1", "y0", "y1", "nx", "ny", "unit"},
                 "components": {"u": encoded uint8 (ny, nx) with "missing": 255, ...}}],
     "frames": [{"t": 0.0, "fields": [...]}],                    unsteady runs only
     "series": {"name": {"x": [...], "y": [...], "x_unit", "y_unit"}},
     "scalars": {"name": {"value", "unit"}}}
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..life.playback import encode, mesh_block

LIMIT = 400_000
MISSING = 255


def grid_field(name: str, x, y, comps: dict, units: dict, nx: int = 160, ny: int = 48, unit: str = "m",
               bounds: tuple | None = None, solid=None, plane: dict | None = None, spacing=None) -> dict:
    """Resample scattered cell values onto an nx by ny grid, rows from y0 up and x fastest. comps maps a
    component name to values per cell, units maps it to its unit, and solid(x, y) is True where there is no fluid.
    spacing gives each point's cell size, and grid points further than that from every cell are marked missing.
    plane says where the grid's x and y axes lie in 3D, as an origin and two unit vectors."""
    from scipy.interpolate import griddata
    from scipy.spatial import cKDTree

    x, y = np.asarray(x, float), np.asarray(y, float)
    x0, x1, y0, y1 = bounds or (float(x.min()), float(x.max()), float(y.min()), float(y.max()))
    gx, gy = np.meshgrid(np.linspace(x0, x1, nx), np.linspace(y0, y1, ny))
    out = {"name": name, "grid": {"x0": _r(x0), "x1": _r(x1), "y0": _r(y0), "y1": _r(y1), "nx": nx, "ny": ny,
                                  "unit": unit}, "components": {}}
    if plane:
        out["plane"] = {k: [_r(c) for c in v] for k, v in plane.items()}
    holes = solid(gx, gy) if solid is not None else np.zeros_like(gx, bool)
    if spacing is not None:
        dist, idx = cKDTree(np.c_[x, y]).query(np.c_[gx.ravel(), gy.ravel()])
        holes = holes | (dist > 1.0 * np.asarray(spacing, float)[idx]).reshape(gx.shape)
    for key, v in comps.items():
        g = griddata((x, y), np.asarray(v, float), (gx, gy), method="linear")
        near = griddata((x, y), np.asarray(v, float), (gx, gy), method="nearest")
        g = np.where(np.isnan(g), near, g)
        out["components"][key] = quantize(np.where(holes, np.nan, g), units.get(key, ""))
    return out


def quantize(values: np.ndarray, unit: str = "") -> dict:
    """One byte per value between the finite values' extremes, 255 where a value is missing."""
    v = np.asarray(values, float)
    ok = np.isfinite(v)
    lo, hi = (float(v[ok].min()), float(v[ok].max())) if ok.any() else (0.0, 1.0)
    span = hi - lo if hi > lo else 1.0
    q = np.full(v.shape, MISSING, np.uint8)
    q[ok] = np.round(254 * (v[ok] - lo) / span).astype(np.uint8)
    d = encode(q, "uint8", scale=_r(span / 254), offset=_r(lo))
    d.update(missing=MISSING, unit=unit)
    return d


def series(x, y, x_unit: str, y_unit: str) -> dict:
    return {"x": [_r(v) for v in x], "y": [_r(v) for v in y], "x_unit": x_unit, "y_unit": y_unit}


def surface_block(X, tris, values: dict, units: dict, max_vertices: int = 2500) -> tuple[dict, dict]:
    """The shared mesh block of a triangulated surface in mm, decimated to whole vertices, and each per vertex
    value quantized on it."""
    from ..life.playback import surface

    class _M:
        pass
    m = _M()
    m.X, m.tris = np.asarray(X, float), np.asarray(tris, int)
    pts, faces, nodes = surface(m, max_vertices)
    return mesh_block(pts, faces), {k: quantize(np.asarray(v, float)[nodes], units.get(k, "")) for k, v in
                                    values.items()}


def write(out_dir: str | Path, kind: str, title: str, about: str, fields: list | None = None,
          frames: list | None = None, curves: dict | None = None, scalars: dict | None = None,
          mesh: dict | None = None, surface: dict | None = None, units: dict | None = None) -> str:
    doc = {"format": "wobbl-playback", "version": 1, "kind": kind, "title": title, "about": about,
           "units": units or {}, "fields": fields or [], "frames": frames or [], "series": curves or {},
           "scalars": {k: {"value": _r(v[0]), "unit": v[1]} for k, v in (scalars or {}).items()}}
    if mesh:
        doc["mesh"], doc["surface"] = mesh, surface or {}
    text = json.dumps(doc, separators=(",", ":"))
    if len(text) > LIMIT:
        raise ValueError(f"playback.json would be {len(text) / 1000:.0f} kB, over the {LIMIT / 1000:.0f} kB limit")
    path = Path(out_dir) / "playback.json"
    path.write_text(text)
    from .scene import write as write_scene
    write_scene(out_dir)
    return str(path)


def _r(v) -> float:
    return float(f"{float(v):.6g}")
