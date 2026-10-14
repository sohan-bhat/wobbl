"""playback.json: a small, self describing record of a run that a page can animate truthfully.

Each manufacturing study writes one next to its outcome.json. It holds a
decimated triangle surface of the part (or of the part cut open where the
action is), a few fields per vertex taken from the solve, and a manifest
saying what each field means, its unit and how frames map to it. Nothing in
it is invented for the animation: a casting gives each vertex the times it
started and finished freezing, a molding the time the melt reached it, a print
the height it was laid at and how far it moved when the part came off the bed.

The file is deterministic for a given run (no timestamps, fixed rounding) and
kept under about 300 KB by decimating the surface.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

MAX_TRIANGLES = 6000
VERSION = 1


def surface(X: np.ndarray, tets: np.ndarray, fields: dict[str, np.ndarray], cut=None):
    """Outer triangles of a tetrahedral mesh, optionally cut by a plane (origin, normal) keeping the side the
    normal points away from, with the nodal fields carried to its vertices."""
    import pyvista as pv

    from ..report.render import _surface

    cells = np.hstack([np.full((len(tets), 1), tets.shape[1]), tets]).ravel()
    ctype = pv.CellType.TETRA if tets.shape[1] == 4 else pv.CellType.QUADRATIC_TETRA
    grid = pv.UnstructuredGrid(cells, np.full(len(tets), ctype, dtype=np.uint8), np.asarray(X, float))
    for k, v in fields.items():
        grid.point_data[k] = np.nan_to_num(np.asarray(v, float), nan=-1.0)
    if cut is not None:
        grid = grid.clip(normal=tuple(cut[1]), origin=tuple(cut[0]), invert=True)
    surf = _surface(grid).triangulate().clean()
    return surf


def decimate(surf, max_triangles: int = MAX_TRIANGLES):
    """Fewer triangles, same shape; fields move to the new vertices from the nearest old ones."""
    from scipy.spatial import cKDTree

    n = surf.n_cells
    if n <= max_triangles:
        out = surf
    else:
        out = surf.decimate(1.0 - max_triangles / n, volume_preservation=True)
    V = np.asarray(out.points)
    F = np.asarray(out.faces).reshape(-1, 4)[:, 1:]
    idx = cKDTree(np.asarray(surf.points)).query(V)[1]
    vals = {k: np.asarray(surf.point_data[k])[idx] for k in surf.point_data.keys()
            if k not in ("vtkOriginalPointIds", "vtkOriginalCellIds")}
    return V, F, vals


def _round(a, digits: int) -> list:
    a = np.asarray(a, float)
    return [None if not np.isfinite(x) else round(float(x), digits) for x in a.ravel()]


def write(path: str | Path, kind: str, title: str, V: np.ndarray, F: np.ndarray, fields: dict, meta: dict,
          frames: dict, extra: dict | None = None) -> str:
    """fields: name -> (values per vertex, unit, meaning, digits). frames: how a player steps through the run."""
    doc = {
        "manifest": {"format": "wobbl-playback", "version": VERSION, "kind": kind, "title": title,
                     "units": {"length": "mm", **{k: f[1] for k, f in fields.items()}},
                     "fields": {k: f[2] for k, f in fields.items()},
                     "vertex_count": int(len(V)), "triangle_count": int(len(F)), **meta},
        "mesh": {"vertices": _round(V, 2), "triangles": [int(x) for x in np.asarray(F).ravel()]},
        "fields": {k: _round(f[0], f[3]) for k, f in fields.items()},
        "frames": frames,
    }
    if extra:
        doc.update(extra)
    path = Path(path)
    path.write_text(json.dumps(doc, separators=(",", ":")))
    return str(path)
