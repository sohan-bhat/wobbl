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


class Cut:
    """A triangle surface with nodal fields on its vertices."""

    def __init__(self, points: np.ndarray, faces: np.ndarray, point_data: dict[str, np.ndarray]):
        self.points, self.faces, self.point_data = points, faces, point_data


def surface(X: np.ndarray, tets: np.ndarray, fields: dict[str, np.ndarray], cut=None) -> Cut:
    """Outer triangles of a tetrahedral mesh, optionally cut by a plane (origin, normal) keeping the side the
    normal points away from, with the nodal fields carried to its vertices."""
    from ..mesh.clip import carry, clip_surface

    (a, b, t), faces = clip_surface(X, tets, *(cut if cut is not None else (None, None)))
    data = {k: carry(np.nan_to_num(np.asarray(v, float), nan=-1.0), a, b, t) for k, v in fields.items()}
    return Cut(carry(np.asarray(X, float), a, b, t), faces, data)


def decimate(surf: Cut, max_triangles: int = MAX_TRIANGLES):
    """Fewer triangles, same shape. Decimation keeps original vertices, so each carries its fields exactly."""
    from ..life.playback import decimate as collapse

    F = surf.faces
    if len(F) > max_triangles:
        F = collapse(surf.points, F, max_triangles // 2)
    used, F = np.unique(F, return_inverse=True)
    return surf.points[used], F.reshape(-1, 3), {k: v[used] for k, v in surf.point_data.items()}


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
