"""Carry face selections across a geometry edit.

After a fillet or a gusset is added, OpenCASCADE renumbers faces. A face that
kept its plane (or its cylinder) is the same face for loads and supports, so
selections are remapped by geometry, not by id.
"""
from __future__ import annotations

import re

import numpy as np

from .faces import FaceInfo
from .select import Selector

_REGION = re.compile(r"(within\s+[\d.]+\s*(?:mm)?\s+of\s+[-\d.,\s()]+|box\s+[-\d.,\s()]+)", re.I)


def _overlap(b1, b2, tol) -> bool:
    lo1, hi1, lo2, hi2 = (np.asarray(x) for x in (b1[0], b1[1], b2[0], b2[1]))
    return bool((lo1 <= hi2 + tol).all() and (lo2 <= hi1 + tol).all())


def same_surface(f: FaceInfo, g: FaceInfo, tol: float) -> bool:
    if f.kind == "plane" and g.kind == "plane":
        n1, n2 = np.asarray(f.normal), np.asarray(g.normal)
        if n1 @ n2 < 0.999:
            return False
        if abs(n1 @ (np.asarray(g.centroid) - np.asarray(f.centroid))) > tol:
            return False
        return _overlap(f.bbox, g.bbox, tol)
    if f.radius is not None and g.radius is not None:
        a1, a2 = np.asarray(f.axis), np.asarray(g.axis)
        if abs(a1 @ a2) < 0.999 or abs(f.radius - g.radius) > max(0.02 * f.radius, tol):
            return False
        d = np.asarray(g.axis_point) - np.asarray(f.axis_point)
        if np.linalg.norm(d - (d @ a1) * a1) > max(0.05 * f.radius, tol):
            return False
        return _overlap(f.bbox, g.bbox, tol)
    return False


def remap_selector(text: str, old: dict[int, FaceInfo], new: dict[int, FaceInfo], bbox_old=None) -> str:
    """Rewrite a selector so it picks the same physical faces on the edited part."""
    s = Selector(old, bbox_old)
    sel = s.select(text)
    tol = 1e-4 * float(np.linalg.norm(s.hi - s.lo)) + 1e-6
    ids = []
    for fid in sel.faces:
        f = old[fid]
        hits = [g.id for g in new.values() if same_surface(f, g, tol)]
        if not hits:  # fall back to the nearest face of the same kind
            same = [g for g in new.values() if g.kind == f.kind] or list(new.values())
            hits = [min(same, key=lambda g: float(np.linalg.norm(np.asarray(g.centroid) - np.asarray(f.centroid)))).id]
        ids += hits
    ids = sorted(set(ids))
    regions = _REGION.findall(text)
    out = f"faces {','.join(map(str, ids))}"
    for r in regions:
        out += f" and {r.strip()}"
    return out
