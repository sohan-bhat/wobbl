"""Pick faces by what they are and where they are, not by fragile ids.

Selectors are short phrases. Combine them with "and" (both must hold) or "or".

    top, bottom, left, right, front, back     flat outer faces at the extremes (+z is up, -y is front)
    x=min   z=max   y=12.5                    faces lying in that plane
    x>100   z<5                               faces whose center is past that coordinate
    normal +z   normal 0,0.7,0.7   facing down
    holes   hole d=4.5   holes d<6   hole near 55,0,10
    bosses   fillets   cylinders r=3   flats   curved
    largest   smallest   area>100
    near 10,20,5                              the single face closest to a point
    within 15 of 50,-25,35                    only the part of a face within 15 mm of a point (a patch)
    box 0,0,0,10,20,5                         only the part of faces inside a box (a patch)
    face 12   faces 3,5,7                     ids from 'wobbl inspect'
    all

Example: "top and within 14 of 210,0,25" is a 28 mm circle on the top face.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from .faces import FaceInfo


class SelectionError(ValueError):
    pass


@dataclass
class Selection:
    text: str
    faces: list[int]
    regions: list[tuple] = field(default_factory=list)   # ("sphere", c, r) or ("box", lo, hi)

    def tri_mask(self, mesh) -> np.ndarray:
        """Boundary triangles covered by this selection (faces, cut down to any regions)."""
        m = mesh.surface_mask(self.faces)
        if self.regions:
            _, _, c = mesh.tri_geometry()
            for reg in self.regions:
                if reg[0] == "sphere":
                    m &= np.linalg.norm(c - np.asarray(reg[1]), axis=1) <= reg[2]
                else:
                    lo, hi = np.asarray(reg[1]), np.asarray(reg[2])
                    m &= ((c >= lo) & (c <= hi)).all(axis=1)
        return m

    def to_dict(self) -> dict:
        return {"selector": self.text, "faces": self.faces,
                "patch": [[r[0], *[np.round(np.asarray(x), 4).tolist() for x in r[1:]]] for r in self.regions]}


_AX = {"x": 0, "y": 1, "z": 2}
_DIRS = {"+x": (1, 0, 0), "-x": (-1, 0, 0), "+y": (0, 1, 0), "-y": (0, -1, 0), "+z": (0, 0, 1), "-z": (0, 0, -1),
         "x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1),
         "up": (0, 0, 1), "down": (0, 0, -1), "right": (1, 0, 0), "left": (-1, 0, 0),
         "back": (0, 1, 0), "front": (0, -1, 0), "forward": (0, 1, 0), "backward": (0, -1, 0)}
_SIDES = {"top": (2, 1), "bottom": (2, -1), "right": (0, 1), "left": (0, -1), "back": (1, 1), "front": (1, -1)}
_NUM = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[-+]?\d+)?"


def parse_vector(text: str) -> np.ndarray:
    t = text.strip().lower().replace("(", "").replace(")", "").replace("[", "").replace("]", "")
    if t in _DIRS:
        v = np.array(_DIRS[t], float)
    else:
        parts = [p for p in re.split(r"[,\s]+", t) if p]
        if len(parts) != 3:
            raise SelectionError(f"could not read a direction from '{text}'; use +x, -z, down, or 0,0,-1")
        v = np.array([float(p) for p in parts])
    nrm = np.linalg.norm(v)
    if nrm == 0:
        raise SelectionError("direction vector is zero")
    return v / nrm


def parse_point(text: str) -> np.ndarray:
    t = text.strip().replace("(", "").replace(")", "")
    parts = [p for p in re.split(r"[,\s]+", t) if p]
    if len(parts) != 3:
        raise SelectionError(f"expected a point x,y,z but got '{text}'")
    return np.array([float(p) for p in parts])


class Selector:
    def __init__(self, faces: dict[int, FaceInfo], bbox=None):
        self.faces = faces
        allpts = np.array([f.bbox[0] for f in faces.values()] + [f.bbox[1] for f in faces.values()])
        self.lo = np.asarray(bbox[0]) if bbox else allpts.min(0)
        self.hi = np.asarray(bbox[1]) if bbox else allpts.max(0)
        self.tol = 1e-3 * float(np.linalg.norm(self.hi - self.lo)) + 1e-6

    def select(self, text: str) -> Selection:
        text = text.strip()
        if not text:
            raise SelectionError("empty face selector")
        regions: list[tuple] = []
        result: set[int] | None = None
        for alt in re.split(r"\s+or\s+|\|", text, flags=re.I):
            ids: set[int] | None = None
            clauses = [c.strip() for c in re.split(r"\s+and\s+|&|;", alt, flags=re.I)]
            # region clauses go first, so "top and within 14 of P" means the highest face near P
            clauses.sort(key=lambda c: 0 if re.match(r"(within|box)\b", c, re.I) else 1)
            for clause in clauses:
                got, reg = self._atom(clause, restrict=ids)
                if reg is not None:
                    regions.append(reg)
                ids = got if ids is None else ids & got
            result = ids if result is None else result | ids
        faces = sorted(result or [])
        if not faces:
            raise SelectionError(f"'{text}' matched no faces. {self.hint()}")
        return Selection(text, faces, regions)

    def hint(self) -> str:
        holes = [f for f in self.faces.values() if f.role == "hole"]
        flats = sorted([f for f in self.faces.values() if f.kind == "plane"], key=lambda f: -f.area)[:4]
        s = "Run 'wobbl inspect <file>' to list faces."
        if holes:
            ds = sorted({round(2 * f.radius, 2) for f in holes})
            s += f" Holes present: diameters {', '.join(map(str, ds))} mm."
        if flats:
            s += " Largest flat faces: " + "; ".join(f.describe() for f in flats)
        return s

    # ---- atoms -------------------------------------------------------------
    def _atom(self, a: str, restrict: set[int] | None = None) -> tuple[set[int], tuple | None]:
        s = a.lower().strip()
        F = self.faces
        allids = set(F)
        if s in ("all", "everything", "*"):
            return allids, None
        m = re.fullmatch(r"(?:faces?|ids?)?[\s:#]*(\d+(?:\s*,\s*\d+)*)", s)
        if m:
            ids = {int(x) for x in re.split(r"\s*,\s*", m.group(1))}
            missing = ids - allids
            if missing:
                raise SelectionError(f"face ids {sorted(missing)} do not exist (valid: 1 to {max(allids)})")
            return ids, None
        if s in _SIDES:
            ax, sign = _SIDES[s]
            cands = [f for f in F.values() if f.kind == "plane" and f.normal[ax] * sign > 0.98
                     and (restrict is None or f.id in restrict)]
            if not cands:
                raise SelectionError(f"no flat face points {s}" + (" in that region" if restrict else "") +
                                     f". {self.hint()}")
            ext = max(f.centroid[ax] * sign for f in cands)
            return {f.id for f in cands if abs(f.centroid[ax] * sign - ext) <= self.tol}, None
        m = re.fullmatch(r"([xyz])\s*(=|==|<=|>=|<|>)\s*(min|max|" + _NUM + ")", s)
        if m:
            ax = _AX[m.group(1)]
            op = m.group(2)
            v = m.group(3)
            val = self.lo[ax] if v == "min" else self.hi[ax] if v == "max" else float(v)
            if op in ("=", "=="):
                return {f.id for f in F.values()
                        if abs(f.bbox[0][ax] - val) <= self.tol and abs(f.bbox[1][ax] - val) <= self.tol}, None
            c = {f.id: f.centroid[ax] for f in F.values()}
            fn = {"<": lambda x: x < val, ">": lambda x: x > val, "<=": lambda x: x <= val + self.tol,
                  ">=": lambda x: x >= val - self.tol}[op]
            return {i for i, x in c.items() if fn(x)}, None
        m = re.fullmatch(r"(?:normal|facing|faces?|pointing)\s*[=:]?\s*(.+)", s)
        if m:
            d = parse_vector(m.group(1))
            return {f.id for f in F.values() if f.kind == "plane" and float(np.dot(f.normal, d)) > 0.985}, None
        m = re.fullmatch(r"(holes?|bosses|boss|fillets?|cylinders?|flats?|curved)\s*(.*)", s)
        if m:
            return self._features(m.group(1), m.group(2).strip()), None
        m = re.fullmatch(r"(largest|biggest|smallest)(?:\s+(flat|face|curved))?", s)
        if m:
            pool = [f for f in F.values() if m.group(2) != "flat" or f.kind == "plane"]
            f = (max if m.group(1) in ("largest", "biggest") else min)(pool, key=lambda f: f.area)
            return {f.id}, None
        m = re.fullmatch(r"area\s*(<|>|<=|>=)\s*(" + _NUM + ")", s)
        if m:
            v = float(m.group(2))
            fn = {"<": lambda x: x < v, ">": lambda x: x > v, "<=": lambda x: x <= v, ">=": lambda x: x >= v}[m.group(1)]
            return {f.id for f in F.values() if fn(f.area)}, None
        m = re.fullmatch(r"(?:near|closest to|at)\s+(.+)", s)
        if m:
            p = parse_point(m.group(1))
            return {self._nearest(p)}, None
        m = re.fullmatch(r"within\s+(" + _NUM + r")\s*(?:mm)?\s+of\s+(.+)", s)
        if m:
            r = float(m.group(1))
            p = parse_point(m.group(2))
            ids = {f.id for f in F.values() if _box_dist(f.bbox, p) <= r}
            return ids, ("sphere", p, r)
        m = re.fullmatch(r"box\s+(.+)", s)
        if m:
            vals = [float(x) for x in re.split(r"[,\s]+", m.group(1).strip("() ")) if x]
            if len(vals) != 6:
                raise SelectionError("box needs six numbers: xmin,ymin,zmin,xmax,ymax,zmax")
            lo, hi = np.array(vals[:3]), np.array(vals[3:])
            ids = {f.id for f in F.values()
                   if (np.asarray(f.bbox[1]) >= lo - self.tol).all() and (np.asarray(f.bbox[0]) <= hi + self.tol).all()}
            return ids, ("box", lo, hi)
        raise SelectionError(f"did not understand the face selector '{a}'. Examples: top, x=min, holes d=4.5, "
                             "normal -z, near 10,0,5, within 15 of 50,-25,35, face 12")

    def _nearest(self, p: np.ndarray) -> int:
        best, bid = np.inf, None
        for f in self.faces.values():
            d = _box_dist(f.bbox, p)
            # break ties between faces sharing a bbox region by centroid distance
            d2 = d + 1e-3 * float(np.linalg.norm(np.asarray(f.centroid) - p))
            if d2 < best:
                best, bid = d2, f.id
        return bid

    def _features(self, word: str, rest: str) -> set[int]:
        F = self.faces
        role = {"hole": "hole", "holes": "hole", "boss": "boss", "bosses": "boss", "fillet": "fillet",
                "fillets": "fillet"}.get(word)
        if role:
            pool = [f for f in F.values() if f.role == role]
        elif word.startswith("cylinder"):
            pool = [f for f in F.values() if f.radius is not None]
        elif word.startswith("flat"):
            pool = [f for f in F.values() if f.kind == "plane"]
        else:
            pool = [f for f in F.values() if f.kind != "plane"]
        if not pool:
            raise SelectionError(f"the part has no {word}. {self.hint()}")
        if not rest:
            return {f.id for f in pool}
        m = re.fullmatch(r"(d|dia|diameter|r|radius)\s*(=|<|>|<=|>=)\s*(" + _NUM + r")\s*(mm)?", rest)
        if m:
            v = float(m.group(3))
            key = (lambda f: 2 * f.radius) if m.group(1).startswith("d") else (lambda f: f.radius)
            op = m.group(2)
            tol = max(0.05 * v, 0.1)
            fn = {"=": lambda x: abs(x - v) <= tol, "<": lambda x: x < v, ">": lambda x: x > v,
                  "<=": lambda x: x <= v + tol, ">=": lambda x: x >= v - tol}[op]
            out = {f.id for f in pool if f.radius is not None and fn(key(f))}
            if not out:
                sizes = sorted({round(key(f), 2) for f in pool if f.radius is not None})
                raise SelectionError(f"no {word} with {m.group(1)}{op}{v}; sizes present: {sizes}")
            return out
        m = re.fullmatch(r"(?:near|at|closest to)\s+(.+)", rest)
        if m:
            p = parse_point(m.group(1))
            best = min(pool, key=lambda f: float(np.linalg.norm(np.asarray(f.axis_point or f.centroid) - p)))
            return {best.id}
        raise SelectionError(f"did not understand '{word} {rest}'; try '{word} d=6' or '{word} near 10,0,5'")


def _box_dist(bbox, p) -> float:
    lo, hi = np.asarray(bbox[0]), np.asarray(bbox[1])
    d = np.maximum(np.maximum(lo - p, 0), p - hi)
    return float(np.linalg.norm(d))
