"""The starting shapes of a crack, as the section curves a Crack is built from.

Every shape lies in the plane through its origin with the given normal. Its fronts reach a little past the part's
surfaces and its mouth sits a little outside the part, so that trimming the ruled surface to the part leaves
exactly the intended crack.
"""
from __future__ import annotations

import math

import numpy as np

from .crackmesh import Crack


def _unit(v) -> np.ndarray:
    v = np.asarray(v, float)
    return v / max(float(np.linalg.norm(v)), 1e-300)


def _in_plane(v, normal) -> np.ndarray:
    n = _unit(normal)
    v = np.asarray(v, float)
    return _unit(v - (v @ n) * n)


def _line(p0, p1, k: int) -> np.ndarray:
    t = np.linspace(0.0, 1.0, k)[:, None]
    return np.asarray(p0, float) + t * (np.asarray(p1, float) - np.asarray(p0, float))


def through(origin, normal, toward, a: float, span: tuple[float, float], margin: float) -> Crack:
    """A crack through a wall from a surface, its straight front a from the origin. ``toward`` points into the
    material and ``span`` is how far the wall reaches each way along the front from the origin."""
    n, d = _unit(normal), _in_plane(toward, normal)
    w = np.cross(n, d)
    o = np.asarray(origin, float)
    lo, hi = -(span[0] + margin), span[1] + margin
    front = _line(o + a * d + lo * w, o + a * d + hi * w, 2)
    mouth = _line(o - margin * d + lo * w, o - margin * d + hi * w, 2)
    return Crack("through", [mouth, front], (False, True), n, o, a)


def center(origin, normal, along, a: float, span: tuple[float, float], margin: float) -> Crack:
    """A crack through a wall with two tips, each a from the origin along ``along``, as in a plate's middle."""
    n, d = _unit(normal), _in_plane(along, normal)
    w = np.cross(n, d)
    o = np.asarray(origin, float)
    lo, hi = -(span[0] + margin), span[1] + margin
    left = _line(o - a * d + lo * w, o - a * d + hi * w, 2)
    right = _line(o + a * d + lo * w, o + a * d + hi * w, 2)
    return Crack("center", [left, right], (True, True), n, o, a)


def surface(origin, normal, inward, a: float, c: float, margin: float, k: int = 41) -> Crack:
    """A semi elliptical crack of depth a and half length c on a surface whose inward direction is given."""
    n, d = _unit(normal), _in_plane(inward, normal)
    w = np.cross(n, d)
    o = np.asarray(origin, float)
    ext = math.asin(min(margin / a, 0.9))
    phi = np.linspace(-ext, math.pi + ext, k)
    front = o + np.outer(a * np.sin(phi), d) + np.outer(c * np.cos(phi), w)
    lift = o - (a * math.sin(ext) + 0.5 * margin) * d
    mouth = _line(lift + c * math.cos(ext) * w, lift - c * math.cos(ext) * w, 2)
    return Crack("surface", [mouth, front], (False, True), n, o, a)


def corner(origin, normal, along_a, along_c, a: float, c: float, margin: float, k: int = 33) -> Crack:
    """A quarter elliptical crack at an edge, reaching a into the material along one face (``along_a``) and c
    along the other (``along_c``)."""
    n = _unit(normal)
    d1, d2 = _in_plane(along_a, n), _in_plane(along_c, n)
    o = np.asarray(origin, float)
    e1, e2 = math.asin(min(margin / c, 0.9)), math.asin(min(margin / a, 0.9))
    phi = np.linspace(-e1, math.pi / 2 + e2, k)
    front = o + np.outer(a * np.cos(phi), d1) + np.outer(c * np.sin(phi), d2)
    out = o - margin * d1 - margin * d2
    mouth = np.vstack([_line(front[0] - 0.5 * margin * d2, out, 6), _line(out, front[-1] - 0.5 * margin * d1, 6)[1:]])
    return Crack("corner", [mouth, front], (False, True), n, o, a)
