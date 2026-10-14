"""Screw bosses and pins on a shell model: straight beams with the post's own round or pipe section.

A boss is too chunky to be a wall, but it holds a lid up near its corners, so leaving it out makes the shell
model too soft (the enclosure example's lid sags 15% more without its four bosses). Each outside cylinder the
midsurface left out, longer than it is wide, becomes a line of 3 node beams along its axis, with a pipe section
when a hole runs down its middle. The end that meets a wall reaches on to the wall's midsurface and grips the
shell nodes inside its outline through a rigid joint; the other end is free unless a support picks it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .mesh import ShellMesh


@dataclass
class Post:
    face: int
    start: np.ndarray              # free end, mm
    end: np.ndarray                # joined end, on the midsurface
    r_out: float
    r_in: float                    # 0 for a solid post
    footprint: np.ndarray          # shell nodes the joined end grips
    nodes: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))   # its beam nodes in the model
    spokes: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))  # midside nodes of its spider
    elements: list = field(default_factory=list)   # its beam element numbers in the deck

    @property
    def length(self) -> float:
        return float(np.linalg.norm(self.end - self.start))

    @property
    def area(self) -> float:
        return float(np.pi * (self.r_out ** 2 - self.r_in ** 2))

    def points(self, h: float) -> np.ndarray:
        """Beam nodes from the free end to the joined one, 2 per element plus the last, about h apart."""
        k = max(2, int(np.ceil(self.length / max(h, 1e-9))))
        s = np.linspace(0.0, 1.0, 2 * k + 1)
        return self.start[None] + s[:, None] * (self.end - self.start)[None]


def find_posts(shell: ShellMesh, faces: dict, h: float) -> list[Post]:
    """Posts for the outside cylinders that the midsurface leaves out and that end at one of its walls."""
    sm = shell.solid
    used = set(int(s) for s in np.unique(shell.tri_surf))
    holes = [f for f in faces.values() if f.role == "hole" and f.axis is not None]
    tmax = float(shell.thickness.max())
    out = []
    for f in faces.values():
        if f.role != "boss" or f.id in used or f.axis is None or not f.radius:
            continue
        if any(np.linalg.norm(np.cross(np.asarray(f.axis_point) - q.start, q.end - q.start)) < 0.1 * f.radius *
               q.length for q in out):
            continue   # the other half of a cylinder split in two
        a = np.asarray(f.axis, float)
        a /= np.linalg.norm(a)
        p0 = np.asarray(f.axis_point, float)
        pts = sm.X[np.unique(sm.tris[sm.tri_surf == f.id].ravel())]
        s = (pts - p0) @ a
        lo, hi = float(s.min()), float(s.max())
        if hi - lo < 1.5 * f.radius:
            continue
        r_in = 0.0
        for g in holes:
            ga = np.asarray(g.axis, float)
            off = np.asarray(g.axis_point, float) - p0
            if abs(abs(ga @ a) - 1.0) < 1e-3 and np.linalg.norm(off - (off @ a) * a) < 0.1 * f.radius \
                    and g.radius < f.radius:
                r_in = max(r_in, float(g.radius))
        rel = shell.X - p0
        along = rel @ a
        radial = np.linalg.norm(rel - np.outer(along, a), axis=1)
        inside = radial <= f.radius
        joined = None
        for end, sign in ((hi, 1.0), (lo, -1.0)):
            near = inside & ((along - end) * sign >= -0.25 * tmax) & ((along - end) * sign <= 0.75 * tmax)
            if near.sum() >= 3:
                joined = (float(along[near].mean()), np.flatnonzero(near), lo if sign > 0 else hi)
                break
        if joined is None:
            continue
        reach, grip, free = joined
        # a node on the axis itself would make a spoke of no length; the rest of the outline holds the post
        grip = grip[radial[grip] > 0.15 * f.radius]
        if len(grip) < 3:
            continue
        out.append(Post(f.id, p0 + free * a, p0 + reach * a, float(f.radius), r_in, grip))
    return out


def beam_normal(axis: np.ndarray) -> np.ndarray:
    """A unit vector square to the beam axis, for the section's first direction."""
    a = axis / np.linalg.norm(axis)
    e = np.eye(3)[int(np.argmin(np.abs(a)))]
    n = e - (e @ a) * a
    return n / np.linalg.norm(n)


__all__ = ["Post", "find_posts", "beam_normal"]
