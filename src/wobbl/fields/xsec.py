"""Cross sections of real board traces: any stack of dielectric layers, any reference planes, one or more traces.

A trace on a board sees the layers of its stackup, the copper planes above and below it (or none), and its
neighbors. The quasi static field of that cross section, solved once with the dielectrics and once in air,
gives the capacitance matrix per length and, through the air solve, the inductance matrix (L = mu0 eps0 C_air^-1,
exact for TEM lines), from which come the impedance, the delay and the coupling to a neighbor.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import solve
from .forms import C0, EPS0, MU0, ElecProblem, electrostatic
from .mesh2d import Model2D, Shape, mesh

UM = 1e-6


@dataclass(frozen=True)
class Section:
    slabs: tuple = ()             # (y0, y1, er): dielectric layers across the whole width, m
    planes: tuple = ()            # y of copper planes that span the width, held at 0 V
    traces: tuple = ()            # (x0, y0, x1, y1) of each trace, m. The first is driven, the others held at 0

    def key(self, digits: int = 7) -> tuple:
        r = lambda v: round(float(v), digits)  # noqa: E731
        return (tuple(tuple(r(x) for x in s) for s in self.slabs), tuple(r(p) for p in self.planes),
                tuple(tuple(r(x) for x in t) for t in self.traces))


def _model(sec: Section, k: float, air: bool) -> tuple[Model2D, ElecProblem]:
    tr = np.array(sec.traces, float)
    span = float(tr[:, 2].max() - tr[:, 0].min())
    ys = [y for s in sec.slabs for y in s[:2]] + list(sec.planes) + list(tr[:, 1]) + list(tr[:, 3])
    height = max(ys) - min(ys)
    xc = 0.5 * (tr[:, 2].max() + tr[:, 0].min())
    R = 25 * max(span, height, 1e-4)
    clip, lines = [], {"outer": "outer"}
    planes = sorted(set(sec.planes))
    lo = min(tr[:, 1].min(), *[s[0] for s in sec.slabs]) if sec.slabs else tr[:, 1].min()
    hi = max(tr[:, 3].max(), *[s[1] for s in sec.slabs]) if sec.slabs else tr[:, 3].max()
    below = [p for p in planes if p <= tr[:, 1].min() + 1e-12]
    above = [p for p in planes if p >= tr[:, 3].max() - 1e-12]
    yc = 0.5 * (lo + hi)
    if below:
        clip.append(f"y>={max(below)!r}")
    if above:
        clip.append(f"y<={min(above)!r}")
    for i, p in enumerate(planes):
        lines[f"plane{i}"] = f"y={p!r}"
    shapes = []
    for i, (y0, y1, er) in enumerate(sec.slabs):
        if below and y1 <= max(below) + 1e-12 or above and y0 >= min(above) - 1e-12:
            continue
        shapes.append(Shape(f"slab{i}", "rect", (xc - 1.05 * R, y0, xc + 1.05 * R, y1)))
    t = float(np.min(tr[:, 3] - tr[:, 1]))
    w = float(np.min(tr[:, 2] - tr[:, 0]))
    gaps = [tr[j + 1, 0] - tr[j, 2] for j in range(len(tr) - 1)] if len(tr) > 1 else []
    hmin = min([abs(tr[0, 1] - p) for p in planes] + [abs(p - tr[0, 3]) for p in planes] + [w] + gaps)
    edge = max(min(w, hmin) / 8, 1 * UM)
    corner = max(t / 2, 0.5 * UM)
    spots = []
    for j, (x0, y0, x1, y1) in enumerate(tr):
        shapes.append(Shape(f"trace{j}", "rect", (x0, y0, x1, y1), hole=True, size=edge))
        spots += [(x, y, corner) for x in (x0, x1) for y in (y0, y1)]
    model = Model2D("planar", shapes, R, center=(xc, yc), clip=",".join(clip), size=min(hmin / 3, w / 3),
                    far=R / 8, growth=0.25, lines=lines, spots=spots).scaled(k)
    eps = {f"slab{i}": (1.0 if air else s[2]) for i, s in enumerate(sec.slabs)}
    pots = {f"trace{j}": 0.0 for j in range(len(tr))}
    pots.update({n: 0.0 for n in lines})
    return model, ElecProblem("planar", eps, pots, fields=False)


def solve_section(sec: Section, work: Path, k: float = 1.0) -> dict:
    """Capacitance matrices per length with the dielectrics and in air, and the line constants of the first trace."""
    n = len(sec.traces)
    out = {}
    for name, air in (("C", False), ("C0", True)):
        M = np.zeros((n, n))
        for j in range(n):
            model, prob = _model(sec, k, air)
            prob.potentials[f"trace{j}"] = 1.0
            d = Path(work) / f"{name}{j}"
            d.mkdir(parents=True, exist_ok=True)
            msh = mesh(model, d / "m.msh")
            solve.run(electrostatic(prob, msh.tags), msh.path, d, resolution="E")
            for i in range(n):
                M[i, j] = float(solve.read_global(d, f"Q_trace{i}.txt")[0])
            out["nodes"] = msh.n_nodes
        out[name] = 0.5 * (M + M.T)
    Lm = MU0 * EPS0 * np.linalg.inv(out["C0"])
    out["L"] = Lm
    out["z0"] = float(math.sqrt(Lm[0, 0] / out["C"][0, 0]))
    out["eps_eff"] = float(out["C"][0, 0] / out["C0"][0, 0])
    out["delay"] = float(math.sqrt(Lm[0, 0] * out["C"][0, 0]))
    out["c_per_m"] = float(out["C"][0, 0])
    out["l_per_m"] = float(Lm[0, 0])
    if n > 1:
        out["cm_per_m"] = float(-out["C"][0, 1])
        out["lm_per_m"] = float(Lm[0, 1])
        out["kb"] = 0.25 * (out["lm_per_m"] / Lm[0, 0] + out["cm_per_m"] / out["C"][0, 0])
        out["kf"] = 0.5 * (out["cm_per_m"] / out["C"][0, 0] - out["lm_per_m"] / Lm[0, 0])
    _ = C0
    return out
