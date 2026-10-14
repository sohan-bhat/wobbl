"""Fracture validation cases: cracked plates with handbook answers, run through the same crack insertion, mesh
and K extraction a crack study uses, plus the life integration against its closed form.

The through cracks use Poisson's ratio 0, which makes the 3D plate exactly a 2D one (nothing pulls across the
thickness), so K is the same all along the front and equals the handbook's 2D value. The plates are pulled by the
same tension on both ends and held at three corners in a way that resists no load, so they are free to bend, as
the handbook strips are. The surface crack keeps Poisson's ratio 0.3, as in Newman and Raju's analyses.
"""
from __future__ import annotations

import math
import shutil
from pathlib import Path

import numpy as np

from . import handbook as H
from . import shapes
from .crackmesh import Crack, Sizing, build_cracked_mesh


def _plate(path: Path, w: float, h: float, t: float) -> tuple:
    from ..structural.run import describe_part
    from ..validation import geometry as G

    part = G.box(path / "plate.step", w, h, t)
    return describe_part(part, path / "geo")


def pulled_plate(out: Path, w: float, h: float, t: float, crack: Crack, sizing: Sizing, stress: float, E: float,
                 nu: float) -> list:
    """Solve a w by h by t plate pulled along y at both ends with the crack in it, and return the fronts' K."""
    from ..fe import NodeConstraint, StaticModel
    from ..solvers import base as solvers
    from ..structural.build import traction_forces
    from .sif import extract
    from .solve import solve_displacements

    out.mkdir(parents=True, exist_ok=True)
    prep, feat = _plate(out, w, h, t)
    cm = build_cracked_mesh(prep, crack, sizing)
    mesh = cm.mesh
    ends = [np.isin(mesh.tri_surf, [f.id for f in feat.faces.values() if f.normal and f.normal[1] * s > 0.9])
            for s in (1, -1)]
    F = traction_forces(mesh, ends[0], np.array([0, stress, 0])) + traction_forces(mesh, ends[1],
                                                                                   np.array([0, -stress, 0]))
    pick = [int(np.argmin(np.linalg.norm(mesh.X - p, axis=1))) for p in ((0, 0, 0), (w, 0, 0), (0, h, 0))]
    model = StaticModel(mesh, E, nu, 2.7e-9, [NodeConstraint("a", np.array([pick[0]]), (1, 2, 3)),
                                              NodeConstraint("b", np.array([pick[1]]), (2, 3)),
                                              NodeConstraint("c", np.array([pick[2]]), (3,))], forces=F)
    U, _ = solve_displacements(solvers.for_domain("structural"), model, out / "solve")
    fronts = extract(cm, U, E, nu)
    shutil.rmtree(out / "geo", ignore_errors=True)
    return fronts, cm


def _info(cm, fronts) -> dict:
    return {"nodes": cm.mesh.n_nodes, "front_elements_mm": round(cm.sizing.h_front, 4),
            "front_points": [len(f.s) for f in fronts]}


def center_crack(out: Path):
    """A 30 mm crack across the middle of a 100 mm wide plate (2a / W = 0.3), K at the middle of each front."""
    w, h, t, a, s = 100.0, 150.0, 4.0, 15.0, 100.0
    crack = shapes.center((w / 2, h / 2, t / 2), (0, 1, 0), (1, 0, 0), a, (t / 2, t / 2), 1.0)
    fronts, cm = pulled_plate(out, w, h, t, crack, Sizing(h=8.0, h_front=a / 30, h_crack=a / 4), s, 70000.0, 0.0)
    got = float(np.mean([f.at(0.5)[0] for f in fronts]))
    return got, H.center_crack(s, a, w), _info(cm, fronts)


def edge_crack(out: Path):
    """A 12 mm edge crack in a 40 mm wide strip (a / W = 0.3), four widths long, free to bend."""
    w, h, t, a, s = 40.0, 160.0, 4.0, 12.0, 100.0
    crack = shapes.through((0.0, h / 2, t / 2), (0, 1, 0), (1, 0, 0), a, (t / 2, t / 2), 1.0)
    fronts, cm = pulled_plate(out, w, h, t, crack, Sizing(h=6.0, h_front=a / 30, h_crack=a / 4), s, 70000.0, 0.0)
    return float(fronts[0].at(0.5)[0]), H.edge_crack(s, a, w), _info(cm, fronts)


def surface_crack(out: Path):
    """A semi elliptical surface crack 3 mm deep and 12 mm long (a / c = 0.5, a / t = 0.25) in a 12 mm plate,
    K at the deepest point against Newman and Raju."""
    w, h, t, a, c, s = 48.0, 64.0, 12.0, 3.0, 6.0, 100.0
    crack = shapes.surface((w / 2, h / 2, t), (0, 1, 0), (0, 0, -1), a, c, 0.75)
    fronts, cm = pulled_plate(out, w, h, t, crack, Sizing(h=6.0, h_front=a / 10, h_crack=a / 2), s, 70000.0, 0.3)
    f = fronts[0]
    deepest = int(np.argmin(f.xyz[:, 2]))
    return float(f.k[deepest, 0]), H.newman_raju(s, a, c, t, w), _info(cm, fronts)


def kink_angle(out: Path):
    """A 20 mm crack at 45 degrees to the pull across a 200 mm plate: K_I and K_II are equal, so the maximum
    tangential stress rule turns the crack 53.13 degrees. The FE turn, from its own K_I and K_II at the middle of
    each front, is compared with that, as an angle."""
    w, h, t, a, s = 200.0, 200.0, 4.0, 10.0, 100.0
    beta = math.pi / 4
    normal = (math.cos(beta), math.sin(beta), 0.0)
    crack = shapes.center((w / 2, h / 2, t / 2), normal, (math.sin(beta), -math.cos(beta), 0.0), a, (t / 2, t / 2),
                          1.0)
    fronts, cm = pulled_plate(out, w, h, t, crack, Sizing(h=12.0, h_front=a / 30, h_crack=a / 4), s, 70000.0, 0.0)
    angles = [abs(math.degrees(H.mts_angle(*f.at(0.5)[:2]))) for f in fronts]
    k1, k2 = H.inclined_crack(s, a, beta)
    want = abs(math.degrees(H.mts_angle(k1, k2)))
    info = _info(cm, fronts)
    info["k_ratio"] = [round(float(f.at(0.5)[1] / f.at(0.5)[0]), 4) for f in fronts]
    return float(np.mean(angles)), want, info


def paris_life(out: Path):
    """Cycles for an edge crack with constant beta 1.12 to grow from 1 to 10 mm under 100 MPa at R = 0 with the
    7075-T6 Paris constants, against N = (a0^(1 - m/2) - af^(1 - m/2)) / ((m/2 - 1) C (beta S sqrt(pi))^m)."""
    from .growth import law_for, parse_spectrum
    from .life import CrackPath, LifeInputs, lives

    law = law_for("7075-t6", "paris")
    law = type(law)("paris", law.C, law.n)
    beta, stress, a0, af = 1.12, 100.0, 1.0, 10.0
    s = np.geomspace(a0, af, 6)
    y = beta * stress * math.sqrt(math.pi * 1e-3)
    path = CrackPath(s, y * np.sqrt(s), np.zeros_like(s))
    n, _ = lives(path, LifeInputs(law, parse_spectrum(r_ratio=0.0), 0.0), a0, 1.0, 1.0, 0.0)
    m = law.n
    exact = (a0 ** (1 - m / 2) - af ** (1 - m / 2)) / ((m / 2 - 1) * law.C * y ** m)
    return float(n[0]), exact, {"C": law.C, "m": m}
