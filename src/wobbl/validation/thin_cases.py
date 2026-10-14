"""Validation cases for shells, laminates and drops, each against a closed form."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import geometry as G


def bar_impact(out: Path):
    """A 10 x 10 x 100 mm steel bar landing on its end at 1 m/s on a rigid floor. While the stress wave runs up the
    bar and back, the floor pushes with rho c A v0, for 2 L / c."""
    from ..drop import tet10
    from ..drop.explicit import Contact, integrate, omega_max
    from ..drop.study import Model, surface_weights
    from ..geometry.cad import prepare
    from ..mesh.mesher import Sizing, build_mesh

    L, a, E, nu, rho, v0 = 100.0, 10.0, 200000.0, 0.3, 7.85e-9, 1000.0
    part = G.box(out / "bar.step", a, a, L)
    mesh = build_mesh(prepare(str(part), out / "geo"), Sizing(h=3.0, order=2, curvature=0))
    K = tet10.stiffness(mesh.X, mesh.tets, E, nu)
    m = tet10.lumped_mass(mesh.X, mesh.tets, rho)
    w = omega_max(K, np.repeat(m, 3))
    bot = mesh.tri_geometry()[2][:, 2] < 1e-6
    wgt = surface_weights(Model(mesh.X, mesh.tets, mesh.tris[bot], mesh.tri_surf[bot], K, m, 3.0, w, 2))
    nodes = np.flatnonzero(wgt > 0)
    k = np.minimum(10 * E / 3.0 * wgt[nodes], 0.5 * m[nodes] * w ** 2)
    c = np.sqrt(E / rho)
    top = np.flatnonzero(mesh.X[:, 2] > L - 1e-6)
    hist = integrate(K, m, mesh.X, Contact(np.array([0, 0, 1.0]), 0.0, nodes, k), np.array([0, 0, -v0]),
                     np.zeros(3), top, 3 * L / c, wmax=w)
    t0, t1 = hist.t_contact
    mid = (hist.t > t0 + 0.5 * L / c) & (hist.t < t0 + 1.5 * L / c)
    impulse = float(np.trapezoid(hist.force, hist.t))
    return float(hist.force[mid].mean()), float(rho * c * a * a * v0), {
        "contact_us": round(1e6 * (t1 - t0), 3), "theory_us": round(2e6 * L / c, 3), "nodes": len(mesh.X),
        "impulse_Ns": round(impulse, 6), "momentum_change_Ns": round(m.sum() * (hist.com_v[-1, 2] + v0), 6),
        "energy_drift": round(float(np.abs(hist.energy).max() / hist.kinetic0), 6)}


def clamped_disk(out: Path):
    """A steel disk 100 mm across and 1 mm thick, clamped round its edge, under 0.01 MPa, solved as a shell:
    the center deflects q a^4 / (64 D)."""
    from ..structural.shell_path import analyze_shell
    from ..structural.spec import StructuralStudy, parse_fixture, parse_load

    a, t, q, E, nu = 50.0, 1.0, 0.01, 200000.0, 0.3
    part = G._session(out / "disk.step", lambda o: o.addCylinder(0, 0, 0, 0, 0, t, a))
    st = StructuralStudy(str(part), f"custom:E={E},nu={nu},yield=250", [parse_load(f"{q} MPa @ top")],
                         [parse_fixture("cylinders")], quality="fast")
    o = analyze_shell(st, out / "run", render=False)
    D = E * t ** 3 / (12 * (1 - nu ** 2))
    return float(o.max_disp_mm), q * a ** 4 / (64 * D), {"confidence": o.confidence,
                                                          "checks": {c.name: c.status for c in o.checks}}


def cross_ply_strip(out: Path):
    """A [0/90]s T300/5208 strip, 120 x 20 mm and 1 mm thick, clamped at one end with 25 N on the other. Beam
    theory gives the moment per width at the governing point and lamination theory, for a plate in cylindrical
    bending, the first ply failure there."""
    from ..composites.laminate import clt_first_ply_failure, parse_layup
    from ..composites.study import run_laminate
    from ..composites.tools import study_from

    part = out / "strip.py"
    part.write_text('PARAMS = {"length": 120.0, "width": 20.0, "thickness": 1.0}\n\n\n'
                    "def build(occ, length, width, thickness):\n    occ.addBox(0, 0, 0, length, width, thickness)\n")
    F, L, b = 25.0, 120.0, 20.0
    st = study_from(str(part), "[0/90]s", "t300-5208", [f"{F} N +z @ x=max"], ["x=min"], ply_thickness=0.25,
                    quality="fast")
    o = run_laminate(st, out / "run", render=False)
    x = o.values["governing_xyz"][0]
    lam = parse_layup("[0/90]s", "t300-5208", 0.25)
    # near the clamp the strip cannot curl across, so it bends as a plate in cylindrical bending
    want = clt_first_ply_failure(lam, np.zeros(3), np.array([F * (L - x) / b, 0.0, 0.0]), cylindrical=True)
    return float(o.values["strength_ratio"]), float(min(want["tsai_wu"], want["hashin"])), {
        "governing_x_mm": round(x, 3), "ply": o.values["governing_ply"], "confidence": o.confidence}


__all__ = ["bar_impact", "clamped_disk", "cross_ply_strip"]
