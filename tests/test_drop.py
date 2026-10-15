"""Drop tests: the element matrices against CalculiX, the integrator against wave theory, and a whole study."""
from __future__ import annotations

import json
import os
import subprocess

import numpy as np
import pytest

from conftest import _write, needs_ccx


def _tet10(scale=(2.0, 1.3, 0.9)):
    X = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [.5, 0, 0], [.5, .5, 0], [0, .5, 0], [0, 0, .5],
                  [.5, 0, .5], [0, .5, .5]], float) * scale
    return X + 0.05 * np.sin(np.arange(30).reshape(10, 3)), np.arange(10)[None]


@pytest.mark.solver
@needs_ccx
def test_stiffness_matches_calculix(tmp_path):
    """The numpy C3D10 stiffness of a distorted element equals the matrix CalculiX writes for it."""
    from wobbl.drop.tet10 import stiffness
    from wobbl.solvers import base as solvers

    X, T = _tet10()
    lines = ["*NODE,NSET=NALL"] + [f"{i + 1},{p[0]:.12g},{p[1]:.12g},{p[2]:.12g}" for i, p in enumerate(X)]
    lines += ["*ELEMENT,TYPE=C3D10,ELSET=E", "1," + ",".join(map(str, range(1, 11))), "*MATERIAL,NAME=M",
              "*ELASTIC", "1000,0.3", "*DENSITY", "1e-9", "*SOLID SECTION,ELSET=E,MATERIAL=M", "*STEP",
              "*FREQUENCY,SOLVER=MATRIXSTORAGE", "1", "*END STEP"]
    (tmp_path / "m.inp").write_text("\n".join(lines) + "\n")
    ccx = solvers.for_domain("structural").require()
    subprocess.run([str(ccx), "-i", "m"], cwd=tmp_path, capture_output=True, env=dict(os.environ))
    dof = (tmp_path / "m.dof").read_text().split()
    idx = [3 * (int(d.split(".")[0]) - 1) + int(d.split(".")[1]) - 1 for d in dof]
    a = np.loadtxt(tmp_path / "m.sti")
    Kc = np.zeros((len(dof), len(dof)))
    Kc[a[:, 0].astype(int) - 1, a[:, 1].astype(int) - 1] = a[:, 2]
    Kc = Kc + np.triu(Kc, 1).T
    K = stiffness(X, T, 1000.0, 0.3).toarray()[np.ix_(idx, idx)]
    assert np.abs(K - Kc).max() <= 1e-9 * np.abs(Kc).max()


def test_mass_and_rigid_motion():
    """Lumped masses add up to the element's mass, and a rigid motion strains nothing."""
    from wobbl.drop.tet10 import StressRecovery, lumped_mass, stiffness, volumes

    X, T = _tet10()
    for tets in (T, T[:, :4]):
        m = lumped_mass(X, tets, 2.0)
        assert m.sum() == pytest.approx(2.0 * volumes(X, tets).sum()) and (m[np.unique(tets)] > 0).all()
        K = stiffness(X, tets, 1000.0, 0.3)
        theta = np.array([0.01, -0.02, 0.03])
        U = np.cross(theta, X) + [1.0, 2.0, 3.0]
        assert np.abs(K @ U.ravel()).max() < 1e-9 * abs(K).max()
        assert np.abs(StressRecovery(X, tets, 1000.0, 0.3).stress(U)).max() < 5e-3   # float32 rounding of u ~ 3


def test_bar_end_impact_matches_wave_theory(tmp_path):
    """A steel bar landing on its end: the floor force is rho c A v0 and lasts 2 L / c."""
    from wobbl.drop import tet10
    from wobbl.drop.explicit import Contact, integrate, omega_max
    from wobbl.geometry.cad import prepare
    from wobbl.mesh.mesher import Sizing, build_mesh

    L, a, E, nu, rho, v0 = 100.0, 10.0, 200000.0, 0.3, 7.85e-9, 1000.0
    part = _write(tmp_path / "bar.step", lambda occ: occ.addBox(0, 0, 0, a, a, L))
    mesh = build_mesh(prepare(str(part), tmp_path / "geo"), Sizing(h=4.0, order=2, curvature=0))
    K = tet10.stiffness(mesh.X, mesh.tets, E, nu)
    m = tet10.lumped_mass(mesh.X, mesh.tets, rho)
    w = omega_max(K, np.repeat(m, 3))
    bot = mesh.tri_geometry()[2][:, 2] < 1e-6
    from wobbl.drop.study import Model, surface_weights

    model = Model(mesh.X, mesh.tets, mesh.tris[bot], mesh.tri_surf[bot], K, m, 4.0, w, 2)
    wgt = surface_weights(model)
    nodes = np.flatnonzero(wgt > 0)
    k = np.minimum(10 * E / 4.0 * wgt[nodes], 0.5 * m[nodes] * w ** 2)
    c = np.sqrt(E / rho)
    top = np.flatnonzero(mesh.X[:, 2] > L - 1e-6)
    hist = integrate(K, m, mesh.X, Contact(np.array([0, 0, 1.0]), 0.0, nodes, k), np.array([0, 0, -v0]),
                     np.zeros(3), top, 3 * L / c, wmax=w)
    t0, t1 = hist.t_contact
    assert t1 - t0 == pytest.approx(2 * L / c, rel=0.05)
    mid = (hist.t > t0 + 0.5 * L / c) & (hist.t < t0 + 1.5 * L / c)
    assert hist.force[mid].mean() == pytest.approx(rho * c * a * a * v0, rel=0.03)
    impulse = np.trapezoid(hist.force, hist.t)
    assert impulse == pytest.approx(m.sum() * (hist.com_v[-1, 2] + v0), rel=0.01)
    assert np.abs(hist.energy).max() <= 0.01 * hist.kinetic0


def test_landings_floors_and_words():
    from wobbl.drop.domain import floor_in, height_in, landing_in
    from wobbl.drop.floors import get_floor
    from wobbl.drop.study import candidates, parse_height, unique_landings

    X = np.array([[x, y, z] for x in (0, 40) for y in (0, 20) for z in (0, 10)], float)
    assert len(candidates("worst", X, np.ones(8))) == 14       # faces and corners; edges too at fine quality
    cands = candidates("all", X, np.ones(8))
    assert len(cands) == 26 and all(abs(np.linalg.norm(d) - 1) < 1e-12 for _, d in cands)
    keep, same = unique_landings(cands, [0, 1, 2])
    assert len(keep) == 7      # three faces, three edges and one corner for a box symmetric three ways
    assert len(candidates("corners", X, np.ones(8))) == 8 and len(candidates("flat", X, np.ones(8))) == 2
    assert np.allclose(candidates("1,1,-1", X, np.ones(8))[0][1], np.array([1, 1, -1]) / np.sqrt(3))
    assert parse_height("1.2") == pytest.approx(1200.0) and parse_height("4 ft") == pytest.approx(1219.2)
    assert height_in("will it survive a 1.2 m drop onto tile?") == "1.2 m"
    assert height_in("what if it falls from 80 cm") == "80 cm"
    assert floor_in("dropped onto a hardwood floor") == "hardwood" and get_floor("hardwood").key == "wood"
    assert landing_in("lands on a corner") == "corners"
    f = get_floor("layer:E=5,t=3")
    assert not f.hard and f.modulus == 5 and f.thickness == 3
    with pytest.raises(KeyError):
        get_floor("lava")


def test_flat_block_drop_study(tmp_path):
    """A block landing flat on concrete behaves like a short bar: its peak stress is about rho c v0."""
    from wobbl.drop.study import DropStudy, run_drop
    from wobbl.materials import resolve_material

    part = tmp_path / "block.py"
    part.write_text('PARAMS = {"a": 40.0, "b": 20.0, "c": 10.0}\n\n\n'
                    "def build(occ, a, b, c):\n    occ.addBox(0, 0, 0, a, b, c)\n")
    o = run_drop(DropStudy(str(part), "pc", "1 m", "-z", "concrete", quality="fast"), tmp_path / "out", render=False)
    mat = resolve_material("pc").internal()
    v0 = np.sqrt(2 * 9810.0 * 1000.0)
    expect = mat["density"] * np.sqrt(mat["E"] / mat["density"]) * v0
    assert o.values["governing_MPa"] == pytest.approx(expect, rel=0.25)
    checks = {c.name: c.status for c in o.checks}
    assert checks["momentum"] == "pass" and checks["energy"] == "pass"
    doc = json.loads((tmp_path / "out" / "playback.json").read_text())
    assert doc["kind"] == "drop" and (tmp_path / "out" / "playback.json").stat().st_size < 300_000
    assert len(doc["frames"]["time_s"]) >= 10 and doc["fall"]["down"] == [0.0, 0.0, -1.0]


def test_budget_estimate_scales_like_an_explicit_march():
    """The cost is steps times matrix work: twice the impact time doubles it, and a shorter step lengthens it."""
    from wobbl.drop import budget as B

    a = B.study_cost(20_000, 4e6, 1e7, 1e-4, 6, 2)
    b = B.study_cost(20_000, 4e6, 1e7, 2e-4, 6, 2)
    assert a.steps == pytest.approx(1e-4 / B.stable_dt(1e7), abs=1)
    assert 1.4 < b.runs["quadratic"] / a.runs["quadratic"] < 2.0      # the stress frames do not grow with time
    c = B.study_cost(20_000, 4e6, 2e7, 1e-4, 6, 2)          # a mesh twice as stiff steps half as long
    assert c.steps == pytest.approx(2 * a.steps, abs=2)
    assert B.study_cost(20_000, 4e6, 1e7, 1e-4, 1, 1, refine=False).seconds < a.seconds
    assert list(a.runs) == ["setup", "screen", "quadratic", "refined"]


def test_budget_coarsens_the_mesh_and_says_so(tmp_path):
    """A plate with a hole over a tiny budget is meshed with fewer elements round the hole and more mass scaling,
    and the budget is reported; with a generous budget the first mesh is kept."""
    from wobbl.drop import budget as B
    from wobbl.drop.floors import get_floor
    from wobbl.drop.study import DropStudy, fit_mesh
    from wobbl.materials import resolve_material
    from wobbl.structural.run import describe_part

    part = tmp_path / "plate.py"
    part.write_text('PARAMS = {"a": 60.0, "b": 40.0, "c": 4.0}\n\n\n'
                    "def build(occ, a, b, c):\n    box = occ.addBox(0, 0, 0, a, b, c)\n"
                    "    hole = occ.addCylinder(a / 2, b / 2, -1, 0, 0, c + 2, 6)\n    occ.cut([(3, box)], [(3, hole)])\n")
    mat = resolve_material("pc").internal()
    prep, feats = describe_part(str(part), tmp_path / "w", 1.0, None, {})
    h = 3.0
    floor = get_floor("tile")
    roomy = DropStudy(str(part), "pc", "0.8 m", "-z", "tile", quality="fast", budget=1e6)
    q, *_, rung, cost = fit_mesh(roomy, prep, feats, mat, floor, h)
    assert rung == B.RUNGS[0] and "left as meshed" in cost.fit
    tight = DropStudy(str(part), "pc", "0.8 m", "-z", "tile", quality="fast", budget=1.0)
    q2, *_, rung2, cost2 = fit_mesh(tight, prep, feats, mat, floor, h)
    assert rung2 == B.RUNGS[-1] and len(q2.X) < len(q.X)
    assert "coarsened" in cost2.fit and "still over the budget" in cost2.fit
    assert cost2.seconds < cost.seconds
    held = DropStudy(str(part), "pc", "0.8 m", "-z", "tile", quality="fast", budget=1.0, mesh_size=h)
    *_, rung3, cost3 = fit_mesh(held, prep, feats, mat, floor, h)
    assert rung3 == B.RUNGS[0] and "was set" in cost3.fit


def test_estimate_prices_a_drop_from_its_mesh(tmp_path):
    """wobbl estimate prices a drop from the mesh it would march."""
    from wobbl.jobs import workload as W

    part = tmp_path / "block.py"
    part.write_text('PARAMS = {"a": 60.0, "b": 40.0, "c": 10.0}\n\n\n'
                    "def build(occ, a, b, c):\n    occ.addBox(0, 0, 0, a, b, c)\n")
    w = W.Workload([], "drop", {"part": str(part), "quality": "fast", "budget": 1e6}, [part], None, [str(part)])
    c = W.drop_cost(w)
    assert c.solver == "wobbl drop" and c.core_seconds > 0 and c.size["nodes"] > 100
    assert "time step" in c.basis[0] and "budget" in c.basis[0]
    assert W.REGISTRY["drop"].cost is W.drop_cost
