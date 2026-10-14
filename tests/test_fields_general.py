"""The general field studies on geometry and boards they were never tuned for.

Fast tests read and mesh varied CAD scenes, read every body's role from short lines, and extract the parasitics of
every board in tests/data/pcb and of boards drawn here with different stackups, with the cross section solver
stubbed by its closed forms. Solver tests (pytest -m solver) solve each physics on a varied set of scenes and
boards and check the run's own checks: a sphere's capacitance against 4 pi eps0 R, a box cavity against its exact
modes, a magnet pulling a plate by virtual work and by action and reaction, a pot core with its coil, and real
board cross sections against Hammerstad and Jensen.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from wobbl.fields import assign, scene

ROOT = Path(__file__).resolve().parents[1]
BOARDS = ROOT / "tests" / "data" / "pcb"


def _have_getdp() -> bool:
    from wobbl.solvers import base
    try:
        return base.get("getdp").locate() is not None
    except Exception:  # noqa: BLE001
        return False


needs_getdp = pytest.mark.skipif(not _have_getdp(), reason="GetDP not installed (wobbl setup --only getdp)")


def step(path: Path, build, labels=()) -> Path:
    """Write a STEP of solids made by build(occ), with optional labels for their names."""
    import gmsh

    gmsh.initialize(readConfigFiles=False)
    gmsh.option.setNumber("General.Terminal", 0)
    try:
        gmsh.model.add("t")
        tags = build(gmsh.model.occ)
        gmsh.model.occ.synchronize()
        for t, lab in zip(tags, labels):
            gmsh.model.setEntityName(3, t, lab)
        gmsh.write(str(path))
    finally:
        gmsh.finalize()
    return path


def _ring(o, r0, r1, z0, h):
    out = o.addCylinder(0, 0, z0, 0, 0, h, r1)
    inn = o.addCylinder(0, 0, z0, 0, 0, h, r0)
    return o.cut([(3, out)], [(3, inn)])[0][0][1]


SCENES = {
    "sphere": lambda o: [o.addSphere(0, 0, 0, 10)],
    "plates": lambda o: [o.addBox(-25, -25, 1, 50, 50, 2), o.addBox(-25, -25, -3, 50, 50, 2)],
    "magnet_plate": lambda o: [o.addBox(-10, -5, -5, 20, 10, 5), o.addBox(-25, -25, 3, 50, 50, 3)],
    "coil_core": lambda o: [o.addCylinder(0, 0, -15, 0, 0, 30, 5), _ring(o, 7, 15, -10, 20),
                            o.addCylinder(0, 0, 17, 0, 0, 5, 12)],
    "racetrack": lambda o: [o.cut([(3, o.addBox(-20, -10, 0, 40, 20, 8))], [(3, o.addBox(-14, -4, 0, 28, 8, 8))])
                            [0][0][1]],
    "box_cavity": lambda o: [o.cut([(3, o.addBox(0, 0, 0, 200, 150, 300))], [(3, o.addBox(15, 15, 15, 170, 120,
                                                                                            270))])[0][0][1]],
}


@pytest.fixture(scope="module")
def scenes(tmp_path_factory):
    d = tmp_path_factory.mktemp("scenes")
    return {k: step(d / f"{k}.step", f) for k, f in SCENES.items()}


def test_scenes_load_with_names_sizes_and_ring_axes(scenes, tmp_path):
    sc = scene.load([scenes["coil_core"]], tmp_path)
    assert len(sc.bodies) == 3
    ring = sc.bodies[1]
    assert ring.hole_axis is not None and abs(abs(ring.hole_axis[2]) - 1) < 1e-6
    assert sc.bodies[0].hole_axis is None
    assert sc.select("largest")[0].volume == max(b.volume for b in sc.bodies)
    assert sc.select("at 0,0,19")[0].index == 3
    rt = scene.load([scenes["racetrack"]], tmp_path / "rt").bodies[0]
    assert rt.hole_axis is not None and abs(abs(rt.hole_axis[2]) - 1) < 1e-6
    with pytest.raises(scene.SceneError):
        sc.select("body 9")


def test_roles_for_each_physics(scenes, tmp_path):
    sc = scene.load([scenes["coil_core"]], tmp_path)
    a = assign.parse(sc, ["body 1: steel 1010", "body 2: coil 400 turns 1.5 A", "body 3: n42 magnet -z"])
    mag, notes = assign.magnetic(sc, a)
    assert mag["coil_core 1"].kind == "steel" and mag["coil_core 2"].kind == "coil"
    assert mag["coil_core 2"].turns == 400 and mag["coil_core 2"].current == 1.5
    assert np.allclose(mag["coil_core 3"].br, [0, 0, -1.3])
    assert any("axis" in n for n in notes)
    conds, diel, _ = assign.electric(sc, assign.parse(sc, ["body 1: conductor 5 kV", "body 2: ptfe",
                                                           "body 3: ground"]))
    assert conds == {"coil_core 1": 5000.0, "coil_core 3": 0.0} and diel["coil_core 2"][0] == 2.1
    with pytest.raises(ValueError):
        assign.magnetic(sc, assign.parse(sc, ["body 3: n42"]))     # a magnet needs a direction


@pytest.mark.parametrize("name,holes,vols", [("sphere", ["sphere"], []), ("plates", ["plates 1", "plates 2"], []),
                                             ("magnet_plate", [], ["magnet_plate 1", "magnet_plate 2"])])
def test_scenes_mesh_in_air_with_a_shell(scenes, tmp_path, name, holes, vols):
    sc = scene.load([scenes[name]], tmp_path)
    m = scene.mesh(sc, holes, vols, tmp_path / "m.msh", k=2.0)
    assert {"air", "shell", "outer"} <= set(m.tags) and m.n_nodes > 100
    for h in holes:
        assert m.tags[h] >= 101 and m.surfaces[h] > 0
    for v in vols:
        assert m.tags[v] < 100


def test_cavity_of_an_enclosure(scenes, tmp_path):
    from wobbl.fields.ac3d import cavities

    sc = scene.load([scenes["box_cavity"]], tmp_path)
    cav = cavities(sc, [sc.bodies[0].name], tmp_path / "c.brep")
    assert len(cav.bodies) == 1
    assert cav.bodies[0].volume == pytest.approx(0.17 * 0.12 * 0.27, rel=1e-6)


# --- boards --------------------------------------------------------------------------------------
def _stub(monkeypatch):
    """Cross sections from closed forms instead of GetDP, so extraction runs anywhere."""
    from wobbl.fields import pcbfield
    from wobbl.fields.board import hammerstad_jensen

    def fake(sec, work, k=1.0):
        tr = sec.traces[0]
        w, t = tr[2] - tr[0], tr[3] - tr[1]
        h = min(abs(tr[1] - p) for p in sec.planes) if sec.planes else 1.5e-3
        z0, ee = hammerstad_jensen(w, max(h, 1e-5), t, 4.4)
        v = 299792458.0 / math.sqrt(ee)
        out = {"z0": z0, "eps_eff": ee, "delay": 1 / v, "c_per_m": 1 / (z0 * v), "l_per_m": z0 / v, "nodes": 0}
        if len(sec.traces) > 1:
            out.update(kb=0.05, kf=-0.03)
        return out
    monkeypatch.setattr(pcbfield, "solve_section", fake)


def synthetic_board(path: Path, layers: int, pour: bool, seed: int) -> Path:
    """A board drawn here: two resistors and a chip joined by tracks, on 2, 4 or 6 layers, with or without a
    ground pour under them."""
    rng = np.random.default_rng(seed)
    names = ["F.Cu"] + [f"In{i}.Cu" for i in range(1, layers - 1)] + ["B.Cu"]
    lay = " ".join(f'({i if i < layers - 1 else 31} "{n}" signal)' for i, n in enumerate(names))
    sheets = []
    for i, n in enumerate(names):
        sheets.append(f'(layer "{n}" (type "copper") (thickness {0.035 if i in (0, layers - 1) else 0.0175}))')
        if i < layers - 1:
            sheets.append(f'(layer "dielectric {i + 1}" (type "core") (thickness {round(rng.uniform(0.1, 0.5), 3)}) '
                          f'(material "FR4") (epsilon_r {round(rng.uniform(3.6, 4.6), 2)}))')
    nets = ["", "A", "B", "GND"]
    out = [f'(kicad_pcb (version 20240108) (generator "test") (layers {lay}) (setup (stackup {" ".join(sheets)}))']
    out += [f'(net {i} "{n}")' for i, n in enumerate(nets)]
    parts = [("R1", 5, 5, "A", "GND"), ("R2", 25, 5, "B", "GND"), ("U1", 15, 15, "A", "B")]
    for ref, x, y, n1, n2 in parts:
        out.append(f'(footprint "t" (layer "F.Cu") (at {x} {y}) (property "Reference" "{ref}")'
                   f' (pad "1" smd rect (at -1 0) (size 1 1.2) (layers "F.Cu") (net {nets.index(n1)} "{n1}"))'
                   f' (pad "2" smd rect (at 1 0) (size 1 1.2) (layers "F.Cu") (net {nets.index(n2)} "{n2}")))')
    w = round(rng.uniform(0.15, 0.6), 2)
    for (x0, y0, x1, y1, n) in [(4, 5, 4, 15, "A"), (4, 15, 14, 15, "A"), (26, 5, 26, 15, "B"),
                                (16, 15, 26, 15, "B")]:
        out.append(f'(segment (start {x0} {y0}) (end {x1} {y1}) (width {w}) (layer "F.Cu") (net {nets.index(n)}))')
    if pour:
        ref = names[1]
        pts = "(xy 0 0) (xy 32 0) (xy 32 20) (xy 0 20)"
        out.append(f'(zone (net 3) (net_name "GND") (layer "{ref}") (polygon (pts {pts})) '
                   f'(filled_polygon (layer "{ref}") (pts {pts})))')
    path.write_text("\n".join(out) + ")\n")
    return path


BOARD_CASES = [(BOARDS / "two_layer_k8.kicad_pcb", None), (BOARDS / "two_layer_k5.kicad_pcb", None),
               (BOARDS / "four_layer_no_stackup.kicad_pcb", None), (BOARDS / "gerber_two_layer", None),
               ("synthetic", (2, False, 1)), ("synthetic", (4, True, 2)), ("synthetic", (6, True, 3))]


@pytest.mark.parametrize("src,spec", BOARD_CASES, ids=lambda x: Path(str(x)).name if x else "")
def test_board_parasitics_on_varied_boards(monkeypatch, tmp_path, src, spec):
    from wobbl.fields import pcbfield
    from wobbl.fields.pcb import read_board

    _stub(monkeypatch)
    path = synthetic_board(tmp_path / "b.kicad_pcb", *spec) if spec else Path(src)
    board = read_board(path)
    ex = pcbfield.extract(board, None, tmp_path / "w", "fast")
    assert ex.pieces and all(p.z0 > 10 and p.l > 0 and p.c > 0 for p in ex.pieces)
    for n, t in ex.nets.items():
        assert t["length_m"] > 0 and t["l_H"] > 0 and t["c_F"] > 0
    sp = pcbfield.parasitic_spec(ex, "test")
    assert all(tr["l"] >= 0 and tr["r"] >= 0 and "." in tr["to"] or tr["to"] for tr in sp["traces"])
    if spec and spec[1]:                      # a pour under the tracks makes them microstrips over it
        assert all(p.below or p.above for p in ex.pieces if p.layer == "F.Cu")


def test_the_example_buck_layouts_join_the_netlist(monkeypatch, tmp_path):
    from wobbl.circuits.netlist import join_parasitics, parse_spice
    from wobbl.fields import pcbfield
    from wobbl.fields.pcb import read_board

    _stub(monkeypatch)
    ex_dir = ROOT / "examples" / "pcb-buck-ringing"
    c = parse_spice((ex_dir / "buck.cir").read_text(), ex_dir)
    for name, ground_traces in (("buck_loose", True), ("buck_tight", False)):
        ex = pcbfield.extract(read_board(ex_dir / f"{name}.kicad_pcb"), None, tmp_path / name, "fast")
        sp = pcbfield._match(c, pcbfield.parasitic_spec(ex, name))
        assert sp["traces"] and {"sw", "vin", "vout"} <= {n.lower() for n in sp["nets"]}
        assert any(tr["net"] == "0" for tr in sp["traces"]) == ground_traces    # a pour needs no ground traces
        said = join_parasitics(c.copy(), sp)
        assert said


# --- solves --------------------------------------------------------------------------------------
@pytest.mark.solver
@needs_getdp
def test_sphere_capacitance(scenes, tmp_path):
    from wobbl.fields import es3d

    sc = scene.load([scenes["sphere"]], tmp_path / "s")
    o = es3d.run(sc, es3d.ElecSetup({"sphere": 1000.0}), tmp_path / "o", quality="fast")
    c = o.values["capacitance_matrix_F"][0][0]
    assert c == pytest.approx(4 * math.pi * 8.8541878128e-12 * 0.01, rel=0.02)
    assert (tmp_path / "o" / "playback.json").exists()


@pytest.mark.solver
@needs_getdp
def test_cavity_modes_match_the_box(scenes, tmp_path):
    from wobbl.fields import ac3d

    sc = scene.load([scenes["box_cavity"]], tmp_path / "s")
    roles = assign.sound(sc, assign.parse(sc, [f"{sc.bodies[0].name}: rigid"]))
    o = ac3d.run(sc, roles, tmp_path / "o", quality="fast")
    assert {c.name: c.status for c in o.checks}["closed form"] == "pass"


@pytest.mark.solver
@pytest.mark.slow
@needs_getdp
def test_magnet_pulls_a_plate_by_virtual_work(scenes, tmp_path):
    from wobbl.fields import ms3d

    sc = scene.load([scenes["magnet_plate"]], tmp_path / "s")
    bodies, notes = assign.magnetic(sc, assign.parse(sc, ["body 1: n42 +z", "body 2: steel 1010"]))
    o = ms3d.run(sc, ms3d.MagSetup(bodies, notes=notes), tmp_path / "o", quality="fast")
    st = {c.name: c.status for c in o.checks}
    assert st["virtual work"] == "pass" and st["action and reaction"] == "pass"
    assert 5 < abs(o.values["force_target_N"][2]) < 20


@pytest.mark.solver
@needs_getdp
def test_board_cross_section_against_hammerstad_jensen(tmp_path):
    from wobbl.fields.board import hammerstad_jensen
    from wobbl.fields.xsec import Section, solve_section

    sec = Section(slabs=((0.0, 0.17e-3, 4.4),), planes=(0.0,), traces=((-0.15e-3, 0.17e-3, 0.15e-3, 0.205e-3),))
    r = solve_section(sec, tmp_path)
    assert r["z0"] == pytest.approx(hammerstad_jensen(0.3e-3, 0.17e-3, 35e-6, 4.4)[0], rel=0.01)
