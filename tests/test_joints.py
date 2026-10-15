"""Bolted joints and contact without a solver: the VDI 2230 hand calculation, bolt sizes and the assembly reader."""
import math

import numpy as np
import pytest

from wobbl.joints import vdi2230 as V


@pytest.fixture(scope="module")
def stack(tmp_path_factory):
    """A block resting on a base, and a pin in a plate's hole with 0.02 mm of radial clearance."""
    from conftest import _write

    def build(occ):
        occ.addBox(0, 0, 0, 60, 40, 10)
        occ.addBox(15, 10, 10, 30, 20, 10)
        b = occ.addBox(70, 0, 0, 30, 30, 10)
        h = occ.addCylinder(85, 15, -1, 0, 0, 12, 4)
        occ.cut([(3, b)], [(3, h)])
        occ.addCylinder(85, 15, 0, 0, 0, 10, 3.98)

    d = tmp_path_factory.mktemp("stack")
    return _write(d / "stack.step", build), d


def test_assembly_keeps_bodies_apart_and_pairs_the_faces_that_touch(stack):
    from wobbl.joints.assembly import build_assembly_mesh, prepare_assembly
    from wobbl.mesh.mesher import Sizing

    path, d = stack
    asm = prepare_assembly(path, d / "work")
    assert asm.n_bodies == 4
    f = asm.features.faces
    by = {(p.body_slave, p.body_master): p for p in asm.pairs}
    assert set(by) == {(2, 1), (4, 3)}
    block = by[(2, 1)]
    # each body's faces point out of that body, even where it presses on another
    assert f[block.slave].normal == pytest.approx([0, 0, -1]) and f[block.master].normal == pytest.approx([0, 0, 1])
    assert block.area == pytest.approx(600.0) and block.gap == pytest.approx(0.0, abs=1e-9)
    assert by[(4, 3)].gap == pytest.approx(0.02, abs=1e-4)          # read from the radii, not the facets
    mesh = build_assembly_mesh(asm, Sizing(h=6.0))
    assert set(np.unique(mesh.body)) == {1, 2, 3, 4}
    # the faces where bodies meet are on the boundary of both, each with its own nodes
    for p in asm.pairs:
        a, b = mesh.surface_nodes([p.slave]), mesh.surface_nodes([p.master])
        assert len(a) and len(b) and not np.intersect1d(a, b).size


def test_faces_are_picked_body_by_body(stack):
    from wobbl.joints.assembly import prepare_assembly
    from wobbl.joints.setup import parse_move, select

    path, d = stack
    asm = prepare_assembly(path, d / "work2")
    top2 = select(asm, "body 2 and top").faces
    assert len(top2) == 1 and asm.features.faces[top2[0]].centroid[2] == pytest.approx(20.0)
    assert select(asm, "top").faces != top2 or len(select(asm, "top").faces) == 1
    m = parse_move("0.05 mm +x @ body 2 and x=min")
    assert m.vector == pytest.approx([0.05, 0, 0]) and m.faces == "body 2 and x=min"


@pytest.mark.parametrize("size,area", [("M6", 20.1), ("M8", 36.6), ("M10", 58.0), ("M12", 84.3), ("M16", 157.0)])
def test_stress_areas_match_iso_898(size, area):
    assert V.bolt_from(f"{size} 8.8").stress_area == pytest.approx(area, rel=0.005)


def test_bolt_text_reads_size_pitch_and_class():
    b = V.bolt_from("M10x1.25 10.9")
    assert (b.d, b.pitch, b.grade, b.proof, b.dw, b.hole) == (10, 1.25, "10.9", 830, 14.63, 11.0)
    assert V.bolt_from("M20 8.8").proof == 600          # 8.8 above M16
    assert V.bolt_from("m6 a2-70").E == 193000
    with pytest.raises(ValueError):
        V.bolt_from("M7 8.8")


def _cone_by_integration(dw, hole, clamp, outer, E, t):
    """Two frustums from the head and nut toward the middle, then a sleeve once they reach the outside."""
    z = np.linspace(0, clamp / 2, 200001)
    D = np.minimum(dw + 2 * z * t, outer)
    f = 1.0 / (E * math.pi / 4 * (D ** 2 - hole ** 2))
    return 2 * float(np.sum(0.5 * (f[1:] + f[:-1]) * np.diff(z)))


@pytest.mark.parametrize("outer", [40.0, 22.0])
def test_plate_compliance_is_the_cone_integrated(outer):
    """A wide part gets the full cones, a narrow one cones and a sleeve: both match the integral of dz / (E A)."""
    dp, t, limit = V.plate_compliance(15.0, 11.0, 30.0, outer, 205000.0)
    assert t == pytest.approx(0.362 + 0.153 * math.log(outer / 15.0))
    assert (outer >= limit) == (outer == 40.0)
    assert dp == pytest.approx(_cone_by_integration(15.0, 11.0, 30.0, outer, 205000.0, t), rel=1e-6)


def test_bolt_compliance_adds_head_thread_and_nut():
    b = V.bolt_from("M10 8.8")
    an, a3 = math.pi * 25, math.pi / 4 * (10 - 1.226869 * 1.5) ** 2
    want = (5 / an + 30 / a3 + 5 / a3 + 4 / an) / 205000
    assert V.bolt_compliance(b, 30.0) == pytest.approx(want)


def test_joint_check_shares_the_load_and_opens_at_fv_over_one_minus_phi():
    b = V.bolt_from("M10 8.8", dw=15.0)
    c = V.check(b, 30.0, 30.0, 205000.0, 20000.0, axial=10000.0, shear=1000.0, mu=0.15)
    assert c.phi == pytest.approx(c.delta_P / (c.delta_S + c.delta_P))
    assert c.bolt_force == pytest.approx(20000 + c.phi * 10000)
    assert c.clamp_force == pytest.approx(20000 - (1 - c.phi) * 10000)
    assert c.separation_load == pytest.approx(20000 / (1 - c.phi))
    assert c.slip_safety == pytest.approx(0.15 * c.clamp_force / 1000)
    assert c.strip_safety > 1                       # a matching nut outlasts its bolt
    # VDI's torque for the preload with mu = 0.12 lands near the nut factor rule T = 0.2 F d
    assert V.torque_for(b, 20000) == pytest.approx(0.2 * 20000 * 10, rel=0.25)


def test_bolt_specs_are_optional_and_read_in_any_order():
    from wobbl.joints.bolts import parse_bolt, size_for_hole

    s = parse_bolt("M8 8.8 @ holes d=9 torque 25 N*m")
    assert (s.size, s.grade, s.faces, s.torque) == ("M8", "8.8", "holes d=9", 25000.0)
    s = parse_bolt("10.9 preload 30 kN")
    assert (s.size, s.grade, s.faces, s.preload) == ("", "10.9", None, 30000.0)
    assert parse_bolt("").grade == "8.8" and parse_bolt("washer 16 @ x>40").washer == 16.0
    assert [size_for_hole(d) for d in (3.4, 6.6, 9.0, 10.0, 13.5, 22.0)] == [3, 6, 8, 8, 12, 20]


@pytest.mark.parametrize("name", ["tapped_block", "flange_pair"])
def test_bolts_are_found_from_the_holes(name, tmp_path):
    from wobbl.joints.assembly import prepare_assembly
    from wobbl.joints.bolts import bolt_lines, parse_bolt
    from wobbl.validation import generality as G

    asm = prepare_assembly(G._step(tmp_path / "a.step", getattr(G, name)), tmp_path / "work")
    if name == "tapped_block":
        lines = bolt_lines(asm, [])
        assert [ln.bolt.name for ln in lines] == ["M8 8.8", "M8 8.8"]
        # a 9 mm hole in the 10 mm plate over a 6.8 mm hole tapped 20 mm deep in the block
        assert all(ln.tapped and ln.bodies == [2, 1] and ln.clamp == pytest.approx(10) for ln in lines)
        assert lines[0].engagement == pytest.approx(8.0) and lines[0].head[2] == pytest.approx(40)
        assert lines[0].preload(0.2)[0] == pytest.approx(0.7 * 580 * 36.6, rel=0.01)
    else:
        lines = bolt_lines(asm, [parse_bolt("M8 8.8 torque 20 N*m")])
        assert len(lines) == 4 and all(ln.clamp == pytest.approx(24) for ln in lines)
        assert lines[0].preload(0.2)[0] == pytest.approx(20000 / (0.2 * 8))
        assert any("40 mm hole" in n and "bore" in n for n in asm.prep.notes)


def test_bolts_are_read_from_plain_words():
    from wobbl.joints.bolts import bolt_from_words

    assert bolt_from_words("Do four M8 bolts at 20 N m keep it closed?") == "M8 torque 20 N*m"
    assert bolt_from_words("two M10 10.9 screws torqued to 45 Nm") == "M10 10.9 torque 45 N*m"
    assert bolt_from_words("M6 bolts preloaded to 8 kN") == "M6 preload 8 kN"
    assert bolt_from_words("M12x1.5 8.8 at 30 lbf ft") == "M12x1.5 8.8 torque 30 lbf*ft"
    assert bolt_from_words("will it hold 200 N?") == ""


# ----------------------------------------------------------------------------------------- unfamiliar assemblies
# Five assemblies built here that no default was tuned for. Each is found, meshed and solved the way a user's would be.
def lapped_plates(o):
    a = o.addBox(0, 0, 0, 70, 30, 6)
    b = o.addBox(40, 0, 6, 70, 30, 6)
    for body, z in ((a, -1), (b, 5)):
        for x in (50, 62):
            o.cut([(3, body)], [(3, o.addCylinder(x, 15, z, 0, 0, 8, 3.3))])


def six_bolt_flanges(o):
    for k in range(2):
        d = o.addCylinder(0, 0, 10 * k, 0, 0, 10, 50)
        o.cut([(3, d)], [(3, o.addCylinder(0, 0, 10 * k - 1, 0, 0, 12, 15))])
        for j in range(6):
            a = math.pi / 6 + j * math.pi / 3
            o.cut([(3, d)], [(3, o.addCylinder(38 * math.cos(a), 38 * math.sin(a), 10 * k - 1, 0, 0, 12, 5.5))])


def shaft_clamp(o):
    """A split collar on a square shaft along y, its halves meeting on the plane z = 0 and drawn together by two
    bolts. The shaft has flats so that the clamp and the faces that touch it can be meshed to match."""
    o.addBox(-10, 0, -10, 20, 80, 20)
    for z0, z1, p0, p1 in ((-20.0, 0.0, -10.0, 0.0), (0.0, 20.0, 0.0, 10.0)):
        h = o.addBox(-20, 25, z0, 40, 30, z1 - z0)
        o.cut([(3, h)], [(3, o.addBox(-10, 24, p0, 20, 32, p1 - p0))])
        for x in (-15, 15):
            o.cut([(3, h)], [(3, o.addCylinder(x, 40, z0 - 1, 0, 0, z1 - z0 + 2, 3.3))])


def press_fit_hub(o):
    """A hub of 22 mm outer radius pressed on a shaft 0.01 mm larger in radius than its bore."""
    o.addCylinder(0, 0, 0, 0, 0, 70, 10.01)
    hub = o.addCylinder(0, 0, 20, 0, 0, 25, 22)
    o.cut([(3, hub)], [(3, o.addCylinder(0, 0, 19, 0, 0, 27, 10.0))])


def wall_bracket(o):
    """A bracket whose flange is bolted to a wall plate by four bolts along y, with a shelf out of its top edge."""
    wall = o.addBox(0, 0, 0, 100, 10, 80)
    flange = o.addBox(20, 10, 15, 60, 8, 50)
    shelf = o.addBox(20, 18, 57, 60, 50, 8)
    br, _ = o.fuse([(3, flange)], [(3, shelf)])
    for x in (35, 65):
        for z in (25, 45):
            o.cut([(3, wall)], [(3, o.addCylinder(x, -1, z, 0, 12, 0, 4.5))])
            o.cut(br, [(3, o.addCylinder(x, 9, z, 0, 10, 0, 4.5))])


ASSEMBLIES = [
    dict(id="lapped-plates", build=lapped_plates, kind="bolts", material="mild-steel", fix=["body 1 and x=min"],
         load=["1.5 kN +x @ body 2 and x=max"], bolt=["M6 8.8 torque 8 N*m"], bodies=2, pairs=1,
         names={"bolt strength", "separation", "slip", "force balance"}),
    dict(id="six-bolt-flanges", build=six_bolt_flanges, kind="bolts", material="mild-steel",
         fix=["body 1 and bottom"], load=["20 kN +z @ body 2 and top"], bolt=["M10 8.8 torque 30 N*m"], bodies=2,
         pairs=1, names={"bolt strength", "separation", "force balance"}),
    dict(id="shaft-clamp", build=shaft_clamp, kind="bolts", material="mild-steel", fix=["body 2 and z=min"],
         load=["600 N +y @ body 1 and y=max"], bolt=["M6 8.8 torque 8 N*m"], bodies=3, pairs=3,
         names={"bolt strength", "separation", "slip", "force balance"}),
    dict(id="press-fit-hub", build=press_fit_hub, kind="contact", material="mild-steel", fix=["body 1 and z=min"],
         load=["2 kN +z @ body 2 and top"], bodies=2, pairs=1, names={"force balance"}),
    dict(id="wall-bracket", build=wall_bracket, kind="bolts", material="6061-t6", fix=["body 1 and y=min"],
         load=["1 kN -z @ body 2 and y=max"], bolt=["M8 8.8 torque 20 N*m"], bodies=2, pairs=1,
         names={"bolt strength", "separation", "force balance"}),
]


@pytest.mark.slow
@pytest.mark.parametrize("case", ASSEMBLIES, ids=[c["id"] for c in ASSEMBLIES])
def test_unfamiliar_assemblies_are_found_solved_and_checked(case, tmp_path):
    from wobbl.joints.assembly import prepare_assembly
    from wobbl.joints.bolts import BoltStudy, run_bolts
    from wobbl.joints.contact import ContactStudy, run_contact
    from wobbl.validation import generality as G

    name = case["id"]
    part = G._step(tmp_path / f"{name}.step", case["build"])
    asm = prepare_assembly(part, tmp_path / "pre")
    assert asm.n_bodies == case["bodies"] and len(asm.pairs) >= case["pairs"]
    if case["kind"] == "bolts":
        o = run_bolts(BoltStudy(str(part), case["material"], case["bolt"], case["load"], case["fix"], quality="fast",
                                budget=900), tmp_path / "run", render_images=False)
        assert len(o.values["bolts"]) >= 2 and o.values["least_clamp_N"] >= 0 and o.values["proof_use"] > 0
    else:
        o = run_contact(ContactStudy(str(part), case["material"], case["load"], case["fix"], case.get("move", []),
                                     0.2, quality="fast", budget=900), tmp_path / "run", render_images=False)
        assert o.values["pairs"] and o.values["pairs"][0]["normal_force_N"] > 0
    assert len(o.values["levels"]) == 2 and o.values["budget"]["predicted_nodes"][1] > 0
    names = {c.name for c in o.checks}
    assert case["names"] <= names, names
    assert [c.name for c in o.checks if c.status == "fail" and c.name == "force balance"] == []
    assert o.verdict and o.headline and o.confidence
    if name == "press-fit-hub":
        # Lame: p = E d (1 - a^2/b^2) / (2 a) for a solid shaft in a hub of the same steel, over the bore
        pressing = 210000 * 0.01 * (1 - (10 / 22) ** 2) / 20 * math.pi * 20 * 25
        pair = o.values["pairs"][0]
        assert 0.3 * pressing < pair["normal_force_N"] < 4.0 * pressing
        assert pair["friction_use"] < 1.0
    print(name, o.verdict, o.headline)


# ------------------------------------------------------------------------------------------------- time budget
def _fake_world():
    """An assembly of a fixed volume whose 'mesh' has a node count that follows the element size and the number of
    elements round a circle, so the budget can be tested without gmsh or a solver."""
    from types import SimpleNamespace

    from wobbl.mesh.mesher import Sizing

    asm = SimpleNamespace(prep=SimpleNamespace(volume=3.0e5))

    def make(h, refine=0.75, curv=16):
        return [Sizing(h=h * refine ** k, curvature=curv) for k in range(2)]

    def mesh_fn(_asm, sizing):
        return SimpleNamespace(n_nodes=int(1.45 * 3.0e5 * 6.5 / sizing.h ** 3 * 1.1 + 40 * sizing.curvature))

    return asm, make, mesh_fn


def test_a_mesh_is_coarsened_until_two_levels_fit_the_budget():
    from wobbl.joints import budget as B

    asm, make, mesh_fn = _fake_world()
    loose = B.fit(asm, make, 10.0, None, 1e9, 60, 1, mesh_fn=mesh_fn)
    tight = B.fit(asm, make, 10.0, None, 600.0, 60, 1, mesh_fn=mesh_fn)
    assert loose.rung_index == 0 and loose.h[0] == pytest.approx(10.0) and not loose.notes
    assert tight.rung_index > 0 and tight.h[0] > loose.h[0] and tight.nodes[0] < loose.nodes[0]
    assert tight.predicted <= 600.0 and tight.mesh0.n_nodes == tight.nodes[0]
    assert "coarsened" in tight.notes[0]
    assert tight.timeout() >= 3 * 600.0 - tight.spent


def test_a_mesh_size_the_user_sets_is_kept_and_only_reported():
    from wobbl.joints import budget as B

    asm, make, mesh_fn = _fake_world()
    plan = B.fit(asm, make, 10.0, 6.0, 100.0, 60, 1, mesh_fn=mesh_fn)
    assert plan.h[0] == 6.0 and plan.rung_index == 0 and plan.user_size
    assert plan.predicted > 100.0 and "was kept" in plan.notes[0]


def test_the_second_level_is_made_less_fine_when_the_first_was_slow():
    from wobbl.joints import budget as B

    asm, make, mesh_fn = _fake_world()
    plan = B.fit(asm, make, 10.0, None, 1e9, 60, 1, mesh_fn=mesh_fn)
    fast = B.second_level(plan, asm, make, 5.0, plan.nodes[0], 60, mesh_fn)
    n_fast = fast.n_nodes
    plan.budget, plan.spent = 1200.0, 0.0
    slow = B.second_level(plan, asm, make, 600.0, plan.nodes[0], 60, mesh_fn)
    assert slow.n_nodes < n_fast and plan.h[1] > 0.75 * plan.h[0] - 1e-9


@pytest.mark.parametrize("physics", ["bolts", "contact"])
def test_simulate_accepts_a_budget_and_a_mesh_size_for_joints(physics):
    from wobbl.simulate import registry

    registry.load()
    a = registry.get("part", physics)
    assert "budget" in {o.name for o in a.options}
    assert "mesh-size" in {o.name for o in registry.domain("part").options}


def test_round_faces_that_press_on_each_other_keep_their_elements_round_the_circle():
    from types import SimpleNamespace

    from wobbl.joints import budget as B

    asm, make, mesh_fn = _fake_world()
    asm.pairs = [SimpleNamespace(slave=1, master=2)]
    asm.features = SimpleNamespace(faces={1: SimpleNamespace(radius=10.0), 2: SimpleNamespace(radius=10.0)})
    plan = B.fit(asm, make, 10.0, None, 100.0, 60, 1, mesh_fn=mesh_fn)
    assert plan.curv >= B.CURVED_CONTACT and plan.rung_index > 0
    flat = _fake_world()[0]
    flat.pairs = [SimpleNamespace(slave=1, master=2)]
    flat.features = SimpleNamespace(faces={1: SimpleNamespace(radius=None), 2: SimpleNamespace(radius=None)})
    assert B.fit(flat, make, 10.0, None, 100.0, 60, 1, mesh_fn=mesh_fn).curv < B.CURVED_CONTACT
