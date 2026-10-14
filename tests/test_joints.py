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
