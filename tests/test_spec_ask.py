import numpy as np
import pytest

from wobbl.ask import detect_domain, find_material
from wobbl.structural.spec import (FIXTURE_ALIASES, FIXTURE_KINDS, LOAD_ALIASES, SpecError, StructuralStudy,
                                   parse_fixture, parse_load)


def test_force_load():
    ld = parse_load("200 N down @ top")
    assert ld.kind == "force" and ld.magnitude == 200 and ld.direction == [0, 0, -1] and ld.faces == "top"


def test_imperial_and_pressure():
    ld = parse_load("50 lbf -z @ x=max")
    assert ld.imperial and ld.magnitude == pytest.approx(222.411, rel=1e-4)
    p = parse_load("0.2 MPa @ top")
    assert p.kind == "pressure" and p.direction is None


def test_remote_rigid_and_moment():
    ld = parse_load("2000 N -0.866,0,-0.5 at 0,-30,66 @ holes d=6.6 rigid")
    assert ld.kind == "remote" and ld.coupling == "rigid" and ld.point == [0, -30, 66] and ld.faces == "holes d=6.6"
    m = parse_load("0.8 N*m +z at 210,0,25 @ top and within 14 of 210,0,25")
    assert m.kind == "moment" and m.magnitude == pytest.approx(800)
    assert m.faces == "top and within 14 of 210,0,25"


def test_gravity_has_no_faces():
    g = parse_load("3 g down")
    assert g.kind == "gravity" and g.faces == ""


@pytest.mark.parametrize("bad", ["200 @ top", "200 N @ top", "200 N down", "5 parsecs @ top"])
def test_bad_loads(bad):
    with pytest.raises(SpecError):
        parse_load(bad)


def test_fixtures():
    assert parse_fixture("holes d=4.5").kind == "fixed"
    assert parse_fixture("bolt holes d=6.6").kind == "bolt"
    assert parse_fixture("wall x=min").kind == "wall"
    assert parse_fixture("roller y=0").kind == "roller"


@pytest.mark.parametrize("word", sorted(w for w in {*FIXTURE_KINDS, *FIXTURE_ALIASES}
                                         if FIXTURE_ALIASES.get(w, w) != "elastic"))
def test_every_fixture_word_is_read(word):
    """Each support kind and every alias in the map starts a fixture string, in any case and with a colon. An
    elastic support also needs its stiffness, so tests/test_elastic_support.py reads those words."""
    want = FIXTURE_ALIASES.get(word, word)
    for text in (f"{word} x=min", f"{word.upper()}: x=min", f"{word} @ x=min"):
        fx = parse_fixture(text)
        assert (fx.kind, fx.faces) == (want, "x=min"), text


def test_clamp_reads_as_fixed():
    fx = parse_fixture("clamp x=min")
    assert fx.kind == "fixed" and fx.faces == "x=min" and fx.dofs == "xyz"
    fx = parse_fixture("clamped-z x=max")
    assert fx.kind == "fixed" and fx.dofs == "z" and fx.text == "fixed-z x=max"


@pytest.mark.parametrize("alias", sorted(LOAD_ALIASES))
def test_every_load_alias_is_read(alias):
    sample = {"moment": "0.8 N*m +z @ top", "bolt": "300 N -x @ hole d=6.6", "gravity": "3 g down"}
    want = LOAD_ALIASES[alias]
    assert parse_load(sample[want], kind=alias).kind == want


@pytest.mark.parametrize("text", ["300 N -x bearing @ hole d=6.6", "300 N -x @ hole d=6.6 bearing",
                                  "bearing 300 N -x @ hole d=6.6", "300 N -x Bearing on hole d=6.6"])
def test_bearing_keyword_makes_a_bolt_load(text):
    ld = parse_load(text)
    assert (ld.kind, ld.magnitude, ld.direction, ld.faces) == ("bolt", 300, [-1, 0, 0], "hole d=6.6")
    assert ld.text == text


def test_force_into_holes_without_bearing_is_unchanged():
    ld = parse_load("300 N -x @ hole d=6.6")
    assert ld.kind == "force" and ld.faces == "hole d=6.6"
    assert parse_load({"kind": "bearing", "load": "300 N", "direction": "-x", "faces": "hole d=6.6"}).kind == "bolt"


@pytest.mark.parametrize("bad", ["0.2 MPa bearing @ hole d=6.6", "300 N -x at 0,0,10 bearing @ hole d=6.6",
                                 "3 g down bearing"])
def test_bearing_only_goes_with_a_force(bad):
    with pytest.raises(SpecError, match="bearing"):
        parse_load(bad)


def test_cli_reads_a_bearing_load():
    from wobbl.cli.main import _study_from

    st = _study_from("plate.step", "", {"loads": ["300 N -x bearing @ hole d=8", "50 N down @ top"],
                                        "fixes": ["clamp x=min"]})
    assert [ld.kind for ld in st.loads] == ["bolt", "force"] and st.fixtures[0].kind == "fixed"


def test_bearing_load_presses_on_the_loaded_side_only(parts, tmp_path):
    """Built on a real mesh, a bearing load pushes only on the half of the hole it faces, with the same total
    force, while the plain force string still spreads over the whole hole."""
    from wobbl.geometry.select import Selector
    from wobbl.materials import resolve_material
    from wobbl.mesh.mesher import Sizing, build_mesh
    from wobbl.structural.build import build_static
    from wobbl.structural.run import describe_part

    prep, feats = describe_part(parts["plate_hole"], tmp_path)
    mesh = build_mesh(prep, Sizing(h=4.0))
    sel = Selector(feats.faces, feats.bbox)
    fix = parse_fixture("x=max")
    far_side = {}
    for text in ("300 N -x bearing @ hole d=8", "300 N -x @ hole d=8"):
        ld = parse_load(text)
        built = build_static(mesh, [(ld, sel.select(ld.faces))], [(fix, sel.select(fix.faces))],
                             resolve_material("6061-t6"), feats.faces)
        F = built.model.forces
        assert F.sum(0) == pytest.approx([-300, 0, 0], abs=1e-6)
        f = np.linalg.norm(F, axis=1)
        # share of the load on the +x half of the hole (axis at x = 50), past one element edge of the middle
        far_side[text] = f[mesh.X[:, 0] > 51.0].sum() / f.sum()
    assert far_side["300 N -x bearing @ hole d=8"] < 0.01
    assert far_side["300 N -x @ hole d=8"] > 0.3


def test_study_roundtrip():
    st = StructuralStudy("p.step", "pla", [parse_load("10 N down @ top")], [parse_fixture("bottom")])
    st2 = StructuralStudy.from_dict(st.to_dict())
    assert st2.loads[0].magnitude == 10 and st2.fixtures[0].faces == "bottom"


def test_question_reading():
    assert detect_domain("will this LED fry on 5V?") == "circuit"
    assert detect_domain("will the battery pack overheat at 12 W?") == "thermal"
    assert detect_domain("will this bracket crack under 200 N?") == "strength"
    assert find_material("a 3D printed PETG mount") == "petg"
    assert find_material("6061 aluminum arm") == "6061-t6"
    assert find_material("a printed part") == "pla"
