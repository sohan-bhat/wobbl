import pytest

from wobbl.ask import detect_domain, find_material
from wobbl.structural.spec import SpecError, StructuralStudy, parse_fixture, parse_load


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
