from pathlib import Path

import pytest
from click.testing import CliRunner

from wobbl.circuits.tools import read_circuit_question
from wobbl.cli.main import cli
from wobbl.fix.apply import FIX_KINDS, check_kinds, material_swap_resolve
from wobbl.geometry.parametric import build_step, load_params, radius_param, thickness_param
from wobbl.geometry.remap import remap_selector, same_surface
from wobbl.materials import resolve_material
from wobbl.structural.run import describe_part
from wobbl.structural.spec import Load, StructuralStudy, parse_fixture, parse_load
from wobbl.thermal.tools import read_thermal_question

PART = '''
PARAMS = {"thickness": 5.0, "fillet": 1.0}

def build(occ, thickness, fillet):
    occ.addBox(0, 0, 0, 60, 20, thickness)
'''


@pytest.fixture()
def part(tmp_path):
    p = tmp_path / "plate.py"
    p.write_text(PART)
    return p


def test_parametric_build_and_override(part, tmp_path):
    assert load_params(part) == {"thickness": 5.0, "fillet": 1.0}
    step, used = build_step(part, {"thickness": "8"}, tmp_path / "geo")
    assert used["thickness"] == 8.0 and Path(step).exists()
    with pytest.raises(ValueError):
        build_step(part, {"nope": 1}, tmp_path / "geo")
    assert thickness_param(used) == "thickness" and radius_param(used) == "fillet"


def test_describe_parametric_part(part, tmp_path):
    prep, feats = describe_part(str(part), tmp_path / "w", params={"thickness": 7})
    assert prep.size[2] == pytest.approx(7, rel=1e-3)
    assert any("Built from plate.py" in n for n in prep.notes)


def test_remap_keeps_meaningful_selectors(part, tmp_path):
    _, a = describe_part(str(part), tmp_path / "a")
    _, b = describe_part(str(part), tmp_path / "b", params={"thickness": 9})
    assert remap_selector("x=max", a.faces, b.faces, a.bbox) == "x=max"
    top_a = max(a.faces.values(), key=lambda f: f.centroid[2])
    out = remap_selector(f"face {top_a.id}", a.faces, a.faces, a.bbox)
    assert out == f"faces {top_a.id}"
    assert same_surface(top_a, top_a, 1e-6)


def test_fix_help_names_every_kind_it_applies():
    text = " ".join(CliRunner().invoke(cli, ["fix", "--help"]).output.split())
    assert "fillet, thicken, gusset and material" in text
    assert "thickness and fillet parameters" in text and "any parameter" not in text
    assert check_kinds(["Thicken ", "fillet"]) == ["thicken", "fillet"] and check_kinds(None) is None
    assert set(FIX_KINDS) == {"fillet", "thicken", "gusset", "material"}


def test_fix_refuses_an_unknown_kind(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    r = CliRunner().invoke(cli, ["fix", "--apply", "--only", "fillet,bogus"])
    assert r.exit_code != 0
    assert "cannot apply bogus, pick from fillet, thicken, gusset, material" in " ".join(r.output.split())


def _study(*loads, print_direction=None):
    return StructuralStudy("p.step", "6061-t6", [parse_load(x) for x in loads], [parse_fixture("x=min")],
                           print_direction=print_direction)


@pytest.mark.parametrize("old,new,loads,layers,why", [
    ("6061-t6", "7075-t6", ["200 N down @ top"], None, ""),
    ("6061-t6", "mild-steel", ["0.2 MPa @ top", "300 N -x bearing @ hole d=6", "2 kN -z at 0,0,50 @ top",
                               "1 N*m +z @ top"], None, ""),          # Poisson 0.33 against 0.29 is close enough
    ("petg", "abs", ["20 N down @ top"], None, ""),
    ("6061-t6", "6061-t6:nu=0.2", ["200 N down @ top"], None, "Poisson's ratio of 0.2 differs from 0.33"),
    ("6061-t6", "7075-t6", ["200 N down @ top", "3 g down"], None, "self weight grows with density"),
    ("pla", "petg", ["20 N down @ top"], None, "judged by von Mises vs yield instead of max principal"),
    ("petg", "abs", ["20 N down @ top"], "z", "across print layers"),
])
def test_material_swap_scales_only_when_exact(old, new, loads, layers, why):
    reason = material_swap_resolve(_study(*loads, print_direction=layers), resolve_material(old),
                                   resolve_material(new))
    assert (reason == "") if not why else (why in reason), reason


def test_material_swap_solves_again_with_a_prescribed_displacement():
    st = _study("200 N down @ top")
    st.loads.append(Load("displacement", "top", 0.1, [0, 0, -1]))
    reason = material_swap_resolve(st, resolve_material("6061-t6"), resolve_material("7075-t6"))
    assert "displacement loads" in reason


def test_question_readers():
    tpl, p, guesses = read_circuit_question("will a blue LED on 5V with a 47 ohm resistor fry?")
    assert tpl == "led" and p["led"] == "blue" and p["resistor"] == "47" and p["supply"] == "5v"
    tpl, p, _ = read_circuit_question("what is the ripple on a 12 V to 5 V buck at 2 A, 500 kHz?")
    assert tpl == "buck" and p["vin"] == "12" and p["vout"] == "5" and p["current"] == "2"
    r = read_thermal_question("will this 12 W pack overheat in a sealed box at 35 C? over 60 C is bad")
    assert r["power"] == "12 w" and r["ambient"] == "35 C" and r["limit"] == "60 C"
    assert r["film"].startswith("5.0")
