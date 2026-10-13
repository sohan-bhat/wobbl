from pathlib import Path

import pytest

from wobbl.circuits.tools import read_circuit_question
from wobbl.geometry.parametric import build_step, load_params, radius_param, thickness_param
from wobbl.geometry.remap import remap_selector, same_surface
from wobbl.structural.run import describe_part
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


def test_question_readers():
    tpl, p, guesses = read_circuit_question("will a blue LED on 5V with a 47 ohm resistor fry?")
    assert tpl == "led" and p["led"] == "blue" and p["resistor"] == "47" and p["supply"] == "5v"
    tpl, p, _ = read_circuit_question("what is the ripple on a 12 V to 5 V buck at 2 A, 500 kHz?")
    assert tpl == "buck" and p["vin"] == "12" and p["vout"] == "5" and p["current"] == "2"
    r = read_thermal_question("will this 12 W pack overheat in a sealed box at 35 C? over 60 C is bad")
    assert r["power"] == "12 w" and r["ambient"] == "35 C" and r["limit"] == "60 C"
    assert r["film"].startswith("5.0")
