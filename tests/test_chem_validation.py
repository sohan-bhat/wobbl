"""The chemistry and motor validation cases run in seconds, so they run with every test pass. The motor cases need
no Cantera, and the flame and propellant cases run where it is installed."""
import json
from pathlib import Path

import pytest

from wobbl.chem import install
from wobbl.validation.cases import CASES, run_cases

ROOT = Path(__file__).resolve().parents[1]
MOTOR = [c.id for c in CASES if c.domain == "motor"]
COMBUSTION = [c.id for c in CASES if c.domain == "combustion"]


def test_motor_cases_land_on_their_closed_forms_and_static_fires(tmp_path):
    assert MOTOR == ["steady-kn", "neutral-grain", "mit-o3800", "mit-o3100"]
    rows = {r["id"]: r for r in run_cases(tmp_path, MOTOR, log=lambda m: None)}
    assert all(r["pass"] for r in rows.values()), rows
    assert rows["steady-kn"]["error"] < 1e-6 and rows["neutral-grain"]["error"] < 0.005
    assert all(r["wall_s"] < 30 for r in rows.values())


@pytest.mark.skipif(not install.available(), reason="Cantera not installed")
def test_combustion_cases_match_cea_and_the_published_values(tmp_path):
    assert COMBUSTION == ["methane-flame", "lox-lh2-cstar", "lox-lh2-isp", "knsu-cstar"]
    rows = run_cases(tmp_path, COMBUSTION, log=lambda m: None)
    assert all(r["pass"] and r["error"] < 0.002 for r in rows), rows


def test_published_results_include_the_chemistry_cases():
    rows = {r["id"]: r for r in json.loads((ROOT / "docs" / "validation-results.json").read_text())}
    assert set(MOTOR + COMBUSTION) <= set(rows) and all(rows[i]["pass"] for i in MOTOR + COMBUSTION)
    assert all(rows[c.id]["title"] == c.title for c in CASES if c.id in MOTOR + COMBUSTION)
