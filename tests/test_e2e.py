"""End to end runs with the real solvers (installed by 'wobbl setup')."""
import json

import pytest
from click.testing import CliRunner

from conftest import needs_ccx, needs_spice
from wobbl.cli.main import cli

pytestmark = pytest.mark.solver


@needs_ccx
def test_cantilever_matches_beam_theory(parts, tmp_path):
    from wobbl.structural.pipeline import analyze_strength
    from wobbl.structural.spec import StructuralStudy, parse_fixture, parse_load

    st = StructuralStudy(parts["cantilever"], "mild-steel", [parse_load("100 N -z @ x=max")],
                         [parse_fixture("x=min")], quality="fast", budget=120)
    o = analyze_strength(st, tmp_path / "out", render=False)
    E, I, L, F = 200000.0, 20 * 10 ** 3 / 12, 200.0, 100.0
    tip = F * L ** 3 / (3 * E * I)
    assert o.max_disp_mm == pytest.approx(tip, rel=0.05)
    beam = next(c for c in o.checks if c.name == "hand calculation")
    assert beam.status == "pass"
    bal = next(c for c in o.checks if c.name == "force balance")
    assert bal.status == "pass"
    assert o.verdict == "holds"
    assert (tmp_path / "out" / "report.html").exists()


@needs_ccx
def test_bolted_plate_cli_json(parts, tmp_path):
    r = CliRunner().invoke(cli, ["run", parts["mount"], "-m", "pla", "--fix", "bolt holes d=4.5",
                                 "--load", "50 N down @ top and within 8 of 30,20,5", "--quality", "fast",
                                 "--no-images", "--json", "-o", str(tmp_path / "o")])
    assert r.exit_code == 0, r.output
    d = json.loads(r.output[r.output.index("{"):])
    assert d["verdict"] in ("holds", "marginal", "fails")
    assert d["confidence"] in ("high", "medium", "low")
    assert any(c["name"] == "bolt bearing" for c in d["checks"])


@needs_ccx
def test_unheld_part_is_refused(parts, tmp_path):
    r = CliRunner().invoke(cli, ["run", parts["cantilever"], "-m", "pla", "--fix", "roller z=min",
                                 "--load", "10 N down @ top", "--quality", "fast", "-o", str(tmp_path / "x")])
    assert r.exit_code != 0
    assert "free to" in r.output


@needs_ccx
def test_thermal_block_energy_balance(parts, tmp_path):
    from wobbl.thermal.study import run_thermal
    from wobbl.thermal.tools import study_from

    st = study_from(parts["cantilever"], "6061-t6", power="5 W", films=["10 W/m2K @ all"], ambient="25 C",
                    quality="fast")
    o = run_thermal(st, tmp_path / "heat")
    area = 2 * (200 * 20 + 200 * 10 + 20 * 10) * 1e-6
    expect = 25 + 5 / (10 * area)
    assert o.values["t_max_C"] == pytest.approx(expect, rel=0.02)  # aluminum: nearly isothermal
    assert next(c for c in o.checks if c.name == "energy balance").status == "pass"


@needs_spice
def test_led_and_rc(tmp_path):
    from wobbl.circuits.run import run_circuit

    led = run_circuit("led", {"supply": "5", "resistor": "150", "led": "red"}, tmp_path / "led")
    assert led.verdict == "ok"
    assert led.values["i_led_A"] == pytest.approx(0.0198, rel=0.05)
    assert all(c.status in ("pass", "info") for c in led.checks)
    rc = run_circuit("rc", {"r": "1k", "c": "1u", "v": "5"}, tmp_path / "rc")
    assert rc.values["v_at_tau"] == pytest.approx(5 * 0.63212, rel=0.005)


@needs_ccx
@needs_spice
def test_validation_subset(tmp_path):
    """A quick slice of the validation suite: exact answers and high confidence, including roller supports."""
    from wobbl.validation.cases import run_cases

    rows = run_cases(tmp_path, ["cantilever-tip", "lame", "slab", "led", "rc"], log=lambda m: None)
    for r in rows:
        assert r["pass"], r
        assert r["confidence"] == "high", r
