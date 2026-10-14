"""Flames and propellant performance, and the three chemistry commands. Cantera parts skip without Cantera."""
import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from wobbl.chem import install
from wobbl.cli.main import cli

ROOT = Path(__file__).resolve().parents[1]
needs_cantera = pytest.mark.skipif(not install.available(), reason="Cantera not installed (wobbl setup --only cantera)")


def _check(o, name):
    return next(c for c in o.checks if c.name == name)


@needs_cantera
def test_methane_air_flame_matches_the_textbook_and_both_thermo_datasets(tmp_path):
    from wobbl.chem.combustion import run_flame

    o = run_flame("methane", "air", 1.0, 298.15, 101325.0, tmp_path, render=True)
    assert o.values["T_ad_K"] == pytest.approx(2226.0, rel=0.003)        # Turns, equilibrium at 1 atm and 298 K
    assert o.values["T_no_dissociation_K"] == pytest.approx(2326.35, rel=0.002)   # complete combustion, Marzouk 2025
    assert o.values["T_gri30_K"] == pytest.approx(o.values["T_ad_K"], rel=0.002)
    assert o.verdict == "ok" and o.confidence == "high"
    assert {c.name: c.status for c in o.checks}["flammability"] == "pass"
    d = json.loads((tmp_path / o.values["files"]["playback"]).read_text())
    here = d["frames"][d["manifest"]["operating_frame"]]
    assert d["manifest"]["kind"] == "flame" and here["phi"] == 1.0
    assert here["T_K"] == pytest.approx(o.values["T_ad_K"], abs=0.5) and "CO2" in here["x"]


@needs_cantera
def test_a_hot_flame_gets_a_leaner_mixture_and_a_lean_one_will_not_burn(tmp_path):
    from wobbl.chem.combustion import adiabatic, fuel_info, run_flame

    o = run_flame("propane", "air", 0.9, 298.15, 101325.0, tmp_path / "hot", limit=2000.0, render=False)
    assert o.verdict == "hot"
    phi = o.fixes[0]["after"]["phi"]
    assert adiabatic(fuel_info("propane")[1], "air", phi, 298.15, 101325.0)[1].T == pytest.approx(2000.0, abs=5.0)
    cold = run_flame("methane", "air", 0.3, 298.15, 101325.0, tmp_path / "cold", render=False)
    assert cold.verdict == "fails" and _check(cold, "flammability").status == "fail"


@needs_cantera
def test_propellant_performance_checks_itself_and_flags_an_overexpanded_nozzle(tmp_path):
    from wobbl.chem.combustion import run_propellant

    o = run_propellant("knsb", 5e6, 8.0, None, tmp_path / "knsb", render=True)
    assert o.verdict == "ok" and o.confidence == "high"
    assert o.values["cstar_frozen_m_s"] == pytest.approx(908, rel=0.01)
    assert o.values["isp_shifting_s"] > o.values["isp_frozen_s"]
    d = json.loads((tmp_path / "knsb" / "playback.json").read_text())
    assert d["manifest"]["kind"] == "nozzle" and len(d["frames"]) == 16
    assert d["frames"][0]["area_ratio"] < d["frames"][-1]["area_ratio"]
    big = run_propellant("lox:6,lh2:1", 6.894757e6, 120.0, None, tmp_path / "lox", render=False)
    assert big.verdict == "marginal" and big.fixes[0]["kind"] == "area_ratio"
    with pytest.raises(ValueError, match="no published composition"):
        run_propellant("ocean-water", 5e6, 8.0, None, tmp_path / "ow", render=False)


def test_motor_command_from_a_spec_and_from_flags(tmp_path):
    r = CliRunner().invoke(cli, ["motor", str(ROOT / "examples" / "knsb-motor" / "motor.toml"), "-o",
                                 str(tmp_path / "a"), "--json"])
    assert r.exit_code == 0, r.output
    d = json.loads(r.output[r.output.index("{"):])
    assert d["domain"] == "motor" and d["verdict"] == "marginal" and d["values"]["designation"].startswith("J")
    r = CliRunner().invoke(cli, ["motor", "--propellant", "knsb", "--grain", "bates d=48 core=20 L=75 x4",
                                 "--throat", "12", "--exit", "30", "--casing", "6061-t6 od=54 wall=1.6", "-o",
                                 str(tmp_path / "b"), "--json"])
    assert r.exit_code == 0, r.output
    assert (tmp_path / "b" / "motor.eng").exists()
    r = CliRunner().invoke(cli, ["motor", "--propellant", "knsb", "--throat", "12"])
    assert r.exit_code != 0 and "--grain" in r.output


def test_flame_and_propellant_commands_parse(tmp_path):
    r = CliRunner().invoke(cli, ["flame", "methane", "--phi", "0", "-o", str(tmp_path / "f")])
    assert r.exit_code != 0 and "phi" in r.output
    r = CliRunner().invoke(cli, ["propellant", "knsb", "--area-ratio", "0.5", "-o", str(tmp_path / "p")])
    assert r.exit_code != 0 and "area ratio" in r.output.replace("-", " ")
