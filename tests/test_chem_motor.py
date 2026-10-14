"""Solid motor runs: the verified burn, the casing and port checks, simulated fixes and the RASP file the motion
domain reads. The chemistry cross check runs only where Cantera is installed."""
from pathlib import Path

import pytest

from wobbl.chem import install
from wobbl.chem.motor import casing_stress, run_motor
from wobbl.chem.spec import Casing, read_spec

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "knsb-motor" / "motor.toml"

DESIGN = """
name = "test"
propellant = "knsb"
[[grain]]
shape = "bates"
diameter = "48 mm"
core = "20 mm"
length = "75 mm"
count = 4
[nozzle]
throat = "12 mm"
exit = "30 mm"
[casing]
material = "6061-t6"
outer_diameter = "54 mm"
wall = "{wall}"
length = "320 mm"
target_sf = 2
"""


def _check(o, name):
    return next(c for c in o.checks if c.name == name)


def test_the_example_runs_verified_and_flags_its_narrow_ports(tmp_path):
    o = run_motor(read_spec(EXAMPLE), tmp_path, render=False)
    for name in ("time step convergence", "mass balance", "closed form", "geometry", "burn rate data", "casing"):
        assert _check(o, name).status == "pass", (name, _check(o, name).summary)
    assert _check(o, "erosive burning").status == "warn" and o.verdict == "marginal"
    assert o.values["port_throat"] == pytest.approx((16 / 12.7) ** 2, rel=1e-6)
    assert o.values["designation"].startswith("J") and 1100 < o.values["impulse_Ns"] < 1400
    core = next(f for f in o.fixes if f["kind"] == "core")
    assert "18 mm" in core["title"] and "Simulated" in core["summary"]
    assert (tmp_path / "thrust.csv").read_text().startswith("time_s,thrust_N")
    if not install.available():
        assert _check(o, "chemistry").status == "skip"


def test_a_thin_casing_fails_and_the_throat_fix_is_simulated(tmp_path):
    o = run_motor(read_spec(DESIGN.format(wall="0.3 mm")), tmp_path, render=False)
    assert o.verdict == "fails" and _check(o, "casing").status == "fail"
    assert o.values["safety_factor"] < 1
    throat = next(f for f in o.fixes if f["kind"] == "throat")
    assert throat["after"]["safety_factor"] == pytest.approx(2.0, rel=0.03)
    assert throat["after"]["throat_mm"] > 12
    assert any(f["kind"] == "wall" for f in o.fixes) and any(f["kind"] == "segments" for f in o.fixes)
    ok = run_motor(read_spec(DESIGN.format(wall="1.6 mm")), tmp_path / "ok", render=False)
    assert ok.verdict == "ok" and not ok.fixes


def test_hoop_stress_matches_lame_and_the_thin_wall_hand_calculation():
    v = casing_stress(Casing("6061-t6", 0.054, 0.0508), 5e6)
    ro, ri = 0.027, 0.0254
    assert v["hoop_Pa"] == pytest.approx(5e6 * (ro ** 2 + ri ** 2) / (ro ** 2 - ri ** 2))
    assert v["hoop_thin_Pa"] == pytest.approx(5e6 * ri / 0.0016)
    assert v["burst_pressure_Pa"] == pytest.approx(2 * 310e6 * 0.0016 / 0.054)
    assert v["safety_factor"] == pytest.approx(276e6 / v["hoop_Pa"])


def test_the_motor_file_reads_back_in_the_motion_domain(tmp_path):
    pytest.importorskip("wobbl.motion.spec")
    from wobbl.motion.spec import read_eng

    o = run_motor(read_spec(EXAMPLE), tmp_path, render=False)
    eng = read_eng(tmp_path / "motor.eng")
    assert eng.impulse == pytest.approx(o.values["impulse_Ns"], rel=0.02)
    assert eng.length == pytest.approx(0.4) and eng.propellant == pytest.approx(o.values["propellant_kg"], rel=1e-3)


def test_agents_get_the_chemistry_tools(tmp_path, monkeypatch):
    import asyncio

    from wobbl.chem.tools import solve_motor_tool
    from wobbl.mcp import tools
    from wobbl.mcp.server import build_server

    names = {t.name for t in asyncio.run(build_server().list_tools())}
    assert {"solve_motor", "solve_flame", "solve_propellant"} <= names
    monkeypatch.setenv("WOBBL_HOME", str(tmp_path))
    r = solve_motor_tool(propellant="knsb", grains=["bates d=48 core=20 L=75 x4"], throat="12", exit="30",
                         casing="6061-t6 od=54 wall=1.6")
    assert "error" not in r, r
    assert r["verdict"] == "ok" and Path(r["eng"]).exists() and "time_step_runs" not in r["values"]
    assert "error" in solve_motor_tool(propellant="knsb", throat="12")
    bad = solve_motor_tool(propellant="knsb", grains=["bates d=48 core=20 L=75 x4"], throat="12", exit="30",
                           casing="6061-t6 od=54 wall=0.3")
    fix = tools.suggest_fix(bad["result_dir"])
    assert fix["domain"] == "motor" and "solve_motor" in fix["note"] and fix["fixes"]


@pytest.mark.skipif(not install.available(), reason="Cantera not installed")
def test_the_chemistry_cross_check_agrees_with_the_published_knsb_data(tmp_path):
    o = run_motor(read_spec(EXAMPLE), tmp_path, render=False)
    c = _check(o, "chemistry")
    assert c.status == "pass", c.summary
    assert o.values["equilibrium_cstar_m_s"] == pytest.approx(908, rel=0.01)
    assert 0.6 < o.values["delivered_fraction"] < 0.95
