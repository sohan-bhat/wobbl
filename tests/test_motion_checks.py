"""The checks behind a motion answer: verification sets the confidence, rocketry sets the verdict, and each fix,
flown again, does what it says."""
from dataclasses import replace
from pathlib import Path

import pytest

from wobbl.motion.checks import step_convergence
from wobbl.motion.flight import VZ, fly
from wobbl.motion.spec import Chute, MotionStudy, margin, read_spec
from wobbl.motion.study import analyze

ROCKET = Path(__file__).resolve().parents[1] / "examples" / "rocket-launch" / "rocket.toml"


@pytest.fixture(scope="module")
def rocket():
    return read_spec(ROCKET)


def _by_name(checks):
    return {c.name: c for c in checks}


def test_a_dropped_rock_passes_every_check():
    st = MotionStudy(name="rock", mass=1.03, diameter=0.09, cd=0.6, height=300.0)
    a = analyze(st)
    c = _by_name(a.checks)
    assert list(c) == ["step convergence", "hand calculation", "vacuum drop", "energy balance", "units and sanity"]
    assert all(x.status == "pass" for x in a.checks) and a.confidence == "high" and a.verdict == "ok"
    assert c["vacuum drop"].detail["time_s"] == pytest.approx(7.822, abs=1e-3)
    assert c["energy balance"].detail["error"] < 1e-6
    assert "without air" in dict(a.metrics)


def test_the_example_rocket_is_ok_with_closed_budgets(rocket):
    a = analyze(rocket)
    v, d = _by_name(a.checks), _by_name(a.design)
    assert a.confidence == "high" and a.verdict == "ok" and not a.fixes
    assert v["delta v budget"].detail["error"] < 1e-6 and v["energy balance"].detail["error"] < 1e-6
    assert v["parachute descent"].status == "pass"
    assert {n: c.status for n, c in d.items()} == {
        "stability margin": "pass", "rail exit speed": "pass", "max dynamic pressure": "info",
        "max acceleration": "info", "apogee": "pass", "landing speed": "pass", "drift": "info"}
    assert 1.0 < a.values["max_mach"] < 1.5 and a.values["margin_cal"] > 1.5


def test_an_unstable_rocket_fails_and_its_fixes_reach_one_and_a_half_calibers(rocket):
    st = replace(rocket, airframe=replace(rocket.airframe, cg=1.50))
    a = analyze(st)
    assert a.verdict == "fails" and _by_name(a.design)["stability margin"].status == "fail"
    assert a.headline.startswith("Unstable off the rail")
    assert a.confidence == "high"          # the design fails, the numbers are still right
    fixes = {f["kind"]: f for f in a.fixes}
    nose = fixes["nose weight"]["params"]
    heavier = replace(st, mass=st.mass + nose["mass_kg"], airframe=replace(st.airframe, cg=nose["cg_m"]))
    cg = (st.liftoff_mass * 1.50 + nose["mass_kg"] * nose["at_m"]) / heavier.liftoff_mass
    assert cg == pytest.approx(nose["cg_m"]) and margin(heavier) == pytest.approx(1.5)
    fins = replace(st.airframe.fins, span=fixes["fin area"]["params"]["span_m"])
    assert margin(st, fins=fins) == pytest.approx(1.5, abs=1e-6)


def test_a_short_rail_is_marginal_and_the_longer_rail_fixes_it(rocket):
    st = replace(rocket, rail_length=0.5)
    a = analyze(st)
    assert a.verdict == "marginal" and _by_name(a.design)["rail exit speed"].status == "warn"
    need = next(f for f in a.fixes if f["kind"] == "rail")["params"]["rail_length_m"]
    fl = fly(replace(st, rail_length=need))
    y = fl.event("rail exit").y
    assert (y[2] ** 2 + y[3] ** 2) ** 0.5 == pytest.approx(15.0, rel=0.02)


def test_a_small_main_lands_too_fast_and_the_bigger_one_lands_at_5_m_s(rocket):
    small = [rocket.chutes[0], Chute("main", 0.9, 1.5, 150.0)]
    a = analyze(replace(rocket, chutes=small))
    assert a.verdict in ("marginal", "fails") and a.headline.startswith("Lands at")
    d = next(f for f in a.fixes if f["kind"] == "main parachute")["params"]["diameter_m"]
    fl = fly(replace(rocket, chutes=[rocket.chutes[0], Chute("main", d, 1.5, 150.0)]))
    assert -fl.end.y[VZ] == pytest.approx(5.0, rel=0.005)


def test_step_convergence_grades_the_change():
    a = {"apogee": (1000.0, "m"), "landing speed": (5.0, "m/s")}
    assert step_convergence(a, {"apogee": (1000.0001, "m"), "landing speed": (5.0, "m/s")}).status == "pass"
    assert step_convergence(a, {"apogee": (1005.0, "m"), "landing speed": (5.0, "m/s")}).status == "warn"
    bad = step_convergence(a, {"apogee": (1000.0, "m"), "landing speed": (6.0, "m/s")})
    assert bad.status == "fail" and "landing speed" in bad.summary


def test_a_thrown_ball_is_checked_against_the_throw_without_air():
    a = analyze(MotionStudy(name="ball", mass=0.145, diameter=0.074, speed=40.0, angle=35.0))
    c = _by_name(a.checks)
    assert "hand calculation" not in c and c["vacuum flight"].status == "pass"
    assert c["vacuum flight"].detail["range_m"] == pytest.approx(40.0 ** 2 * 0.9396926 / 9.80665, rel=1e-3)
    assert a.confidence == "high" and dict(a.metrics)["range"].endswith(" m")
