"""Four bar linkages end to end: geometry against Freudenstein, torque against virtual work, and the motor.
Skipped when MuJoCo is not installed."""
import pytest

from wobbl.mechanism import install

pytestmark = pytest.mark.skipif(not install.available(), reason="MuJoCo is not installed (wobbl setup --only mujoco)")
MOTOR = {"stall_torque": "5 N*m", "no_load_speed": "60 rpm", "turns": 1}


def test_a_crank_rocker_sweeps_as_freudenstein_says_and_its_motor_keeps_up(tmp_path):
    from wobbl.mechanism.study import run_mechanism

    o = run_mechanism("fourbar", {**MOTOR, "load": "1 N*m"}, tmp_path)
    st = {c.name: c.status for c in o.checks}
    assert st["hand calculation"] == "pass" and st["virtual work"] == "pass" and st["energy account"] == "pass"
    assert o.values["full_turn"] and not o.values["stalled"]
    assert o.values["sweep_deg"] == pytest.approx(87.97, abs=0.1)
    lo, hi = o.values["transmission_deg"]
    assert lo == pytest.approx(31.9, abs=0.3) and o.verdict == "marginal"      # under 40 degrees
    assert o.fixes[0]["kind"] == "lengths" and o.fixes[0]["after"]["transmission_deg"][0] > 40
    assert any(e["name"] == "crank turn 1" for e in o.events)


def test_a_linkage_whose_crank_cannot_turn_fails_and_gets_new_lengths(tmp_path):
    from wobbl.mechanism.study import run_mechanism

    o = run_mechanism("fourbar", {"ground": "100 mm", "crank": "80 mm", "coupler": "40 mm", "rocker": "90 mm",
                                  "start": "60 deg"}, tmp_path)
    assert o.verdict == "fails" and "cannot turn all the way" in o.headline
    assert not o.values["full_turn"]


def test_a_weak_motor_stalls_and_is_told_the_torque_it_needs(tmp_path):
    from wobbl.mechanism.study import run_mechanism

    o = run_mechanism("fourbar", {**MOTOR, "stall_torque": "0.3 N*m", "load": "2 N*m"}, tmp_path)
    assert o.verdict == "fails" and o.values["stalled"]
    fix = next(f for f in o.fixes if f["kind"] == "motor")
    assert fix["after"]["stall_torque_Nm"] > 1.0
