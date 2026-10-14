"""Gear trains driven by a DC motor, end to end. Skipped when MuJoCo is not installed."""
import pytest

from wobbl.mechanism import install

pytestmark = pytest.mark.skipif(not install.available(), reason="MuJoCo is not installed (wobbl setup --only mujoco)")


def _st(o):
    return {c.name: c.status for c in o.checks}


def test_a_gear_train_matches_its_ratio_losses_and_motor_line(tmp_path):
    from wobbl.mechanism.study import run_mechanism

    o = run_mechanism("gears", {}, tmp_path)
    st = _st(o)
    assert st["hand calculation"] == "pass" and st["mesh losses"] == "pass" and st["energy account"] == "pass"
    assert o.values["ratio_simulated"] == pytest.approx(12.0, rel=1e-4)
    assert o.verdict in ("ok", "marginal") and o.values["t95_s"] > 0


def test_a_load_past_stall_stalls_the_train_and_asks_for_more_motor(tmp_path):
    from wobbl.mechanism.study import run_mechanism

    o = run_mechanism("gears", {"load": "0.1 N*m", "time": "0.3 s"}, tmp_path)
    assert o.verdict == "fails" and "stalls" in o.headline
    assert {f["kind"] for f in o.fixes} >= {"ratio", "motor"}
