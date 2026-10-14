"""Impacts and tipping end to end. Skipped when MuJoCo is not installed."""
import pytest

from wobbl.mechanism import install

pytestmark = pytest.mark.skipif(not install.available(), reason="MuJoCo is not installed (wobbl setup --only mujoco)")


def _st(o):
    return {c.name: c.status for c in o.checks}


def test_a_head_on_hit_keeps_momentum_and_its_peak_follows_the_contact_stiffness(tmp_path):
    from wobbl.mechanism.study import run_mechanism

    o = run_mechanism("impact", {"mass": "1 kg", "target_mass": "3 kg", "speed": "1 m/s", "restitution": 0.8},
                      tmp_path)
    st = _st(o)
    for name in ("restitution", "momentum", "hand calculation", "peak force", "contact softness", "energy lost"):
        assert st[name] == "pass", (name, st)
    assert o.confidence == "medium"            # the peak force rests on the Hertz stiffness, and the report says so
    a1, a2 = o.values["speeds_after_mps"]
    assert (a1, a2) == pytest.approx((-0.35, 0.45), abs=0.005)      # (1 - 3 e) / 4 and (1 + e) / 4


def test_a_drop_past_its_force_limit_gets_a_pad_and_a_lower_drop(tmp_path):
    from wobbl.mechanism.study import run_mechanism

    o = run_mechanism("impact", {"mass": "200 g", "drop": "500 mm", "radius": "5 mm", "limit": "500 N",
                                 "restitution": 0.5}, tmp_path)
    assert o.verdict == "fails" and o.values["peak_force_N"] > 500
    assert {f["kind"] for f in o.fixes} == {"pad", "speed"}
    assert o.values["impact_speed_mps"] == pytest.approx((2 * 9.80665 * 0.5) ** 0.5)


def test_a_stand_pushed_past_its_tipping_force_tips_and_ballast_holds_it(tmp_path):
    from wobbl.mechanism.study import run_mechanism

    o = run_mechanism("tip", {"force": "5 N"}, tmp_path)
    st = _st(o)
    assert o.verdict == "fails" and o.values["tipped"]
    assert st["tipping statics"] == "pass" and st["hand statics"] == "pass"
    assert o.values["critical_force_N"] == pytest.approx(3.27, abs=0.01)
    cw = next(f for f in o.fixes if f["kind"] == "counterweight")
    assert cw["after"]["stays"]
    ok = run_mechanism("tip", {"force": "2.5 N"}, tmp_path / "ok")
    assert ok.verdict == "marginal" and not ok.values["tipped"]      # a margin of 1.31, under 1.5
