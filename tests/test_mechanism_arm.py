"""A robot arm's pick and place against its servos, end to end. Skipped when MuJoCo is not installed."""
import pytest

from wobbl.mechanism import install

pytestmark = pytest.mark.skipif(not install.available(), reason="MuJoCo is not installed (wobbl setup --only mujoco)")
FAST = {"move_time": "0.7 s", "dwell": "0.2 s", "timestep": "2 ms"}


def _checks(o):
    return {c.name: c.status for c in o.checks}


def test_strong_servos_carry_a_light_payload(tmp_path):
    from wobbl.mechanism.study import run_mechanism

    o = run_mechanism("arm", {**FAST, "servo": "xm430", "payload": "100 g"}, tmp_path)
    assert o.verdict == "ok" and not o.fixes
    st = _checks(o)
    assert st["hand statics"] == "pass" and st["forward run"] == "pass" and st["energy account"] == "pass"
    j = o.values["joints"]
    assert set(j) == {"base", "shoulder", "elbow", "wrist"} and all(v["peak_usage"] < 0.8 for v in j.values())
    assert j["shoulder"]["lag_deg"] < 2
    assert [e["name"] for e in o.events][:2] == ["pick", "lifted"]


def test_hobby_servos_stall_and_each_fix_is_solved_again(tmp_path):
    from wobbl.mechanism.arm import needs
    from wobbl.mechanism.study import run_mechanism

    o = run_mechanism("arm", {**FAST, "servo": "sg90", "payload": "200 g", "links": "100,90", "link_mass": "30,25",
                              "pick": "150,0,30", "place": "0,150,30"},
                      tmp_path)
    assert o.verdict == "fails" and "stalls" in o.headline
    assert _checks(o)["forward run"] == "pass"            # the run falls behind, as the inverse dynamics says
    assert any(e["name"].endswith("at its limit") for e in o.events)
    kinds = {f["kind"] for f in o.fixes}
    assert "reduction" in kinds and "payload" in kinds
    red = next(f for f in o.fixes if f["kind"] == "reduction")
    raw = {**FAST, "servo": "sg90", "payload": "200 g", "links": "100,90", "link_mass": "30,25", "pick": "150,0,30",
           "place": "0,150,30",
           **{f"{j}_ratio": r for j, r in red["after"]["ratios"].items()}}
    again = needs(raw)
    assert again["worst"] == pytest.approx(red["after"]["worst_usage"])
    for j in red["after"]["ratios"]:                      # each geared joint is now inside its servo
        assert again["joints"][j]["peak"] <= 0.8
    solved = again["worst"] <= 1 and again["hot"] <= 1
    assert ("still not enough" in red["summary"]) == (not solved)


def test_a_free_standing_arm_tips_and_ballast_holds_it_down(tmp_path):
    from wobbl.mechanism.study import run_mechanism

    o = run_mechanism("arm", {**FAST, "servo": "xm430", "payload": "400 g", "mount": "free", "base_mass": "150 g",
                              "base_size": "80,80"}, tmp_path)
    assert o.verdict == "fails" and o.values["tipped"]
    assert _checks(o)["hand statics, tipping"] in ("pass", "info")
    ballast = next(f for f in o.fixes if f["kind"] == "ballast")
    assert ballast["after"]["base_mass_kg"] > 0.15
