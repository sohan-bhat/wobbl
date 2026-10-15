"""Mechanisms through wobbl simulate: the physics registered by name, a plan that shows every guess before it
runs, and a run that writes outcome.json and request.json and repeats from it. Skipped without MuJoCo."""
import json

import pytest

from wobbl.mechanism import install
from wobbl.simulate import registry, route
from wobbl.simulate.runner import preview, run_plan

from test_mechanism_general import ARM

needs_mujoco = pytest.mark.skipif(not install.available(), reason="MuJoCo is not installed")


def test_the_physics_are_registered_by_name():
    names = {a.name for a in registry.analyses("mechanism")}
    assert names == {"torque", "kinematics", "collision", "stability", "contact", "dynamics", "overview",
                     "drive"}
    assert all(a.dry for a in registry.analyses("mechanism"))
    assert [a.name for a in registry.analyses("mechanism") if a.default] == ["overview"]
    assert not registry.errors


@needs_mujoco
def test_a_plan_shows_the_payload_guess_and_a_run_repeats_from_its_request(tmp_path):
    (tmp_path / "arm.xml").write_text(ARM)
    plan = route(str(tmp_path / "arm.xml"), "can the motors hold 50 g at the end of the arm?")
    assert plan.domain == "mechanism" and plan.kind == "mjcf" and plan.names == ["torque"]
    steps = preview(plan, {})
    guessed = " ".join(t for s in steps for t, _ in s.job.assumptions)
    assert "payload is held at the end of the fore" in guessed
    assert steps[0].job.used["payload"] == ["50 g"]
    out = tmp_path / "run"
    s = run_plan(plan, {}, out)[0]
    assert s.get("verdict") == "ok"
    req = json.loads((out / "request.json").read_text())
    assert req["domain"] == "mechanism" and req["analysis"] == "torque"
    again = registry.get("mechanism", "torque").run(req["spec"], tmp_path / "again", quick=True)
    assert again["values"]["joints"]["elbow"]["needed"] == pytest.approx(
        s.get("values")["joints"]["elbow"]["needed"], rel=0.01)


@needs_mujoco
def test_words_run_the_ready_made_mechanism_for_their_physics(tmp_path):
    plan = route("a pendulum 1 m long released at 30 degrees", "what is its period?")
    assert plan.names == ["dynamics"] and plan.kind == "pendulum"
    s = run_plan(plan, {}, tmp_path)[0]
    assert s.get("values")["period_s"] == pytest.approx(2.0, rel=0.05)
