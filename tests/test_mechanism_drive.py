"""Can this motor drive it, asked through wobbl simulate of mechanisms nothing was tuned for: a crank rocker, a drag
link, a double rocker that cannot turn, a slider crank that stalls a small servo and one a bigger servo drives, a quick
return and a URDF elbow behind a servo. Every case is held to a number worked out here by hand."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from wobbl.mechanism import install
from wobbl.mechanism import theory as T
from wobbl.mechanism.motors import get_motor
from wobbl.simulate import route, run_plan

pytestmark = pytest.mark.skipif(not install.available(), reason="MuJoCo is not installed (wobbl setup --only mujoco)")
ROOT = Path(__file__).resolve().parents[1]

URDF = """<robot name="arm">
  <link name="base"><inertial><mass value="1"/><inertia ixx="0.01" iyy="0.01" izz="0.01" ixy="0" ixz="0" iyz="0"/></inertial></link>
  <link name="upper">
    <inertial><origin xyz="0.1 0 0"/><mass value="0.3"/><inertia ixx="0.0005" iyy="0.001" izz="0.001" ixy="0" ixz="0" iyz="0"/></inertial>
    <visual><origin xyz="0.1 0 0" rpy="0 1.5708 0"/><geometry><cylinder radius="0.01" length="0.2"/></geometry></visual>
  </link>
  <link name="fore">
    <inertial><origin xyz="0.08 0 0"/><mass value="0.2"/><inertia ixx="0.0003" iyy="0.0005" izz="0.0005" ixy="0" ixz="0" iyz="0"/></inertial>
    <visual><origin xyz="0.08 0 0" rpy="0 1.5708 0"/><geometry><cylinder radius="0.008" length="0.16"/></geometry></visual>
  </link>
  <joint name="shoulder" type="revolute"><parent link="base"/><child link="upper"/><origin xyz="0 0 0.05"/><axis xyz="0 1 0"/><limit lower="-1.5" upper="1.5" effort="3" velocity="3"/></joint>
  <joint name="elbow" type="revolute"><parent link="upper"/><child link="fore"/><origin xyz="0.2 0 0"/><axis xyz="0 1 0"/><limit lower="-2" upper="2" effort="3" velocity="3"/></joint>
</robot>
"""


def ask(thing, question, out, **opts):
    plan = route(thing, question)
    steps = run_plan(plan, {k.replace("_", "-"): v if isinstance(v, list) else [v] for k, v in opts.items()}, out)
    assert len(steps) == 1 and not steps[0].error, steps[0].error
    return plan, steps[0].outcome


def checks(o):
    return {c.name: c for c in o.checks}


def test_drive_questions_route_to_the_drive_analysis():
    for thing, q in [("fourbar", "can a 45 rpm wiper motor drive this linkage?"),
                     ("fourbar", "does the motor stall, and what current does it draw?"),
                     ("a slider crank pushing 40 N", "can an mg996r drive it at 60 rpm?"),
                     ("fourbar", "is this a quick return linkage, and what torque does it take?")]:
        assert route(thing, q).names == ["drive"], (thing, q)
    assert route("fourbar", "does it sweep 95 degrees?").names == ["kinematics"]
    # the sweep and the motor are one linkage run, so the drive analysis answers both clauses
    assert route("fourbar", "does it sweep 95 degrees, and can a 45 rpm motor drive it?").names == ["drive"]


@pytest.mark.slow
def test_a_crank_rocker_against_virtual_work(tmp_path):
    sets = ["ground=200", "crank=50", "coupler=180", "rocker=120", "stall_torque=20", "no_load_speed=60rpm",
            "load=4", "voltage=12"]
    plan, o = ask("fourbar", "can this 60 rpm motor drive the linkage against 4 N m, what current?", tmp_path,
                  set=sets)
    assert plan.names == ["drive"] and o.verdict in ("ok", "marginal")
    # the slow turn needs the load times the rocker's speed over the crank's, at most
    th = np.linspace(0, 2 * math.pi, 3600)
    t4, t3 = T.freudenstein(0.2, 0.05, 0.18, 0.12, th)
    hand = 4 * float(np.nanmax(np.abs(T.velocity_ratio(0.05, 0.12, th, t3, t4))))
    assert o.values["peak_torque_Nm"] == pytest.approx(hand, rel=0.15)
    c = checks(o)
    assert c["virtual work from the weights and loads"].status == "pass" and c["virtual work"].status == "pass"
    # a brushed motor's current is linear in torque, with a stall current of Ts w0 / V
    i_s = 20 * (60 * 2 * math.pi / 60) / 12
    assert o.values["stall_current_A"] == pytest.approx(i_s, rel=1e-6)
    assert o.values["peak_current_A"] == pytest.approx(0.05 * i_s + 0.95 * i_s * o.values["peak_torque_Nm"] / 20, rel=0.05)


@pytest.mark.slow
def test_a_drag_link_turns_fully_and_a_motor_too_weak_for_it_stalls(tmp_path):
    base = ["ground=60", "crank=100", "coupler=120", "rocker=110", "load=3"]
    plan, o = ask("fourbar", "can the motor drive this drag link?", tmp_path / "a",
                  set=base + ["stall_torque=12", "no_load_speed=40rpm"])
    assert o.values["grashof"].startswith("double crank") and o.values["full_turn"] and not o.values["stalled"]
    plan, o = ask("fourbar", "does the motor stall on this drag link?", tmp_path / "b",
                  set=base + ["stall_torque=0.8", "no_load_speed=40rpm"])
    assert o.verdict == "fails" and o.values["stalled"]


@pytest.mark.slow
def test_a_double_rocker_cannot_be_turned_by_any_motor(tmp_path):
    plan, o = ask("fourbar", "can this motor drive the linkage?", tmp_path,
                  set=["ground=100", "crank=60", "coupler=50", "rocker=70", "stall_torque=5", "no_load_speed=30rpm"])
    assert not o.values["full_turn"] and o.verdict == "fails" and "cannot turn all the way round" in o.headline


@pytest.mark.slow
def test_a_slider_crank_against_the_closed_form_torque(tmp_path):
    sets = ["crank=40", "rod=140", "offset=15", "load=30", "motor=ds3218"]
    plan, o = ask("a slider crank", "can a ds3218 servo drive this slider crank against 30 N, what current?", tmp_path,
                  set=sets)
    assert plan.names == ["drive"] and not o.values["stalled"]
    th = np.linspace(0, 2 * math.pi, 7200)
    hand = 30 * float(np.nanmax(np.abs(T.slider_ratio(0.04, 0.14, 0.015, th))))     # N times metres per radian
    assert o.values["needed_slow_Nm"] == pytest.approx(hand, rel=0.03)
    stroke = float(np.nanmax(T.slider_x(0.04, 0.14, 0.015, th)) - np.nanmin(T.slider_x(0.04, 0.14, 0.015, th)))
    assert o.values["stroke_m"] == pytest.approx(stroke, rel=1e-3)
    assert all(c.status == "pass" for c in o.checks), [(c.name, c.summary) for c in o.checks if c.status != "pass"]
    assert o.values["peak_current_A"] > get_motor("ds3218").currents()[0]
    # the playback carries the drive's torque against its limit, the speed and the current, for the 3D scene
    pb = json.loads((tmp_path / "playback.json").read_text())
    assert {"crank speed", "crank current"} <= set(pb["series"]) and "needed" in pb["joints"]["crank"]
    assert pb["joints"]["slider"]["angle"]["unit"] == "m" and (tmp_path / "scene.wobbl").exists()


@pytest.mark.slow
def test_a_small_servo_stalls_on_the_same_slider_crank(tmp_path):
    plan, o = ask("a slider crank", "will an sg90 stall driving this slider crank?", tmp_path,
                  set=["crank=40", "rod=140", "offset=15", "load=30", "motor=sg90"])
    need = o.values["needed_slow_Nm"]
    assert need > get_motor("sg90").stall_torque and o.verdict == "fails" and o.values["stalled"]


@pytest.mark.slow
def test_a_quick_return_linkage_time_ratio_against_its_geometry(tmp_path):
    d, a, b, c = 0.1, 0.04, 0.11, 0.08
    plan, o = ask("fourbar", "what is the quick return ratio of this linkage, and can the motor drive it?", tmp_path,
                  set=["ground=100", "crank=40", "coupler=110", "rocker=80", "stall_torque=3", "no_load_speed=60rpm"])
    th = np.linspace(0, 2 * math.pi, 72000, endpoint=False)
    t4, _ = T.freudenstein(d, a, b, c, th)
    lo, hi = int(np.argmin(t4)), int(np.argmax(t4))
    sweep = ((th[hi] - th[lo]) % (2 * math.pi))
    hand = max(sweep, 2 * math.pi - sweep) / min(sweep, 2 * math.pi - sweep)
    assert hand > 1.1
    assert o.values["quick_return_ratio"] == pytest.approx(hand, rel=0.03)


@pytest.mark.slow
def test_a_urdf_elbow_behind_a_servo(tmp_path):
    f = tmp_path / "arm.urdf"
    f.write_text(URDF)
    plan, o = ask(str(f), "can an mg996r servo drive the elbow from end to end at 20 rpm, what current?",
                  tmp_path / "out")
    assert plan.names == ["drive"]
    vals = o.values
    hold = 0.2 * 9.80665 * 0.08                  # the forearm's weight at its centre, 80 mm out, with the arm level
    assert hold <= vals["drive_Nm"] <= 1.2 * hold
    mot = get_motor("mg996r")
    assert vals["peak_current_A"] == pytest.approx(float(mot.current(vals["peak_torque_Nm"])), rel=0.02)
    assert not vals["stalled"] and vals["peak_rpm"] == pytest.approx(20, rel=0.1)


@pytest.mark.slow
def test_the_wiper_matches_the_legacy_command(tmp_path):
    stored = json.loads((ROOT / "examples/wiper-linkage/output/outcome.json").read_text())["values"]
    sets = ["ground=300", "crank=80", "coupler=300", "rocker=110", "link_mass=80,250,300", "stall_torque=25",
            "no_load_speed=45rpm", "load=6", "sweep=95"]
    plan, o = ask("fourbar", "Does this wiper linkage sweep 95 degrees, and can a 45 rpm wiper motor drive it over "
                  "dry glass?", tmp_path, set=sets)
    assert plan.names == ["drive"]
    for k in ("sweep_deg", "peak_torque_Nm", "rms_torque_Nm", "mean_crank_rpm"):
        assert o.values[k] == pytest.approx(stored[k], rel=1e-3), k
    assert o.values["worst_usage"] < 1 and not o.values["stalled"]
    assert checks(o)["virtual work from the weights and loads"].status == "pass"
