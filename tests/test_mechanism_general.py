"""The general mechanism analyses on small models with exact answers: holding torque, reach, collisions, standing
forces, tipping and balance. Skipped when MuJoCo is not installed."""
import math

import pytest

from wobbl.mechanism import install

pytestmark = pytest.mark.skipif(not install.available(), reason="MuJoCo is not installed (wobbl setup --only mujoco)")
G = 9.81

ARM = """<mujoco>
  <option gravity="0 0 -9.81"/>
  <worldbody>
    <geom name="table" type="box" size="0.3 0.3 0.02" pos="0 0 -0.02"/>
    <body name="base" pos="0 0 0.05">
      <geom type="box" size="0.04 0.04 0.05" mass="1"/>
      <body name="upper" pos="0 0 0.05">
        <joint name="shoulder" axis="0 1 0" range="-90 90"/>
        <geom type="capsule" fromto="0 0 0 0.2 0 0" size="0.01" mass="0.5"/>
        <body name="fore" pos="0.2 0 0">
          <joint name="elbow" axis="0 1 0" range="-150 150"/>
          <geom type="capsule" fromto="0 0 0 0.15 0 0" size="0.01" mass="0.3"/>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor joint="shoulder" forcerange="-2 2" forcelimited="true"/>
    <motor joint="elbow" forcerange="-0.5 0.5" forcelimited="true"/>
  </actuator>
</mujoco>
"""

TABLE = """<mujoco>
  <worldbody>
    <geom name="floor" type="plane" size="2 2 0.1"/>
    <body name="top" pos="0 0 0.5">
      <freejoint/>
      <geom type="box" size="0.4 0.2 0.02" mass="8"/>
      <body name="legs" pos="0 0 0">
        <joint name="fold" axis="1 0 0" range="-0.01 0.01"/>
        <geom name="l1" type="capsule" fromto="0.35 0.15 0 0.35 0.15 -0.48" size="0.02" mass="0.5"/>
        <geom name="l2" type="capsule" fromto="-0.35 0.15 0 -0.35 0.15 -0.48" size="0.02" mass="0.5"/>
        <geom name="l3" type="capsule" fromto="0.35 -0.15 0 0.35 -0.15 -0.48" size="0.02" mass="0.5"/>
        <geom name="l4" type="capsule" fromto="-0.35 -0.15 0 -0.35 -0.15 -0.48" size="0.02" mass="0.5"/>
      </body>
    </body>
  </worldbody>
  <actuator><motor joint="fold" forcerange="-100 100" forcelimited="true"/></actuator>
</mujoco>
"""

ROD = """<mujoco>
  <option gravity="0 0 -9.81"/>
  <worldbody>
    <body name="rod" pos="0 0 0">
      <joint name="pivot" axis="0 1 0"/>
      <geom type="capsule" fromto="0 0 0 0 0 {z}" size="0.002" mass="1"/>
    </body>
  </worldbody>
</mujoco>
"""


def _run(tmp_path, xml, question, params=None, name="m.xml"):
    from wobbl.mechanism.study import run_mechanism

    (tmp_path / name).write_text(xml)
    return run_mechanism(str(tmp_path / name), params or {}, tmp_path / "out", question)


def test_the_worst_holding_torque_is_the_arm_held_level_and_the_weak_elbow_fails(tmp_path):
    o = _run(tmp_path, ARM.replace("-0.5 0.5", "-0.02 0.02"), "can the motors hold the arm?")
    j = o.values["joints"]
    # level, the shoulder carries the upper arm's middle and the whole forearm, the elbow its own middle
    shoulder = G * (0.5 * 0.1 + 0.3 * (0.2 + 0.075))
    elbow = G * 0.3 * 0.075
    assert abs(j["shoulder"]["needed"]) == pytest.approx(shoulder, rel=2e-3)
    assert abs(j["elbow"]["needed"]) == pytest.approx(elbow, rel=2e-3)
    assert o.verdict == "fails" and "elbow cannot hold" in o.headline      # 0.22 N m against 0.02
    st = {c.name: c.status for c in o.checks}
    assert st["hand statics"] == "pass" and st["worst pose search"] == "pass"


def test_a_payload_moves_the_worst_torque_and_its_limit_is_solved_for(tmp_path):
    o = _run(tmp_path, ARM, "can it hold 300 g at the tip?")
    pay = next(f for f in o.fixes if f["kind"] == "payload")
    top = pay["after"]["max_payload_kg"]
    # held at the end of the forearm, 0.16 m past the elbow, the elbow's 0.5 N m runs out first
    assert top == pytest.approx((0.5 - G * 0.3 * 0.075) / (G * 0.16), rel=0.02)


def test_reach_is_the_straight_arm_and_it_can_strike_the_table(tmp_path):
    o = _run(tmp_path, ARM, "how far does it reach, and does it hit anything?")
    k = o.values["kinematics"]
    assert k["reach_m"] == pytest.approx(0.05 + 0.2 + 0.16, abs=2e-3)       # pointing straight up
    c = o.values["collision"]
    assert c["floor_share"] > 0                       # swung down past level, the forearm meets the table
    assert {x.split(", ")[-1] for x in (ch.name for ch in o.checks)} >= {"chain length", "pose sampling"}


def test_a_table_on_four_legs_shares_its_weight_and_stands(tmp_path):
    o = _run(tmp_path, TABLE, "what does each leg carry, and is it stable?")
    f = o.values["contact"]["forces_N"]
    weight = 10 * G
    assert sum(f.values()) == pytest.approx(weight, rel=0.02)
    assert o.values["stability"]["margin_m"] > 0.1 and o.verdict in ("ok", "marginal")
    st = {c.name: c.status for c in o.checks}
    assert st["contact, hand statics"] == "pass" and st["stability, tipping statics"] == "pass"


@pytest.mark.parametrize("up", [True, False])
def test_a_rod_balanced_upright_falls_and_a_hanging_one_swings(tmp_path, up):
    L = 0.5
    o = _run(tmp_path, ROD.replace("{z}", str(L if up else -L)), "does it fall over if nothing holds it?")
    if up:
        assert o.verdict == "fails" and "falls over" in o.headline
        st = {c.name: c.status for c in o.checks}
        assert st["linear stability"] == "pass"
        # a uniform rod on its end: theta'' = (3 g / 2 L) theta
        rate = math.sqrt(3 * G / (2 * L))
        assert f"{math.acosh(10) / rate:.3g}"[:3] in o.headline
    else:
        assert o.verdict == "ok" and "stable" in o.headline
        assert f"{2 * math.pi * math.sqrt(2 * L / (3 * G)):.3g}"[:3] in o.headline


def test_questions_pick_the_physics():
    from types import SimpleNamespace

    from wobbl.mechanism.general import choose

    fixed = SimpleNamespace(floating=False, joints=[{"name": "a"}])
    legs = SimpleNamespace(floating=True, joints=[{"name": "a"}])
    assert choose("can these servos lift 500 g?", fixed, False, False) == ["torque"]
    assert choose("how far can it reach?", fixed, False, False) == ["kinematics"]
    assert choose("does it collide with itself?", fixed, False, False) == ["collision"]
    assert choose("what does each foot carry?", legs, False, False) == ["contact"]
    assert choose("will it tip over?", legs, False, False) == ["stability"]
    assert choose("", fixed, False, False) == ["torque", "kinematics", "collision"]
    assert choose("", legs, False, False) == ["contact", "stability", "torque"]
    assert choose("does it fall over if nothing holds it?", fixed, False, True) == ["dynamics"]
