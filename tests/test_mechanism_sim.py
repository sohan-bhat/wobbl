"""MuJoCo runs of small mechanisms against exact answers. Skipped when MuJoCo is not installed."""
import math

import numpy as np
import pytest

from wobbl.mechanism import install
from wobbl.mechanism import theory as T

pytestmark = pytest.mark.skipif(not install.available(), reason="MuJoCo is not installed (wobbl setup --only mujoco)")


def test_a_large_swing_keeps_the_elliptic_period_and_its_energy():
    from wobbl.mechanism.sim import simulate
    from wobbl.mechanism.templates import template

    b = template("pendulum", {"angle": "150 deg", "periods": 2})
    run = simulate(b.mech, samples=4000, frames=0)
    v = run.v[:, 0]
    turns = np.nonzero(np.sign(v[:-1]) * np.sign(v[1:]) < 0)[0]          # the turning points, two a period
    tc = run.t[turns] + (run.t[turns + 1] - run.t[turns]) * v[turns] / (v[turns] - v[turns + 1])
    assert 2 * np.mean(np.diff(tc)) == pytest.approx(b.info["period_s"], rel=1e-4)
    end, worst, _ = run.energy_residual()
    assert worst < 0.01


def test_a_four_bar_follows_freudenstein_and_closes_its_energy_account():
    from wobbl.mechanism.sim import simulate
    from wobbl.mechanism.templates import template

    b = template("fourbar", {"speed": "30 rpm", "turns": 1, "load": "0.5 N*m"})
    run = simulate(b.mech, frames=0)
    d, a, bl, c = b.info["lengths"]
    t2s, _, t4s = b.info["start"]
    th2 = t2s + run.q[:, run.col("crank")]
    th4 = t4s + run.q[:, run.col("rocker")]
    t4f, _ = T.freudenstein(d, a, bl, c, th2)
    err = np.degrees(np.abs(np.angle(np.exp(1j * (th4 - t4f)))))
    assert err.max() < 0.05
    assert run.eq_err.max() < 1e-5
    assert run.energy_residual()[1] < 0.01


def test_a_gear_train_turns_at_its_ratio_and_balances_the_motor_line():
    from wobbl.mechanism.sim import simulate
    from wobbl.mechanism.templates import template

    b = template("gears", {"teeth": "12:36,10:40", "load": "0.005 N*m"})
    run = simulate(b.mech, frames=0)
    last = run.t > 0.9 * run.t[-1]
    w_in, w_out = run.v[last, run.col("shaft0")].mean(), run.v[last, run.col("shaft2")].mean()
    assert abs(w_in / w_out) == pytest.approx(12.0, rel=1e-4)
    assert w_in == pytest.approx(b.info["hand_motor_speed"], rel=2e-3)
    assert run.energy_residual()[1] < 1e-3


@pytest.mark.parametrize("e", [1.0, 0.5])
def test_two_balls_trade_momentum_and_keep_their_restitution(e):
    from wobbl.mechanism.sim import simulate
    from wobbl.mechanism.templates import template

    b = template("impact", {"mass": "1 kg", "target_mass": "3 kg", "speed": "1 m/s", "restitution": e,
                            "stiffness": "2e5 N/m"})
    run = simulate(b.mech, samples=2000, frames=0)
    v1, v2 = run.free["part_free"]["vel"][-1][0], run.free["target_free"]["vel"][-1][0]
    x1, x2 = T.collision(1.0, 1.0, 3.0, 0.0, e)
    assert (v1, v2) == pytest.approx((x1, x2), abs=2e-3)
    assert run.momentum[-1][0] == pytest.approx(1.0, abs=1e-9)
    ke = 0.5 * v1 ** 2 + 1.5 * v2 ** 2
    assert ke == pytest.approx(0.5 * (1 - (1 - e * e) * 0.75), rel=5e-3)


def test_inverse_dynamics_holds_a_two_link_arm_like_hand_statics():
    from wobbl.mechanism.sim import inverse_along, simulate
    from wobbl.mechanism.spec import read_spec

    spec = read_spec("""
hold = "0.5 s"
[[body]]
name = "upper"
to = [150, 0, 0]
mass = 80
joint = { name = "shoulder", axis = "-y" }
[[body]]
name = "fore"
parent = "upper"
pos = [150, 0, 0]
to = [120, 0, 0]
mass = 50
joint = { name = "elbow", axis = "-y" }
[[mass]]
name = "payload"
body = "fore"
mass = 200
pos = [120, 0, 0]
[[drive]]
joint = "shoulder"
servo = "ds3218"
[[drive]]
joint = "elbow"
servo = "ds3218"
[[move]]
t = 0
shoulder = 30
elbow = -45
""")
    tau = T.arm_statics([0.15, 0.12], [math.radians(30), math.radians(-45)],
                        [(0, 0.075, 0.08), (1, 0.06, 0.05), (1, 0.12, 0.2)])
    names, need, com = inverse_along(spec, np.array([0.0, 0.25]))
    assert names == ["elbow", "shoulder"]
    assert need[1] == pytest.approx([tau[1], tau[0]], rel=1e-9)
    run = simulate(spec, frames=0)               # strong enough servos hold the pose
    assert np.nanmax(np.abs(run.q - run.q_des)) < math.radians(0.1)
    assert run.joints == ["shoulder", "elbow"]
    assert run.tau[-1] == pytest.approx(tau, rel=1e-3)


def test_a_weak_servo_falls_behind_and_a_spin_keeps_its_speed():
    import dataclasses

    from wobbl.mechanism.sim import simulate
    from wobbl.mechanism.spec import Drive
    from wobbl.mechanism.templates import template

    b = template("arm", {"servo": "sg90", "payload": "200 g"})
    run = simulate(b.mech, frames=0)
    c = run.col("shoulder")
    assert np.nanmax(np.abs(run.q[:, c] - run.q_des[:, c])) > math.radians(10)
    assert np.any(np.abs(run.tau_need[:, c]) > np.abs(run.tau[:, c]) + 1e-6)     # pinned at its line
    fb = template("fourbar", {"speed": "20 rpm", "turns": 1})
    spun = simulate(dataclasses.replace(fb.mech, drives=[Drive("crank", None, "servo")]), frames=0)
    late = spun.t > 0.5
    assert spun.v[late, spun.col("crank")] == pytest.approx(20 * 2 * math.pi / 60, rel=1e-3)
