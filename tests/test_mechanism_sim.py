"""MuJoCo runs of small mechanisms against exact answers. Skipped when MuJoCo is not installed."""
import math

import numpy as np
import pytest

from wobbl.mechanism import install
from wobbl.mechanism import theory as T

pytestmark = pytest.mark.skipif(not install.available(), reason="MuJoCo is not installed (wobbl setup --only mujoco)")


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
