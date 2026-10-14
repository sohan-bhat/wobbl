"""Mechanism spec files read into SI, and the errors a bad one gets. Reading needs no MuJoCo."""
import math

import pytest

from wobbl.mechanism.spec import SpecError, read_spec

ARM = """
name = "two link arm"
timestep = "0.5 ms"
hold = "1 s"

[[body]]
name = "upper"
from = [0, 0, 0]
to = [150, 0, 0]
radius = 10
mass = 80
joint = { name = "shoulder", axis = "-y", range = [-90, 90] }

[[body]]
name = "fore"
parent = "upper"
pos = [150, 0, 0]
to = [120, 0, 0]
mass = "50 g"
joint = "elbow"

[[mass]]
name = "payload"
body = "fore"
mass = "0.5 kg"
pos = [120, 0, 0]

[[drive]]
joint = "shoulder"
servo = "mg996r"

[[drive]]
joint = "elbow"
stall_torque = "20 kg*cm"
no_load_speed = "0.2 s/60deg"
ratio = 2
efficiency = 0.9

[[load]]
joint = "elbow"
torque = "0.1 N*m"

[[move]]
t = 0
shoulder = 0
elbow = 0

[[move]]
t = "1.5 s"
shoulder = 45
elbow = "-0.5 rad"
"""


def test_a_spec_reads_into_si():
    m = read_spec(ARM)
    assert m.name == "two link arm" and m.timestep == pytest.approx(5e-4) and m.hold == 1.0
    up = m.body("upper")
    assert up.fromto == pytest.approx((0, 0, 0, 0.15, 0, 0)) and up.mass == pytest.approx(0.08)
    assert m.joint("shoulder").axis == (0.0, -1.0, 0.0)
    assert m.joint("shoulder").range == pytest.approx((-math.pi / 2, math.pi / 2))
    assert m.joint("elbow").body == "fore" and m.masses[0].mass == pytest.approx(0.5)
    el = m.drive("elbow")
    assert el.motor.stall_torque == pytest.approx(20 * 0.0980665)
    assert el.at_joint.stall_torque == pytest.approx(20 * 0.0980665 * 2 * 0.9)
    assert m.move[1] == (1.5, pytest.approx({"shoulder": math.pi / 4, "elbow": -0.5}))
    assert m.end_time == pytest.approx(2.5)
    assert m.loads[0].kind == "drag" and m.loads[0].torque == pytest.approx(0.1)


@pytest.mark.parametrize("broken,message", [
    ('[[body]]\nname = "a"\nparent = "nowhere"', "not a body"),
    ('[[body]]\nname = "a"\nfrm = [0, 0, 0]', "unknown settings"),
    ('[[drive]]\njoint = "ghost"\nservo = "sg90"', "does not exist"),
    ('[[body]]\nname = "a"\njoint = "j"\n[[move]]\nt = 1\nj = 10\n[[move]]\nt = 0.5\nj = 20', "increasing"),
    ('colour = "red"', "unknown settings"),
])
def test_bad_specs_say_what_is_wrong(broken, message):
    with pytest.raises(SpecError, match=message):
        read_spec(broken)
