"""The MJCF Wobbl writes for MuJoCo from a mechanism spec. Writing it needs no MuJoCo."""
import xml.etree.ElementTree as ET

import pytest

from test_mechanism_spec import ARM
from wobbl.mechanism.mjcf import HARD, to_mjcf
from wobbl.mechanism.spec import read_spec


def test_mjcf_has_the_bodies_joints_and_constraints():
    m = read_spec(ARM + '\n[[loop]]\nbody1 = "fore"\nbody2 = "upper"\npoint = [120, 0, 0]\n')
    root = ET.fromstring(to_mjcf(m))
    assert root.find("option").get("timestep") == "0.0005"
    assert root.find("option/flag").get("energy") == "enable"
    upper = root.find("worldbody/body[@name='upper']")
    assert upper.find("joint").get("axis") == "0 -1 0" and upper.find("joint").get("range") is not None
    fore = upper.find("body[@name='fore']")
    assert fore.get("pos") == "0.15 0 0"
    payload = fore.find("geom[@name='payload']")
    assert payload.get("mass") == "0.5" and payload.get("contype") == "0"
    connect = root.find("equality/connect")
    assert connect.get("solimp") == HARD and connect.get("anchor") == "0.12 0 0"
    assert root.find("contact/exclude").get("body1") == "fore"
    assert root.find("actuator") is None          # Wobbl applies every drive itself


def test_gear_couplings_and_direct_contacts():
    m = read_spec("""
[contact]
stiffness = 1e4
damping = 20
condim = 1
[[body]]
name = "a"
shape = "none"
joint = "a"
[[body]]
name = "b"
shape = "none"
pos = [30, 0, 0]
joint = "b"
[[coupling]]
driver = "a"
driven = "b"
ratio = 3
""")
    root = ET.fromstring(to_mjcf(m))
    eq = root.find("equality/joint")
    assert eq.get("joint1") == "b" and eq.get("joint2") == "a"
    assert float(eq.get("polycoef").split()[1]) == pytest.approx(-1 / 3)     # a gear turns the other way
    g = root.find("default/geom")
    assert g.get("solref") == "-10000 -20" and g.get("condim") == "1"
