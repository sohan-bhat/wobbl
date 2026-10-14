"""Mechanism units and motors: reading values into SI, torque speed lines and the servo catalog."""
import math

import numpy as np
import pytest

from wobbl.mechanism import quantities as Q
from wobbl.mechanism.motors import CATALOG, get_motor, motor_from, stronger


@pytest.mark.parametrize("text,read,value", [
    ("120", Q.length, 0.12),
    ("2 in", Q.length, 0.0508),
    ("500", Q.mass, 0.5),
    ("1.2 kg", Q.mass, 1.2),
    ("11 kg*cm", Q.torque, 1.0787),
    ("11 kgcm", Q.torque, 1.0787),
    ("1.5", Q.torque, 1.5),
    ("150 N*mm", Q.torque, 0.15),
    ("25 oz*in", Q.torque, 0.17654),
    ("90", Q.angle, math.pi / 2),
    ("1 rad", Q.angle, 1.0),
    ("60 rpm", Q.speed, 2 * math.pi),
    ("0.14 s/60deg", Q.speed, 7.4800),
    ("0.14 s/60°", Q.speed, 7.4800),
    ("2 m/s", Q.velocity, 2.0),
    ("1 g", Q.accel, 9.80665),
])
def test_quantities_read_into_si(text, read, value):
    assert read(text) == pytest.approx(value, rel=1e-4)


def test_quantities_refuse_nonsense():
    with pytest.raises(ValueError):
        Q.speed("3 parsecs")
    with pytest.raises(ValueError):
        Q.angle("90 furlongs")
    assert Q.listed("120, 95 mm,60", Q.length) == pytest.approx([0.12, 0.095, 0.06])


def test_a_dc_motor_line_and_its_quadrants():
    m = get_motor("mg996r")
    assert m.stall_torque == pytest.approx(11 * Q.KGF_CM)
    assert m.no_load_speed == pytest.approx(math.radians(60) / 0.14)
    lo, hi = m.limit(0.0)
    assert hi == pytest.approx(m.stall_torque) and lo == pytest.approx(-m.stall_torque)
    w = 0.5 * m.no_load_speed
    lo, hi = m.limit(w)
    assert hi == pytest.approx(0.5 * m.stall_torque)          # driving: half speed leaves half the torque
    assert lo == pytest.approx(-m.stall_torque)               # braking is capped at stall
    assert m.limit(1.2 * m.no_load_speed)[1] < 0               # past no load speed it can only brake
    np.testing.assert_allclose(m.usage([0.5 * m.stall_torque, -m.stall_torque], [w, w]), [1.0, 1.0])
    assert m.rated == pytest.approx(m.stall_torque / 3)


def test_box_limits_from_a_urdf_style_rating():
    m = motor_from({"stall_torque": "2 N*m", "no_load_speed": "30 rpm", "curve": "box"})
    assert m.limit(0.5 * m.no_load_speed) == (-2.0, 2.0)
    assert m.limit(2 * m.no_load_speed)[1] == 0.0


def test_motor_tables_catalog_names_and_gearing():
    m = motor_from({"servo": "mg996r", "continuous_torque": "0.4 N*m"})
    assert m.rated == pytest.approx(0.4) and m.stall_torque == CATALOG["mg996r"].stall_torque
    g = get_motor("mg996r").geared(3, 0.9)
    assert g.stall_torque == pytest.approx(CATALOG["mg996r"].stall_torque * 2.7)
    assert g.no_load_speed == pytest.approx(CATALOG["mg996r"].no_load_speed / 3)
    with pytest.raises(ValueError, match="mg996r"):
        get_motor("mg969r")
    with pytest.raises(ValueError):
        motor_from({"no_load_speed": "60 rpm"})


def test_the_stronger_servo_is_the_lightest_that_clears_the_need():
    pick = stronger(get_motor("mg996r"), need_torque=1.5, need_speed=1.0)
    assert pick is not None and pick.limit(1.0)[1] >= 1.25 * 1.5
    assert all(m.mass >= pick.mass for m in CATALOG.values() if m.limit(1.0)[1] >= 1.875 and not m.no_load_current
               and m.key != "mg996r")
    assert stronger(get_motor("mg996r"), need_torque=50.0) is None


def test_max_efficiency_point_of_a_dc_motor():
    m = get_motor("fa-130")
    share = 0.15 / 2.2
    want = m.stall_torque * (math.sqrt(share) - share) / (1 - share)
    assert m.max_efficiency_torque() == pytest.approx(want)
    assert 0.1 * m.stall_torque < want < 0.3 * m.stall_torque
