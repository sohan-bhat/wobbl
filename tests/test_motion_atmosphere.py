"""The 1976 standard atmosphere, gravity with altitude, and the speed and angle units motion reads."""
import pytest

from wobbl.motion.atmosphere import G0, R_EARTH, air, gravity
from wobbl.units import UnitError, parse_quantity


# geometric altitude (m): density (kg/m3), pressure (Pa), temperature (K), from the published 1976 tables
@pytest.mark.parametrize("h,rho,p,t", [
    (0.0, 1.2250, 101325.0, 288.15),
    (1000.0, 1.1117, 89876.0, 281.65),
    (11000.0, 0.36480, 22700.0, 216.77),
    (20000.0, 0.088910, 5529.3, 216.65),
    (32000.0, 0.013555, 889.06, 228.49),
    (51000.0, 9.0690e-4, 70.458, 270.65),
    (71000.0, 7.1966e-5, 4.4795, 216.85),
    (86000.0, 6.958e-6, 0.37338, 186.95),
])
def test_standard_atmosphere_matches_the_tables(h, rho, p, t):
    r, pp, tt, a = air(h)
    assert r == pytest.approx(rho, rel=2e-4)
    assert pp == pytest.approx(p, rel=2e-4)
    assert tt == pytest.approx(t, abs=0.01)
    assert a == pytest.approx((1.4 * 287.0531 * tt) ** 0.5, rel=1e-4)


def test_atmosphere_carries_on_above_86_km_and_below_sea_level():
    hi, top = air(100000.0), air(86000.0)
    assert hi[2] == pytest.approx(top[2]) and hi[0] < top[0] / 5
    assert air(-200.0)[0] > 1.225 and air(-200.0)[2] > 288.15


def test_gravity_falls_off_with_the_square_of_the_distance():
    assert gravity(0.0) == G0
    assert gravity(R_EARTH) == pytest.approx(G0 / 4)
    assert gravity(3000.0) == pytest.approx(G0 * (1 - 2 * 3000.0 / R_EARTH), rel=1e-6)


@pytest.mark.parametrize("text,kind,value", [
    ("5 m/s", "speed", 5000.0),
    ("10 mph", "speed", 4470.4),
    ("36 km/h", "speed", 10000.0),
    ("50 ft/s", "speed", 15240.0),
    ("20 knots", "speed", 10288.9),
    ("5 deg", "angle", 5.0),
    ("5°", "angle", 5.0),
    ("0.5 rad", "angle", 28.6479),
])
def test_speed_and_angle_units(text, kind, value):
    assert parse_quantity(text, kind).value == pytest.approx(value, rel=1e-4)


def test_speed_needs_a_unit():
    with pytest.raises(UnitError):
        parse_quantity("5", "speed")
    with pytest.raises(UnitError):
        parse_quantity("5 furlongs/fortnight", "speed")
