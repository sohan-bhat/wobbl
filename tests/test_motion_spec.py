"""What flies: motors and their curves, drag against Mach and Barrowman's center of pressure."""
import math

import pytest

from wobbl.motion.spec import TRANSONIC, Airframe, Fins, MotionStudy, Motor, SpecError, barrowman


def test_motor_impulse_exhaust_speed_and_burn():
    m = Motor("test", [(0.1, 1000.0), (1.0, 1000.0), (1.1, 0.0)], 0.5, 1.2)
    assert m.curve[0] == (0.0, 0.0)                 # ramps from zero at ignition
    assert m.impulse == pytest.approx(1000.0)
    assert m.exhaust_speed == pytest.approx(2000.0)
    assert m.burn_time == 1.1
    assert m.thrust(0.05) == pytest.approx(500.0) and m.thrust(1.2) == 0.0
    assert m.burned(0.55) == pytest.approx(0.5 * 500.0 / 1000.0)
    assert m.burned(0.1) == pytest.approx(0.5 * 50.0 / 1000.0) and m.burned(5.0) == 0.5


def test_motor_rejects_bad_curves_and_masses():
    with pytest.raises(SpecError):
        Motor("back", [(0.0, 0.0), (0.5, 100.0), (0.4, 50.0)], 0.1, 0.2)
    with pytest.raises(SpecError):
        Motor("heavy", [(0.0, 0.0), (1.0, 100.0)], 0.3, 0.2)


def test_drag_coefficient_against_mach():
    cd = MotionStudy(cd=0.45, transonic=True).drag_coefficient()
    peak = dict(TRANSONIC)[1.1]
    assert cd(0.3) == pytest.approx(0.45) and cd(1.1) == pytest.approx(0.45 * peak)
    assert cd(5.0) == pytest.approx(0.45 * TRANSONIC[-1][1])
    table = MotionStudy(cd=[(0.0, 0.5), (1.0, 0.9), (2.0, 0.6)]).drag_coefficient()
    assert table(0.0) == 0.5 and table(1.0) == pytest.approx(0.9) and table(3.0) == 0.6
    assert 0.5 < table(0.5) < 0.9
    assert MotionStudy(cd=0.6).drag_coefficient()(2.0) == 0.6


@pytest.mark.parametrize("shape,fraction", [("cone", 2 / 3), ("elliptical", 1 / 3), ("parabolic", 7 / 15),
                                            ("haack", 0.5)])
def test_nose_center_of_pressure_from_its_volume(shape, fraction):
    af = Airframe(1.0, 0.3, shape, Fins(3, 0.1, 0.05, 0.05, 0.05, 0.9))
    assert barrowman(af, 0.1)["x_nose"] == pytest.approx(fraction * 0.3, rel=1e-4)


def test_tangent_ogive_sits_where_barrowman_puts_it():
    af = Airframe(2.1, 0.45, "ogive", Fins(3, 0.2, 0.07, 0.09, 0.11, 1.9))
    assert barrowman(af, 0.098)["x_nose"] / 0.45 == pytest.approx(0.466, abs=0.01)


def test_fin_normal_force_and_center_by_hand():
    """Barrowman's fin terms for the example rocket's fins, written out step by step."""
    n, cr, ct, s, xr, xb, d = 3, 0.20, 0.07, 0.09, 0.11, 1.90, 0.098
    r = d / 2
    lf = math.sqrt(s ** 2 + (xr + ct / 2 - cr / 2) ** 2)
    cn = (1 + r / (s + r)) * (4 * n * (s / d) ** 2) / (1 + math.sqrt(1 + (2 * lf / (cr + ct)) ** 2))
    xf = xb + xr / 3 * (cr + 2 * ct) / (cr + ct) + (cr + ct - cr * ct / (cr + ct)) / 6
    bm = barrowman(Airframe(2.1, 0.45, "ogive", Fins(n, cr, ct, s, xr, xb)), d)
    assert bm["cn_fins"] == pytest.approx(cn) and bm["x_fins"] == pytest.approx(xf)
    assert bm["cp"] == pytest.approx((2 * bm["x_nose"] + cn * xf) / (2 + cn))
