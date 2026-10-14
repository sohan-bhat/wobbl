"""The closed forms the mechanism checks compare MuJoCo against, each tested on its own."""
import math

import numpy as np
import pytest

from wobbl.mechanism import theory as T


def test_pendulum_period_reaches_the_small_angle_limit():
    assert T.pendulum_period(1.0, 1e-4) == pytest.approx(2 * math.pi / math.sqrt(T.G), rel=1e-8)
    # the classic table value: 90 degrees swings 18.0% slower than small angles
    assert T.pendulum_period(1.0, math.pi / 2) / (2 * math.pi / math.sqrt(T.G)) == pytest.approx(1.18034, rel=1e-5)


def test_freudenstein_closes_the_loop_at_every_crank_angle():
    d, a, b, c = 0.12, 0.03, 0.125, 0.045
    t2 = np.linspace(0, 2 * np.pi, 73)
    t4, t3 = T.freudenstein(d, a, b, c, t2)
    pin_a = np.stack([a * np.cos(t2), a * np.sin(t2)], 1)
    pin_b = np.stack([d + c * np.cos(t4), c * np.sin(t4)], 1)
    np.testing.assert_allclose(np.linalg.norm(pin_b - pin_a, axis=1), b, rtol=1e-10)
    np.testing.assert_allclose(np.arctan2(*(pin_b - pin_a)[:, ::-1].T), t3, atol=1e-10)
    assert T.grashof(d, a, b, c) == ("crank rocker", True)
    assert T.grashof(0.1, 0.08, 0.03, 0.09)[1] is False


def test_velocity_ratio_matches_a_finite_difference():
    d, a, b, c = 0.12, 0.03, 0.125, 0.045
    t2 = np.linspace(0.1, 6.0, 30)
    h = 1e-6
    t4, t3 = T.freudenstein(d, a, b, c, t2)
    t4h, _ = T.freudenstein(d, a, b, c, t2 + h)
    np.testing.assert_allclose(T.velocity_ratio(a, c, t2, t3, t4), (t4h - t4) / h, rtol=1e-4, atol=1e-6)


def test_restitution_of_a_one_sided_dashpot():
    assert T.dashpot_restitution(0.0) == 1.0
    # Schwager and Poschel's result, which a direct integration of x'' = -2 zeta x' - x until the force reaches
    # zero reproduces to 1e-7, and which differs from the naive exp(-pi zeta / sqrt(1 - zeta^2)) = 0.729 and 0.373
    assert T.dashpot_restitution(0.1) == pytest.approx(0.744079, abs=1e-6)
    assert T.dashpot_restitution(0.3) == pytest.approx(0.450975, abs=1e-6)
    for e in (0.95, 0.6, 0.3, 0.1):
        assert T.dashpot_restitution(T.zeta_for_restitution(e)) == pytest.approx(e, abs=1e-9)
    # continuous across both branch points
    for z in (1 / math.sqrt(2), 1.0):
        assert T.dashpot_restitution(z - 1e-7) == pytest.approx(T.dashpot_restitution(z + 1e-7), abs=1e-5)


def test_collision_conserves_momentum_and_honors_restitution():
    v1, v2 = T.collision(1.0, 1.0, 3.0, 0.0, 1.0)
    assert (v1, v2) == pytest.approx((-0.5, 0.5))
    v1, v2 = T.collision(2.0, 3.0, 1.0, -1.0, 0.5)
    assert 2 * v1 + v2 == pytest.approx(5.0) and v2 - v1 == pytest.approx(0.5 * 4.0)


def test_hertz_impact_energy_and_linear_peak():
    hz = T.hertz_impact(0.75, 1.0, 1e11, 0.025)
    k = 4 / 3 * 1e11 * math.sqrt(0.025)
    assert 0.4 * k * hz["delta_m"] ** 2.5 == pytest.approx(0.5 * 0.75 * 1.0, rel=1e-9)   # all the energy stored
    w0 = 100.0
    assert T.dashpot_peak(1.0, 2.0, w0, 0.0) == pytest.approx(w0 * 2.0, rel=1e-6)


def test_tipping_moments_and_arm_statics():
    r = T.tipping([(1.0, 0.0, 0.1)], edge=0.05, force=1.0, force_height=0.2)
    assert r["restoring_Nm"] == pytest.approx(T.G * 0.05) and r["critical_force_N"] == pytest.approx(T.G * 0.25)
    tau = T.arm_statics([0.15, 0.12], [math.radians(30), math.radians(-45)],
                        [(0, 0.075, 0.08), (0, 0.15, 0.055), (1, 0.06, 0.05), (1, 0.12, 0.5)])
    c1, c12 = math.cos(math.radians(30)), math.cos(math.radians(-15))
    assert tau[1] == pytest.approx(T.G * c12 * (0.05 * 0.06 + 0.5 * 0.12))
    assert tau[0] == pytest.approx(T.G * (c1 * (0.08 * 0.075 + (0.055 + 0.05 + 0.5) * 0.15)) + tau[1])
    t1, t2 = T.arm_ik(0.15, 0.12, 0.2, 0.05)
    assert 0.15 * math.cos(t1) + 0.12 * math.cos(t1 + t2) == pytest.approx(0.2)
    assert 0.15 * math.sin(t1) + 0.12 * math.sin(t1 + t2) == pytest.approx(0.05)
    assert t2 < 0 and T.arm_ik(0.15, 0.12, 0.5, 0.0) is None
