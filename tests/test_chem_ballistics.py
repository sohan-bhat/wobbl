"""Internal ballistics against closed forms: a steady Kn holds the closed form pressure, mass is conserved."""
import math

import numpy as np
import pytest

from wobbl.chem.ballistics import Motor, Nozzle, converge, motor_class, simulate, steady_pressure, thrust_coefficient
from wobbl.chem.grains import make_grain
from wobbl.chem.propellants import PROPELLANTS

KNSU = PROPELLANTS["knsu"]       # one burn rate band, so the closed form holds at every pressure


def _closed_form(prop, kn):
    """(a rho c* Kn)^(1/(1-n)) with the gas density term, by fixed point."""
    a, n = prop.burn_rate.bands[0][2:]
    a_si = a / 1000 * 1e-6 ** n
    k = prop.k
    gam = math.sqrt(k) * (2 / (k + 1)) ** ((k + 1) / (2 * (k - 1)))
    RT = (prop.cstar * gam) ** 2
    P = 3e6
    for _ in range(200):
        P = (a_si * (prop.density - P / RT) * prop.cstar * kn) ** (1 / (1 - n))
    return P


def test_end_burner_holds_the_closed_form_pressure():
    g = make_grain("end", 0.05, 0.10)
    At = g.burn_area(0) / 150
    noz = Nozzle(math.sqrt(4 * At / math.pi), 0.012)
    m = Motor([g], KNSU, noz, chamber_volume=math.pi / 4 * 0.05 ** 2 * 0.11)
    b, runs = converge(m)
    P_cf = _closed_form(KNSU, 150)
    mid = (b.t > 0.3 * b.t[-1]) & (b.t < 0.7 * b.t[-1])
    assert np.median(b.P[mid]) == pytest.approx(P_cf, rel=0.002)
    assert steady_pressure(m, 0.05) == pytest.approx(P_cf, rel=1e-6)
    assert b.burn_time == pytest.approx(0.10 / KNSU.burn_rate.rate(P_cf), rel=0.02)
    assert abs(b.expelled + b.gas_end - b.gas_start - b.mass) / b.mass < 1e-3
    assert len(runs) >= 2 and runs[-1]["change"] < 0.003


def test_neutral_tube_burns_flat_for_its_web():
    g = make_grain("tube", 0.04, 0.10, ends=0, core=0.015)
    At = g.burn_area(0) / 150
    m = Motor([g], KNSU, Nozzle(math.sqrt(4 * At / math.pi), 0.03), chamber_volume=math.pi / 4 * 0.05 ** 2 * 0.1)
    b = simulate(m)
    P_cf = _closed_form(KNSU, 150)
    assert b.peak_pressure == pytest.approx(P_cf, rel=0.003)
    assert b.burn_time == pytest.approx(g.web / KNSU.burn_rate.rate(P_cf), rel=0.04)


def test_thrust_coefficient_and_classes():
    k, eps = 1.2, 8.0
    from wobbl.chem.thermo import _ideal_exit_ratio
    pr = _ideal_exit_ratio(eps, k)
    P = 101325 / pr                           # matched exit: no pressure term
    gam = math.sqrt(k) * (2 / (k + 1)) ** ((k + 1) / (2 * (k - 1)))
    ideal = gam * math.sqrt(2 * k / (k - 1) * (1 - pr ** ((k - 1) / k)))
    assert thrust_coefficient(k, eps, P, 101325, pr) == pytest.approx(ideal)
    assert thrust_coefficient(k, eps, 101325 * 1.2, 101325, pr) < thrust_coefficient(k, eps, P, 101325, pr)
    assert thrust_coefficient(k, eps, 90000, 101325, pr) == 0.0
    assert [motor_class(i) for i in (2.0, 2.6, 640, 641, 1300, 31540)] == ["A", "B", "I", "J", "K", "O"]
