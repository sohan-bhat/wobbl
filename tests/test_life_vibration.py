"""Vibration inputs, the resonance verdict and the rigid attachment fit, without a solver."""
import numpy as np
import pytest

from wobbl.life import ccx, modal
from wobbl.life import vibchecks as vc
from wobbl.life.vibspec import (default_duration, parse_frequency, parse_mass, parse_rpm, parse_sine,
                                VibrationStudy)
from wobbl.structural.spec import SpecError, StructuralStudy


def test_masses():
    m = parse_mass("45 g at 150,0,25 @ top and within 15 of 150,0,15")
    assert m.mass == pytest.approx(45e-6) and m.point == [150, 0, 25] and m.rigid
    assert m.faces == "top and within 15 of 150,0,15"
    d = parse_mass("0.3 kg @ top distributed")
    assert d.mass == pytest.approx(0.3e-3) and not d.rigid and d.faces == "top"
    assert not parse_mass("20 g @ x=max").rigid       # no center given: spread over the faces
    with pytest.raises(SpecError):
        parse_mass("45 g")
    with pytest.raises(SpecError):
        parse_mass("45 N @ top")


def test_rotor_speeds_and_tones():
    once, blade = parse_rpm("4000-9000 rpm", blades=2)
    assert (once.lo, once.hi) == pytest.approx((4000 / 60, 150.0)) and not once.tone
    assert (blade.lo, blade.hi) == pytest.approx((8000 / 60, 300.0)) and "2 blades" in blade.label
    single = parse_rpm("6000", blades=1)
    assert len(single) == 1 and single[0].tone and single[0].lo == pytest.approx(100.0)
    assert parse_frequency("2 kHz").lo == 2000.0 and parse_frequency("100-150 Hz").hi == 150.0
    with pytest.raises(SpecError):
        parse_rpm("fast")
    st = VibrationStudy(StructuralStudy("p.step"), rpm=["6000 rpm"], frequencies=["120 Hz"], blades=3)
    assert [round(e.lo) for e in st.excitations()] == [100, 300, 120]


def test_sine_sweep_and_durations():
    s = parse_sine("2 g z 10-500 Hz")
    assert s.accel == pytest.approx(2 * modal.G0) and s.direction == [0, 0, 1] and (s.lo, s.hi) == (10, 500)
    assert (parse_sine("1 g 1,0,0 from 20 Hz to 2 kHz").lo, parse_sine("1 g 1,0,0 20 Hz to 2 kHz").hi) == (20, 2000)
    with pytest.raises(SpecError):
        parse_sine("2 g z")
    with pytest.raises(SpecError):
        parse_sine("1 g x 20 to 2 khz")      # both ends in kHz: 20 kHz down to 2 kHz is backwards
    assert default_duration("gevs-qual")[0] == 120 and default_duration("gevs-accept")[0] == 60
    assert default_duration("20:0.1, 100:0.1")[0] == 0


def _modes(freqs, frac_z):
    m = len(freqs)
    part = np.zeros((m, 6))
    part[:, 2] = np.sqrt(frac_z)
    eff = part ** 2
    return modal.Modes(np.array(freqs, float), np.zeros((m, 2, 3), np.float32), None, part, eff,
                       np.array([1.0, 1.0, 1.0, 0, 0, 0]), 2)


def test_resonance_verdicts():
    ex = parse_rpm("6000 rpm", blades=2)                 # 100 Hz and 200 Hz
    modes = _modes([150.0, 400.0], [0.6, 0.2])
    rows, v = vc.resonance(modes, modes.effective_mass[:, :3], ex, 0.2, 0.02)
    assert v == "holds" and all(r["status"] == "clear" for r in rows)
    modes = _modes([105.0, 400.0], [0.6, 0.2])          # 5% from the once per turn tone
    rows, v = vc.resonance(modes, modes.effective_mass[:, :3], ex, 0.2, 0.02)
    assert v == "fails" and rows[0]["status"] == "resonant" and rows[0]["amplification"] > 9
    modes = _modes([117.0, 400.0], [0.6, 0.2])          # 17%: inside the margin but outside half of it
    assert vc.resonance(modes, modes.effective_mass[:, :3], ex, 0.2, 0.02)[1] == "marginal"
    # a mode that moves almost no mass is listed, but does not decide the verdict
    modes = _modes([101.0, 400.0], [0.001, 0.6])
    rows, v = vc.resonance(modes, modes.effective_mass[:, :3], ex, 0.2, 0.02)
    assert v == "holds" and rows[0]["status"] == "barely excited"
    assert vc.separation(120.0, 100.0, 150.0) == 0.0 and vc.separation(80.0, 100.0, 150.0) == pytest.approx(0.2)


def test_rigid_fit_recovers_the_attachment_motion():
    rng = np.random.default_rng(1)
    X = rng.uniform(-5, 5, (30, 3))
    nodes = np.arange(10, 30)
    point = np.array([1.0, 2.0, 3.0])
    u0, theta = np.array([0.1, -0.2, 0.3]), np.array([0.01, 0.02, -0.03])
    U = np.zeros((31, 3))
    U[nodes] = u0 + np.cross(theta, X[nodes] - point)
    deck = ccx.Deck("", "", "", 30, rigid_sets={30: (nodes, point)})
    filled = ccx.fill_rigid(U, np.vstack([X, point]), deck)
    assert filled[30] == pytest.approx(u0, abs=1e-12)
    many = ccx.fill_rigid(np.stack([U, 2 * U]), np.vstack([X, point]), deck)
    assert many[1, 30] == pytest.approx(2 * u0, abs=1e-12)
