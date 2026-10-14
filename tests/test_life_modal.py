"""Modal sums against closed forms: the GEVS level, Miles' equation, a damped resonance and von Mises amplitudes."""
import numpy as np
import pytest

from wobbl.life import modal


def _one_mode(fn=100.0, gamma=0.01, n=2):
    shapes = np.zeros((1, n, 3), np.float32)
    shapes[0, :, 2] = 1.0
    stress = np.zeros((1, n, 6), np.float32)
    stress[0, :, 0] = 1000.0
    part = np.zeros((1, 6))
    part[0, 2] = gamma
    return modal.Modes(np.array([fn]), shapes, stress, part, part ** 2, np.array([gamma ** 2] * 3 + [0] * 3), n)


def test_gevs_qualification_is_14_1_grms():
    psd = modal.parse_psd("gevs-qual")
    assert psd.grms() == pytest.approx(14.1, abs=0.05)
    assert modal.parse_psd("gevs-accept").grms() == pytest.approx(psd.grms() / np.sqrt(2), rel=1e-6)
    # the published breakpoints round +6 dB per octave, so the ramp is a straight line between them on log axes
    slope = np.log(0.16 / 0.026) / np.log(50 / 20)
    assert psd(np.array([30.0, 400.0, 3000.0])) == pytest.approx([0.026 * 1.5 ** slope, 0.16, 0.0], rel=1e-6)


def test_custom_psd_and_errors():
    p = modal.parse_psd("20:0.01, 100 Hz 0.04, 2000:0.04")
    assert list(p.f) == [20, 100, 2000] and p(np.array([1000.0]))[0] == pytest.approx(0.04)
    flat = modal.parse_psd("10:0.1, 1010:0.1")
    assert flat.grms() == pytest.approx(10.0)
    with pytest.raises(ValueError):
        modal.parse_psd("loud")
    with pytest.raises(ValueError):
        modal.parse_psd("100:0.1, 50:0.1")


def test_random_response_of_one_mode_matches_the_closed_form():
    m = _one_mode()
    flat = modal.parse_psd("10:0.04, 1000:0.04")
    zeta = 0.03
    R = modal.random_modal(m, zeta, flat, [0, 0, 1])
    q_rms = float(np.sqrt(R.C0[0, 0].real))
    # the closed form integrates |H|^2 from zero to infinity; a band from 10 to 1000 Hz holds all but a sliver
    assert q_rms == pytest.approx(modal.single_mode_rms(0.01, 100.0, zeta, 0.04), rel=0.01)
    # Miles: the absolute acceleration of a single mode is about w^2 times its relative motion
    w = 2 * np.pi * 100.0
    assert w ** 2 * q_rms / modal.G0 == pytest.approx(modal.miles_rms(100.0, zeta, 0.04) * 0.01, rel=0.01)
    # a narrow band process crosses zero at its natural frequency
    assert modal.zero_crossing_rate(m, R, 0) == pytest.approx(100.0, rel=0.03)
    assert modal.vm_rms(m, R.C0)[0] == pytest.approx(1000 * q_rms, rel=1e-6)


def test_harmonic_peak_is_one_over_two_zeta():
    m = _one_mode(fn=50.0, gamma=0.02)
    zeta = 0.02
    f = np.array([1e-3, 50.0])
    q = modal.harmonic_coords(f, m, zeta, modal.base_force(m, modal.G0, [0, 0, 1]))
    static, peak = abs(q[0, 0]), abs(q[1, 0])
    assert peak / static == pytest.approx(1 / (2 * zeta), rel=1e-6)
    assert static == pytest.approx(0.02 * modal.G0 / (2 * np.pi * 50.0) ** 2, rel=1e-6)


def test_von_mises_amplitude_counts_phase():
    s = np.zeros((2, 6), complex)
    s[0, 0] = 100.0                       # uniaxial, in phase
    s[1, 0], s[1, 1] = 100.0, 100.0j      # two normal stresses a quarter cycle apart
    vm = modal.vm_amplitude(s)
    assert vm[0] == pytest.approx(100.0)
    # s = 100 (cos, -sin): vm^2 = 100^2 (1 + cos sin), largest 1.5 x 100^2 an eighth of a cycle in
    assert vm[1] == pytest.approx(100.0 * np.sqrt(1.5), rel=1e-9)
    t = np.linspace(0, 2 * np.pi, 3601)
    a, b = s[1].real, s[1].imag
    brute = max(np.sqrt(np.einsum("i,ij,j", a * np.cos(x) - b * np.sin(x), modal.VM_Q, a * np.cos(x) - b * np.sin(x)))
                for x in t)
    assert vm[1] == pytest.approx(brute, rel=1e-5)


def test_consistent_mass_of_a_tetrahedron():
    from types import SimpleNamespace

    X = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1.0]])
    mids = [(0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3)]
    X = np.vstack([X] + [0.5 * (X[i] + X[j]) for i, j in mids])
    mesh = SimpleNamespace(X=X, tets=np.arange(10)[None, :], n_nodes=10)
    rho = 2.0
    ones = np.zeros((10, 3))
    ones[:, 0] = 1.0
    # a rigid translation carries the whole mass, 2 x 1/6
    assert np.sum(ones * modal.mass_times(mesh, rho, ones)) == pytest.approx(rho / 6)
    assert modal.part_mass(mesh, rho, {"m": np.array([0.5])}) == pytest.approx(rho / 6 + 0.5)
