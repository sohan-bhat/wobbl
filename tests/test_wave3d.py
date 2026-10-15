"""3D full wave antennas: the port strip, the near to far field transform on a known dipole, the closed form
detection, then a wire dipole solve that skips when GetDP is not installed."""
from __future__ import annotations

import math

import numpy as np
import pytest

from wobbl.fields import wave3d
from wobbl.fields.forms import C0
from wobbl.fields.scene import Body, Scene

ETA = wave3d.ETA0


def _have_getdp() -> bool:
    from wobbl.solvers import base

    try:
        return base.get("getdp").locate() is not None
    except Exception:  # noqa: BLE001
        return False


needs_getdp = pytest.mark.skipif(not _have_getdp(), reason="GetDP not installed (wobbl setup --only getdp)")


# --- the port strip ----------------------------------------------------------------------------
def test_port_strip_holds_the_line_and_lies_across_the_metal_normal():
    a, b = np.array([0.0, 0.0, 0.0]), np.array([2e-3, 0.0, 0.0])
    poly = wave3d.port_polygon(a, b, 0.4e-3, normal=(0, 0, 1))
    assert poly.shape == (4, 3)
    # a and b are the middles of the short sides, and the strip is as wide as asked
    assert np.allclose(0.5 * (poly[0] + poly[1]), a) and np.allclose(0.5 * (poly[2] + poly[3]), b)
    assert np.linalg.norm(poly[1] - poly[0]) == pytest.approx(0.4e-3)
    # the width runs perpendicular to the line and to the metal's normal, so the strip lies in the copper plane
    u = poly[1] - poly[0]
    assert abs(np.dot(u, b - a)) < 1e-15 and abs(u[2]) < 1e-15
    assert np.allclose(wave3d.polygon_normal(poly), [0, 0, 1]) or np.allclose(wave3d.polygon_normal(poly), [0, 0, -1])


def test_port_strip_falls_back_to_any_perpendicular():
    a, b = (0.0, 0.0, -0.5e-3), (0.0, 0.0, 0.5e-3)
    # an end cap's normal runs along the gap, so it says nothing about the width direction
    for n in ((0, 0, 1), None):
        poly = wave3d.port_polygon(a, b, 0.2e-3, normal=n)
        u = poly[1] - poly[0]
        assert np.linalg.norm(u) == pytest.approx(0.2e-3)
        assert abs(u[2]) < 1e-15
    assert wave3d.port_width(a, b) == pytest.approx(0.2e-3)          # a tenth of 1 mm is under the 0.2 mm floor
    assert wave3d.port_width((0, 0, 0), (0, 0, 5e-3)) == pytest.approx(0.5e-3)
    assert wave3d.port_width(a, b, 1e-3) == 1e-3
    with pytest.raises(ValueError, match="same point"):
        wave3d.port_polygon(a, a, 1e-3)


def test_closest_point_on_a_copper_polygon():
    sq = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], float)
    d, q = wave3d.closest_on_polygon(sq, (0.5, 0.5, 0.2))
    assert d == pytest.approx(0.2) and np.allclose(q, [0.5, 0.5, 0])
    d, q = wave3d.closest_on_polygon(sq, (2.0, 0.5, 0.0))
    assert d == pytest.approx(1.0) and np.allclose(q, [1, 0.5, 0])


# --- near to far field --------------------------------------------------------------------------
def _hertzian(points: np.ndarray, k: float, il: float, at: np.ndarray):
    """Exact E and H of a z directed Hertzian dipole of moment il (A m) at the point at (e^(j w t))."""
    r_vec = points - at
    r = np.linalg.norm(r_vec, axis=-1)
    ct = r_vec[..., 2] / r
    st = np.sqrt(np.maximum(1 - ct ** 2, 0))
    ph = np.arctan2(r_vec[..., 1], r_vec[..., 0])
    rh = r_vec / r[..., None]
    th = np.stack([ct * np.cos(ph), ct * np.sin(ph), -st], axis=-1)
    fh = np.stack([-np.sin(ph), np.cos(ph), np.zeros_like(ph)], axis=-1)
    x = 1 / (1j * k * r)
    g = np.exp(-1j * k * r)
    hp = 1j * k * il * st / (4 * math.pi * r) * (1 + x) * g
    er = ETA * il * ct / (2 * math.pi * r ** 2) * (1 + x) * g
    et = 1j * ETA * k * il * st / (4 * math.pi * r) * (1 + x + x ** 2) * g
    return er[..., None] * rh + et[..., None] * th, hp[..., None] * fh


def test_hertzian_dipole_gives_directivity_one_and_a_half():
    f = 1e9
    k = 2 * math.pi * f / C0
    rs = 0.15
    sph = wave3d.sphere_grid(24)
    t, p = np.meshgrid(sph["theta"], sph["phi"], indexing="ij")
    pts = rs * np.stack([np.sin(t) * np.cos(p), np.sin(t) * np.sin(p), np.cos(t)], axis=-1)
    # off the sphere's center, so the phase of the transform matters
    E, H = _hertzian(pts, k, 1e-3, np.array([0.02, -0.01, 0.03]))
    exact = ETA * (k * 1e-3) ** 2 / (12 * math.pi)
    assert wave3d.sphere_flux(E, H, sph, rs) == pytest.approx(exact, rel=1e-3)
    pat = wave3d.pattern(E, H, sph, rs, k, axis=(0, 0, 1))
    assert pat["p_ff"] == pytest.approx(exact, rel=0.02)
    assert pat["d_max"] == pytest.approx(1.5, rel=0.02)
    assert pat["theta_deg"] == pytest.approx(90, abs=3)
    # the elevation cut is the sin squared figure eight, the azimuth cut is round
    el = pat["d_el"]
    assert el[0] < 0.01 * el.max() and el[90] == pytest.approx(1.5, rel=0.02)
    assert np.ptp(pat["d_az"]) < 0.02 * pat["d_az"].max()


# --- closed form detection ----------------------------------------------------------------------
def _rod(name, i, z0, z1, a=0.5e-3):
    return Body(name, i, "x.step", math.pi * a * a * (z1 - z0), 2 * math.pi * a * (z1 - z0), (-a, -a, z0, a, a, z1),
                (0, 0, 0.5 * (z0 + z1)))


def test_wire_dipole_and_monopole_are_recognized():
    sc = Scene([_rod("arm 1", 1, -0.071, -0.0005), _rod("arm 2", 2, 0.0005, 0.071)], "")
    s = wave3d.WaveSetup(metals=["arm 1", "arm 2"])
    w = wave3d.wire_geometry(sc, s, (0, 0, 1), 1e9)
    assert w["kind"] == "dipole" and w["length"] == pytest.approx(0.142) and w["a"] == pytest.approx(0.5e-3)
    assert wave3d.wire_geometry(sc, s, (1, 0, 0), 1e9) is None        # fed across, not along the wire
    plate = Body("plate", 2, "x.step", 0.3 * 0.3 * 0.002, 2 * 0.09, (-0.15, -0.15, -0.002, 0.15, 0.15, 0.0),
                 (0, 0, -0.001))
    sc = Scene([_rod("rod", 1, 0.001, 0.075), plate], "")
    w = wave3d.wire_geometry(sc, wave3d.WaveSetup(metals=["rod", "plate"]), (0, 0, 1), 1e9)
    assert w["kind"] == "monopole" and w["length"] == pytest.approx(0.150) and w["plate_wavelengths"] > 0.9
    sheet = {"ground": [np.array([[-0.15, -0.15, 0], [0.15, -0.15, 0], [0.15, 0.15, 0], [-0.15, 0.15, 0]])]}
    w = wave3d.wire_geometry(Scene([_rod("rod", 1, 0.001, 0.075)], ""),
                             wave3d.WaveSetup(metals=["rod"], sheets=sheet), (0, 0, 1), 1e9)
    assert w["kind"] == "monopole" and w["length"] == pytest.approx(0.150)


def test_resonance_from_reactance_or_resistance_peak():
    f = [0.9e9, 1.0e9, 1.1e9]
    r = wave3d.resonance(f, [60 - 40j, 70 + 10j, 85 + 60j], 50)
    assert r[2] == "reactance zero" and r[0] == pytest.approx(0.98e9)
    # a patch seen through probe inductance: reactance stays positive, resistance peaks
    r = wave3d.resonance(f, [20 + 30j, 90 + 20j, 30 + 40j], 50)
    assert r[2] == "resistance peak" and 0.95e9 < r[0] < 1.05e9
    assert wave3d.resonance(f, [20 + 30j, 30 + 40j, 40 + 50j], 50) is None


def test_setup_mistakes_say_what_to_do(tmp_path):
    sc = Scene([_rod("arm 1", 1, -0.071, -0.0005)], "")
    with pytest.raises(ValueError, match="needs a port"):
        wave3d.run(sc, wave3d.WaveSetup(metals=["arm 1"], f0=1e9), tmp_path)
    with pytest.raises(ValueError, match="no body called"):
        wave3d.run(sc, wave3d.WaveSetup(metals=["arm 9"], f0=1e9, port=wave3d.Port((0, 0, 0), (0, 0, 1e-3))), tmp_path)


# --- a solve --------------------------------------------------------------------------------------
@pytest.mark.solver
@needs_getdp
def test_half_wave_wire_dipole(tmp_path):
    sc, port = wave3d.wire_dipole_scene(142, 1.0, 1.0, tmp_path / "geo")
    arms = [b.name for b in sc.bodies]
    s = wave3d.WaveSetup(metals=arms, port=port, f0=1e9, span=0.1, points=3,
                         sizes={n: 0.5e-3 for n in arms})              # coarse, about 50 thousand unknowns
    o = wave3d.run(sc, s, tmp_path / "out", quality="fast")
    assert o.values["unknowns"][0] < 70_000
    assert 60 < o.values["z_resonance_ohm"][0] < 90
    assert {c.name: c.status for c in o.checks}["power balance"] == "pass"
    assert o.values["wire"]["kind"] == "dipole"
    assert 1.5 < o.values["gain_dBi"] < 2.8
    assert (tmp_path / "out" / "playback.json").stat().st_size < 300_000
