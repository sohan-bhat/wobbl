"""Full wave antennas against the references they are cited to, and on antennas nothing was tuned for. These solve in
3D, so each takes tens of minutes and is marked slow. Each test states the error it measured and where it comes from.

Half wave dipole (Balanis, Antenna Theory, chapter 4 and Kraus, Antennas): a thin dipole resonates at about 0.48
wavelength, has 73 ohm there and a gain of 2.15 dBi.
Rectangular patch (Balanis, chapter 14): the transmission line model gives f = c / (2 (L + 2 dL) sqrt(e_eff)).
"""
from __future__ import annotations

import math

import pytest

from wobbl.fields import antenna3d, radiate, wave3d
from wobbl.fields.forms import C0

pytestmark = [pytest.mark.solver, pytest.mark.slow]


def _have_getdp() -> bool:
    from wobbl.solvers import base

    try:
        return base.get("getdp").locate() is not None
    except Exception:  # noqa: BLE001
        return False


needs_getdp = pytest.mark.skipif(not _have_getdp(), reason="GetDP not installed (wobbl setup --only getdp)")


@needs_getdp
def test_half_wave_dipole_against_the_textbook(tmp_path):
    """0.48 wavelength of 1 mm wire at 1 GHz (a/lambda 0.0017). Measured at normal quality: the resonance is 2 to 3%
    low, which is the mesh (about 1% by the finer mesh check) and the feed strip's inductance, resistance and gain
    are within 5% and 0.15 dB of the thin wire values."""
    b = antenna3d.build("dipole", {"frequency": "1GHz", "length": "144mm", "diameter": "1mm", "gap": "1mm", "span": 0.12,
                                   "points": 5}, tmp_path / "geo")
    o = wave3d.run(b.scene, b.setup, tmp_path / "out", quality="normal", budget=0)
    v = o.values
    f_res, (r_res, _) = v["resonance_Hz"], v["z_resonance_ohm"]
    length_in_wavelengths = 0.144 * f_res / C0
    assert v["resonance_kind"] == "reactance zero"
    assert length_in_wavelengths == pytest.approx(0.48, abs=0.025)       # the textbook's 0.48, within 5%
    assert r_res == pytest.approx(73.1, rel=0.08)
    assert v["gain_dBi"] == pytest.approx(2.15, abs=0.25)
    st = {c.name: c.status for c in o.checks}
    assert st["power balance"] == "pass" and st["mesh convergence"] in ("pass", "warn")


@needs_getdp
def test_patch_resonance_against_the_transmission_line_model(tmp_path):
    """Balanis example 14.1, 10 GHz on 1.588 mm of er 2.2, at normal quality."""
    b = antenna3d.build("patch", {"frequency": "10GHz", "er": "2.2", "height": "1.588mm", "loss_tangent": "0.0009"},
                        tmp_path / "geo")
    fr = b.ref["transmission_line_Hz"]
    assert fr == pytest.approx(10e9, rel=1e-3)
    o = wave3d.run(b.scene, b.setup, tmp_path / "out", quality="normal", budget=0)
    res = o.values["resonance_Hz"]
    assert res is not None
    assert abs(res - fr) / fr < 0.08          # see the docstring of the comparison in the report for where this goes


@needs_getdp
def test_a_loop_and_an_inverted_f_trace_run_end_to_end(tmp_path):
    for kind in ("loop", "invf"):
        o = radiate.run_radiate(kind, {"frequency": "2.4GHz", "points": "5", "budget": "60 min"}, tmp_path / kind, "",
                                quality="fast")
        v = o.values
        assert v["gain_dBi"] > -3 and math.isfinite(v["s11_at_f0_dB"])
        assert {c.name: c.status for c in o.checks}["power balance"] == "pass"
        assert (tmp_path / kind / "scene.wobbl").stat().st_size > 20_000
