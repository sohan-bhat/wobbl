"""Buckling, vibration and fatigue end to end with CalculiX, on parts small enough to run in seconds."""
import math

import numpy as np
import pytest

from conftest import needs_ccx

pytestmark = pytest.mark.solver

STEEL = "custom:E=200000,nu=0.3,yield=250,density=7850"


@needs_ccx
def test_euler_column_buckles_where_euler_says(tmp_path):
    from wobbl.life.buckling import BucklingStudy, run_buckling
    from wobbl.structural.spec import StructuralStudy, parse_fixture, parse_load
    from wobbl.validation import geometry as G

    part = G.box(tmp_path / "col.step", 400, 10, 10)
    base = StructuralStudy(str(part), STEEL, [parse_load("1000 N -x @ x=max")], [parse_fixture("x=min")],
                           quality="fast")
    o = run_buckling(BucklingStudy(base), tmp_path / "out")
    euler = math.pi ** 2 * 200000 * (10 ** 4 / 12) / (2 * 400) ** 2 / 1000
    assert o.values["load_factor"] == pytest.approx(euler, rel=0.01)
    assert o.values["shape"] == "column" and o.verdict == "holds"
    checks = {c.name: c.status for c in o.checks}
    assert checks["hand calculation"] == "pass" and checks["force balance"] == "pass"
    # the column curve cuts a slender steel column by about a tenth
    assert 0.85 < o.values["knockdown"] < 0.95
    assert (tmp_path / "out" / "report.html").exists() and not (tmp_path / "out" / "work").exists()
    _playback_ok(tmp_path / "out", "buckling")


def _playback_ok(out, kind):
    import json

    p = out / "playback.json"
    assert p.exists() and p.stat().st_size < 300_000
    d = json.loads(p.read_text())
    assert d["kind"] == kind and d["format"] == "wobbl-playback" and d["mesh"]["vertices"]["data"]


@needs_ccx
def test_tip_mass_beam_resonates_and_every_check_agrees(tmp_path):
    """A 300 mm steel strip carrying 200 g at its tip: nearly a single degree of freedom, k = 3EI/L^3 with a third
    of the strip's mass, so the frequency, Rayleigh's quotient and CalculiX's own steady state step must agree."""
    from wobbl.life.vibration import run_vibration
    from wobbl.life.vibspec import VibrationStudy
    from wobbl.structural.spec import StructuralStudy, parse_fixture
    from wobbl.validation import geometry as G

    L, b, h = 300.0, 12.0, 6.0
    part = G.box(tmp_path / "strip.step", L, b, h)
    base = StructuralStudy(str(part), STEEL, [], [parse_fixture("x=min")], quality="fast")
    st = VibrationStudy(base, masses=[f"200 g at {L},6,3 @ x=max"], modes=6, rpm=["1300-1500 rpm"], blades=2,
                        sine="0.5 g z 5-200 Hz")
    o = run_vibration(st, tmp_path / "out")
    k = 3 * 200000 * (b * h ** 3 / 12) / L ** 3
    f_sdof = math.sqrt(k / (0.2e-3 + 33 / 140 * 7850e-12 * L * b * h)) / (2 * math.pi)
    assert o.values["frequencies"][0] == pytest.approx(f_sdof, rel=0.01)
    checks = {c.name: c.status for c in o.checks}
    for name in ("mass balance", "hand calculation", "solver cross-check", "single mode estimate"):
        assert checks[name] == "pass", (name, o.checks)
    # the rotor's once per turn band, 21.7 to 25 Hz, holds the first mode
    assert o.verdict == "fails" and o.values["resonance"][0]["status"] == "resonant"
    assert o.values["harmonic"]["frequency_Hz"] == pytest.approx(f_sdof, rel=0.01)
    assert not (tmp_path / "out" / "work").exists()
    _playback_ok(tmp_path / "out", "vibration")
