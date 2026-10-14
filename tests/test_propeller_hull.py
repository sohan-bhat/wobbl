"""Blade element momentum theory against UIUC propeller measurements, and Holtrop and Mennen against their paper's
worked example. Neither needs a solver."""
from __future__ import annotations

import math

import pytest

from wobbl.flow import bem, hull


def test_bundled_props_carry_geometry_and_measurements():
    names = bem.prop_names()
    assert "apcsf_10x4.7" in names and len(names) >= 5
    p = bem.load_prop("apcsf_10x4.7")
    assert p.diameter == pytest.approx(0.254) and p.blades == 2
    assert len(p.r_R) == len(p.c_R) == len(p.beta_deg) == 18
    assert "Brandt and M. S. Selig" in bem.uiuc_source()
    assert bem.load_prop("APC 10x4.7 Slow Flyer").key == "apcsf_10x4.7"
    q = bem.load_prop("12x6x3")
    assert q.blades == 3 and q.diameter == pytest.approx(12 * 0.0254)
    # a constant geometric pitch: tan(beta) 2 pi r = pitch at every station
    r = q.r_R[10] * q.radius
    assert math.tan(math.radians(q.beta_deg[10])) * 2 * math.pi * r == pytest.approx(6 * 0.0254)
    with pytest.raises(ValueError):
        bem.load_prop("banana")


def test_static_thrust_and_power_against_uiuc():
    """APC 10x4.7 Slow Flyer at 4319 rpm, near a 1.2 kg quadcopter's hover, measured CT 0.1177 and CP 0.0474
    (Brandt and Selig 2011). The generic section lands about 7% low on thrust and 10% low on power there."""
    p = bem.load_prop("apcsf_10x4.7")
    r = bem.solve(p, 4319)
    ct, cp = bem.measured_static(p, 4319)
    assert (ct, cp) == pytest.approx((0.1177, 0.0474))
    assert r.CT == pytest.approx(ct, rel=0.10)
    assert r.CP == pytest.approx(cp, rel=0.15)
    assert 0.5 < r.figure_of_merit < 0.8
    assert r.tip_mach == pytest.approx(4319 / 60 * 2 * math.pi * 0.127 / 343, rel=1e-6)
    assert bem.measured_static(p, 20000) is None


def test_bem_settles_with_stations_and_finds_hover_speed():
    p = bem.load_prop("apcsf_10x4.7")
    cts = [bem.solve(p, 5000, stations=n).CT for n in (20, 40, 80)]
    assert abs(cts[2] - cts[1]) / cts[2] < 0.01
    h = bem.rpm_for_thrust(p, 2.94)
    assert h.thrust == pytest.approx(2.94, rel=1e-3)
    assert 3500 < h.rpm < 5500 and 10 < h.power < 40
    # forward flight: thrust falls with speed, and the efficiency is a fraction
    fwd = bem.solve(p, 6000, 8.0)
    assert fwd.thrust < bem.solve(p, 6000).thrust and 0 < fwd.eta < 1


def test_holtrop_mennen_reproduces_the_paper_example():
    """Holtrop and Mennen (1982): 205 m, 25 knots, 1793.26 kN, with S 7381.45 m2 and 1 + k1 1.156."""
    r = hull.resistance(hull.EXAMPLE, 25 * hull.KNOT)
    assert r.wetted == pytest.approx(7381.45, rel=1e-5)
    assert r.form_factor == pytest.approx(1.156, abs=1e-3)
    assert r.parts["friction"] / r.form_factor == pytest.approx(869.63e3, rel=1e-3)
    assert r.parts["waves"] == pytest.approx(557.11e3, rel=0.002)
    assert r.parts["transom"] == 0.0
    assert r.total == pytest.approx(hull.EXAMPLE_KN * 1e3, rel=0.002)
    assert not r.out_of_range


def test_the_cli_and_the_mcp_tool_read_flow_values_the_same_way(tmp_path, monkeypatch):
    import json

    from click.testing import CliRunner

    from wobbl.cli.main import cli
    from wobbl.flow.tools import build, solve_flow_tool

    monkeypatch.setenv("WOBBL_HOME", str(tmp_path / "home"))
    boat = {"length": "12 m", "beam": "3.6 m", "draft": "0.9 m", "displacement": "16 t", "speed": "8 kn"}
    st, _ = build("hull", boat)
    assert st.hull.displacement == pytest.approx(16000 / 1025) and st.speed == pytest.approx(8 * 0.514444)
    assert build("hull", {**boat, "displacement": "15.6 m3"})[0].hull.displacement == 15.6
    r = solve_flow_tool("hull", {**boat, "power": "60 kW"})
    assert r["verdict"] == "ok" and r["confidence"] == "medium" and r["values"]["resistance_N"] > 0
    assert "error" in solve_flow_tool("kite", {})
    res = CliRunner().invoke(cli, ["flow", "hull", "--length", "12 m", "--beam", "3.6 m", "--draft", "0.9 m",
                                   "--displacement", "16 t", "--speed", "8 kn", "-o", str(tmp_path / "h"), "--json"])
    assert res.exit_code == 0, res.output
    assert json.loads(res.output)["domain"] == "flow"
    assert (tmp_path / "h" / "report.html").exists()


def test_hull_estimates_what_is_missing_and_flags_its_range():
    h = hull.Hull(12.0, 3.5, 0.9, 18.0).fill()
    assert 0.9 < h.cm < 1.0 and 0.6 < h.cwp < 0.95 and len(h.estimated) == 2
    slow, fast = hull.resistance(h, 2.0), hull.resistance(h, 7.0)
    assert fast.total > slow.total > 0
    assert any("Froude" in s for s in fast.out_of_range)
