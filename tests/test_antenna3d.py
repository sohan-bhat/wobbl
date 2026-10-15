"""3D antennas from dimensions: which solver a question needs, the geometry each builder makes and the cost plan.
The solves themselves are in test_antenna3d_solves.py."""
from __future__ import annotations

import math

import numpy as np
import pytest

from wobbl.fields import antenna3d, radiate, wave3d, wavecost
from wobbl.fields.forms import C0


# --- routing ------------------------------------------------------------------------------------
@pytest.mark.parametrize("kind,params,question,mode", [
    ("dipole", {}, "", "axisymmetric"),
    ("dipole", {}, "what is its gain and S11?", "axisymmetric"),
    ("dipole", {}, "how does it do near a human hand?", "3d"),
    ("monopole", {}, "", "axisymmetric"),
    ("monopole", {"ground": "100mm"}, "", "3d"),
    ("patch", {}, "", "closed form"),
    ("patch", {}, "does it match at 2.4 GHz? what is its S11", "3d"),
    ("patch", {"length": "29mm"}, "", "3d"),
    ("loop", {}, "", "3d"),
    ("invf", {}, "", "3d"),
])
def test_route_follows_the_dimensions_and_the_question(kind, params, question, mode):
    r = radiate.route(kind, params, question)
    assert r.mode == mode and r.reasons


# --- geometry -----------------------------------------------------------------------------------
def test_patch_defaults_follow_balanis_example_14_1():
    # 10 GHz on 1.588 mm of er 2.2: W = 11.86 mm, L = 9.06 mm, er effective 1.972 (Balanis, Antenna Theory)
    W, L, ee, dl = antenna3d.patch_guess(10e9, 2.2, 1.588e-3)
    assert W * 1e3 == pytest.approx(11.86, abs=0.02)
    assert L * 1e3 == pytest.approx(9.06, abs=0.02)
    assert ee == pytest.approx(1.972, abs=0.002)
    assert antenna3d.patch_resonance(W, L, 2.2, 1.588e-3) == pytest.approx(10e9, rel=1e-3)


def test_builders_make_a_feedable_antenna_for_each_kind(tmp_path):
    for kind in antenna3d.KINDS:
        b = antenna3d.build(kind, {"frequency": "2.4GHz"}, tmp_path / kind)
        s = b.setup
        a, c = np.asarray(s.port.a), np.asarray(s.port.b)
        assert 0 < np.linalg.norm(c - a) < 0.1 * C0 / 2.4e9, kind
        assert s.metals or s.sheets
        near = wave3d.feed(b.scene, s)           # the port ends land on two different conductors
        assert near["gap"] > 0, kind


def test_bad_dimensions_say_what_to_change(tmp_path):
    with pytest.raises(ValueError, match="feed"):
        antenna3d.build("patch", {"feed": "40mm"}, tmp_path / "a")
    with pytest.raises(ValueError, match="unknown"):
        antenna3d.build("dipole", {"colour": "red"}, tmp_path / "b")
    with pytest.raises(ValueError, match="too small"):
        antenna3d.build("loop", {"radius": "1mm", "diameter": "1mm"}, tmp_path / "c")


# --- the cost plan --------------------------------------------------------------------------------
def test_a_study_over_budget_gives_up_in_order_and_far_over_refuses():
    ok = wavecost.assess(5_000, "normal", 9, 4, budget=3600)
    assert not ok.gave_up and ok.verify and ok.quality == "normal"
    tight = wavecost.assess(20_000, "normal", 9, 4, budget=1500)
    assert tight.gave_up and tight.seconds <= 1500
    assert tight.points < 9 or tight.quality == "fast"
    with pytest.raises(wavecost.CostRefusal) as e:
        wavecost.assess(400_000, "normal", 9, 4, budget=300)
    assert "budget" in str(e.value) and e.value.hint
    assert wavecost.assess(400_000, "normal", 9, 4, budget=0).seconds > 3600     # 0 means no limit


def test_cost_grows_with_the_mesh():
    a = wavecost.cost(10_000, 7, 4, True)[1]
    b = wavecost.cost(40_000, 7, 4, True)[1]
    assert 4 < b / a < 12
    assert math.isclose(wavecost.assess(8_000, "fast", 5, 2, 0).unknowns, 8_000 * wavecost.UNKNOWNS_PER_NODE, rel_tol=1e-6)


# --- antennas that arrive as files ----------------------------------------------------------------
def test_port_text_is_two_points_in_mm():
    p = antenna3d.parse_port("1,2,3:1,2,4.5:0.4")
    assert p.a == pytest.approx((1e-3, 2e-3, 3e-3)) and p.b[2] == pytest.approx(4.5e-3) and p.width == pytest.approx(0.4e-3)
    with pytest.raises(ValueError, match="two points"):
        antenna3d.parse_port("1,2,3")


def test_a_step_dipole_finds_its_own_feed_gap(tmp_path):
    antenna3d._scene([("cyl", "arm one", (0, 0, -30), (0, 0, -0.5), 0.5), ("cyl", "arm two", (0, 0, 0.5), (0, 0, 30), 0.5)],
                     tmp_path / "geo", "dipole")
    b = antenna3d.from_step(tmp_path / "geo" / "dipole.step", {"frequency": "2.4GHz"}, tmp_path / "w")
    a, c = np.asarray(b.setup.port.a), np.asarray(b.setup.port.b)
    assert abs(a[2]) < 1e-3 and abs(c[2]) < 1e-3 and 0 < np.linalg.norm(c - a) < 2e-3
    assert len(b.setup.metals) == 2 and not b.setup.dielectrics
    with pytest.raises(ValueError, match="frequency"):
        antenna3d.from_step(tmp_path / "geo" / "dipole.step", {}, tmp_path / "w2")


def test_a_kicad_trace_antenna_becomes_sheets_a_board_and_a_port(tmp_path):
    import sys
    from pathlib import Path

    ex = Path(__file__).resolve().parents[1] / "examples" / "antenna-pcb"
    sys.path.insert(0, str(ex))
    try:
        from make_board import build
    finally:
        sys.path.remove(str(ex))
    pcb = tmp_path / "ifa.kicad_pcb"
    pcb.write_text(build())
    b = antenna3d.from_board(pcb, {"frequency": "2.4GHz"}, tmp_path / "w")
    s = b.setup
    assert {"antenna_F_Cu", "ground_F_Cu", "ground_B_Cu"} <= set(s.sheets)
    assert list(s.dielectrics) == ["board"] and s.dielectrics["board"][0] == pytest.approx(4.4)
    gap = np.linalg.norm(np.asarray(s.port.b) - np.asarray(s.port.a))
    assert 0.3e-3 < gap < 1e-3                       # the 0.6 mm between the feed pad and the pour
    fd = wave3d.feed(b.scene, s)                     # both ends land on different copper
    assert fd["gap"] == pytest.approx(gap, rel=0.05)
    with pytest.raises(ValueError, match="feed"):
        antenna3d.from_board(pcb, {"frequency": "2.4GHz", "feed": "NOPE"}, tmp_path / "w2")
