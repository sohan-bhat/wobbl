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
        from make_board import board_text as build
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


# --- the 3D scene -------------------------------------------------------------------------------
def test_pattern_balloon_grows_with_the_field_not_the_power():
    from wobbl.fields import antscene

    th = np.radians(np.arange(0, 181, 15.0))
    ph = np.radians(np.arange(0, 360, 30.0))
    d = 1.5 * np.sin(th)[:, None] ** 2 * np.ones((1, len(ph)))        # a short dipole's doughnut
    d = np.maximum(d, 1e-9)
    X, T, gdb = antscene.balloon(d, th, ph, 1.0, 10.0, (1.0, 2.0, 3.0))
    r = np.linalg.norm(X - np.array([1.0, 2.0, 3.0]), axis=1)
    assert r.max() == pytest.approx(10.0, rel=1e-6)                      # the peak reaches the stated radius
    assert gdb.max() == pytest.approx(10 * math.log10(1.5), abs=1e-6)    # colored by gain in dBi
    assert T.max() < len(X) and T.shape[1] == 3
    ring = np.isclose(X[:, 2], 3.0, atol=1e-6)
    assert np.allclose(r[ring], 10.0, rtol=1e-6)                         # the equator is the full radius


# --- the simulate registry ----------------------------------------------------------------------
def test_simulate_sends_antenna_files_and_words_to_the_antenna_studies(tmp_path):
    from pathlib import Path

    from wobbl.simulate import route

    ex = Path(__file__).resolve().parents[1] / "examples" / "antenna-pcb"
    import sys
    sys.path.insert(0, str(ex))
    try:
        from make_board import board_text as build
    finally:
        sys.path.remove(str(ex))
    pcb = tmp_path / "ifa.kicad_pcb"
    pcb.write_text(build())
    p = route(str(pcb), "does the trace antenna resonate at 2.4 GHz and what is its S11?", [], None, None)
    assert [a.key for a in p.analyses] == ["board.antenna"]            # the word trace does not ask for parasitics too
    antenna3d._scene([("cyl", "arm one", (0, 0, -30), (0, 0, -0.5), 0.5), ("cyl", "arm two", (0, 0, 0.5), (0, 0, 30), 0.5)],
                     tmp_path / "geo", "dipole")
    p = route(str(tmp_path / "geo" / "dipole.step"), "what is the S11 and gain of this antenna at 2.4 GHz?", [], None, None)
    assert "part.antenna" in [a.key for a in p.analyses]
    p = route("a loop antenna 100 mm across at 868 MHz, what is its gain?", "", [], None, None)
    assert [a.key for a in p.analyses] == ["field.antenna"] and p.kind == "loop"
    p = route("an inverted F antenna on FR4 for 2.4 GHz", "what is the S11?", [], None, None)
    assert p.kind == "invf"


def test_the_estimate_prices_a_3d_antenna_without_meshing_it(tmp_path):
    from wobbl.jobs import workload as W
    from wobbl.simulate import route

    plan = route("a patch antenna for 2.4 GHz with a 29 mm long patch", "does it match, what is its S11?", [], None, None)
    c = W.plan_cost(plan, {}, False)
    assert c.known and c.core_seconds > 10 and "3D full wave" in " ".join(c.basis)
    plan = route("a half wave dipole for 2.4 GHz", "what is its length?", [], None, None)
    c = W.plan_cost(plan, {}, False)
    assert c.known and c.core_seconds < 200 and "axisymmetric" in " ".join(c.basis)


def test_a_study_far_over_its_budget_is_refused_before_it_meshes(tmp_path):
    import time

    from wobbl.fields.wavecost import CostRefusal

    t = time.time()
    with pytest.raises(CostRefusal) as e:
        radiate.run_radiate("loop", {"frequency": "2.4GHz", "budget": "5 s"}, tmp_path / "out", quality="fine")
    assert time.time() - t < 90                      # building the loop took seconds, no mesh or solve was started
    assert "budget of 5 s" in str(e.value) and "lower frequency" in e.value.hint
    assert not list((tmp_path / "out").glob("work/*/m.msh"))


def test_the_recorded_generality_runs_balance_their_power():
    import json
    from pathlib import Path

    d = json.loads((Path(__file__).resolve().parents[1] / "docs" / "antenna-generality.json").read_text())
    names = {r["antenna"] for r in d["rows"]}
    assert {"dipole", "monopole", "patch", "loop", "invf", "ifa-kicad"} <= names
    for r in d["rows"]:
        assert r["checks"]["power balance"] == "pass", r["antenna"]
        assert r["gain_dBi"] <= r["directivity_dBi"] + 1e-6 and r["peak_theta_deg"] >= 0


def test_a_slice_with_short_rows_still_reads(tmp_path):
    from wobbl.fields import pro3d

    pl = {"normal": "y", "at": 0.0, "iu": 0, "iv": 2, "ik": 1, "u0": 0.0, "u1": 1.0, "v0": 0.0, "v1": 1.0, "nu": 2, "nv": 2}
    (tmp_path / "s.txt").write_text("0 0 0 1 2 3 4 5 6\n1 0 0 1 2 3 4 5 6\n0 0 1 0 0 0\n1 0 1 9 8 7 6 5 4\n")
    a = pro3d.read_plane(tmp_path / "s.txt", pl, comps=6)
    assert a.shape == (2, 2, 6)
    assert np.isnan(a[1, 0, 0]) and a[1, 1, 0] == 9 and a[0, 0, 5] == 6      # the short row left its point empty
