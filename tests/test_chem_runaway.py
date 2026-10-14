"""Lithium ion thermal runaway onset: Semenov's criterion, the transient with reactants used up, and the commands."""
import json
import math

import pytest

from wobbl.chem.runaway import Cell, cell_from, critical, generation, loss, run_runaway, semenov, transient


def test_semenov_point_is_a_tangency():
    c = Cell(0.018, 0.065)
    Ta, Ts = semenov(c)
    assert abs(loss(c, Ts, Ta) - generation(c, Ts)) < 1e-3 * generation(c, Ts)      # the curves touch
    dT = 0.05
    slope_g = (generation(c, Ts + dT) - generation(c, Ts - dT)) / (2 * dT)
    slope_l = (loss(c, Ts + dT, Ta) - loss(c, Ts - dT, Ta)) / (2 * dT)
    assert slope_g == pytest.approx(slope_l, rel=0.02)                              # with the same slope
    better = Cell(0.018, 0.065, h=30.0)
    assert semenov(better)[0] > Ta                                                  # cooling raises the onset


def test_the_18650_runs_away_where_hatchard_and_hu_put_it():
    """Hu, He, Rein and Restuccia (Fire Technology 2020) put a single LCO 18650's critical oven temperature at
    155 C, in line with Hatchard's oven tests."""
    c = Cell(0.018, 0.065)
    ta, stay, ran = critical(c, semenov(c)[0])
    assert ta - 273.15 == pytest.approx(155.0, abs=3.0)
    assert not stay.runaway and ran.runaway and ran.t_runaway < 3 * 3600
    assert stay.energy_error < 1e-6 and ran.energy_error < 1e-6
    hot = transient(c, ta + 10.0)
    assert hot.runaway and hot.t_runaway < ran.t_runaway


def test_run_gives_a_verdict_fix_and_playback(tmp_path):
    o = run_runaway(Cell(0.018, 0.065), 150 + 273.15, tmp_path)
    assert o.verdict == "warm" and o.domain == "battery"
    assert {f["kind"] for f in o.fixes} == {"cooling", "temperature"}
    d = json.loads((tmp_path / o.values["files"]["playback"]).read_text())
    assert d["manifest"]["kind"] == "cell-runaway" and len(d["frames"]) == 150
    assert max(f["above"]["T_C"] for f in d["frames"]) > max(f["below"]["T_C"] for f in d["frames"]) + 50
    ok = run_runaway(Cell(0.018, 0.065), 40 + 273.15, tmp_path / "ok", render=False)
    assert ok.verdict == "ok" and not ok.fixes and ok.confidence == "high"


def test_cells_and_cooling_read_like_the_heat_domain():
    c = cell_from("21700", "a small fan")
    assert (c.diameter, c.length, c.h) == (0.021, 0.070, 25.0)
    assert cell_from("18x65", "12 W/m2K").h == pytest.approx(12.0)
    assert cell_from("18650").h == pytest.approx(7.17)
    with pytest.raises(ValueError):
        cell_from("aa")
    assert math.isclose(cell_from("18650", "sealed box").h, 5.0)
