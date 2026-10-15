"""Mesh levels fitted to a budget, and the time limit of the generality harness."""
import os
import sys
import time

import pytest

from wobbl.process import budget as B


def _pil(a, b):
    return [B.Pilot(3.0, 4, a), B.Pilot(2.0, 4, b)]


def test_a_pilot_that_fails_to_mesh_is_skipped():
    tried = []

    def mesh_at(f, c):
        tried.append((f, c))
        if f > 2.5:
            raise RuntimeError("a segment and a facet intersect")
        return [0] * int(100 * f)

    got = B.pilots(mesh_at)
    assert tried[0] == (3.0, 4) and [p.factor for p in got] == [2.0, 1.4]
    with pytest.raises(RuntimeError):
        B.pilots(lambda f, c: (_ for _ in ()).throw(RuntimeError("no mesh")))


def test_a_cheap_part_keeps_the_finest_rung_and_two_levels():
    f = B.fit(B.PRINT, _pil(400, 600), 5.0, [1.4, 1.0], B.DEFAULT_BUDGET)
    assert f.rung == 0 and not f.over and len(f.nodes) == B.MIN_LEVELS
    assert f.seconds <= f.budget


def test_a_costly_part_gives_up_curves_first_and_then_size():
    # a slender truss: the curved tubes set the node count, whatever the element size
    f = B.fit(B.PRINT, _pil(12000, 12300), 12.0, [1.4, 1.0], B.DEFAULT_BUDGET)
    assert f.rung > 0 and f.curvature < B.RUNGS[0][1] and f.seconds <= f.budget
    tight = B.fit(B.PRINT, _pil(12000, 12300), 12.0, [1.4, 1.0], 20.0)
    assert tight.over and tight.rung == len(B.RUNGS) - 1 and len(tight.nodes) == B.MIN_LEVELS


def test_a_set_mesh_size_is_not_coarsened():
    f = B.fit(B.PRINT, _pil(12000, 12300), 12.0, [1.4, 1.0], 20.0, fixed_size=True)
    assert f.rung == 0 and f.kept and not f.over


def test_a_bigger_budget_never_picks_a_coarser_rung():
    rungs = [B.fit(B.CASTING, _pil(3000, 4000), 3.0, [1.0, 0.72], b).rung for b in (30, 100, 300, 1000, 3000)]
    assert rungs == sorted(rungs, reverse=True)


def test_node_growth_follows_the_size_the_pilots_measured():
    assert B.size_power(500, 2000, 6.0, 3.0) == pytest.approx(2.0)
    assert B.size_power(500, 480, 6.0, 3.0) == 0.3                    # a count that does not move is held up
    assert B.grow(1000, 6.0, 4, 3.0, 4, 2.0) == pytest.approx(4000)
    assert B.grow(1000, 6.0, 4, 6.0, 16, 2.0) == pytest.approx(1000 * 4 ** B.CURVE_POWER)


def test_the_note_says_what_was_given_up():
    f = B.fit(B.PRINT, _pil(12000, 12300), 12.0, [1.4, 1.0], B.DEFAULT_BUDGET)
    text = B.note(f)
    assert "coarsened" in text and "elements round each curve" in text and "\u2014" not in text


@pytest.mark.skipif(os.name != "posix", reason="process groups")
def test_a_case_that_outlives_its_limit_is_stopped_with_its_children(tmp_path):
    from wobbl.validation.process_generality import run_group

    pidfile = tmp_path / "child.pid"
    code = ("import subprocess, sys, time; p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)']); "
            f"open({str(pidfile)!r}, 'w').write(str(p.pid)); time.sleep(120)")
    t0 = time.time()
    timed_out, _ = run_group([sys.executable, "-c", code], 3.0, tmp_path / "log")
    assert timed_out and time.time() - t0 < 30
    pid = int(pidfile.read_text())
    for _ in range(50):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.1)
    else:
        raise AssertionError("the child of the case is still running")


def test_a_solve_past_its_hard_limit_stops_with_a_clear_message():
    with pytest.raises(B.OverBudget, match="over four times the budget") as e:
        B.check(time.time() - 1.0, "the mold fill", 600.0)
    assert "40 min" in str(e.value) and "--budget" in e.value.hint
    B.check(None, "the mold fill", 600.0)
    B.check(B.deadline(600.0), "the mold fill", 600.0)


def test_a_mold_fill_given_an_expired_deadline_stops_at_once(tmp_path):
    import numpy as np

    from wobbl.geometry.cad import prepare
    from wobbl.mesh.mesher import Sizing, build_mesh
    from wobbl.process import hele_shaw as H
    from wobbl.validation import geometry as G

    mesh = build_mesh(prepare(G.box(tmp_path / "strip.step", 40, 10, 2), tmp_path / "geo"), Sizing(h=4.0, order=1))
    hm = H.Mesh(mesh.X, mesh.tets, np.full(len(mesh.tets), 2.0))
    gate = np.nonzero(mesh.X[:, 0] < 1e-6)[0]
    late = H.FillSettings([gate], 800.0, deadline=time.time() - 1.0, budget=600.0)
    with pytest.raises(B.OverBudget):
        H.fill(hm, H.Viscosity(m=2e4, n=0.35), late)


def test_an_extra_level_is_priced_at_the_pace_the_last_one_ran():
    n = 5000.0
    base = B.PRINT.level(n)
    fits = dict(law=B.PRINT, n_last=n, h_last=4.0, curv=8, h_next=3.0, spent=base, budget=3 * base, power=1.5)
    assert B.room_for(**fits)
    assert not B.room_for(**fits, last_seconds=20 * base)        # this part ran twenty times slower than the law
