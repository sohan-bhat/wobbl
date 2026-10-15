"""Mesh levels fitted to a budget, and the time limit of the generality harness."""
import os
import sys
import time

import pytest

from wobbl.process import budget as B


def test_a_cheap_part_keeps_the_finest_rung_and_two_levels():
    f = B.fit(B.PRINT, (400, 600), 5.0, [1.4, 1.0], B.DEFAULT_BUDGET)
    assert f.rung == 0 and not f.over and len(f.nodes) == B.MIN_LEVELS
    assert f.seconds <= f.budget


def test_a_costly_part_gives_up_curves_first_and_then_size():
    # a slender truss: the curved tubes set the node count, whatever the element size
    f = B.fit(B.PRINT, (12000, 12300), 12.0, [1.4, 1.0], B.DEFAULT_BUDGET)
    assert f.rung > 0 and f.curvature < B.RUNGS[0][1] and f.seconds <= f.budget
    tight = B.fit(B.PRINT, (12000, 12300), 12.0, [1.4, 1.0], 20.0)
    assert tight.over and tight.rung == len(B.RUNGS) - 1 and len(tight.nodes) == B.MIN_LEVELS


def test_a_set_mesh_size_is_not_coarsened():
    f = B.fit(B.PRINT, (12000, 12300), 12.0, [1.4, 1.0], 20.0, fixed_size=True)
    assert f.rung == 0 and f.kept and not f.over


def test_a_bigger_budget_never_picks_a_coarser_rung():
    rungs = [B.fit(B.CASTING, (3000, 4000), 3.0, [1.0, 0.72], b).rung for b in (30, 100, 300, 1000, 3000)]
    assert rungs == sorted(rungs, reverse=True)


def test_node_growth_follows_the_size_the_pilots_measured():
    assert B.size_power(500, 2000, 6.0, 3.0) == pytest.approx(2.0)
    assert B.size_power(500, 480, 6.0, 3.0) == 0.3                    # a count that does not move is held up
    assert B.grow(1000, 6.0, 4, 3.0, 4, 2.0) == pytest.approx(4000)
    assert B.grow(1000, 6.0, 4, 6.0, 16, 2.0) == pytest.approx(1000 * 4 ** B.CURVE_POWER)


def test_the_note_says_what_was_given_up():
    f = B.fit(B.PRINT, (12000, 12300), 12.0, [1.4, 1.0], B.DEFAULT_BUDGET)
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
