"""Advancing a crack front: each point turns by the maximum tangential stress angle and moves at its own rate."""
import math

import numpy as np
import pytest

from wobbl.fracture.growth import law_for, parse_spectrum
from wobbl.fracture.handbook import mts_angle
from wobbl.fracture.path import advance, front_size, orient, resample
from wobbl.fracture.sif import FrontK, smooth


def _front(k1, k2=0.0, n=9):
    """A straight front along z at x = 0 on a crack in the plane y = 0, growing toward +x."""
    z = np.linspace(0, 5, n)
    xyz = np.column_stack([np.zeros(n), np.zeros(n), z])
    k = np.column_stack([np.broadcast_to(k1, n), np.broadcast_to(k2, n), np.zeros(n)]).astype(float)
    return {"xyz": xyz, "forward": np.tile([1.0, 0, 0], (n, 1)), "normal": np.tile([0, 1.0, 0], (n, 1)),
            "tangent": np.tile([0, 0, 1.0], (n, 1)), "k": k}


def test_mode_one_front_moves_straight_ahead():
    law, sp = law_for("7075-t6"), parse_spectrum(r_ratio=0.1)
    pts, da, theta = advance(_front(10.0), law, sp, 0.33, 2.0, 0.5, 1.0)
    assert np.allclose(da, 0.5) and np.allclose(theta, 0.0)
    assert np.allclose(pts[1:-1, 0], 0.5) and np.allclose(pts[1:-1, 1], 0.0)
    # the new front reaches past both surfaces by the given amount
    assert pts[0, 2] == pytest.approx(-1.0) and pts[-1, 2] == pytest.approx(6.0)


def test_mixed_mode_front_turns_by_mts():
    law, sp = law_for("7075-t6"), parse_spectrum(r_ratio=0.1)
    pts, da, theta = advance(_front(10.0, 10.0), law, sp, 0.33, 2.0, 0.5, 1.0)
    want = mts_angle(10.0, 10.0)
    assert np.allclose(theta, want)
    step = pts[1] - _front(10.0)["xyz"][0]
    assert math.atan2(step[1], step[0]) == pytest.approx(want)


def test_slow_points_move_less_and_points_below_threshold_stay():
    law, sp = law_for("7075-t6"), parse_spectrum(r_ratio=0.1)
    f = _front(np.linspace(1.0, 10.0, 9))
    _, da, _ = advance(f, law, sp, 0.33, 2.0, 0.5, 1.0)
    assert da[-1] == pytest.approx(0.5) and da[0] == 0.0
    assert np.all(np.diff(da) >= 0)
    _, none, _ = advance(_front(1.0), law, sp, 0.33, 2.0, 0.5, 1.0)
    assert np.all(none == 0)


def test_resample_orient_and_smooth():
    s = np.linspace(0, 5, 11)
    xyz = np.column_stack([np.zeros(11), np.zeros(11), s])
    k = np.column_stack([10 + s, np.zeros(11), np.zeros(11)])
    fk = FrontK(np.arange(11), xyz, s, np.tile([1.0, 0, 0], (11, 1)), np.tile([0, 1.0, 0], (11, 1)),
                np.tile([0, 0, 1.0], (11, 1)), k, smooth(s, k), 0.33)
    assert fk.k[:, 0] == pytest.approx(10 + s)      # a straight line survives the smoothing
    r = resample(fk, 5)
    assert r["xyz"][:, 2] == pytest.approx([0, 1.25, 2.5, 3.75, 5])
    flipped = orient(fk, np.array([[0, 0, 6.0], [0, 0, -1.0]]))
    assert flipped.xyz[0, 2] == 5 and flipped.k[0, 0] == pytest.approx(15)


def test_front_element_size():
    assert front_size(1.4, None, 6.0, 0.0, "normal") == pytest.approx(0.1)
    assert front_size(30.0, None, 6.0, 5.0, "normal") == pytest.approx(5.0 / 6)
    assert front_size(100.0, None, 6.0, 0.0, "normal") == pytest.approx(1.5)
