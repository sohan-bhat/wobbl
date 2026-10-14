"""The pieces of CFD on a CAD part that need no solver: openings and lids, the stream's frame, frontal area, the
shedding frequency from a force history, and the thermal wall function."""
from __future__ import annotations

import numpy as np
import pytest

from wobbl.flow import shape
from wobbl.flow.cooling import jayatilleke_tplus
from wobbl.flow.transient import dominant


def _cube_tris(size=1.0):
    v = np.array([[x, y, z] for x in (0, size) for y in (0, size) for z in (0, size)], float)
    quads = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    return v[np.array([t for a, b, c, d in quads for t in ((a, b, c), (a, c, d))])]


def test_lids_fill_any_simple_loop():
    ell = np.array([[0, 0], [3, 0], [3, 1], [1, 1], [1, 3], [0, 3]], float)       # an L, not convex
    tris = shape.ear_clip(ell)
    def cross(u, v):
        return u[0] * v[1] - u[1] * v[0]
    area = sum(abs(cross(ell[b] - ell[a], ell[c] - ell[a])) / 2 for a, b, c in tris)
    assert len(tris) == 4 and area == pytest.approx(5.0)
    square = np.array([[0, 1, 2], [0, 2, 3]])
    assert [sorted(lp) for lp in shape.boundary_loops(square)] == [[0, 1, 2, 3]]


def test_inside_and_frontal_area_of_a_cube():
    t = _cube_tris(2.0)
    assert list(shape.inside_surface(t, np.array([[1, 1, 1], [3, 1, 1], [1.9, 0.1, 1.5]]))) == [True, False, True]
    assert shape.frontal_area(t, [1, 0, 0]) == pytest.approx(4.0, rel=0.01)
    assert shape.frontal_area(t, [1, 1, 0]) == pytest.approx(4 * 2 ** 0.5, rel=0.01)


def test_the_stream_frame_puts_the_flow_on_x_and_up_on_z():
    R = shape.frame([0, -1, 0])
    assert R @ np.array([0, -1, 0]) == pytest.approx([1, 0, 0])
    assert R @ np.array([0, 0, 1.0]) == pytest.approx([0, 0, 1])
    assert np.linalg.det(R) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        shape.frame([0, 0, 1], up=[0, 0, 2])


def test_shedding_frequency_from_an_uneven_history():
    rng = np.random.default_rng(1)
    t = np.sort(rng.uniform(0, 4, 900))
    f, cycles = dominant(t, 0.3 * np.sin(2 * np.pi * 3.2 * t) + 0.05 * rng.standard_normal(len(t)))
    assert f == pytest.approx(3.2, rel=0.01) and cycles == pytest.approx(3.2 * np.ptp(t), rel=0.01)


def test_thermal_wall_function_meets_the_sublayer():
    y = np.array([0.5, 2.0])
    assert jayatilleke_tplus(y, 0.71) == pytest.approx(0.71 * y)        # conduction alone near the wall
    assert jayatilleke_tplus(np.array([1.0]), 7.0)[0] == pytest.approx(7.0)
    far = jayatilleke_tplus(np.array([100.0]), 7.0)[0]
    air = jayatilleke_tplus(np.array([100.0]), 0.71)[0]
    assert far > 3 * air            # water's thin thermal sublayer adds resistance a plain eddy model misses
