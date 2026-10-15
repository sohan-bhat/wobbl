"""The bluff body reference check: classes from the shape, the published curves and the bands around them."""
from __future__ import annotations

import numpy as np
import pytest

from wobbl.flow import bluff
from wobbl.flow import correlations as C


def _sphere(r, n=24, m=48):
    th, ph = np.linspace(0, np.pi, n + 1), np.linspace(0, 2 * np.pi, m + 1)

    def P(i, j):
        return (r * np.sin(th[i]) * np.cos(ph[j]), r * np.sin(th[i]) * np.sin(ph[j]), r * np.cos(th[i]))
    t = []
    for i in range(n):
        for j in range(m):
            a, b, c, d = P(i, j), P(i + 1, j), P(i + 1, j + 1), P(i, j + 1)
            if i > 0:
                t.append((a, b, d))
            if i < n - 1:
                t.append((b, c, d))
    return np.array(t)


def _box(a, b, c):
    v = np.array([[x, y, z] for x in (0, a) for y in (0, b) for z in (0, c)], float)
    q = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    return v[np.array([t for a_, b_, c_, d in q for t in ((a_, b_, c_), (a_, c_, d))])]


def _cylinder(r, length, n=48):
    ph = np.linspace(0, 2 * np.pi, n + 1)
    t = []
    for j in range(n):
        a, b = (r * np.cos(ph[j]), r * np.sin(ph[j])), (r * np.cos(ph[j + 1]), r * np.sin(ph[j + 1]))
        p = lambda q, z: (q[0], q[1], z)      # noqa: E731
        t += [(p(a, 0), p(b, 0), p(b, length)), (p(a, 0), p(b, length), p(a, length)),
              ((0, 0, 0), p(b, 0), p(a, 0)), ((0, 0, length), p(a, length), p(b, length))]
    return np.array(t)


def test_shapes_are_recognised():
    s = _sphere(0.05)
    assert bluff.classify(s, np.pi * 0.05 ** 2).kind == "sphere"
    c = _cylinder(0.05, 0.4)[:, :, [2, 0, 1]]        # axis along x: along the stream, not across it
    assert bluff.classify(c, np.pi * 0.05 ** 2).kind == "slender"
    across = _cylinder(0.05, 0.4)[:, :, [0, 2, 1]]    # axis along y
    sh = bluff.classify(across, 0.1 * 0.4)
    assert sh.kind == "cylinder" and sh.length == pytest.approx(0.4) and sh.width == pytest.approx(0.1, rel=0.01)
    assert bluff.classify(_box(0.1, 0.1, 0.1), 0.01).kind == "blunt"
    assert bluff.classify(_box(0.01, 0.3, 0.3), 0.09).kind == "flat"


def test_a_sphere_that_reads_half_its_drag_fails_and_the_curve_stands_in():
    sh = bluff.classify(_sphere(0.05), np.pi * 0.05 ** 2)
    ck = bluff.check(sh, 25000.0, 0.25)
    assert ck.status == "fail" and ck.detail["use_reference"]
    assert ck.detail["cd_reference"] == pytest.approx(C.sphere_cd(25000.0))
    assert "attached too" in ck.summary
    ok = bluff.check(sh, 25000.0, 0.47)
    assert ok.status == "pass" and not ok.detail["use_reference"]


def test_the_crisis_widens_the_band_and_classes_have_bands():
    sh = bluff.classify(_sphere(0.05), np.pi * 0.05 ** 2)
    assert bluff.check(sh, 3e5, 0.2).status == "pass"
    cube = bluff.classify(_box(0.1, 0.1, 0.1), 0.01)
    assert bluff.check(cube, 2e4, 1.05).status == "pass"
    high = bluff.check(cube, 2e4, 2.5)
    assert high.status == "fail" and not high.detail["use_reference"]
    plate = bluff.classify(_box(0.01, 0.3, 0.3), 0.09)
    assert bluff.check(plate, 2e4, 0.2).status == "fail"
