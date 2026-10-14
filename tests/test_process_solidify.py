"""The enthalpy solver against exact answers: Neumann's freezing front, a lumped block, and energy bookkeeping."""
import math

import numpy as np
import pytest

from wobbl.process import solidify as S
from wobbl.process.analytic import neumann_front, neumann_lambda


@pytest.fixture(scope="module")
def bar(tmp_path_factory):
    from wobbl.geometry.cad import prepare
    from wobbl.mesh.mesher import Sizing, build_mesh
    from wobbl.validation import geometry as G

    d = tmp_path_factory.mktemp("bar")
    step = G.box(d / "bar.step", 60, 6, 6)
    return build_mesh(prepare(step, d / "geo"), Sizing(h=1.5, order=1))


def _front(X, T, tm):
    x = X[:, 0]
    bins = np.linspace(0, x.max(), 241)
    idx = np.digitize(x, bins)
    keep = [i for i in range(1, len(bins)) if (idx == i).any()]
    xs = np.array([x[idx == i].mean() for i in keep])
    ts = np.array([T[idx == i].mean() for i in keep])
    j = np.nonzero(ts >= tm)[0][0]
    return float(np.interp(tm, [ts[j - 1], ts[j]], [xs[j - 1], xs[j]]))


def test_freezing_front_follows_neumann(bar):
    k, rho, c, L, tm, ti, tw, w = 200.0, 2700.0, 900.0, 397e3, 660.0, 700.0, 600.0, 4.0
    mat = S.Material("m", rho, k, c, L, lambda T: np.clip((tm + w / 2 - np.asarray(T)) / w, 0, 1), tm + w / 2,
                     tm - w / 2)
    held = np.nonzero(bar.X[:, 0] < 1e-6)[0]
    m = S.Model(bar.X, bar.tets, np.zeros(len(bar.tets), int), [mat], np.full(len(bar.X), ti),
                fixed=[(held, tw)], duration=3.0, save_times=[3.0])
    r = S.solve(m)
    a = k / (rho * c) * 1e6
    lam = neumann_lambda(k, k, a, a, c, L, tm, ti, tw)
    assert _front(bar.X, r.saved[3.0], tm) == pytest.approx(neumann_front(lam, a, 3.0), rel=0.02)
    assert r.energy["error"] < 1e-6
    assert r.energy["supplied_J"] < 0          # the cold face takes heat out


def test_block_with_a_film_cools_like_the_lumped_solution(bar):
    rho, c, k, h = 2700.0, 900.0, 200.0, 50.0
    mat = S.Material("al", rho, k, c)
    tris = bar.tris[:, :3]
    m = S.Model(bar.X, bar.tets, np.zeros(len(bar.tets), int), [mat], np.full(len(bar.X), 200.0),
                films=[(tris, h, 20.0)], duration=300.0, save_times=[300.0])
    r = S.solve(m)
    vol = 60 * 6 * 6 * 1e-9
    area = 2 * (60 * 6 * 2 + 36) * 1e-6
    tau = rho * c * vol / (h * area)
    expected = 20.0 + 180.0 * math.exp(-300.0 / tau)
    assert float(r.saved[300.0].mean()) == pytest.approx(expected, rel=0.01)
    assert r.energy["error"] < 1e-6
    assert r.energy["lost_J"] == pytest.approx(-r.energy["stored_J"], rel=1e-6)
