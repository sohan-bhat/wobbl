"""Mold filling: the Cross-WLF fluidity, a strip and a weld line behind a hole, and where a gate goes."""
import math

import numpy as np
import pytest

from wobbl.process import hele_shaw as H
from wobbl.process.analytic import strip_pressure
from wobbl.process.data import get_resin


def _mesh(tmp_path, build, h):
    from wobbl.geometry.cad import prepare
    from wobbl.mesh.mesher import Sizing, build_mesh
    from wobbl.validation import geometry as G

    step = G._session(tmp_path / "part.step", build)
    return build_mesh(prepare(step, tmp_path / "geo"), Sizing(h=h, order=1))


def test_cross_fluidity_runs_from_newtonian_to_power_law():
    ps = get_resin("gpps")
    visc = H.Viscosity(ps, 230.0)
    e0 = float(visc.eta0(0.0))
    b = np.array([2.0])
    # tiny stress: the gap flow of a Newtonian melt at the zero shear viscosity
    assert H.fluidity(visc, b, np.array([1e-9]), np.zeros(1))[0] == pytest.approx(8 / (12 * e0 * 1e-6), rel=1e-3)
    # large stress: eta -> eta0^n tau*^(1-n) gamma_dot^(n-1), a power law with that consistency
    G = np.array([ps.tau * 1e-6 * 2e4])            # wall stress 2e4 tau*
    m = e0 ** ps.n * ps.tau ** (1 - ps.n)
    assert H.fluidity(visc, b, G, np.zeros(1))[0] == pytest.approx(
        H.fluidity(H.Viscosity(m=m, n=ps.n), b, G, np.zeros(1))[0], rel=0.03)


def test_strip_fills_like_the_power_law_channel(tmp_path):
    L, W, gap = 80.0, 10.0, 2.0
    mesh = _mesh(tmp_path, lambda o: o.addBox(0, 0, 0, L, W, gap), 2.0)
    hm = H.Mesh(mesh.X, mesh.tets, np.full(len(mesh.tets), gap))
    gate = np.nonzero(mesh.X[:, 0] < 1e-6)[0]
    m, n, Q = 2e4, 0.35, L * W * gap / 1.0
    res = H.fill(hm, H.Viscosity(m=m, n=n), H.FillSettings([gate], Q))
    assert not res.short and res.filled == pytest.approx(1.0)
    assert res.injected == pytest.approx(res.in_cavity, rel=1e-9)
    i = int(np.searchsorted(res.share, 0.6))
    expected = strip_pressure(m, n, Q, gap / 2, W, res.share[i] * L) / 1e6
    assert res.gate_pressure[i] == pytest.approx(expected, rel=0.06)
    # the front reaches x at t = x W gap / Q
    mid = np.abs(mesh.X[:, 0] - 40.0) < 1.0
    assert float(np.nanmean(res.t_fill[mid])) == pytest.approx(40.0 * W * gap / Q, rel=0.08)


def test_a_pressure_limit_slows_the_fill_and_can_stop_it(tmp_path):
    mesh = _mesh(tmp_path, lambda o: o.addBox(0, 0, 0, 60, 10, 1.0), 2.0)
    hm = H.Mesh(mesh.X, mesh.tets, np.full(len(mesh.tets), 1.0))
    gate = np.nonzero(mesh.X[:, 0] < 1e-6)[0]
    free = H.fill(hm, H.Viscosity(m=2e4, n=0.35), H.FillSettings([gate], 600.0))
    capped = H.fill(hm, H.Viscosity(m=2e4, n=0.35), H.FillSettings([gate], 600.0, p_max=0.5 * free.gate_pressure.max(),
                                                                   t_limit=3.0))
    assert capped.gate_pressure.max() <= 0.5 * free.gate_pressure.max() * (1 + 1e-9)
    assert capped.short or capped.times[-1] > free.times[-1] * 1.2


def test_weld_line_forms_behind_a_hole_and_the_gate_goes_in_the_middle(tmp_path):
    def plate(o):
        a = o.addBox(0, 0, 0, 80, 40, 2)
        c = o.addCylinder(40, 20, -1, 0, 0, 4, 6)
        o.cut([(3, a)], [(3, c)])

    mesh = _mesh(tmp_path, plate, 2.0)
    hm = H.Mesh(mesh.X, mesh.tets, np.full(len(mesh.tets), 2.0))
    gate = np.nonzero(mesh.X[:, 0] < 1e-6)[0]
    res = H.fill(hm, H.Viscosity(m=2e4, n=0.35), H.FillSettings([gate], 80 * 40 * 2 / 1.0))
    weld = H.weld_lines(hm, res.t_fill)
    assert len(weld)
    # the fronts split by the hole meet first past its upstream side, in the band the hole shadows. Where the last of
    # the fill meets the far walls the fill time flattens and the mesh each platform's gmsh makes decides which
    # nodes read as a weld, so the line is judged where the fronts meet first
    # (the nodes on the side walls have gradients from one side only, so they are left out)
    inner = weld[(mesh.X[weld, 1] > 3) & (mesh.X[weld, 1] < 37)]
    assert len(inner)
    first = inner[np.argsort(res.t_fill[inner])[:max(1, len(inner) // 2)]]
    behind = mesh.X[first]
    assert (behind[:, 0] >= 40).mean() > 0.5
    assert np.abs(behind[:, 1] - 20).max() < 10, np.round(behind, 1).tolist()
    g = H.infer_gate(hm, np.unique(mesh.tris[:, :3]))
    assert abs(mesh.X[g, 0] - 40) < 15


def test_frozen_skin_depth_from_the_conduction_solution():
    ps = get_resin("gpps")
    r = H.frozen_skin_ratio(ps, 230.0, 50.0)
    # erf(r) is how far the no flow temperature sits from the mold toward the melt
    assert math.erf(r) == pytest.approx((ps.no_flow - 50) / (230 - 50), rel=1e-6)
