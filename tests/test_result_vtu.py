from types import SimpleNamespace

import numpy as np
import pyvista as pv

from wobbl.fe import StaticResult
from wobbl.geometry.cad import prepare
from wobbl.mesh.mesher import Sizing, build_mesh
from wobbl.structural.pipeline import _save_vtu


def test_result_vtu_keeps_the_stress_tensor(parts, tmp_path):
    prep = prepare(parts["plate_hole"], tmp_path / "g")
    m = build_mesh(prep, Sizing(h=8.0))
    rng = np.random.default_rng(1)
    S = rng.normal(0.0, 50.0, (m.n_nodes, 6))
    res = StaticResult(U=rng.normal(0.0, 0.1, (m.n_nodes, 3)), S=S, RF=np.zeros((m.n_nodes, 3)))
    run = SimpleNamespace(mesh=m, result=res, measure=res.von_mises(), excluded=np.zeros(m.n_nodes, bool))
    _save_vtu(run, tmp_path / "result.vtu")

    g = pv.read(tmp_path / "result.vtu")
    T = np.asarray(g.point_data["stress_MPa"])
    assert T.shape == (m.n_nodes, 6)
    assert np.allclose(T, S)
    # the tensor is what the von Mises field was computed from
    sxx, syy, szz, sxy, syz, szx = T.T
    vm = np.sqrt(0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2) + 3 * (sxy ** 2 + syz ** 2 + szx ** 2))
    assert np.allclose(vm, g.point_data["von_mises_MPa"])
