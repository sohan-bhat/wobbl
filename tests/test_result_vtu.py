import re
import zlib
from types import SimpleNamespace

import numpy as np
import pytest

from wobbl.fe import StaticResult
from wobbl.geometry.cad import prepare
from wobbl.mesh.mesher import Sizing, build_mesh
from wobbl.structural.pipeline import _save_vtu


def _run(parts, tmp_path):
    prep = prepare(parts["plate_hole"], tmp_path / "g")
    m = build_mesh(prep, Sizing(h=8.0))
    rng = np.random.default_rng(1)
    S = rng.normal(0.0, 50.0, (m.n_nodes, 6))
    res = StaticResult(U=rng.normal(0.0, 0.1, (m.n_nodes, 3)), S=S, RF=np.zeros((m.n_nodes, 3)))
    run = SimpleNamespace(mesh=m, result=res, measure=res.von_mises(), excluded=np.zeros(m.n_nodes, bool))
    _save_vtu(run, tmp_path / "result.vtu")
    return m, S


def _read(path):
    """A minimal reader for the appended, zlib compressed layout Wobbl writes: name -> array, plus cells."""
    data = path.read_bytes()
    head, raw = data.split(b"<AppendedData encoding=\"raw\">\n_", 1)
    out = {}
    for tag in re.findall(rb"<DataArray [^>]*/>", head):
        attrs = dict(re.findall(rb'(\w+)="([^"]*)"', tag))
        off = int(attrs[b"offset"])
        nb, size, last, comp = np.frombuffer(raw[off:off + 32], "<u8")
        assert nb == 1 and size == last
        dtype = {b"Float32": "<f4", b"Int32": "<i4", b"Int64": "<i8", b"UInt8": "u1"}[attrs[b"type"]]
        a = np.frombuffer(zlib.decompress(raw[off + 32:off + 32 + int(comp)]), dtype)
        k = int(attrs.get(b"NumberOfComponents", b"1"))
        out[attrs.get(b"Name", b"points").decode()] = a.reshape(-1, k) if k > 1 else a
    return out


def test_result_vtu_keeps_the_stress_tensor(parts, tmp_path):
    m, S = _run(parts, tmp_path)
    g = _read(tmp_path / "result.vtu")
    assert g["points"].shape == (m.n_nodes, 3) and np.allclose(g["points"], m.X, atol=1e-4)
    assert np.array_equal(g["connectivity"].reshape(-1, 10), m.tets) and set(g["types"]) == {24}
    T = g["stress_MPa"]
    assert T.shape == (m.n_nodes, 6)
    assert np.allclose(T, S, rtol=1e-6, atol=1e-4)
    # the tensor is what the von Mises field was computed from
    sxx, syy, szz, sxy, syz, szx = T.astype(float).T
    vm = np.sqrt(0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2) + 3 * (sxy ** 2 + syz ** 2 + szx ** 2))
    assert np.allclose(vm, g["von_mises_MPa"], rtol=1e-5)


def test_paraview_reads_the_result_vtu(parts, tmp_path):
    """VTK itself, the reader inside ParaView, opens the file and finds the same fields."""
    pv = pytest.importorskip("pyvista")
    m, S = _run(parts, tmp_path)
    g = pv.read(tmp_path / "result.vtu")
    assert g.n_points == m.n_nodes and g.n_cells == m.n_elems
    assert int(g.celltypes[0]) == pv.CellType.QUADRATIC_TETRA
    T = np.asarray(g.point_data["stress_MPa"])
    assert T.shape == (m.n_nodes, 6)
    assert np.allclose(T, S, rtol=1e-6, atol=1e-4)
    for name in ("displacement_mm", "von_mises_MPa", "failure_measure_MPa", "max_principal_MPa", "excluded_zone"):
        assert name in g.point_data
