"""Submodeling: a fine ball around the peak of a coarse strength result, held on its cut by the global
displacements, gives the peak of a converged fine model at a fraction of its size, and says so only when its cut
is far enough from the peak."""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

from conftest import needs_ccx
from wobbl.structural import submodel as S

# A 160 x 60 x 4 mm plate with a 16 mm hole, fixed at one end and pulled at the other by 10 MPa. A fine study of
# it converges at 33.07 MPa over 24,405, 41,464 and 59,025 nodes. Heywood's finite width formula gives 32.65 MPa in
# plane stress, and the middle of a plate a quarter as thick as the hole is wide runs about a percent higher.
REFERENCE_MPA = 33.07


def cylinder(r=8.0, h=10.0, n=48, m=6):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    z = np.linspace(0, h, m)
    X = np.array([[r * np.cos(a), r * np.sin(a), zz] for zz in z for a in t])
    tris = []
    for j in range(m - 1):
        for i in range(n):
            a, b = j * n + i, j * n + (i + 1) % n
            tris += [[a, b, a + n], [b, b + n, a + n]]
    return X, np.array(tris), np.zeros(len(tris), int)


def test_curvature_radius_reads_a_hole_and_a_flat_face():
    X, tris, surf = cylinder()
    r, face = S.curvature_radius(X, tris, surf, np.array([8.0, 0.0, 5.0]), 4.0)
    assert r == pytest.approx(8.0, rel=0.1) and face == 0
    flat = np.array([[x, y, 0.0] for y in range(6) for x in range(6)], float)
    ft = np.array([[j * 6 + i, j * 6 + i + 1, (j + 1) * 6 + i] for j in range(5) for i in range(5)])
    assert S.curvature_radius(flat, ft, np.zeros(len(ft), int), np.array([2.0, 2.0, 0.0]), 3.0)[0] is None


def test_submodel_cards_go_before_and_inside_the_step():
    deck = "*NODE\n1,0,0,0\n*NSET, NSET=NCUT\n1\n*STEP\n*STATIC\n*CLOAD\n1,1,1.\n*NODE FILE\nU\n*END STEP\n"
    out = S.submodel_cards(deck, "NCUT", "global.frd")
    assert out.index("*SUBMODEL, TYPE=NODE, INPUT=global.frd\nNCUT\n") < out.index("*STEP")
    assert out.index("*STATIC") < out.index("*BOUNDARY, SUBMODEL, STEP=1\nNCUT,1,3\n") < out.index("*NODE FILE")


def test_a_result_it_cannot_use_is_refused(tmp_path):
    with pytest.raises(S.SubmodelError, match="no Wobbl strength result"):
        S.load_global(tmp_path)
    (tmp_path / "outcome.json").write_text(json.dumps({"domain": "strength"}))
    (tmp_path / "study.json").write_text(json.dumps({"part": "p.step", "shell": True}))
    with pytest.raises(S.SubmodelError, match="shell") as e:
        S.load_global(tmp_path)
    assert e.value.hint and "solid" in e.value.hint
    frd = tmp_path / "x.frd"
    frd.write_text("    1C\n    2C                          3922                                     1\n")
    assert S.frd_nodes(frd) == 3922 and S.frd_nodes(tmp_path / "none.frd") == -1


def test_a_result_is_refined_by_its_own_domain(tmp_path):
    from wobbl.jobs import workload as WL

    (tmp_path / "outcome.json").write_text(json.dumps({"domain": "motion"}))
    with pytest.raises(WL.WorkloadError, match="motion results have no local refinement") as e:
        WL.refine(tmp_path)
    assert "strength" in e.value.hint
    with pytest.raises(WL.WorkloadError, match="no Wobbl result"):
        WL.refine(tmp_path / "missing")


@pytest.fixture(scope="module")
def coarse(tmp_path_factory):
    from wobbl.structural.pipeline import analyze_strength
    from wobbl.structural.spec import StructuralStudy, parse_fixture, parse_load
    from wobbl.validation import geometry as G

    os.environ.setdefault("OMP_NUM_THREADS", "2")
    d = tmp_path_factory.mktemp("submodel")
    part = G.plate_with_hole(d / "plate.step", 160.0, 60.0, 4.0, 16.0)
    st = StructuralStudy(part, "custom:E=200000,nu=0.3,yield=250", [parse_load("-10 MPa @ x=max")],
                         [parse_fixture("fixed x=min")], quality="fast", max_levels=1, mesh_size=8.0)
    o = analyze_strength(st, d / "global", render=False, fixes=False)
    return d / "global", o.governing_MPa


@pytest.mark.solver
@needs_ccx
def test_the_local_peak_matches_a_converged_fine_model(coarse):
    folder, global_peak = coarse
    r = S.run_submodel(folder)
    assert r["trusted"] and [c["status"] for c in r["checks"]] == ["pass", "pass"]
    assert abs(r["submodel_peak_MPa"] / REFERENCE_MPA - 1) < 0.015
    assert abs(global_peak / REFERENCE_MPA - 1) > 2 * abs(r["submodel_peak_MPa"] / REFERENCE_MPA - 1)
    assert "radius of face" in r["feature"] and r["radius_mm"] == pytest.approx(2.5 * 8.0, rel=0.15)
    assert r["nodes"] < 59025 and (folder / "submodel" / "submodel.json").exists()


@pytest.mark.solver
@needs_ccx
def test_a_cut_too_close_to_the_peak_is_flagged(coarse, tmp_path):
    folder, _ = coarse
    r = S.run_submodel(folder, radius=9.0, size=1.5, out=tmp_path / "small")
    cut = next(c for c in r["checks"] if c["name"] == "cut distance")
    assert cut["status"] == "fail" and not r["trusted"]
