"""Elastic supports: a bed of springs over a face, read from text, built with exact total stiffness, written for
CalculiX and solved against beam theory on a Winkler bed."""
from __future__ import annotations

import numpy as np
import pytest

from conftest import needs_ccx
from wobbl.structural.build import binned_stiffness
from wobbl.structural.spec import SpecError, StructuralStudy, parse_fixture


def test_elastic_support_reads_its_stiffness():
    fx = parse_fixture("elastic 50 N/mm3 @ z=min")
    assert fx.kind == "elastic" and fx.faces == "z=min" and fx.stiffness == 50.0
    assert parse_fixture("springs 50 MN/m3 @ holes d=6.6").stiffness == pytest.approx(0.05)
    assert parse_fixture("elastic 100 pci @ bottom").stiffness == pytest.approx(0.0271447)
    for bad, words in (("elastic z=min", "stiffness per area"), ("elastic 50 N @ z=min", "50 N/mm3"),
                       ("elastic -5 N/mm3 @ z=min", "positive")):
        with pytest.raises(SpecError, match=words):
            parse_fixture(bad)


@pytest.mark.parametrize("word", ["elastic", "spring", "springs", "foundation", "ELASTIC:"])
def test_every_elastic_word_is_read(word):
    fx = parse_fixture(f"{word} 50 N/mm3 @ x=min")
    assert (fx.kind, fx.faces, fx.stiffness) == ("elastic", "x=min", 50.0)


def test_elastic_support_survives_study_json():
    st = StructuralStudy("part.step", "6061-t6", [], [parse_fixture("elastic 50 N/mm3 @ z=min")])
    again = StructuralStudy.from_dict(st.to_dict())
    assert again.fixtures[0].kind == "elastic" and again.fixtures[0].stiffness == 50.0


def test_binned_stiffness_keeps_the_total():
    k = np.random.default_rng(0).lognormal(0.0, 1.5, 5000)
    b = binned_stiffness(k)
    assert b.sum() == pytest.approx(k.sum(), rel=1e-12)
    assert len(np.unique(b)) <= 64 and np.max(np.abs(b / k - 1)) < 0.2
    assert np.all(binned_stiffness(np.full(10, 3.0)) == 3.0)


def test_springs_are_written_for_calculix(tmp_path):
    from wobbl.fe import Spring, StaticModel
    from wobbl.mesh.mesher import Mesh
    from wobbl.solvers.calculix import write_static_inp

    X = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], float)
    mesh = Mesh(X, np.array([[0, 1, 2, 3]]), np.zeros((0, 3), int), np.zeros(0, int), np.zeros(0, int),
                np.zeros(0, int), 1)
    model = StaticModel(mesh, 1000.0, 0.3, 1e-9, springs=[Spring("fix1", np.array([0, 1, 2]),
                                                                 np.array([2.0, 2.0, 5.0]))])
    write_static_inp(tmp_path / "m.inp", model)
    text = (tmp_path / "m.inp").read_text()
    assert text.count("TYPE=SPRING1") == 6 and text.count("*SPRING, ELSET") == 6
    assert "2,1\n3,2\n" in text   # element numbers start after the one tetrahedron, nodes are 1 based


@pytest.mark.solver
@needs_ccx
def test_elastic_root_against_a_winkler_bed(parts, tmp_path):
    """A softer bed turns the root more, which adds F L^2 / (k I) at the tip, and the reactions still balance."""
    from wobbl.structural.pipeline import analyze_strength
    from wobbl.structural.spec import parse_load

    st = StructuralStudy(parts["cantilever"], "6061-t6", [parse_load("100 N -z @ x=max")],
                         [parse_fixture("elastic 500 N/mm3 @ x=min")], quality="fast", budget=120)
    o = analyze_strength(st, tmp_path / "run", render=False, fixes=False)
    E, b, h, L, F, k = 68900.0, 20.0, 10.0, 200.0, 100.0, 500.0
    I = b * h ** 3 / 12
    tip = F * L ** 3 / (3 * E * I) + F * L / (5 / 6 * E / 2.66 * b * h) + F * L ** 2 / (k * I) + F / (k * b * h)
    assert o.max_disp_mm == pytest.approx(tip, rel=0.03)
    assert next(c for c in o.checks if c.name == "force balance").status == "pass"
    assert any("bed of springs of 500 N/mm3" in a for a in o.assumptions)
