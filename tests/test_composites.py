"""Composite layups on shell models: a cantilever strip against classical lamination theory, end to end."""
from __future__ import annotations

import json

import numpy as np
import pytest

from conftest import needs_ccx


@pytest.mark.solver
@needs_ccx
def test_cross_ply_strip_matches_lamination_theory(tmp_path):
    """A [0/90]s strip, 120 x 20 mm and 1 mm thick, clamped at one end with 25 N on the other. Beam theory gives
    the moment per width at the governing point, and lamination theory turns it into the first ply failure."""
    from wobbl.composites.laminate import clt_first_ply_failure, parse_layup
    from wobbl.composites.study import run_laminate
    from wobbl.composites.tools import study_from

    part = tmp_path / "strip.py"
    part.write_text('PARAMS = {"length": 120.0, "width": 20.0, "thickness": 1.0}\n\n\n'
                    "def build(occ, length, width, thickness):\n    occ.addBox(0, 0, 0, length, width, thickness)\n")
    F, L, b = 25.0, 120.0, 20.0
    st = study_from(str(part), "[0/90]s", "t300-5208", [f"{F} N +z @ x=max"], ["x=min"], ply_thickness=0.25,
                    quality="fast")
    o = run_laminate(st, tmp_path / "out", render=False)
    v = o.values
    lam = parse_layup("[0/90]s", "t300-5208", 0.25)
    x = v["governing_xyz"][0]
    want = clt_first_ply_failure(lam, np.zeros(3), np.array([F * (L - x) / b, 0.0, 0.0]))
    assert v["strength_ratio"] == pytest.approx(min(want["tsai_wu"], want["hashin"]), rel=0.08)
    checks = {c.name: c.status for c in o.checks}
    assert checks["hand calculation"] == "pass" and checks["force balance"] == "pass"
    # below the 1.5 target, so the layup changes come with their own lamination theory check
    assert o.verdict == "marginal" and o.fixes
    assert all(f["strength_ratio"] >= 1.5 for f in o.fixes)
    pb = tmp_path / "out" / "playback.json"
    assert pb.stat().st_size < 300_000
    doc = json.loads(pb.read_text())
    assert doc["kind"] == "laminate" and doc["first_ply_failure"]["load_factor"] == pytest.approx(v["strength_ratio"])
    assert doc["frames"]["load_factor"][-1] == pytest.approx(v["strength_ratio"], rel=1e-4)
    assert len(doc["plies"]) == 4
