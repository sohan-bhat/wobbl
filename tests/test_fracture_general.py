"""Crack growth on parts it was never tuned for: every finished strength or fatigue example in examples/, started
from its result folder with an automatic crack at the governing point.

The plastic parts have no crack growth data built in (none is published that fits printed or molded parts well
enough), so they are run once as they are, which must refuse with a clear message, and once as 6061-T6, which
tests the geometry. Run it with `pytest -m slow tests/test_fracture_general.py`; each part takes a few minutes.
"""
import json
import math
import shutil
from pathlib import Path

import pytest

from conftest import needs_ccx

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
TUNED = {"bracket-crack"}        # built alongside crack growth, so it proves nothing here


def results() -> list[str]:
    out = []
    for f in sorted(EXAMPLES.glob("*/output/outcome.json")):
        name = f.parts[-3]
        if name in TUNED:
            continue
        if json.loads(f.read_text()).get("domain") in ("strength", "fatigue"):
            out.append(name)
    return out


def run_case(name: str, out: Path, material: str = "", progress=None) -> dict:
    """Grow a crack in one example from its result folder, with a short path and a small Monte Carlo."""
    from wobbl.fracture.api import from_result, run

    st = from_result(EXAMPLES / name / "output", "auto", material, steps=2, quality="fast", samples=2000,
                     budget=900.0)
    o = run(st, out, progress)
    v = o.values
    row = {"example": name, "material": st.material, "verdict": o.verdict, "confidence": o.confidence,
           "crack": v.get("crack", ""), "where": v["where"], "steps": v["steps"], "path_end": v["path_end"],
           "k_start": v["k_table"]["k_mpa_sqrt_m"][0], "nominal_cycles": v["nominal_cycles"],
           "checks": {c.name: c.status for c in o.checks}}
    shutil.rmtree(out / "work", ignore_errors=True)
    return row


def test_the_plastic_examples_are_refused_with_a_reason(tmp_path):
    from wobbl.fracture.api import from_result
    from wobbl.fracture.growth import law_for
    from wobbl.materials import resolve_material

    st = from_result(EXAMPLES / "shelf-bracket" / "output")
    with pytest.raises(KeyError, match="only metals can be estimated"):
        law_for(resolve_material(st.material))


@pytest.mark.slow
@pytest.mark.solver
@needs_ccx
@pytest.mark.parametrize("name", results())
def test_crack_growth_runs_on_an_untuned_example(name, tmp_path):
    from wobbl.materials import resolve_material

    from wobbl.fracture.api import result_inputs

    mat = resolve_material(result_inputs(EXAMPLES / name / "output")["material"])
    row = run_case(name, tmp_path / name, "" if mat.category == "metal" else "6061-t6")
    assert row["steps"] >= 1 and row["k_start"] > 0 and math.isfinite(row["k_start"])
    assert row["checks"].get("SIF mesh convergence") in ("pass", "warn"), row
