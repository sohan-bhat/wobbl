"""Plasticity, contact and bolted joints on parts and assemblies they were not tuned on: every example part, every
validation part, and assemblies built in wobbl.validation.generality. Each must run to an answer and pass its own
checks. Slow (minutes a case); 'python -m wobbl.validation.generality OUT --workers 3' runs them all in parallel."""
import pytest
from conftest import needs_ccx

from wobbl.validation.generality import CASES, run_case

pytestmark = [pytest.mark.solver, needs_ccx]


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_case_runs_and_checks_itself(case, tmp_path):
    row = run_case(case, tmp_path)
    assert row["ran"], row.get("error")
    assert row["ok"], row["failed_checks"]
