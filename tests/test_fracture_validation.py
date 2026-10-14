"""The fracture validation cases are registered, and the quick ones pass."""
import pytest

from conftest import needs_ccx
from wobbl.validation.cases import CASES, run_cases

FRACTURE = {"center-crack", "edge-crack", "surface-crack", "paris-life", "kink-angle"}


def test_fracture_cases_are_registered():
    ids = {c.id for c in CASES if c.domain == "fracture"}
    assert FRACTURE <= ids
    assert len({c.id for c in CASES}) == len(CASES)


def test_paris_life_case(tmp_path):
    (row,) = run_cases(tmp_path, ["paris-life"], log=lambda m: None)
    assert row["pass"] and row["error"] < 1e-4


@pytest.mark.solver
@needs_ccx
def test_edge_crack_case(tmp_path):
    (row,) = run_cases(tmp_path, ["edge-crack"], log=lambda m: None)
    assert row["pass"], row
    assert not list(tmp_path.rglob("*.frd"))      # solver files are removed after each case
