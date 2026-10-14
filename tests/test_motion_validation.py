"""The motion validation cases run in seconds and land on their closed forms, so they run with every test pass."""
import json
from pathlib import Path

from wobbl.validation.cases import CASES, run_cases

MOTION = [c.id for c in CASES if c.domain == "motion"]
ROOT = Path(__file__).resolve().parents[1]


def test_every_motion_case_lands_on_its_closed_form(tmp_path):
    assert MOTION == ["free-fall", "drag-fall", "projectile", "rocket-speed", "rocket-height", "chute-descent"]
    rows = run_cases(tmp_path, MOTION, log=lambda m: None)
    for r in rows:
        assert r["pass"] and r["error"] < 1e-6 and r["confidence"] == "high", r
        assert r["wall_s"] < 10


def test_published_results_include_the_motion_cases():
    rows = {r["id"]: r for r in json.loads((ROOT / "docs" / "validation-results.json").read_text())}
    assert set(MOTION) <= set(rows) and all(rows[i]["pass"] for i in MOTION)
    assert all(rows[c.id]["title"] == c.title for c in CASES if c.id in MOTION)
