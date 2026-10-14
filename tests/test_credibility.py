"""Uncertainty, comparison with tests, credibility and test plans on finished results. A strength result scales
exactly, so these run on the outcome files of the bundled examples without a solver."""
from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import numpy as np
import pytest
from scipy import integrate, stats

from wobbl.credibility import uq
from wobbl.credibility.models import StrengthModel, for_result
from wobbl.credibility.uncertainty import run_uncertainty

ROOT = Path(__file__).resolve().parents[1]


def _copy(folder: str, tmp_path: Path, files=("outcome.json", "study.json")) -> Path:
    d = tmp_path / Path(folder).parent.name
    d.mkdir()
    for f in files:
        if (ROOT / folder / f).exists():
            shutil.copy(ROOT / folder / f, d / f)
    return d


def test_strength_model_knows_what_it_can_vary(tmp_path):
    d = _copy("examples/cast-motor-mount/output", tmp_path)
    m = for_result(d)
    assert isinstance(m, StrengthModel)
    assert m.known["load"].nominal == 2000.0 and m.known["load"].unit == "N"
    assert m.known["yield"].nominal == 165.0 and m.known["E"].nominal == 72400.0
    assert [i.name for i in m.parse(["load 10%", "strength 8%", "young 3%"])] == ["load", "yield", "E"]
    with pytest.raises(uq.VaryError, match="can vary"):
        m.parse(["thickness 3%"])
    with pytest.raises(uq.VaryError, match="twice"):
        m.parse(["load 10%", "force 5%"])


def test_exact_scaling_matches_direct_integration(tmp_path):
    """Safety factor = yield / (load factor x converged stress), so the failure probability is a one dimensional
    integral that quadrature gets to many digits."""
    d = _copy("examples/cast-motor-mount/output", tmp_path)
    r = run_uncertainty(d, ["load 10%", "yield 8%"], samples=100000, numerical=False)
    o = json.loads((d / "outcome.json").read_text())
    g0 = o["governing_MPa"]
    want, _ = integrate.quad(lambda a: stats.norm.cdf(a * g0, 165, 13.2) * stats.norm.pdf(a, 1, 0.1), 0.4, 1.6)
    pf = r["probability"]["failure"]
    assert pf["ci95"][0] - 2e-4 <= want <= pf["ci95"][1] + 2e-4, (pf, want)
    assert r["outputs"]["governing_MPa"]["mean"] == pytest.approx(g0, rel=2e-3)
    assert r["outputs"]["governing_MPa"]["sd"] == pytest.approx(0.1 * g0, rel=1e-2)
    # the stress spread is all load, the safety factor shares it with the yield scatter
    names = [x["input"] for x in r["sensitivity"]["indices"]]
    assert set(names) == {"load", "yield"}
    total = sum(x["first_order"] for x in r["sensitivity"]["indices"])
    assert total == pytest.approx(1.0, abs=0.05)
    assert r["model"]["kind"] == "exact"


def test_uncertainty_files_and_report(tmp_path):
    d = _copy("examples/cast-motor-mount/output", tmp_path)
    r = run_uncertainty(d, ["load lognormal 10%", "yield 8%", "E 3%"], samples=20000)
    saved = json.loads((d / "uncertainty.json").read_text())
    assert saved["headline"] == r["headline"] and "mesh" in [i["name"] for i in saved["inputs"]]
    # the rigid servo case Wobbl already solved bounds the answer from the other side
    alt = r["alternatives"][0]
    assert "rigid" in alt["label"] and alt["probability"]["value"] < r["probability"]["failure"]["value"]
    html = (d / "report.html").read_text()
    assert "<h2>Uncertainty</h2>" in html and "What matters most" in html
    assert "Uncertainty: The chance of failing" in (d / "summary.txt").read_text()
    assert (d / "images" / "uncertainty.png").stat().st_size < 120_000
    e = next(x for x in r["sensitivity"]["indices"] if x["input"] == "E")
    assert abs(e["total"]) < 0.01        # stiffness moves the deflection, never the safety factor


def test_heat_and_circuit_results_name_their_inputs(tmp_path):
    d = _copy("examples/circuits/led", tmp_path, ("outcome.json",))
    m = for_result(d)
    assert {"supply", "resistor", "vf", "imax"} <= set(m.known)
    assert m.decision == "i_led_A"
    old_heat = _copy("examples/battery-pack/output", tmp_path, ("outcome.json",))
    with pytest.raises(ValueError, match="study.json"):
        for_result(old_heat)


def test_probability_words_and_rounding():
    assert uq.odds({"value": 0.025, "ci95": [0.024, 0.026], "failures": 250, "samples": 10000}).startswith(
        "about 1 in 40")
    assert math.isclose(uq.probability(np.array([True] * 5 + [False] * 95))["value"], 0.05)
