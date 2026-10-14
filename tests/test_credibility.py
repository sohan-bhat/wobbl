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

from conftest import needs_ccx, needs_spice
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


CSV = """# Synthetic readings for a test of the reader, not a real test.
name,quantity,x,y,z,direction,load,value,uncertainty,unit
SG1,strain,50,10,10,x,100,650,13,microstrain
D1,deflection,200,10,0,-z,100,0.091,0.002,in
T1,temperature,10,10,10,,,149,1,F
LED,i_led_A,,,,,,21.5,0.2,mA
"""


def test_readings_convert_units_and_flag_synthetic(tmp_path):
    from wobbl.credibility.readings import read_tests

    p = tmp_path / "tests.csv"
    p.write_text(CSV)
    rs, meta = read_tests(p)
    assert meta["synthetic"] and meta["rows"] == 4
    sg, d1, t1, led = rs
    assert sg.value == pytest.approx(650e-6) and sg.u == pytest.approx(13e-6) and sg.direction == [1, 0, 0]
    assert d1.quantity == "displacement" and d1.value == pytest.approx(0.091 * 25.4) and d1.direction == [0, 0, -1]
    assert t1.value == pytest.approx(65.0) and t1.u == pytest.approx(5 / 9)
    assert led.value == pytest.approx(0.0215) and led.xyz is None


@pytest.mark.parametrize("row,words", [("SG1,strain,,,,x,100,650,13,microstrain", "needs x, y and z"),
                                       ("SG1,strain,1,2,3,x,100,650,13,psi", "unknown strain unit"),
                                       ("SG1,strain,1,2,3,sideways,100,650,13,microstrain", "direction"),
                                       (",strain,1,2,3,x,100,650,13,microstrain", "no name")])
def test_readings_say_which_row_is_wrong(tmp_path, row, words):
    from wobbl.credibility.readings import ReadingError, read_tests

    p = tmp_path / "bad.csv"
    p.write_text("name,quantity,x,y,z,direction,load,value,uncertainty,unit\n" + row + "\n")
    with pytest.raises(ReadingError, match=words):
        read_tests(p)


def test_area_metric_and_u_pooling():
    from wobbl.credibility.readings import area_metric, u_pooled

    z = stats.norm.ppf((np.arange(20000) + 0.5) / 20000)
    assert area_metric(z, [0.0]) == pytest.approx(math.sqrt(2 / math.pi), rel=1e-3)   # E|X| for X ~ N(0, 1)
    assert area_metric(z, [3.0]) == pytest.approx(3.0, rel=1e-3)
    good = u_pooled((np.arange(12) + 0.5) / 12)
    bad = u_pooled(np.full(12, 0.97))
    assert good["area"] < good["reference_95"] < bad["area"]


def test_calibration_matches_the_conjugate_posterior():
    """A reading linear in one normal input has a normal posterior in closed form."""
    from wobbl.credibility.calibrate import calibrate

    theta = uq.Input("k", "normal", 10.0, 2.0)
    a = np.array([1.0, 2.0, 3.0, 4.0])
    sigma = np.full(4, 0.5)
    data = a * 11.3 + np.array([0.3, -0.2, 0.4, -0.1])
    prec = 1 / 4.0 + np.sum(a ** 2) / 0.25
    mean = (10.0 / 4.0 + np.sum(a * data) / 0.25) / prec
    r = calibrate(lambda X: X[:, :1] * a[None, :], [theta], data, sigma, steps=12000)
    c = r["inputs"][0]
    assert c["mean"] == pytest.approx(mean, abs=0.05 / math.sqrt(prec))
    assert c["sd"] == pytest.approx(1 / math.sqrt(prec), rel=0.1)
    assert c["ci95"][0] < 11.3 < c["ci95"][1] and c["narrowed"] > 5
    assert r["residual_after"]["rms_in_sigma"] < r["residual_before"]["rms_in_sigma"]
    assert 0.2 < r["acceptance"] < 0.8


def test_readings_pick_the_idealization_they_came_from():
    """Gauges read from the distributed field, at a modulus 3% low, favour it over the rigid one, and the modulus
    fitted to it comes back."""
    from types import SimpleNamespace

    from wobbl.credibility.against import model_form

    dist = np.array([800e-6, 300e-6, 120e-6, 0.26])
    rigid = np.array([500e-6, 310e-6, 410e-6, 0.21])
    ratio = np.array([1.0, 1.0, 1.0, 1.0])
    pred = SimpleNamespace(unit=dist, alt_unit=rigid, ratio=ratio, E0=72400.0,
                           alt={"assumed": "a distributed attachment", "label": "loads entering through a rigid "
                                                                                  "attachment"})
    data = dist * 72400.0 / 70300.0
    sigma = np.array([10e-6, 6e-6, 5e-6, 0.005])
    mf = model_form(pred, data, sigma)
    assert mf["favours"] == "a distributed attachment" and mf["log10_likelihood_ratio"] > 10
    assert mf["idealizations"][0]["fitted_modulus"] == pytest.approx(70300.0, rel=1e-6)
    assert mf["idealizations"][1]["idealization"] == "a rigid attachment"


def test_credibility_assessment_and_test_plan(tmp_path):
    from wobbl.credibility.assess import model_risk, run_credibility

    assert model_risk("low", "low") == 1 and model_risk("high", "high") == 5
    d = _copy("examples/cast-motor-mount/output", tmp_path)
    run_uncertainty(d, ["load 10%", "yield 8%", "E 3%"], samples=20000)
    r = run_credibility(d, "high", "medium")
    assert r["risk"] == 4 and r["goal"] == "high"
    f = {x["factor"]: x for x in r["factors"]}
    assert f["solution verification"]["level"] == "high"
    assert f["model form"]["level"] == "low" and not f["model form"]["meets"]   # rigid versus distributed servo
    assert f["comparator"]["level"] == "low" and f["model inputs"]["level"] == "high"
    assert "does not certify" in r["disclaimer"]
    plan = json.loads((d / "test-plan.json").read_text())
    kinds = [t["kind"] for t in plan["tests"]]
    assert kinds[:2] == ["validation", "model form"] and "material coupons" in kinds
    val = plan["tests"][0]
    assert val["steps"] == [0.25, 0.5, 0.75, 1.0]     # the peak stays under 80% of yield at the design load
    assert val["gauges"][0]["kind"] == "peak" and val["gauges"][0]["xyz"] == pytest.approx(
        json.loads((d / "outcome.json").read_text())["governing_xyz"])
    verdicts = {x["test"]: x["verdict"] for x in plan["replace"]}
    assert verdicts["a test to failure"] == "cannot" and verdicts["a static proof test at the design load"] == "not yet"
    html = (d / "credibility.html").read_text()
    for words in ("Credibility evidence", "Factors", "Test plan", "Uncertainty", "does not certify"):
        assert words in html
    assert "Test plan:" in (d / "summary.txt").read_text()


@pytest.mark.solver
@needs_ccx
def test_twin_experiment_recovers_the_modulus(parts, tmp_path):
    """Readings made from a cantilever's own solution at a known modulus, with noise, are calibrated back."""
    from wobbl.credibility.against import run_validation
    from wobbl.credibility.probes import gauge, load_fields, surface
    from wobbl.structural.pipeline import analyze_strength
    from wobbl.structural.spec import StructuralStudy, parse_fixture, parse_load

    st = StructuralStudy(parts["cantilever"], "6061-t6", [parse_load("100 N -z @ x=max")], [parse_fixture("x=min")],
                         quality="fast", budget=120)
    out = tmp_path / "run"
    analyze_strength(st, out, render=False, fixes=False)
    f = load_fields(out)
    surf = surface(f)
    rng = np.random.default_rng(1)
    rows = ["# synthetic twin experiment", "name,quantity,x,y,z,direction,load,value,uncertainty,unit"]
    for name, x in (("SG1", 50.0), ("SG2", 100.0), ("SG3", 150.0)):
        g = gauge(f, (x, 10.0, 10.0), 68900.0, 0.33, (1, 0, 0), surf=surf)
        for load in (50.0, 100.0):
            eps = g["strain"] * load / 100.0 * 68900.0 / 66000.0 * 1e6
            rows.append(f"{name},strain,{x},10,10,x,{load},{eps + 5 * rng.standard_normal():.3f},5,microstrain")
    (tmp_path / "t.csv").write_text("\n".join(rows) + "\n")
    r = run_validation(out, tmp_path / "t.csv", ["E 5%"])
    c = r["calibration"]["inputs"][0]
    assert c["input"] == "E" and c["ci95"][0] < 66000 < c["ci95"][1]
    assert r["tests"]["synthetic"] and "synthetic" in r["headline"]
    assert "Comparison with tests" in (out / "report.html").read_text()


@pytest.mark.solver
@needs_ccx
def test_heat_uncertainty_through_new_solves(tmp_path):
    """A small aluminum cube making 2 W in still air is nearly isothermal, so its rise is P / (h A), and a lognormal
    film coefficient gives a mean rise of rise0 (1 + cov^2) in closed form. The surrogate of new heat solves has to
    find that, and say how well it fits."""
    from wobbl.thermal.study import run_thermal
    from wobbl.thermal.tools import study_from
    from wobbl.validation import geometry as G

    part = G.box(tmp_path / "cube.step", 20, 20, 20)
    st = study_from(part, "6061-t6", power="2 W", films=["10 W/m2K @ all"], ambient="25 C", quality="fast")
    run_thermal(st, tmp_path / "run")
    assert json.loads((tmp_path / "run" / "study.json").read_text())["part"] == "../cube.step"
    r = run_uncertainty(tmp_path / "run", ["film lognormal 20%"], samples=20000, numerical=False, solves=6)
    rise0 = json.loads((tmp_path / "run" / "outcome.json").read_text())["values"]["t_max_C"] - 25.0
    mean = r["outputs"]["t_max_C"]["mean"] - 25.0
    assert mean == pytest.approx(rise0 * 1.04, rel=0.01)
    assert r["model"]["kind"] == "polynomial chaos" and r["model"]["surrogate"]["solves"] == 6
    assert r["model"]["surrogate"]["loo_error"]["t_max_C"] < 0.01
    assert r["sensitivity"]["indices"][0]["input"] == "film"


@pytest.mark.solver
@needs_spice
def test_circuit_uncertainty_through_ngspice(tmp_path):
    from wobbl.circuits.run import run_circuit

    run_circuit("led", {"supply": "5", "resistor": "150", "led": "red"}, tmp_path / "led")
    r = run_uncertainty(tmp_path / "led", ["resistor 5%", "supply 2%", "vf sd 0.05"], samples=20000, solves=24)
    i0 = json.loads((tmp_path / "led" / "outcome.json").read_text())["values"]["i_led_A"]
    assert r["outputs"]["i_led_A"]["mean"] == pytest.approx(i0, rel=0.02)
    assert r["probability"]["failure"]["value"] < 0.01         # 20 mA nominal against a 30 mA rating
    assert {x["input"] for x in r["sensitivity"]["indices"]} >= {"resistor", "supply", "vf"}
