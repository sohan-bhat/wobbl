"""The uncertainty math on problems with known answers: input parsing, sampling, polynomial chaos, Sobol indices
and failure probabilities. Nothing here needs a solver."""
from __future__ import annotations

import math

import numpy as np
import pytest

from wobbl.credibility import uq

KNOWN = {"yield": uq.Known(165.0, "MPa", "strength", "yield strength"),
         "load": uq.Known(2000.0, "N", "load", "the belt pull"),
         "ambient": uq.Known(25.0, "C", "boundary", "the air temperature", positive=False)}


@pytest.mark.parametrize("text,dist,mean,sd", [
    ("yield 8%", "normal", 165.0, 13.2),
    ("load lognormal 10%", "lognormal", 2000.0, 200.0),
    ("yield normal mean 180 sd 12", "normal", 180.0, 12.0),
    ("yield sd 12 MPa", "normal", 165.0, 12.0),
    ("ambient tol 3", "normal", 25.0, 1.0),
    ("load uniform 1900..2100", "uniform", 2000.0, 200 / math.sqrt(12)),
    ("load 1900..2100", "uniform", 2000.0, 200 / math.sqrt(12)),
])
def test_vary_reads_every_form(text, dist, mean, sd):
    inp = uq.parse_vary(text, KNOWN)
    assert inp.dist == dist and inp.mean == pytest.approx(mean) and inp.sd == pytest.approx(sd)
    assert inp.role == KNOWN[inp.name].role


@pytest.mark.parametrize("text,words", [("yield", "how much"), ("yield 8% kPa", "MPa"), ("density 3%", "can vary"),
                                        ("yield 60%", "lognormal"), ("load uniform -10..10", "positive")])
def test_vary_says_what_is_wrong(text, words):
    with pytest.raises(uq.VaryError, match=words):
        uq.parse_vary(text, KNOWN)


def test_inputs_map_to_one_standard_normal():
    for inp in (uq.parse_vary("yield 8%", KNOWN), uq.parse_vary("load lognormal 10%", KNOWN),
                uq.parse_vary("load uniform 5%", KNOWN)):
        x = inp.ppf(np.array([0.05, 0.5, 0.95]))
        assert inp.cdf(x) == pytest.approx([0.05, 0.5, 0.95], abs=1e-9)
        assert inp.from_z(inp.to_z(x)) == pytest.approx(x)


def test_latin_hypercube_puts_one_sample_in_each_slice():
    u = uq.unit_samples(50, 3, "lhs", seed=4)
    for j in range(3):
        assert sorted((u[:, j] * 50).astype(int)) == list(range(50))


def test_lognormal_product_matches_the_closed_form():
    """Tip deflection F L^3 / (3 E I) with lognormal F and E is lognormal, so its mean and spread are exact."""
    F = uq.Input("F", "lognormal", 100.0, 10.0)
    E = uq.Input("E", "lognormal", 200000.0, 10000.0)
    X = uq.sample([F, E], 100000, seed=1)
    d = X[:, 0] / X[:, 1]
    s2 = math.log(1.01) + math.log(1.0025)
    mean = 100.0 / 200000.0 * (1.0025)    # E[F] E[1/E], and E[1/E] = (1 + cov^2) / mean for a lognormal
    assert d.mean() == pytest.approx(mean, rel=2e-3)
    assert d.std() == pytest.approx(mean * math.sqrt(math.exp(s2) - 1), rel=1e-2)


def test_pce_is_exact_on_a_polynomial_and_honest_off_it():
    ins = [uq.parse_vary("yield 8%", KNOWN), uq.parse_vary("load uniform 5%", KNOWN)]
    X = uq.sample(ins, 30, seed=2)
    y = 3.0 + 0.2 * X[:, 0] - 1e-4 * X[:, 1] ** 2 + 1e-3 * X[:, 0] * X[:, 1]
    s = uq.fit_pce(ins, X, y, ["y"])
    assert s.loo[0] < 1e-12
    Xt = uq.sample(ins, 200, seed=3)
    want = 3.0 + 0.2 * Xt[:, 0] - 1e-4 * Xt[:, 1] ** 2 + 1e-3 * Xt[:, 0] * Xt[:, 1]
    assert s(Xt)[:, 0] == pytest.approx(want, rel=1e-9)
    rough = uq.fit_pce(ins, X, np.sin(X[:, 0] / 3.0), ["y"], max_degree=2)
    assert rough.loo[0] > 1e-6


def test_sobol_indices_of_the_ishigami_function():
    ins = [uq.Input(f"x{i + 1}", "uniform", 0.0, 2 * math.pi / math.sqrt(12), -math.pi, math.pi) for i in range(3)]
    got = {r["input"]: r for r in uq.sobol_indices(uq.ishigami, ins, n=2 ** 12, boot=20)}
    for name, (s1, st) in uq.ishigami_indices().items():
        assert got[name]["first_order"] == pytest.approx(s1, abs=0.02)
        assert got[name]["total"] == pytest.approx(st, abs=0.02)
    assert [r["input"] for r in uq.sobol_indices(uq.ishigami, ins, n=2 ** 10, boot=5)][0] == "x1"


def test_probability_and_its_interval():
    rng = np.random.default_rng(0)
    pr = uq.probability(rng.random(20000) < 0.03)
    assert pr["ci95"][0] < 0.03 < pr["ci95"][1]
    assert pr["reliability_index"] == pytest.approx(-uq.ndtri(pr["value"]))
    none = uq.probability(np.zeros(10000, bool))
    assert none["value"] == 0 and none["ci95"][1] == pytest.approx(3.7e-4, rel=0.05)
    assert "no failures in 10,000 samples" in uq.odds(none)
    assert uq.odds(pr).startswith("about 1 in 3")
