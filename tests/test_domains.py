"""The one interface credibility uses for every domain: request.json, the numbers a spec is made of, reruns and
the criteria that decide a run."""
from __future__ import annotations

import json

import numpy as np
import pytest

from wobbl.credibility import domains as D


def test_spec_leaves_keep_their_units_and_form():
    spec = {"name": "rock", "mass": "145 g", "diameter": "7.4 cm", "speed": 40, "angle": 35, "air": "vacuum",
            "motor": {"mass": 2.0, "curve": [[0, 10], [1, 10]]}, "question": "how far?"}
    found = D.leaves(spec)
    assert set(found) == {"mass", "diameter", "speed", "angle", "motor.mass"}
    assert (found["mass"].value, found["mass"].unit, found["mass"].text) == (145.0, "g", True)
    out = D.with_values(spec, {"diameter": 8.1, "speed": 42.5, "motor.mass": 2.5}, found)
    assert out["diameter"] == "8.1 cm" and out["speed"] == 42.5 and out["motor"]["mass"] == 2.5
    assert spec["diameter"] == "7.4 cm"          # the original is untouched


def test_roles_come_from_names_or_the_domain():
    assert D.role_of("thrust") == "load" and D.role_of("yield") == "strength" and D.role_of("diameter") == "geometry"
    assert D.role_of("film") == "boundary" and D.role_of("vf") == "material"
    assert D.role_of("vf", {"vf": "load"}) == "load"


def test_criteria_listed_or_read_from_value_names():
    o = {"values": {"t_max_C": 80.0, "limit_C": 85.0, "ripple_V": 0.02, "max_ripple": 0.05, "safety_factor": 1.4}}
    names = {c["value"]: (c.get("limit_key"), c["sense"]) for c in D.criteria(o)}
    assert names == {"t_max_C": ("limit_C", "max"), "ripple_V": ("max_ripple", "max"), "safety_factor": (None, "min")}
    given = dict(o, criteria=[{"name": "landing", "value": "t_max_C", "limit": 90.0, "sense": "max"}])
    assert [c["name"] for c in D.criteria(given)] == ["landing"]
    vals = D.scalar_values(o)
    assert D.governing(D.criteria(o), vals)["value"] == "t_max_C"     # 5.9% from its limit, the closest


def test_numerical_change_from_any_convergence_check():
    o = {"checks": [{"name": "mesh convergence", "status": "pass", "summary": "", "detail": {"last_change": 0.004}},
                    {"name": "time step convergence", "status": "pass", "summary": "agree (0.3% apart)"},
                    {"name": "integrated again tighter", "status": "pass", "summary": "the apogee moved 0.02%"}]}
    assert D.numerical_change(o) == pytest.approx(0.004)
    assert D.numerical_change({"checks": o["checks"][1:]}) == pytest.approx(0.003)


def test_request_round_trip_keeps_files_relative(tmp_path):
    part = tmp_path / "parts" / "bar.step"
    part.parent.mkdir()
    part.write_text("x")
    out = tmp_path / "runs" / "bar"
    out.mkdir(parents=True)
    D.save_request(out, "strength", {"part": str(part), "loads": ["100 N -z @ x=max"]}, "study")
    raw = json.loads((out / "request.json").read_text())
    assert raw["spec"]["part"] == "../../parts/bar.step"
    req = D.load_request(out, {"domain": "strength"})
    assert req.domain == "strength" and req.kind == "study" and req.spec["part"] == str(part.resolve())


def test_a_run_with_no_request_says_so(tmp_path):
    (tmp_path / "outcome.json").write_text(json.dumps({"domain": "acoustics", "values": {}}))
    with pytest.raises(D.DomainError, match="no request.json"):
        D.load_request(tmp_path)


def test_time_series_are_read_from_any_csv(tmp_path):
    (tmp_path / "trajectory.csv").write_text("t_s,altitude_m,speed_mps\n0,0,0\n1,10,20\n2,30,15\n")
    (tmp_path / "parts.csv").write_text("name,count\na,1\nb,2\nc,3\n")
    s = D.read_series(tmp_path)
    assert list(s) == ["trajectory"]
    assert D.series_value(s, "altitude_m", 1.5) == pytest.approx(20.0)
    assert D.series_value(s, "trajectory:speed_mps", 0.5) == pytest.approx(10.0)
    assert D.series_value(s, "missing", 1.0) is None


def test_a_registered_runner_and_a_router(tmp_path):
    calls = []

    @D.register("toy")
    def run(req, out):
        calls.append(req.spec["x"])
        return {"domain": "toy", "values": {"y": req.spec["x"] ** 2}}

    req = D.Request("toy", {"x": 3.0})
    assert D.rerun(req, tmp_path / "a")["values"]["y"] == 9.0
    D.set_router(lambda r, out: {"domain": r.domain, "values": {"y": -1.0}} if r.domain == "toy" else None)
    try:
        assert D.rerun(req, tmp_path / "b")["values"]["y"] == -1.0
    finally:
        D._ROUTER.clear()
    assert calls == [3.0]


def test_generic_model_on_a_toy_domain(tmp_path):
    """Any registered domain gets distributions, failure odds and Sobol indices with no code of its own."""
    from wobbl.credibility.uncertainty import run_uncertainty

    @D.register("toy2")
    def run(req, out):
        a, b = float(req.spec["a"]), float(req.spec["b"].split()[0])
        return {"domain": "toy2", "values": {"peak_C": 20 + a * b, "limit_C": 85.0},
                "checks": [{"name": "time step convergence", "status": "pass", "summary": "(0.1% apart)"}]}

    out = tmp_path / "run"
    out.mkdir()
    o = run(D.Request("toy2", {"a": 5.0, "b": "10 W"}), out)
    o.update(question="does it stay cool?", verdict="ok", headline="ok", metrics=[], confidence="high",
             confidence_reasons=[], assumptions=["the toy is linear"], part="toy")
    (out / "outcome.json").write_text(json.dumps(o))
    D.save_request(out, "toy2", {"a": 5.0, "b": "10 W"})
    r = run_uncertainty(out, ["a 10%", "b 20%"], samples=50000)
    assert r["decision"] == "peak_C" and r["model"]["kind"] == "polynomial chaos"
    assert r["outputs"]["peak_C"]["mean"] == pytest.approx(70.0, rel=0.01)
    # a b > 65 with a ~ N(5, 0.5) and b ~ N(10, 2): about 34% of samples
    rng = np.random.default_rng(0)
    want = np.mean(rng.normal(5, 0.5, 400000) * rng.normal(10, 2, 400000) > 65)
    assert r["probability"]["failure"]["value"] == pytest.approx(want, abs=0.01)
    assert {x["input"] for x in r["sensitivity"]["indices"]} >= {"a", "b"}


def test_readings_of_any_value_keep_their_name_and_convert_their_unit(tmp_path):
    from wobbl.credibility.readings import read_tests

    (tmp_path / "t.csv").write_text("name,quantity,value,uncertainty,unit\nALT,apogee_m,10000,30,ft\n"
                                    "I,i_led_A,20,0.1,mA\nT,t_max_C,212,1,F\n")
    rs, _ = read_tests(tmp_path / "t.csv")
    assert [r.quantity for r in rs] == ["apogee_m", "i_led_A", "t_max_C"]
    assert rs[0].value == pytest.approx(3048.0) and rs[0].u == pytest.approx(9.144)
    assert rs[1].value == pytest.approx(0.02) and rs[2].value == pytest.approx(100.0)


def test_registry_spec_leaves():
    spec = {"thing": "/x/bar.step", "kind": "step", "question": "q",
            "options": {"material": ["6061-t6"], "load": ["220 N down @ top", "60 N +y @ x=max"], "fix": ["x=min"],
                        "param": ["thickness=6"], "target-sf": ["2"], "film": ["10 W/m2K @ all"]}}
    found = D.leaves(spec)
    assert set(found) == {"load1", "load2", "thickness", "target-sf", "film"}
    out = D.with_values(spec, {"load1": 250.0, "thickness": 6.5, "film": 12.0}, found)["options"]
    assert out["load"] == ["250 N down @ top", "60 N +y @ x=max"]
    assert out["param"] == ["thickness=6.5"] and out["film"] == ["12 W/m2K @ all"] and out["fix"] == ["x=min"]


def test_a_registry_run_scatters_through_the_registry(tmp_path):
    """A run 'wobbl simulate' made is run again by its own analysis, and its criteria come from job.criterion."""
    from wobbl.credibility.uncertainty import run_uncertainty
    from wobbl.simulate import Analysis, Option, get, register

    def entry(job):
        a = float(job.value("a", "5"))
        b = float(job.value("b", "10 W").split()[0])
        o = {"domain": "toy3", "part": "toy", "question": job.question, "verdict": "ok", "headline": "ok",
             "metrics": [], "confidence": "high", "confidence_reasons": [], "assumptions": [], "checks": [],
             "values": {"peak_C": 20 + a * b}}
        job.out.mkdir(parents=True, exist_ok=True)
        (job.out / "outcome.json").write_text(json.dumps(o))
        job.criterion("temperature", "peak_C", 85.0, "max")
        return o

    register(Analysis("toydomain", "warmth", "a toy", entry, options=(Option("a"), Option("b"))))
    out = tmp_path / "run"
    get("toydomain", "warmth").run({"thing": "", "kind": "words", "question": "warm?",
                                    "options": {"a": ["5"], "b": ["10 W"]}}, out)
    req = D.load_request(out)
    assert req.registry and req.domain == "toydomain" and req.kind == "warmth"
    assert json.loads((out / "outcome.json").read_text())["criteria"][0]["limit"] == 85.0
    r = run_uncertainty(out, ["a 10%", "b 20%"], samples=50000)
    assert r["decision"] == "peak_C" and "reruns of the toy3 study" in r["model"]["method"]
    rng = np.random.default_rng(0)
    want = np.mean(rng.normal(5, 0.5, 400000) * rng.normal(10, 2, 400000) > 65)
    assert r["probability"]["failure"]["value"] == pytest.approx(want, abs=0.01)
