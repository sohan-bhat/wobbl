"""Flow through wobbl simulate: questions in words read into the right numbers, each fluid analysis runs from a
question alone and records what it guessed, and an OpenFOAM case folder runs as it is."""
from __future__ import annotations

import pytest

from conftest import needs_foam
from wobbl.flow.words import read
from wobbl.simulate import route, run_plan


def test_words_find_flow_numbers():
    r = read("40 m of 25 mm pipe carrying 30 L/min of water at 60 C, what is the pressure drop?")
    assert (r.length, r.size, r.flow, r.fluid, r.temp, r.mode) == ("40 m", "25 mm", "30 l/min", "water", "60 C",
                                                                   "internal")
    r = read("drag on a 22 cm sphere at 20 m/s in the wind")
    assert (r.size, r.speed, r.fluid, r.mode) == ("22 cm", "20 m/s", "air", "external")
    r = read("does a 15 W chip stay cool with a 10 CFM, 25 Pa fan?")
    assert r.power == "15 w" and r.fan == "10 cfm, 25 pa"
    assert read("hydraulic oil through the manifold at 12 gpm").fluid == "iso-vg-46"
    assert read("a 2.5 inch hose").size == "2.5 inch"


def _run(thing, question, tmp_path, **opts):
    plan = route(thing, question)
    steps = run_plan(plan, {k.replace("_", "-"): [v] for k, v in opts.items()}, tmp_path, quick=True)
    assert len(steps) == 1 and not steps[0].error, steps[0].error
    return plan, steps[0]


def test_each_fluid_question_runs_from_words_alone(tmp_path):
    plan, st = _run("", "40 m of 25 mm pipe carrying 30 L/min of water, what is the pressure drop?", tmp_path / "a")
    assert plan.names == ["pressure-drop"] and st.outcome.values["dp_Pa"] == pytest.approx(20.4e3, rel=0.03)
    assert any("temperature" in t for t, _ in st.job.assumptions)
    plan, st = _run("", "what is the drag on a 22 cm sphere at 20 m/s?", tmp_path / "b")
    assert plan.names == ["drag"] and st.outcome.values["drag_N"] == pytest.approx(4.41, rel=0.02)
    plan, st = _run("apcsf_10x4.7", "can a 1.2 kg quad with 4 rotors hover?", tmp_path / "c")
    assert plan.names == ["thrust"] and st.outcome.verdict == "ok"
    plan, st = _run("", "a 12 m boat hull at 8 knots, how much power does it need?", tmp_path / "d")
    assert plan.names == ["resistance"] and {o for _, o in st.job.assumptions} >= {"--beam '3.6 m'"}
    assert (tmp_path / "d" / "request.json").exists()


def test_a_question_missing_its_flow_says_what_to_give(tmp_path):
    plan = route("", "what is the pressure drop in a 25 mm pipe?")
    with pytest.raises(ValueError, match="flow rate.*30 L/min"):
        run_plan(plan, {}, tmp_path, quick=True)


@pytest.mark.solver
@needs_foam
def test_an_openfoam_case_folder_runs_as_it_is(tmp_path):
    from wobbl.flow import cavity

    src = cavity.write(tmp_path / "cavity")
    before = sorted(p.name for p in src.rglob("*"))
    plan, st = _run(str(src), "", tmp_path / "out")
    o = st.outcome
    assert plan.names == ["cfd"] and o.values["application"] == "pimpleFoam" and o.values["steps"] == 100
    assert {c.name: c.status for c in o.checks}["continuity"] == "pass"
    assert sorted(p.name for p in src.rglob("*")) == before          # the user's folder is left alone
