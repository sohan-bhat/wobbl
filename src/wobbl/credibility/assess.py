"""wobbl credibility: map a run's evidence to a risk informed credibility assessment in the manner of ASME V&V 40.

V&V 40 starts from the question and the model's context of use, sets the model risk from how much the decision
leans on the model (model influence) and how bad a wrong decision would be (decision consequence), proposes a
credibility goal for each factor in proportion to that risk, and compares the evidence with the goals. Wobbl has
evidence for most factors already. Its validation suite verifies the code, every run verifies its own solution,
uncertainty.json quantifies the inputs and validation.json compares the model with tests.

Wobbl proposes the levels and the goals, says why for each one, and leaves the judgment to the reviewer. It does
not certify a model, and a regulator or a customer decides what evidence they accept.
"""
from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path

from .. import __version__
from ..units import fmt
from . import uq

LEVELS = ["low", "medium", "high"]
VERB = {"pass": "passes", "warn": "warns", "fail": "fails", "info": "notes", "skip": "was skipped"}
SUITE_NAMES = {"thermal": "heat"}     # a domain whose validation cases go by another name


def suite_domain(dom: str) -> str:
    return SUITE_NAMES.get(dom, dom)


def model_risk(influence: str, consequence: str) -> int:
    """V&V 40's risk map: 1 when the model barely sways a decision that hardly matters, 5 when a decision with
    severe consequences rests on the model alone."""
    return LEVELS.index(influence) + LEVELS.index(consequence) + 1


def goal_for(risk: int) -> str:
    return "low" if risk <= 2 else ("medium" if risk == 3 else "high")


def suite_results(result_dir: Path, suite: str | None = None) -> tuple[list[dict], str]:
    """Validation suite results for code verification: a results file the user names, the one 'wobbl validate'
    wrote under wobbl-out, or the results published with Wobbl's source, in that order."""
    cands = [Path(suite)] if suite else []
    cands += [Path("wobbl-out/validation/results.json"), Path(__file__).resolve().parents[3] / "docs" /
              "validation-results.json"]
    for p in cands:
        if p.exists():
            try:
                return json.loads(p.read_text()), str(p)
            except (OSError, ValueError):
                continue
    return [], ""


def _factor(name: str, group: str, level: str, goal: str, evidence: str, gap: str = "") -> dict:
    meets = LEVELS.index(level) >= LEVELS.index(goal)
    return {"factor": name, "group": group, "level": level, "goal": goal, "meets": meets, "evidence": evidence,
            "gap": "" if meets else gap}


def assess(result_dir, influence: str = "medium", consequence: str = "medium", question: str = "",
           suite: str | None = None) -> dict:
    from .models import load_outcome
    from .uncertainty import load

    d = Path(result_dir)
    o = load_outcome(d)
    u, v = load(d), load(d, "validation.json")
    dom = o.get("domain", "")
    risk = model_risk(influence, consequence)
    goal = goal_for(risk)
    F = []
    model = _model(d)

    # code verification: cases with known answers in this domain
    rows, src = suite_results(d, suite)
    sd = suite_domain(dom)
    mine = [r for r in rows if r.get("domain") in (sd, dom)]
    if mine:
        ok = sum(bool(r.get("pass")) for r in mine)
        worst = max(mine, key=lambda r: r.get("error", 0) or 0)
        lvl = "high" if ok == len(mine) and len(mine) >= 3 else ("medium" if ok == len(mine) else "low")
        ev = (f"{ok} of {len(mine)} {sd} cases with known answers pass their tolerance, run through "
              f"the same pipeline, the largest error {100 * (worst.get('error') or 0):.2g}% ({worst.get('title')})")
    else:
        from ..validation.cases import CASES

        n = len([c for c in CASES if c.domain in (sd, dom)])
        lvl = "low"
        ev = (f"{n} {sd} cases exist in Wobbl's validation suite, but no results were found here, so run "
              "'wobbl validate' and pass its results.json") if n else (
            f"Wobbl's validation suite has no {sd} case with a known answer yet")
    F.append(_factor("code verification", "verification", lvl, goal, ev,
                     "run 'wobbl validate' on this machine and keep its results with the review"))

    # solution verification: the run's own convergence and conservation checks
    conv, bal = solution_checks(o)
    st = [c.get("status") for c in conv + bal]
    if not conv or "fail" in st:
        lvl = "low"
    else:
        lvl = "high" if all(x in ("pass", "info") for x in st) else "medium"
    ev = ". ".join(f"The {c['name']} check {VERB.get(c['status'], c['status'])}, {_sentence(c['summary'])}"
                   for c in conv + bal) or "the run has no convergence check"
    F.append(_factor("solution verification", "verification", lvl, goal, ev,
                     "refine further (--quality fine or a longer --budget) until the answer settles"))

    # use error: did the run rest on guesses
    guesses = [a for a in o.get("assumptions", []) if a.startswith("Wobbl guessed")]
    F.append(_factor("inputs as intended", "verification", "medium" if guesses else "high", goal,
                     ("the run used guesses for " + ", ".join(g.replace("Wobbl guessed: ", "").rstrip(".")
                                                              for g in guesses[:3]))
                     if guesses else "every load, support and material was stated explicitly",
                     "state the guessed inputs explicitly and run again"))

    # model form: idealizations and closed form cross checks the run flags
    flags = idealization_flags(o)
    lvl = "high" if not flags else "medium"
    if any(_decisive(c) for c in flags):
        lvl = "low"
    ev = ("no check found an idealization that moves the answer" if not flags else
          ". ".join(f"The {c['name']} check reports {_sentence(c['summary'])}" for c in flags))
    mf = (v or {}).get("model_form")
    if mf:
        decided = mf["favours"] == mf["assumed"] and mf["log10_likelihood_ratio"] >= 2
        from .against import ratio_words

        ev += f". The readings favour {mf['favours']} over the other idealization {ratio_words(mf)}"
        if v["tests"].get("synthetic"):
            ev += ", but the readings are synthetic, so this shows how a real test would settle it"
        elif decided:
            lvl = "high" if len(flags) == 1 else "medium"
        elif mf["favours"] != mf["assumed"]:
            ev += ", which is not the one the answer assumes, so run the study again with that attachment"
    F.append(_factor("model form", "validation", lvl, goal, ev,
                     "measure where the idealizations differ most, so a test decides between them"))

    # model inputs: uncertainty and sensitivity
    if u:
        roles = {i["role"] for i in u["inputs"]} - {"numerical"}
        can = {k.role for k in model.known.values()} if model else roles
        # the demand and the capacity both: a load, and at least one property of the design or its surroundings
        demand = "load" in roles or "load" not in can
        capacity = bool(roles - {"load"}) or not (can - {"load"})
        lvl = "high" if demand and capacity else "medium"
        top = u["sensitivity"]["indices"][0]
        ev = (f"{len(u['inputs'])} inputs propagated with {u['samples']:,} samples by {u['model']['method']}. "
              f"Of these, {top['input']} drives most of the spread, total Sobol index {top['total']:.2f}")
    else:
        lvl, ev = "low", "no input uncertainty was quantified"
    F.append(_factor("model inputs", "validation", lvl, goal, ev,
                     "run 'wobbl uncertainty' with the scatter of the load and the material strength"))

    # comparator: the tests themselves
    if v:
        n = v["summary"]["readings"]
        stated = sum(1 for r in v["readings"] if r.get("uncertainty") is not None or r.get("u"))
        synthetic = v["tests"].get("synthetic")
        lvl = "low" if synthetic else ("high" if n >= 6 else "medium")
        ev = (f"{n} readings from {v['tests']['file']}" + (", synthetic and not from a real test" if synthetic else "")
              + f", {stated} with a stated uncertainty")
    else:
        lvl, ev = "low", "no test readings were compared"
    F.append(_factor("comparator", "validation", lvl, goal, ev,
                     "test real parts with calibrated gauges (see the test plan)"))

    # output comparison: agreement and rigor
    if v:
        s = v["summary"]
        p = v["u_pooled"]
        agree = s["within"] == s["readings"] and (p["readings"] < 3 or p["area"] <= p["reference_95"])
        lvl = "high" if agree and not v["tests"].get("synthetic") else ("medium" if s["within"] >= 0.8 * s["readings"]
                                                                         else "low")
        ev = (f"{s['within']} of {s['readings']} readings within twice the validation uncertainty (ASME V&V 20), "
              f"pooled area metric {p['area']:.3f} against {p['reference_95']:.3f}")
        if v["tests"].get("synthetic"):
            lvl, ev = "low", ev + ", but the readings are synthetic"
    else:
        lvl, ev = "low", "no comparison with tests"
    F.append(_factor("output comparison", "validation", lvl, goal, ev,
                     "compare with real readings, and agreement within the validation uncertainty sets this high"))

    # applicability: do the tests cover the quantity and the regime of the decision
    if v:
        lvl, ev = _applicability(o, u, v)
    else:
        lvl, ev = "low", "no validation evidence to carry over to the decision"
    F.append(_factor("applicability", "applicability", lvl, goal, ev,
                     "put a gauge at the governing point and test up to the design load"))

    # the answer's own uncertainty against the decision
    lvl, ev = _answer(o, u)
    F.append(_factor("uncertainty in the answer", "applicability", lvl, goal, ev,
                     "the failure probability or the margin leaves too little room, so narrow the dominant input"))

    met = sum(f["meets"] for f in F)
    if src:   # relative to the result folder, as every other path in it is
        from ..verify.checks import rel_up

        src = rel_up(Path(src).resolve(), d.resolve())
    res = {"kind": "credibility", "domain": dom, "wobbl": __version__,
           "created": _dt.datetime.now().strftime("%Y-%m-%d %H:%M"), "question": question or o.get("question", ""),
           "context_of_use": _context(o, model), "influence": influence, "consequence": consequence, "risk": risk,
           "goal": goal, "factors": F, "met": met, "suite": src,
           "disclaimer": "Wobbl supplies credibility evidence and proposes levels. It does not certify the model, "
                         "and the reviewer decides whether the evidence is enough for the decision."}
    res["headline"] = (f"Model risk {risk} of 5 ({influence} influence, {consequence} consequence), so Wobbl proposes "
                       f"a {goal} goal for each factor. The evidence meets it on {met} of {len(F)} factors"
                       + (". Short on " + ", ".join(f["factor"] for f in F if not f["meets"]) + "." if met < len(F)
                          else "."))
    return res


def _sentence(text: str) -> str:
    """The first sentence of a check's summary, which carries its finding."""
    return text.split(". ")[0].rstrip(".")[:300]


def _model(d: Path):
    """The run's model, for its inputs and context of use, or None when it cannot be built."""
    from .models import ModelError, for_result

    try:
        return for_result(d)
    except (ModelError, OSError, ValueError, KeyError):
        return None


def _context(o: dict, model) -> str:
    text = model.context() if model else ""
    if text:
        return text
    part = Path(str(o.get("part", ""))).name
    q = o.get("question") or ""
    return (f"A {o.get('domain', '')} model{' of ' + part if part else ''}"
            + (f", used to answer '{q}'" if q else "") + ".")


_CONV = r"converg|time step|tolerance|refin|step size|integrated again|halved|tighter|resolution|grid"
_BAL = r"balance|conserv|budget|closes|residual|energy|mass|momentum|kirchhoff"


def solution_checks(o: dict) -> tuple[list[dict], list[dict]]:
    """The run's checks of its own numerics: convergence (mesh, time step, tolerance) and conservation."""
    import re

    conv = [c for c in o.get("checks", []) if re.search(_CONV, c["name"], re.I)]
    bal = [c for c in o.get("checks", []) if c not in conv and re.search(_BAL, c["name"], re.I)]
    return conv, bal


def idealization_flags(o: dict) -> list[dict]:
    """Checks outside the numerics that warn or fail: an idealization or a cross check that moves the answer."""
    conv, bal = solution_checks(o)
    own = {id(c) for c in conv + bal}
    return [c for c in o.get("checks", []) if id(c) not in own and c.get("status") in ("warn", "fail")
            and c["name"] not in ("units and sanity",)]


def _decisive(c: dict) -> bool:
    """A flag that says the verdict itself depends on a modeling choice."""
    det = c.get("detail") or {}
    return any(str(k).startswith("alternative") for k in det) or "verdict depends" in c.get("summary", "")


def _applicability(o: dict, u: dict | None, v: dict) -> tuple[str, str]:
    import numpy as np

    rows = v["readings"]
    notes = []
    lvl = "medium"
    if o.get("governing_xyz") is not None:
        g = np.asarray(o["governing_xyz"])
        near = [r for r in rows if r.get("xyz") and np.linalg.norm(np.asarray(r["xyz"]) - g) <= 5.0]
        loads = [r["load"] for r in rows if r.get("load") is not None]
        design = (u or {}).get("study_load", {}).get("nominal")
        reach = max(loads) / design if loads and design else None
        notes.append(f"{len(near)} reading{'s' if len(near) != 1 else ''} within 5 mm of the governing point")
        if reach is not None:
            notes.append(f"test loads reach {100 * reach:.0f}% of the design load")
        lvl = "high" if near and (reach is None or reach >= 0.8) else ("medium" if near or (reach or 0) >= 0.8
                                                                      else "low")
    else:
        notes.append(f"{len(rows)} readings of the quantities the decision uses")
    if v["tests"].get("synthetic"):
        lvl = "low"
        notes.append("the readings are synthetic")
    return lvl, ", ".join(notes)


def _answer(o: dict, u: dict | None) -> tuple[str, str]:
    if not u:
        return "low", "the answer's spread is unknown until the input scatter is propagated"
    pf = u["probability"].get("failure")
    dec = u["outputs"][u["decision"]]
    if pf is None:
        return "medium", f"{u['decision']} {fmt(dec['mean'])} ± {fmt(dec['sd'])}"
    hi = pf["ci95"][1]
    alt = [a["probability"]["ci95"][1] for a in u.get("alternatives", [])]
    worst = max([hi] + alt)
    lvl = "high" if worst < 1e-3 else ("medium" if worst < 1e-2 else "low")
    ev = f"probability of failure {uq.odds(pf)}"
    if alt:
        ev += f", and up to {worst:.2g} across the idealizations Wobbl solved"
    return lvl, ev


def run_credibility(result_dir, influence="medium", consequence="medium", question="", suite=None) -> dict:
    from .report import credibility_page, refresh
    from .testplan import plan
    from .uncertainty import _clean

    d = Path(result_dir)
    res = assess(d, influence, consequence, question, suite)
    (d / "credibility.json").write_text(json.dumps(_clean(res), indent=1))
    tp = plan(d)
    (d / "test-plan.json").write_text(json.dumps(_clean(tp), indent=1))
    refresh(d)
    credibility_page(d)
    return res
