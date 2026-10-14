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
DOMAIN_SUITE = {"strength": "strength", "thermal": "heat", "circuit": "circuit"}


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
    checks = {c["name"]: c for c in o.get("checks", [])}
    F = []

    # code verification: cases with known answers in this domain
    rows, src = suite_results(d, suite)
    mine = [r for r in rows if r.get("domain") == DOMAIN_SUITE.get(dom)]
    if mine:
        ok = sum(bool(r.get("pass")) for r in mine)
        worst = max(mine, key=lambda r: r.get("error", 0) or 0)
        lvl = "high" if ok == len(mine) and len(mine) >= 3 else ("medium" if ok == len(mine) else "low")
        ev = (f"{ok} of {len(mine)} {DOMAIN_SUITE[dom]} cases with known answers pass their tolerance, run through "
              f"the same pipeline, the largest error {100 * (worst.get('error') or 0):.2g}% ({worst.get('title')})")
    else:
        from ..validation.cases import CASES

        n = len([c for c in CASES if c.domain == DOMAIN_SUITE.get(dom)])
        lvl, ev = "low", (f"{n} {DOMAIN_SUITE.get(dom, dom)} cases exist in Wobbl's validation suite, but no results "
                          "were found here; run 'wobbl validate' and pass its results.json")
    F.append(_factor("code verification", "verification", lvl, goal, ev,
                     "run 'wobbl validate' on this machine and keep its results with the review"))

    # solution verification: the run's own checks
    conv = checks.get("mesh convergence") or checks.get("time step convergence") or {}
    bal = checks.get("force balance") or checks.get("energy balance") or checks.get("power balance") or {}
    st = [conv.get("status"), bal.get("status")]
    lvl = "high" if st == ["pass", "pass"] else ("low" if "fail" in st or None in st else "medium")
    if dom == "circuit" and conv.get("status") == "pass" and not bal:
        lvl = "high"
    ev = ". ".join(f"The {c['name']} check {VERB.get(c['status'], c['status'])}, {c['summary']}"
                   for c in (conv, bal) if c) or "no convergence check"
    F.append(_factor("solution verification", "verification", lvl, goal, ev,
                     "refine further (--quality fine or a longer --budget) until the answer settles"))

    # use error: did the run rest on guesses
    guesses = [a for a in o.get("assumptions", []) if a.startswith("Wobbl guessed")]
    F.append(_factor("inputs as intended", "verification", "medium" if guesses else "high", goal,
                     ("the run used guesses for " + ", ".join(g.replace("Wobbl guessed: ", "").rstrip(".")
                                                              for g in guesses[:3]))
                     if guesses else "every load, support and material was stated explicitly",
                     "state the guessed inputs explicitly and run again"))

    # model form: idealizations the checks flag
    flags = [c for c in o.get("checks", []) if c["name"] in ("load introduction", "large deflection",
                                                               "singularities", "hand calculation")
             and c["status"] in ("warn", "fail")]
    alt = (u or {}).get("alternatives", [])
    lvl = "high" if not flags else "medium"
    if any(c["name"] == "load introduction" for c in flags):
        lvl = "low"
    ev = ("no check found an idealization that moves the answer" if not flags else
          ". ".join(f"The {c['name']} check reports {_sentence(c['summary'])}" for c in flags))
    if v and any("idealization" in n for n in v.get("notes", [])):
        lvl = "medium" if lvl == "low" else lvl
    F.append(_factor("model form", "validation", lvl, goal, ev,
                     "measure where the idealizations differ most, so a test decides between them"))

    # model inputs: uncertainty and sensitivity
    if u:
        roles = {i["role"] for i in u["inputs"]}
        need = {"load", "strength"} if dom == "strength" else ({"load", "boundary"} if dom == "thermal" else {"load"})
        lvl = "high" if need <= roles else "medium"
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
                     "the failure probability or the margin leaves too little room; narrow the dominant input"))

    met = sum(f["meets"] for f in F)
    res = {"kind": "credibility", "domain": dom, "wobbl": __version__,
           "created": _dt.datetime.now().strftime("%Y-%m-%d %H:%M"), "question": question or o.get("question", ""),
           "context_of_use": _context(o), "influence": influence, "consequence": consequence, "risk": risk,
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


def _context(o: dict) -> str:
    dom = o.get("domain")
    if dom == "strength":
        return (f"A linear elastic static model of {Path(o.get('part', '')).name} in {o.get('material')}, under "
                f"{'; '.join(o.get('loads', []))}, held by {'; '.join(o.get('fixtures', []))}, used to judge "
                f"the {o.get('measure')} against {fmt(o.get('allowable_MPa', 0))} MPa with a target safety factor "
                f"of {o.get('target_sf'):g}.")
    if dom == "thermal":
        return (f"A heat conduction model of {Path(o.get('part', '')).name} used to judge its hottest point against "
                f"{fmt(o.get('values', {}).get('limit_C', 0))} C.")
    return f"An ngspice model of the {o.get('part')} circuit used to judge it against its rating."


def _applicability(o: dict, u: dict | None, v: dict) -> tuple[str, str]:
    import numpy as np

    rows = v["readings"]
    notes = []
    lvl = "medium"
    if o.get("domain") == "strength":
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
