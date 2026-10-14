"""wobbl test-plan: the smallest set of physical tests that would most reduce the risk left in a decision.

Two kinds of test help. Measuring an input (coupons for the material, a load cell for the service load, the real
spread of a toleranced dimension) removes that input's share of the answer's variance, and its first order Sobol
index is exactly the expected share removed. A validation test (strain gauges, thermocouples or a bench
measurement) supplies the comparator the credibility assessment lacks and, placed where two idealizations
disagree, settles a modeling choice no amount of computing can.

The plan picks greedily. It keeps the validation test whenever a validation factor is short of its goal, the
discriminating gauge whenever the model form is, and input measurements in order of their first order index until
they cover 80% of the variance that measuring could remove. Then it says which usual tests the simulation evidence
can likely stand in for, and which it cannot.
"""
from __future__ import annotations

import datetime as _dt
from pathlib import Path

import numpy as np

from .. import __version__
from ..units import fmt

COVER = 0.8
TITLES = {"validation": "a validation test", "model form": "a gauge where the idealizations differ",
          "service load": "a service load measurement", "material coupons": "material coupons",
          "first article inspection": "a first article inspection",
          "boundary condition": "a boundary condition measurement", "component values": "component measurements"}
INPUT_TESTS = {
    "strength": ("material coupons", "Tensile test coupons from the same lot, casting or print settings, measuring "
                 "the {label} ({name})"),
    "stiffness": ("material coupons", "Measure Young's modulus on the same coupons with an extensometer"),
    "load": ("service load", "Measure {label} in service, with a load cell or a strain gauged link"),
    "geometry": ("first article inspection", "Measure the {name} on the first parts made, to replace the tolerance "
                 "band with the real spread"),
    "boundary": ("boundary condition", "Measure {label} on the real setup, for example from how fast a heated part "
                 "cools"),
    "material": ("component values", "Measure {label} on the parts in hand"),
}


def _load(d: Path, name: str):
    import json

    p = d / name
    return json.loads(p.read_text()) if p.exists() else None


def plan(result_dir) -> dict:
    d = Path(result_dir)
    o = _load(d, "outcome.json")
    u, v, c = _load(d, "uncertainty.json"), _load(d, "validation.json"), _load(d, "credibility.json")
    dom = o.get("domain")
    short = {f["factor"] for f in (c or {}).get("factors", []) if not f["meets"]}
    cands = []

    # 1. input measurements, by the share of variance each would remove
    groups: dict[str, dict] = {}
    if u:
        for x in u["sensitivity"]["indices"]:
            if x["role"] not in INPUT_TESTS or x["first_order"] < 0.02:
                continue
            kind, words = INPUT_TESTS[x["role"]]
            g = groups.setdefault(kind, {"kind": kind, "test": [], "inputs": [], "share": 0.0})
            g["test"].append(words.format(label=x.get("label") or x["input"], name=x["input"]))
            g["inputs"].append(x["input"])
            g["share"] += max(x["first_order"], 0.0)
    measurable = sum(g["share"] for g in groups.values())
    for g in groups.values():
        sd = u["outputs"][u["decision"]]["sd"]
        after = sd * np.sqrt(max(1.0 - g["share"], 0.0))
        g.update({"score": g["share"], "addresses": [f"the scatter of {', '.join(g['inputs'])}"],
                  "what": ". ".join(dict.fromkeys(g["test"])) + ".",
                  "why": (f"Knowing the {' and '.join(g['inputs'])} exactly removes about {100 * g['share']:.0f}% of "
                          f"the variance of the {u['decision'].replace('_', ' ')}, taking its standard deviation "
                          f"from {fmt(sd)} to about {fmt(after)}.")})
        if g["kind"] == "material coupons":
            g["how_many"] = ("At least 10 coupons, since the uncertainty in the lot mean falls with the square root "
                             "of their number, to a third of the scatter at 9.")
        cands.append(g)

    # 2. the validation test, with gauges, loads and predicted readings
    val = _validation_test(dom, o, u, v)
    if val:
        val["score"] = 1.0 if (short & {"comparator", "output comparison", "applicability"} or not v) else 0.2
        cands.append(val)

    # 3. a gauge that decides between the idealizations Wobbl solved
    disc = _discriminator(d, o)
    if disc:
        disc["score"] = 1.0 if ("model form" in short or not c) else 0.3
        cands.append(disc)

    chosen, covered = [], 0.0
    for t in sorted(cands, key=lambda t: -t["score"]):
        if t["kind"] in ("validation", "model form"):
            if t["score"] >= 1.0:
                chosen.append(t)
        elif measurable > 0 and covered < COVER * measurable:
            chosen.append(t)
            covered += t["share"]
    for k, t in enumerate(chosen, start=1):
        t["order"] = k
    res = {"kind": "test plan", "domain": dom, "wobbl": __version__,
           "created": _dt.datetime.now().strftime("%Y-%m-%d %H:%M"), "tests": chosen,
           "considered": [t["kind"] for t in cands], "replace": replaceable(dom, o, u, v, c),
           "rule": "the validation test whenever a validation factor is short of its goal, the discriminating gauge "
                   "whenever the model form is, and input measurements by first order Sobol index until they cover "
                   f"{100 * COVER:.0f}% of the variance measuring could remove"}
    res["headline"] = _headline(res)
    return res


def _headline(r: dict) -> str:
    tests = r["tests"]
    if not tests:
        return "The evidence already meets the proposed goals, so the plan adds no physical test."
    names = [TITLES.get(t["kind"], t["kind"]) for t in tests]
    lead = names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]
    can = [x["test"] for x in r["replace"] if x["verdict"] == "likely"]
    s = f"{len(tests)} test{'s' if len(tests) != 1 else ''} would most reduce the risk, in this order, {lead}."
    if can:
        s += f" The simulation evidence can likely stand in for {', '.join(can)}."
    return s


def _validation_test(dom: str, o: dict, u: dict | None, v: dict | None) -> dict | None:
    """The test that supplies the comparator. A run with a governing point and a strength limit gets gauges, load
    steps and predicted readings, a run with a hottest point gets thermocouples, and any other run gets a
    measurement of what decides it, at the design point, with the prediction and its band."""
    vals = o.get("values") or {}
    if o.get("governing_xyz") is not None and o.get("allowable_MPa"):
        probes = (u or {}).get("probes") or []
        allow, gov = o["allowable_MPa"], o["governing_MPa"]
        cap = min(1.0, 0.8 * allow / gov) if gov > 0 else 1.0
        steps = [s for s in (0.25, 0.5, 0.75, 1.0) if s <= cap + 1e-9]
        if cap < 1.0 and round(cap, 2) not in steps:
            steps.append(round(cap, 2))
        load = (u or {}).get("study_load", {})
        p0, unit = load.get("nominal", 1.0), load.get("unit", "x")
        if not probes:
            probes = [{"kind": "peak", "xyz": o["governing_xyz"], "direction": [],
                       "why": "the governing point, where the answer is decided"}]
        e = next((i for i in (u or {}).get("inputs", []) if i["name"] == "E"), None)
        rel = (e["sd"] / e["mean"]) if e else 0.0
        gauges = []
        for p in probes:
            g = {"kind": p["kind"], "xyz": p["xyz"], "direction": p["direction"], "why": p["why"]}
            if "strain" in p:
                g["quantity"] = "strain"
                g["predicted"] = [{"load": s * p0, "microstrain": p["strain"] * s * 1e6,
                                   "band_95": 2 * rel * abs(p["strain"]) * s * 1e6} for s in steps]
                lo, hi = p.get("measure_range") or (0, 0)
                if hi > 0 and (hi - lo) / hi > 0.1:
                    g["note"] = (f"the stress changes {100 * (hi - lo) / hi:.0f}% under a 3 mm grid here, so use a "
                                 "1 mm grid or read the gauge as an average over its length")
            elif "displacement" in p:
                g["quantity"] = "displacement"
                g["predicted"] = [{"load": s * p0, "mm": p["displacement"] * s,
                                   "band_95": 2 * rel * p["displacement"] * s} for s in steps]
            else:
                g["quantity"] = "strain"
                g["note"] = ("the run kept no fields, so run 'wobbl uncertainty' on a run with its work folder for the "
                             "gauge axis and the predicted readings")
            gauges.append(g)
        words = [f"{100 * s:.0f}% of the design load" if unit == "x" else f"{fmt(s * p0)} {unit}" for s in steps]
        loads = words[0] if len(words) == 1 else ", ".join(words[:-1]) + " and " + words[-1]
        why = ("It supplies the comparison with tests that V&V 20 needs, at the point the decision rests on and up to "
               "the design load.")
        if cap < 1.0:
            why += (f" The steps stop at {100 * cap:.0f}% of the design load, where the peak reaches 80% of the "
                    "strength limit, so the test does not damage the part.")
        return {"kind": "validation", "what": f"A static load test with strain gauges and a dial gauge, loaded in "
                f"steps of {loads}", "gauges": gauges, "steps": steps, "why": why,
                "addresses": ["comparator", "output comparison", "applicability"]}
    if vals.get("hot_xyz") is not None and "t_max_C" in vals:
        hot = vals.get("hot_xyz")
        how = "through the same run" if o.get("values", {}).get("time_over_limit_s") is not None else \
            "until it settles"
        return {"kind": "validation", "what": "Thermocouples at the hottest point and on the cooled faces, with the "
                f"part run at full power {how}",
                "gauges": [{"kind": "hottest point", "xyz": hot, "quantity": "temperature",
                            "predicted": [{"C": o["values"]["t_max_C"]}]}] if hot else [],
                "why": "It checks the film coefficients and the contact paths, the inputs heat models get wrong most.",
                "addresses": ["comparator", "output comparison", "applicability"]}
    return _measure_decision(dom, o, u)


def _measure_decision(dom: str, o: dict, u: dict | None) -> dict | None:
    from .domains import criteria, governing, scalar_values, unit_of

    vals = scalar_values(o)
    crit = criteria(o)
    key = (u or {}).get("decision") or ((governing(crit, vals) or {}).get("value")) or next(iter(vals), None)
    if key is None:
        return None
    keys = list(dict.fromkeys([key] + [c["value"] for c in crit]))[:4]
    gauges = []
    for k in keys:
        st = ((u or {}).get("outputs") or {}).get(k)
        mean = st["mean"] if st else vals.get(k)
        if mean is None:
            continue
        gauges.append({"kind": k.replace("_", " "), "quantity": k, "xyz": [],
                       "predicted": [{"value": mean, "band_95": 2 * st["sd"] if st else None, "unit": unit_of(k)}]})
    top = [x["input"] for x in ((u or {}).get("sensitivity") or {}).get("indices", []) if x["total"] >= 0.1][:2]
    what = (f"A measurement of {', '.join(k.replace('_', ' ') for k in keys)} on the real article at the design "
            "point")
    if top:
        what += f", and again with {' and '.join(top)} at the ends of their range, which tells the slope apart"
    return {"kind": "validation", "what": what, "gauges": gauges,
            "why": f"It supplies the comparison with tests V&V 20 needs, for the quantity this {dom} run is judged on.",
            "addresses": ["comparator", "output comparison", "applicability"]}


def _discriminator(d: Path, o: dict) -> dict | None:
    """A gauge where the two load attachment idealizations disagree most, at the other idealization's own peak,
    with what each of them predicts there."""
    from .probes import load_fields

    lp = next((c.get("detail", {}) for c in o.get("checks", []) if c["name"] == "load introduction"), {})
    if not lp or "alternative_xyz" not in lp:
        return None
    alt, xyz = lp["alternative_peak"], lp["alternative_xyz"]
    here = None
    f = load_fields(d)
    if f is not None and "measure" in f:
        here = float(f["measure"][int(np.argmin(np.linalg.norm(f["X"] - np.asarray(xyz), axis=1)))])
    why = (f"With a {lp['alternative']} attachment the peak sits here at {fmt(alt)} MPa"
           + (f", and with the {lp['primary']} one Wobbl predicts {fmt(here)} MPa at the same spot" if here else "")
           + ". A reading here tells which idealization the real assembly is closer to, a choice that moves the "
             "answer more than any input scatter.")
    return {"kind": "model form", "what": f"A strain gauge at ({', '.join(fmt(x) for x in xyz)}), where the "
            f"{lp['alternative']} attachment puts its peak", "gauges": [{"kind": "discriminating", "xyz": xyz}],
            "why": why, "addresses": ["model form"], "predicted_alternative_MPa": alt,
            "predicted_primary_MPa": here}


def replaceable(dom: str, o: dict, u: dict | None, v: dict | None, c: dict | None) -> list[dict]:
    """Which usual tests the simulation evidence can likely stand in for, and which it cannot, with why."""
    real = bool(v) and not v["tests"].get("synthetic")
    agree = bool(v) and v["summary"]["within"] == v["summary"]["readings"]
    pf = (u or {}).get("probability", {}).get("failure")
    safe = bool(pf) and pf["ci95"][1] < 1e-3 and all(a["probability"]["ci95"][1] < 1e-3
                                                     for a in (u or {}).get("alternatives", []))
    goals = bool(c) and c["met"] == len(c["factors"])
    why_not = []
    if not real:
        why_not.append("no real test has validated the model yet" if not v else "the readings are synthetic")
    elif not agree:
        why_not.append("some readings disagree with the model beyond the validation uncertainty")
    if not safe:
        why_not.append("the failure probability is not small enough to stand on the model alone" if pf else
                       "the failure probability has not been quantified")
    if c and not goals:
        why_not.append("credibility factors are short of their goals")
    likely = not why_not
    out = []
    if o.get("governing_xyz") is not None and o.get("allowable_MPa"):
        out.append({"test": "a static proof test at the design load", "verdict": "likely" if likely else "not yet",
                    "why": "the model is validated in this regime, verified, and the margin holds across the input "
                           "scatter and the idealizations" if likely else ", ".join(why_not)})
        out.append({"test": "strain surveys of later design variants",
                    "verdict": "likely" if real and agree else "not yet",
                    "why": "once one survey agrees, variants in the same family, materials and load range are inside "
                           "the validated regime" if real and agree else "no validated survey exists to carry over"})
        out.append({"test": "a test to failure", "verdict": "cannot",
                    "why": "Wobbl's strength model is linear elastic and stops at the first yield, so it says "
                           "nothing about how the part fails after that"})
        out.append({"test": "material coupons", "verdict": "cannot",
                    "why": "they measure the inputs the model rests on"})
        out.append({"test": "fatigue, impact and vibration tests", "verdict": "cannot",
                    "why": "this run is a static strength study and does not model them"})
    else:
        dec = ((u or {}).get("decision") or "answer").replace("_", " ")
        test = PROOF.get(dom, f"a test of the {dec} at the design point")
        out.append({"test": test, "verdict": "likely" if likely else "not yet",
                    "why": "the model is validated and verified there, and the margin holds across the input scatter"
                    if likely else ", ".join(why_not)})
        out.append({"test": "tests of later variants in the same family", "verdict": "likely" if real and agree
                    else "not yet", "why": "variants inside the validated range of inputs carry the evidence over"
                    if real and agree else "no validated test exists to carry over"})
        roles = sorted({i["role"] for i in (u or {}).get("inputs", []) if i["role"] not in ("numerical",)})
        out.append({"test": "measurements of the inputs themselves", "verdict": "cannot",
                    "why": "they measure what the model takes as given" + (f" (its {', '.join(roles)} inputs)"
                                                                             if roles else "")})
        first = next((a for a in o.get("assumptions", []) if len(a) < 200), "")
        out.append({"test": "anything outside the run's assumptions", "verdict": "cannot",
                    "why": "the model says nothing past them" + (f", starting with this one, {first.rstrip('.')}"
                                                                 if first else "")})
    return out


# the usual test a validated run could stand in for, by domain, when it has a familiar name
PROOF = {"thermal": "a steady soak at full power", "circuit": "a bench check of the operating point",
         "motion": "a flight or drop test at the design conditions", "flow": "a flow or tow test at the design point",
         "motor": "a static fire at the design point", "mechanism": "a run of the mechanism through its move"}
