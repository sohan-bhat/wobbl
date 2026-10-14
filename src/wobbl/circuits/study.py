"""Simulate any circuit a user brings and say whether it works and whether anything burns.

    simulate_circuit("board.kicad_sch" | "amp.cir" | SPICE text | "a 9 V battery and a red LED with 330 ohm"
                     | "led", question, params, out_dir) -> GenericOutcome

The input becomes a Circuit (netlist.py, kicad.py, describe.py; the template names are shortcuts to their
own checks in run.py), every part is bound to a model and its ratings (bind.py), and the analyses are
chosen from the circuit and the question: an operating point for a circuit that holds still, a power on
transient run until it settles for one that moves, AC response and loop stability where there is feedback
or a filter, and supply range and tolerance sweeps. Every part is then judged against its ratings
(stress.py), the run checks itself (checks.py, handcalc.py), fixes are solved again before they are
offered (fixes.py), and playback.json records the run for the site to animate.
"""
from __future__ import annotations

import re
import time
from pathlib import Path

from ..outcome_generic import GenericOutcome
from ..units import fmt, parse_quantity
from ..verify.checks import Check, confidence
from . import checks as ck
from .bind import bind
from .netlist import Circuit, NetlistError, join_parasitics, parse_spice
from .solve import ngspice, run_op, run_tran
from .stress import Stress, evaluate

DYNAMIC_WORDS = r"start.?up|power.?on|inrush|ripple|transient|waveform|blink|flash|oscillat|timing|period|" \
                r"frequency of|how fast|rise time|charge time|switch(?:es|ing)? off|turn.?off|spike|over time"
AC_WORDS = r"frequency response|bode|cut.?off|corner frequency|bandwidth|filter|phase margin|stab(?:le|ility)|" \
           r"\bgain\b|\bac\b|-3 ?db|\bdb\b|resonan"
TOL_WORDS = r"toleranc|worst.?case|monte.?carlo|spread|corner|every board|batch|production|as the battery|" \
            r"battery (?:drains|runs down|range)|supply range|over the range"


def _ambient(params: dict) -> float:
    a = params.get("ambient")
    if a in (None, ""):
        return 25.0
    try:
        return float(a)
    except (TypeError, ValueError):
        return parse_quantity(str(a), "temperature").value


def load(source: str, params: dict | None = None) -> Circuit:
    """A Circuit from a file path, SPICE or KiCad text, or a plain description."""
    from . import inputs
    return inputs.load(source, params or {})


def simulate_circuit(source: str, question: str = "", params: dict | None = None, out_dir: str | Path | None = None,
                     probes: list[str] | tuple = (), focus: str = "ratings", quick: bool = False,
                     log=None) -> GenericOutcome:
    """The general entry: any circuit, any question. focus is ratings (every part against its limits),
    operating-point, transient, ac or tolerance; each still reports every part's stress."""
    from .templates import TEMPLATES

    params = dict(params or {})
    log = log or (lambda stage, msg: None)
    if isinstance(source, str) and source.strip().lower() in TEMPLATES:
        from .run import run_circuit
        return run_circuit(source.strip().lower(), params, out_dir or Path("wobbl-out") / source, question)
    t0 = time.time()
    out = Path(out_dir or Path("wobbl-out") / "circuit")
    out.mkdir(parents=True, exist_ok=True)
    work = out / "work"
    c = load(source, params)
    bind(c, params)
    if params.get("parasitics"):
        join_parasitics(c, _parasitic_spec(params["parasitics"]))
    if not [e for e in c.elements if e.kind in ("supply", "signal", "battery", "isource", "charger") or
            e.letter in "vi"]:
        raise NetlistError("the circuit has no source to power it", "Add a supply, as V1 vcc 0 5, or name one in "
                                                                     "the description")
    ta = _ambient(params)
    q = (question or "").lower()
    log("solve", f"{len(c.elements)} parts on {len(c.nodes())} nets")
    runs: dict = {}
    dynamic, why = _dynamic(c, q, focus)
    if focus in ("operating-point",) or not dynamic:
        runs["op"] = run_op(c, work)
    if dynamic:
        step, stop = _user_tran(c)
        log("solve", "power on transient" + (f" ({why})" if why else ""))
        runs["tran"] = run_tran(c, work, step=step, stop=stop)
    main = runs.get("tran") or runs["op"]
    stress = evaluate(c, main, ta, [w for w in runs.values() if w is not main])
    checks: list[Check] = [ck.kcl(c, main), ck.power_balance(c, main, stress)]
    if not quick:
        checks.append(_convergence(c, main, stress, work, ta))
    from . import handcalc
    checks += handcalc.checks(c, main, stress)
    prov, extra = ck.provenance(c, stress)
    checks.append(prov)
    values: dict = {}
    fixes: list[dict] = []
    sections: list[str] = []
    if focus == "ac" or re.search(AC_WORDS, q) or _has_feedback(c):
        try:
            from . import ac
            res = ac.analyze(c, work, runs.get("op") or run_op(c, work, "op_ac"), q, probes)
            checks += res.get("checks", [])
            values["ac"] = res.get("values", {})
        except Exception as e:  # noqa: BLE001  the AC sweep must not lose the rest of the result
            checks.append(Check("AC response", "skip", f"not run: {e}"))
    if focus == "tolerance" or re.search(TOL_WORDS, q) or (focus == "ratings" and not quick):
        try:
            from . import tolerance
            tol = tolerance.analyze(c, work, ta, dynamic=dynamic, monte_carlo=focus == "tolerance" or
                                    bool(re.search(TOL_WORDS, q)), base=stress)
            checks += tol["checks"]
            values["tolerance"] = tol["values"]
            stress = tol.get("stress", stress)
        except Exception as e:  # noqa: BLE001
            checks.append(Check("tolerance", "skip", f"not run: {e}"))
    if not quick:
        try:
            from . import fixes as fx
            fixes = fx.suggest(c, stress, runs, work, ta)
        except Exception as e:  # noqa: BLE001
            checks.append(Check("fixes", "skip", f"could not try fixes: {e}"))
    verdict, head = verdict_of(stress, c)
    o = _outcome(c, source, question, verdict, head, stress, checks, extra, values, main, probes, fixes, runs,
                 t0, out, sections, ta)
    try:
        from .playback import write as write_playback
        write_playback(out, c, main, stress, o.verdict)
    except Exception as e:  # noqa: BLE001  the drawing data must not lose the result
        o.notes.append(f"playback.json not written: {e}")
    (out / "netlist.cir").write_text(c.spice())
    return o.write(out, sections)


def _parasitic_spec(v) -> dict:
    import json
    if isinstance(v, dict):
        return v
    p = Path(str(v))
    f = p / "parasitics.json" if p.is_dir() else p
    return json.loads(f.read_text())


def _dynamic(c: Circuit, q: str, focus: str) -> tuple[bool, str]:
    if focus == "transient":
        return True, "asked for"
    if focus == "operating-point":
        return False, ""
    for e in c.elements:
        if e.kind == "signal":
            return True, f"{e.ref} changes with time"
        if e.kind == "timer":
            return True, f"{e.ref} is a timer"
        if e.kind == "mcu":
            return True, f"{e.ref} switches its outputs"
        if e.letter == "s" or e.letter == "w":
            return True, f"{e.ref} switches"
    if c.analyses and any(a.lower().startswith(".tran") for a in c.analyses):
        return True, "the netlist asks for a transient"
    reactive = any(e.letter in "cl" and e.kind != "parasitic" for e in c.elements)
    if reactive and re.search(DYNAMIC_WORDS, q):
        return True, "the question asks how it changes in time"
    return False, ""


def _user_tran(c: Circuit) -> tuple[float | None, float | None]:
    from ..units import parse_spice_value
    for a in c.analyses:
        t = a.split()
        if t[0].lower() == ".tran" and len(t) >= 3:
            try:
                return parse_spice_value(t[1]), parse_spice_value(t[2])
            except Exception:  # noqa: BLE001
                return None, None
    return None, None


def _has_feedback(c: Circuit) -> bool:
    return any(e.kind == "opamp" for e in c.elements)


def _convergence(c: Circuit, w, stress: list[Stress], work: Path, ta: float) -> Check:
    a = ck.key_values(stress)
    try:
        if w.kind == "tran":
            w2 = run_tran(c, work, "tran_half", step=w.step / 2, stop=w.stop, tries=1)
            b = ck.key_values(evaluate(c, w2, ta))
            return ck.compare(a, b, "time step convergence", "with half the time step")
        w2 = run_op(c, work, "op_tight", extra=[".options reltol=1e-5 abstol=1e-14 vntol=1e-8"])
        b = ck.key_values(evaluate(c, w2, ta))
        return ck.compare(a, b, "tolerance convergence", "with solver tolerances 100 times tighter")
    except Exception as e:  # noqa: BLE001
        return Check("convergence", "warn", f"the refined run failed: {e}")


def verdict_of(stress: list[Stress], c: Circuit) -> tuple[str, str]:
    """The board's verdict and headline from every part's stress."""
    judged = [s for s in stress if s.kind not in ("supply", "signal", "isource", "source", "switch")
              or any(it.limit for it in s.items)]
    broken = [(s, it) for s in judged for it in s.items if it.status == "fails"]
    over = [(s, it) for s in judged if not s.generic for it in s.items if it.status == "over"]
    warn = [(s, it) for s in judged for it in s.items if it.status == "warn" or (s.generic and it.status == "over")]
    near = [(s, it) for s in judged for it in s.items if it.status == "near"]
    rated = [(s, s.worst) for s in judged if s.worst is not None]
    if over:
        s, it = max(over, key=lambda x: x[1].use or 0)
        others = sorted({x.ref for x, _ in over} - {s.ref})
        more = f" {', '.join(others)} also go over a rating." if others else ""
        return "fails", f"{s.ref} ({s.part}) is over its {it.what} rating: {it.text()}.{more}"
    if broken:
        s, it = broken[0]
        return "fails", f"{s.ref} ({s.part}) does not work as wired: {it.note or it.text()}."
    if warn or near:
        s, it = (warn or near)[0] if not near else max(near, key=lambda x: x[1].use or 0)
        if warn and (not near or warn[0][1].status == "warn"):
            s, it = warn[0]
        what = it.note or it.text()
        return "marginal", f"Nothing burns, but {s.ref} ({s.part}) needs a look: {what}."
    if rated:
        s, it = max(rated, key=lambda x: x[1].use or 0)
        return "ok", (f"Works and nothing is stressed: the hardest working part is {s.ref} ({s.part}) at "
                      f"{100 * (it.use or 0):.0f}% of its {it.what} rating.")
    return "ok", "Works; no part has a rating to judge it against."


def _outcome(c, source, question, verdict, head, stress, checks, extra, values, main, probes, fixes, runs, t0, out,
             sections, ta) -> GenericOutcome:
    from ..report.html import badge, esc, table

    lvl, reasons = confidence(checks, extra)
    rows = []
    for s in sorted(stress, key=lambda s: -(s.worst.use if s.worst and s.worst.use else 0)):
        w = s.worst
        rows.append([badge({"ok": "pass", "near": "warn", "over": "fail", "fails": "fail", "warn": "warn"}
                           .get(s.status, "info")), s.ref, s.part or s.kind,
                     w.text() if w else "; ".join(it.text() for it in s.items[:1]),
                     f"{fmt(s.p)} W", (f"{fmt(s.tj)} C" if s.tj is not None else "")])
    sections.insert(0, "<h2>Every part against its ratings</h2>" +
                    table(["status", "part", "type", "most used rating", "power", "junction"], rows))
    nodes = {c.label(n): main.mean(main.volt(n)) for n in c.nodes() if n != "0"}
    pv = {}
    for pr in probes or []:
        m = re.match(r"v\((\w+)\)", pr.lower())
        if m and m.group(1) in main.v:
            x = main.v[m.group(1)]
            pv[pr] = {"min": float(x.min()), "max": float(x.max()), "mean": main.mean(x), "final": float(x[-1])}
    parts = {s.ref: s.to_dict() for s in stress}
    flat = {}
    for s in stress:
        for it in s.items:
            if it.limit is not None and it.use is not None:
                flat[f"{s.ref}_{it.what.replace(' ', '_')}"] = it.value
    hard = max((s for s in stress if s.worst and s.worst.use is not None), key=lambda s: s.worst.use, default=None)
    metrics = []
    if hard:
        metrics.append(("hardest part", f"{hard.ref} at {100 * hard.worst.use:.0f}%"))
    supplied = sum(s.p for s in stress if s.kind in ("supply", "battery", "signal", "isource"))
    metrics.append(("supply power", f"{fmt(supplied)} W"))
    hot = max((s for s in stress if s.tj is not None), key=lambda s: s.tj, default=None)
    if hot:
        metrics.append(("hottest part", f"{hot.ref} at {fmt(hot.tj)} C"))
    metrics.append(("analysis", "transient" if main.kind == "tran" else "operating point"))
    assumptions = list(c.assumptions)
    assumptions += [f"{s.ref}: {s.source}." for s in stress if s.source and s.kind not in ("resistor", "capacitor")]
    gen_r = [s.ref for s in stress if s.kind == "resistor" and s.generic]
    if gen_r:
        assumptions.append(f"{', '.join(gen_r[:8])}{' and others' if len(gen_r) > 8 else ''}: no package given, so "
                           "rated as 1/4 W leaded resistors.")
    nocap = [s.ref for s in stress if s.kind in ("capacitor", "electrolytic") and any(it.status == "unknown"
                                                                                       for it in s.items)]
    if nocap:
        assumptions.append(f"{', '.join(nocap[:8])}: no voltage rating given, so their voltage is reported against "
                           "the next standard rating instead of judged.")
    assumptions.append(f"Ambient {fmt(ta)} C; temperatures are ambient plus average power times each part's "
                       "thermal resistance.")
    if main.kind == "tran":
        assumptions += [f"Transient: {n}." for n in main.notes]
    vals = {"parts": parts, "nodes": nodes, "probes": pv, **flat, **values,
            "analysis": main.kind, "netlist_source": c.source}
    o = GenericOutcome("circuit", source if len(str(source)) < 200 and "\n" not in str(source) else "circuit",
                       question, verdict, head, metrics, lvl, reasons, checks, assumptions, vals, {}, fixes,
                       solver=f"ngspice {ngspice().version()}", t_total=time.time() - t0)
    return o
