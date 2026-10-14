"""Every part across the supply range and the parts' tolerances.

The supply range comes from what powers the circuit: a battery from full to empty (6 to 9.6 V for a 9 V
battery, 3.0 to 4.2 V for a Li-ion cell), a USB port from 4.75 to 5.25 V, and a plain source 10% either way,
which is a guess the report names. The circuit is solved at both ends with every part nominal, and each part's
stress is the worse of nominal and the two corners, so a fresh battery that pushes an LED past its rating
counts against the board.

Monte Carlo builds vary every resistor, capacitor and inductor inside its tolerance (normal, three sigma at
the tolerance), each MOSFET's threshold across its datasheet range, each diode's and LED's forward drop by
50 mV and each transistor's gain from 0.6 to 1.5 times typical, with the supply anywhere in its range. For
an operating point all builds go into one netlist as separate copies and solve together; for a transient a
few builds run one after another. Seeds are fixed, so the same circuit gives the same builds.
"""
from __future__ import annotations

import math
import re
from dataclasses import replace
from pathlib import Path

import numpy as np

from ..units import fmt
from ..verify.checks import Check
from .bind import dc_value
from .netlist import Circuit, Element
from .solve import Wave, run, run_op, run_tran, materialize
from .stress import Item, Stress, evaluate


def supply_range(e: Element) -> tuple[float, float, bool]:
    """Low and high ends of a supply and whether the range is known (a battery or USB) or a 10% guess."""
    prm = e.part.get("params", {}) if e.part else {}
    if prm.get("range"):
        lo, hi = prm["range"]
        return float(lo), float(hi), True
    v = dc_value(e) if e.letter == "v" else None
    if v is None:
        return 0.0, 0.0, False
    return 0.9 * v, 1.1 * v, False


def _supplies(c: Circuit) -> list[Element]:
    return [e for e in c.elements if (e.kind in ("supply", "battery")) and e.nodes and not e.meta.get("cell")
            and "cell" != e.meta.get("ref") and "transformer" != e.meta.get("ref")]


def _set_supply(c: Circuit, e: Element, v: float) -> None:
    me = c.element(e.name)
    if me.letter == "v":
        me.rest = [f"{v:.6g}"]
    else:                                  # a battery subcircuit: its own copy with the new voltage
        sub = c.subckts[me.model.lower()]
        name = f"{sub.name}_{int(v * 1000)}"
        from .netlist import Subckt
        c.subckts[name.lower()] = Subckt(name, sub.ports, re.sub(r"^(V1 pi n )\S+", lambda m: f"{m.group(1)}{v:.6g}",
                                                                 sub.text.replace(sub.name, name, 1), flags=re.M))
        me.rest = [name]


def _solve(c: Circuit, main: Wave, work: Path, tag: str) -> Wave:
    if main.kind == "tran":
        return run_tran(c, work, tag, step=main.step, stop=main.stop, tries=1)
    return run_op(c, work, tag)


def worst_of(base: list[Stress], others: list[tuple[str, list[Stress]]]) -> list[Stress]:
    """Each part with each of its ratings at the worst of the runs given, saying which run that was."""
    out = []
    for s in base:
        items = []
        for it in s.items:
            best, where = it, ""
            for label, st in others:
                o = next((x for x in st if x.ref == s.ref), None)
                oi = next((x for x in o.items if x.what == it.what), None) if o else None
                if oi is not None and oi.use is not None and (best.use is None or oi.use > best.use):
                    best, where = oi, label
            if where:
                best = replace(best, note=(best.note + "; " if best.note else "") + where)
            items.append(best)
        out.append(replace(s, items=items))
    return out


def analyze(c: Circuit, work: Path, ta: float, dynamic: bool = False, monte_carlo: bool = False,
            base: list[Stress] | None = None, main: Wave | None = None, samples: int | None = None) -> dict:
    checks, values = [], {}
    sups = _supplies(c)
    stress = list(base or [])
    if sups and main is not None:
        corners = []
        known = True
        for end in (0, 1):
            cc = c.copy()
            text = []
            for e in sups:
                lo, hi, k = supply_range(e)
                known = known and k
                v = (lo, hi)[end]
                _set_supply(cc, e, v)
                text.append(f"{e.ref} at {fmt(v)} V")
            w = _solve(cc, main, work, f"corner{end}")
            st = evaluate(cc, w, ta)
            corners.append(("with " + ", ".join(text), st))
        merged = worst_of(stress, corners)
        if not known:      # a 10% guess at a plain source's range: going over there is a warning, not a failure
            for s in merged:
                for it in s.items:
                    if it.status == "over" and any(lbl in it.note for lbl, _ in corners):
                        it.status = "warn"
                        it.note += " (the supply range is a 10% guess)"
        stress = merged
        rng = "; ".join(f"{e.ref} {fmt(supply_range(e)[0])} to {fmt(supply_range(e)[1])} V" +
                        ("" if supply_range(e)[2] else " (10% either way, a guess)") for e in sups)
        moved = sorted(((s.worst.use or 0, s.ref) for s in stress if s.worst is not None), reverse=True)[:3]
        checks.append(Check("supply range", "info", f"solved at both ends of the supply ({rng}); the hardest parts "
                            "there: " + ", ".join(f"{r} at {100 * u:.0f}%" for u, r in moved),
                            {"range": rng}))
        values["supply_range"] = rng
    if monte_carlo:
        n = samples or (24 if not dynamic else 4)
        try:
            res = monte(c, work, ta, n, main)
            values["monte_carlo"] = res["values"]
            checks.append(res["check"])
            for s in stress:
                frac = res["values"]["over_fraction"].get(s.ref, 0.0)
                if frac > 0:
                    s.items.append(Item("tolerance", frac, None, "", None, "warn",
                                        f"over a rating in {round(frac * n)} of {n} builds with every part inside "
                                        "its tolerance"))
        except Exception as e:  # noqa: BLE001
            checks.append(Check("monte carlo", "skip", f"not run: {e}"))
    return {"checks": checks, "values": values, "stress": stress}


def _perturb(c: Circuit, rng: np.random.Generator, k: int) -> Circuit:
    cc = c.copy()
    for e in cc.elements:
        tol = (e.part.get("ratings") or {}).get("tolerance") if e.part else None
        if e.letter in "rcl" and e.value and e.kind not in ("parasitic", "load"):
            t = tol if tol is not None else {"r": 0.05, "c": 0.2, "l": 0.2}[e.letter]
            e.set_value(e.value * (1 + t * float(np.clip(rng.normal(0, 1 / 3), -1, 1))))
        elif e.kind in ("supply", "battery") and e.nodes:
            lo, hi, _ = supply_range(e)
            if hi > lo:
                _set_supply(cc, e, float(rng.uniform(lo, hi)))
        elif e.letter in "dmq" and e.model.lower() in cc.models:
            text = cc.models[e.model.lower()]
            prm = e.part.get("params", {})
            if e.letter == "m" and prm.get("vth_range"):
                lo, hi = prm["vth_range"]
                v = float(rng.uniform(lo, hi))
                text = re.sub(r"Vto=-?[\d.e+-]+", f"Vto={-v if e.kind == 'pmos' else v:.4g}", text)
            elif e.letter == "d":
                n_ = float((re.search(r"\bN=([\d.]+)", text) or [0, "1.8"])[1])
                text = re.sub(r"\bIS=([\d.e+-]+)", lambda m: f"IS={float(m.group(1)) * math.exp(rng.uniform(-0.05, 0.05) / (n_ * 0.02587)):.6e}",
                              text, flags=re.I)
            elif e.letter == "q":
                text = re.sub(r"\bBF=([\d.]+)", lambda m: f"BF={float(m.group(1)) * rng.uniform(0.6, 1.5):.4g}", text)
            name = f"{e.model}_mc{k}"
            cc.models[name.lower()] = re.sub(r"^(\.model\s+)\S+", lambda m: m.group(1) + name, text)
            e.rest[e.rest.index(e.model)] = name
    return cc


def _merge(copies: list[Circuit]) -> Circuit:
    """Every build as its own copy in one netlist: names and nets get the build's number, ground stays shared."""
    out = Circuit(title="monte carlo builds", models={}, subckts={})
    for k, cc in enumerate(copies):
        mc = materialize(cc, "op")
        out.models.update(mc.models)
        out.subckts.update(mc.subckts)
        if k == 0:
            out.directives = list(mc.directives)
        for e in mc.elements:
            ne = Element(f"{e.name}_b{k}", e.letter, [n if n == "0" else f"{n}_b{k}" for n in e.nodes], list(e.rest),
                         list(e.pins), e.kind, e.part, dict(e.meta))
            out.elements.append(ne)
    return out


def monte(c: Circuit, work: Path, ta: float, n: int, main: Wave | None) -> dict:
    rng = np.random.default_rng(7)
    copies = [_perturb(c, rng, k) for k in range(n)]
    stresses = []
    if main is None or main.kind == "op":
        big = _merge(copies)
        w = run(big, ["op"], work, "monte")[0]
        for k, cc in enumerate(copies):
            sub = Wave("op", w.t, {nn[:-len(f"_b{k}")]: v for nn, v in w.v.items() if nn.endswith(f"_b{k}")},
                       {nm[:-len(f"_b{k}")]: v for nm, v in w.i.items() if nm.endswith(f"_b{k}")})
            stresses.append(evaluate(materialize(cc, "op"), sub, ta))
    else:
        for k, cc in enumerate(copies):
            stresses.append(evaluate(cc, run_tran(cc, work, f"mc{k}", step=main.step, stop=main.stop, tries=1), ta))
    over: dict[str, float] = {}
    spread: dict[str, list[float]] = {}
    for st in stresses:
        for s in st:
            if s.worst is not None and s.worst.use is not None:
                spread.setdefault(s.ref, []).append(s.worst.use)
            if any(it.status in ("over", "fails") for it in s.items):
                over[s.ref] = over.get(s.ref, 0) + 1 / n
    top = sorted(((max(v), r) for r, v in spread.items()), reverse=True)[:3]
    bad = {r: f for r, f in over.items() if f > 0}
    text = (f"{n} builds with parts inside their tolerances: " + ", ".join(
        f"{r} up to {100 * u:.0f}% of a rating" for u, r in top) +
            (f"; over a rating in some builds: " + ", ".join(f"{r} ({round(f * n)} of {n})" for r, f in bad.items())
             if bad else "; no build goes over a rating"))
    return {"check": Check("monte carlo", "info", text, {"samples": n}),
            "values": {"samples": n, "over_fraction": over,
                       "use": {r: {"min": min(v), "median": float(np.median(v)), "max": max(v)}
                               for r, v in spread.items()}}}
