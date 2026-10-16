"""A circuit solved across the one input a designer would change for its question, for the interactive 3D bench.

choose() reads the question against the circuit. A load current it names ("a 2 A strip on J2") sweeps that load,
a part it names by word or value ("driving the cable", "a 47 ohm resistor") sweeps that part's value over a
couple of decades, and otherwise the main supply is turned from zero to half again its value. A load goes up to
three times what was asked, to show how much more the board takes. template_input()
does the same for a template run through the template's own parameters, so its netlist is rebuilt exactly as the
template would build it at each value.

solve() runs the circuit at every value the way the run itself was solved (an operating point or the same
transient), judges every part against its ratings there, and returns per value the current in every wire, part
and lead of the board plan, each part's share of its limit, and the readouts that answer the question. A part's
share of its limit is the most used of its ratings, so 1 is at the rating. An op amp's is 1 at the 45 degrees
of phase margin a loop needs, 2 with none left and 0 at twice that.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from ..units import parse_spice_value
from .netlist import Circuit, Element

PM_NEEDED = 45.0                 # degrees of phase margin a loop needs to settle without ringing
LABELS = {"resistor": "series resistor", "supply": "supply", "current": "load current", "vin": "input voltage",
          "vbus": "bus voltage", "r": "resistor", "v": "step voltage"}
UNITS = {"ohm": r"(?:ohms?|Ω|r\b|k\b|kohm)", "V": r"v\b", "A": r"m?a\b"}
TEMPLATE_INPUTS = {"led": (("resistor", "ohm"), ("supply", "V")), "buck": (("current", "A"), ("vin", "V")),
                   "mosfet": (("current", "A"), ("vbus", "V")), "rc": (("r", "ohm"), ("v", "V"))}


@dataclass
class Input:
    """The input the view's slider moves: its name, unit and values, the value the question asked about, and the
    circuit at any value."""
    label: str
    unit: str
    values: np.ndarray
    design: float
    make: Callable[[float], Circuit]
    ref: str = ""
    kind: str = "part"                  # supply, load or part
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"label": self.label, "unit": self.unit, "values": [float(f"{x:.6g}") for x in self.values],
                "design": float(f"{self.design:.6g}"), "ref": self.ref, "kind": self.kind}


def linear(design: float, start: int = 0) -> np.ndarray:
    """Zero (or start sixteenths) to one and a half times the design value, with the design value on the grid."""
    return design * np.arange(start, 25) / 16


def decades(design: float, below: float, above: float) -> np.ndarray:
    """25 values spread evenly on a log scale from design / 10^below to design * 10^above, the design among them."""
    step = (below + above) / 24
    k0 = round(below / step)
    v = design * 10 ** ((np.arange(25) - k0) * step)
    v = np.array([float(f"{x:.3g}") for x in v])
    v[k0] = design
    return v


def named(question: str, unit: str) -> list[float]:
    """The values with that unit the question names, as "2 A", "47 ohm" or "4.7k"."""
    out = []
    for m in re.finditer(r"(\d+(?:\.\d+)?)\s*(" + UNITS[unit] + ")", question, re.I):
        try:
            v = parse_spice_value(m.group(1) + ("k" if m.group(2).lower().startswith("k") else
                                                 "m" if unit == "A" and m.group(2).lower().startswith("m") else ""))
        except Exception:  # noqa: BLE001
            continue
        out.append(v)
    return out


def set_value(e: Element, v: float) -> None:
    """Give a resistor, capacitor or inductor a new value, keeping everything after it (IC=, tolerances)."""
    for k, t in enumerate(e.rest):
        if "=" in t:
            continue
        try:
            parse_spice_value(t)
        except Exception:  # noqa: BLE001
            continue
        e.rest[k] = f"{v:.6g}"
        return
    e.rest = [f"{v:.6g}", *e.rest]


def _main_supply(c: Circuit) -> Element | None:
    from .bind import dc_value
    from .board import source_role
    return next((e for e in c.elements if e.letter == "v" and "0" in e.nodes and e.nodes[0] != e.nodes[1]
                 and source_role(e)[0] == "supply" and dc_value(e)), None)


def choose(c: Circuit, question: str = "") -> Input | None:
    """The input for any circuit, read from its question: a load current it names, a part it names, or else the
    main supply."""
    from .bind import dc_value
    from .tolerance import _set_supply
    q = (question or "").lower()
    amps = named(q, "A")
    loads = [e for e in c.elements if e.kind == "load" and e.letter == "r" and e.value]
    if amps and loads:
        e, i0 = loads[0], amps[0]
        r0 = float(e.value)

        def load(x, e=e, r0=r0, i0=i0):
            cc = c.copy()
            set_value(cc.element(e.name), r0 * i0 / max(x, 1e-12))
            return cc
        return Input(f"{e.ref} current", "A", i0 * np.arange(1, 25) / 8, i0, load, e.name, "load",
                     [f"{e.ref} is set to draw each current at its nominal voltage"])
    words = set(re.findall(r"[a-z]{3,}", q))
    vals = [(u, v) for u in ("ohm",) for v in named(q, u)]
    for e in c.elements:
        if e.letter not in "rcl" or not e.value or e.kind == "load":
            continue
        tag = re.sub(r"^[rcl]", "", e.name.lower())
        by_word = len(tag) >= 3 and (tag in words or any(w.startswith(tag) for w in words))
        by_value = e.letter == "r" and any(abs(v - e.value) <= 0.01 * e.value for _, v in vals)
        if by_word or by_value:
            v0 = float(e.value)
            below, above = (3.0, 1.0) if e.letter == "c" else (1.0, 1.0)

            def part(x, e=e):
                cc = c.copy()
                set_value(cc.element(e.name), x)
                return cc
            unit = {"r": "ohm", "c": "F", "l": "H"}[e.letter]
            what = {"r": "resistance", "c": "capacitance", "l": "inductance"}[e.letter]
            return Input(f"{e.ref} {what}", unit, decades(v0, below, above), v0, part, e.name, "part")
    src = _main_supply(c)
    if src is None:
        return None
    v0 = float(dc_value(src))

    def supply(x, src=src):
        cc = c.copy()
        _set_supply(cc, src, x)
        return cc
    return Input(f"{src.ref} supply", "V", linear(v0), v0, supply, src.name, "supply")


def template_input(name: str, params: dict, question: str, build: Callable[[dict], Circuit]) -> Input | None:
    """The input for a template run: the first of its parameters the question names by unit, rebuilt by the
    template itself at each value. build turns parameters into the solved circuit."""
    opts = TEMPLATE_INPUTS.get(name)
    if not opts:
        return None
    q = (question or "").lower()
    key, unit = next(((k, u) for k, u in opts if named(q, u) and params.get(k)), opts[0])
    v0 = params.get(key)
    if not isinstance(v0, (int, float)) or v0 <= 0:
        return None
    vals = decades(v0, 0.5, 1.0) if unit == "ohm" else linear(v0, 2 if unit == "A" else 0)
    return Input(LABELS.get(key, key), unit, vals, float(v0), lambda x: build({**params, key: float(x)}),
                 key, "load" if unit == "A" else "supply" if unit == "V" else "part")


def solve(inp: Input, w, work: Path, ambient: float = 25.0, *, rerun: Callable | None = None,
          corners: bool = False, measure: Callable | None = None, question: str = "") -> dict:
    """The circuit at every input value: board plan currents, each part's share of its limit and the readouts.
    rerun(circuit, tag) solves one circuit the way the run did; measure(x, circuit, wave) adds a template's own
    readouts and limits as {"readouts": [(name, value, unit, limit)], "heat": {part: share}}."""
    from .board import plan
    from .pictures import wave_data
    from .solve import materialize
    from .stress import evaluate
    from . import tolerance as tol

    def again(cc, tag):
        return rerun(cc, tag) if rerun is not None else tol._solve(cc, w, work, tag)

    if getattr(w, "kind", "") == "tran" and len(inp.values) > 13:     # a transient costs more: every other value
        at = int(np.argmin(np.abs(inp.values - inp.design)))
        inp.values = inp.values[sorted(set(range(at % 2, len(inp.values), 2)) | {at})]
    ac_on = False
    rows, plans, heats, names = [], [], [], {}
    lo, mid = inp.make(float(inp.values[0])), inp.make(float(inp.design))
    changed = [e.name for e in mid.elements if (x := lo.element(e.name)) is not None and
               (x.value != e.value if e.letter in "rcl" else x.rest != e.rest)]
    for k, x in enumerate(inp.values):
        cc = inp.make(float(x))
        wx = again(cc, f"sw{k}")
        st = evaluate(cc, wx, ambient)
        if corners and inp.kind != "supply":
            sups, ends = tol._supplies(cc), []
            for end in (0, 1):
                c2 = cc.copy()
                for e in sups:
                    tol._set_supply(c2, e, tol.supply_range(e)[end])
                ends.append(("", evaluate(c2, again(c2, f"sw{k}c{end}"), ambient)))
            st = tol.worst_of(st, ends) if sups else st
        pm = {}
        if any(e.kind == "opamp" for e in cc.elements):
            try:
                from . import ac
                from .solve import run_op
                op = wx if wx.kind == "op" else run_op(cc, work, f"sw{k}op")
                res = ac.analyze(cc, work, op, question, ())
                pm = {r: s["phase_margin_deg"] for r, s in (res.get("values", {}).get("stability") or {}).items()
                      if s.get("phase_margin_deg") is not None}
                ac_on = ac_on or bool(pm)
            except Exception:  # noqa: BLE001  the loop only adds its margin
                pm = {}
        extra = measure(float(x), cc, wx) if measure else {}
        m = materialize(cc, wx.kind)
        pl = plan(m, wave_data(m, wx))
        plans.append(pl)
        refs = {e.ref: e.name for e in cc.elements}
        heat = {}
        for s in st:
            it = s.worst
            if it is not None and it.use is not None:
                heat[refs.get(s.ref, s.ref)] = float(it.use)
        for r, v in pm.items():
            nm = refs.get(r, r)
            heat[nm] = max(heat.get(nm, 0.0), max(0.0, 2.0 - v / PM_NEEDED))
        heat.update(extra.get("heat", {}))
        heats.append(heat)
        hard = max(((s.worst.use, s) for s in st if s.worst is not None and s.worst.use is not None),
                   key=lambda z: z[0], default=(None, None))
        names.update({p["name"]: p for p in pl["parts"]})
        row = {"readouts": list(extra.get("readouts", [])), "hard": hard[0],
               "hard_ref": hard[1].ref if hard[1] is not None else "", "pm": pm,
               "supply": _supply_current(pl, st, cc)}
        rows.append(row)

    at = int(np.argmin(np.abs(inp.values - inp.design)))
    base = plans[at]
    n = len(plans)

    def series(get):
        return [[round(float(get(p)[j].get("current") or 0.0), 9) for p in plans] for j in range(len(get(base)))]

    out = {"input": inp.to_dict(), "plan": base, "values": inp.to_dict()["values"],
           "source": inp.ref, "unit": inp.unit, "label": inp.label,
           "parts": series(lambda p: p["parts"]), "wires": series(lambda p: p["wires"]),
           "leads": [[[round(float(p["instruments"][i]["leads"][j].get("current") or 0.0), 9) for p in plans]
                      for j in range(len(ins.get("leads", [])))] for i, ins in enumerate(base["instruments"])],
           "heat": [[round(h[p["name"]], 4) if p["name"] in h else None for h in heats] for p in base["parts"]],
           "notes": inp.notes, "changed": changed}
    for j, p in enumerate(base["parts"]):
        col = out["heat"][j]
        if all(v is None for v in col):
            out["heat"][j] = None
        else:
            out["heat"][j] = [0.0 if v is None else v for v in col]
            p["heat"] = col[at]
    out["readouts"] = _readouts(rows, n, ac_on)
    return out


def _supply_current(pl: dict, stress=(), c: Circuit | None = None) -> float | None:
    """What the main supply delivers: its own judged current over the settled window, else its lead's current."""
    src = _main_supply(c) if c is not None else None
    if src is not None:
        for s in stress:
            it = next((x for x in s.items if x.what == "current drawn"), None)
            if s.ref == src.ref and it is not None and it.value:
                return abs(float(it.value))
    sup = next((i for i in pl.get("instruments", []) if i.get("kind") == "supply" and i.get("leads")), None)
    return abs(float(sup["leads"][0].get("current") or 0.0)) if sup else None


def _readouts(rows: list[dict], n: int, ac_on: bool) -> list[dict]:
    """The readouts per value, the one that answers the question first: a template's own, else the loop's phase
    margin, else the hardest working part's share of its rating."""
    out = []
    names = [r[0] for r in rows[0]["readouts"]]
    for k, name in enumerate(names):
        vals = [r["readouts"][k][1] for r in rows]
        unit, lim = rows[0]["readouts"][k][2], rows[0]["readouts"][k][3]
        out.append({"name": name, "unit": unit, "values": vals, "limit": lim})
    if ac_on:
        refs = sorted({r for row in rows for r in row["pm"]})
        for ref in refs[:1]:
            out.append({"name": f"{ref} phase margin", "unit": "deg",
                        "values": [row["pm"].get(ref, float("nan")) for row in rows], "limit": None})
    if not names and any(r["hard"] is not None for r in rows):
        out.append({"name": "hardest part", "unit": "% of rating",
                    "values": [100 * (r["hard"] or 0.0) for r in rows], "limit": 100,
                    "parts": [r["hard_ref"] for r in rows]})
    if any(r["supply"] is not None for r in rows):
        out.append({"name": "supply current", "unit": "mA", "values": [1000 * (r["supply"] or 0.0) for r in rows],
                    "limit": None})
    return out


def template_measure(name: str, params: dict, key: str, ambient: float = 25.0) -> Callable | None:
    """A template's own answer at each value, from the instrumented run: the LED's current against its rating,
    the buck's output ripple against its limit, the MOSFET's junction temperature against its maximum. The part
    that carries the answer is shaded by it."""
    def at(x):
        return {**params, key: x}

    if name == "led":
        def led(x, c, w):
            p = at(x)
            i = abs(float(np.ravel(w.i["D1"][0])[-1]))
            imax = float(p.get("imax") or 0.03)
            return {"readouts": [("LED current", 1000 * i, "mA", 1000 * imax)], "heat": {"D1": i / imax}}
        return led
    if name == "buck":
        def buck(x, c, w):
            p = at(x)
            f = float(p.get("frequency") or 5e5)
            m = w.t >= w.t[-1] - 20 / f
            vo, il = w.v["out"][m], np.abs(w.i["L1"][0][m])
            spec = float(p.get("max_ripple") or 0) or 0.01 * float(p.get("vout") or 5)
            rip = float(np.ptp(vo))
            return {"readouts": [("output ripple", 1000 * rip, "mV", 1000 * spec), ("output", float(np.mean(vo)),
                                 "V", None), ("inductor peak", float(il.max()), "A", None)],
                    "heat": {"C1": rip / spec}}
        return buck
    if name == "mosfet":
        from .run import junction

        def mosfet(x, c, w):
            p = at(x)
            f = float(p.get("frequency") or 0)
            m = w.t >= w.t[-1] - (10 / f if f > 0 else (w.t[-1] - w.t[0]) / 2)
            pc = abs(float(np.mean(w.v["d"][m] * w.i["VS"][0][m])))
            tj, ptot, psw, _ = junction(p, pc)
            ta, tjmax = float(p.get("ambient", 25)), float(p.get("tj_max", 150))
            return {"readouts": [("junction", tj, "C", tjmax), ("loss", ptot, "W", None)],
                    "heat": {"S1": (tj - ta) / max(tjmax - ta, 1e-9)}}
        return mosfet
    return None
