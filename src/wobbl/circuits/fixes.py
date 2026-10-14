"""Fixes for the parts that fail or run near a limit, each solved again before it is offered.

A fix changes the circuit the way a designer would and with real parts: a series resistor raised to the E24
value that brings an LED, a pin or a zener back to about half its rating, a resistor or capacitor of the next
standard power or voltage rating, a diode, MOSFET, transistor or regulator from the parts library that meets
what the run asked of the old one (a logic level MOSFET for a 3.3 V gate), a flyback diode across a relay or
motor coil, a regulator in front of a microcontroller wired to too much voltage, an isolation resistor between
an op amp and a capacitive load, or a heatsink sized from the part's own power. The changed circuit is solved
with the same analysis, every part is judged again, and the fix is offered only when the part it was for is
within its ratings afterwards and nothing else went over.
"""
from __future__ import annotations

import math
from pathlib import Path

from ..units import fmt
from .netlist import Circuit, Element
from .parts import CAP_VOLTS, DIODES, MOSFETS, POWERS, e_series, lookup
from .solve import run_op, run_tran
from .stress import Stress, evaluate

PKG_FOR_POWER = {0.0625: "0402", 0.1: "0603", 0.125: "0805", 0.25: "1206 or a 1/4 W leaded part",
                 0.5: "2010 or a 1/2 W leaded part", 1.0: "2512 or a 1 W leaded part", 2.0: "a 2 W leaded part",
                 3.0: "a 3 W wirewound", 5.0: "a 5 W wirewound", 10.0: "a 10 W wirewound"}


def _bad(stress: list[Stress]) -> list[Stress]:
    order = {"fails": 0, "over": 1, "warn": 2, "near": 3}
    return sorted([s for s in stress if s.status in order], key=lambda s: (order[s.status], -(s.worst.use or 0)
                                                                          if s.worst else 0))


def suggest(c: Circuit, stress: list[Stress], runs: dict, work: Path, ta: float, limit: int = 4) -> list[dict]:
    main = runs.get("tran") or runs.get("op")
    fixes, tried = [], 0

    def resolve(c2: Circuit, loop: bool) -> list[Stress]:
        nonlocal tried
        tried += 1
        if main.kind == "tran":
            w = run_tran(c2, work, f"fix{tried}", step=main.step, stop=main.stop, tries=1)
        else:
            w = run_op(c2, work, f"fix{tried}")
        st = evaluate(c2, w, ta)
        if loop:                  # a stability fix is judged on the loop gain, swept again
            from . import ac
            ac.attach(st, ac.analyze(c2, work, run_op(c2, work, f"fix{tried}op"), "", ()))
        return st

    before_over = {s.ref for s in stress if s.status in ("over", "fails")}
    for s in _bad(stress)[:6]:
        if len(fixes) >= limit or tried > 10:
            break
        for title, what, apply, solve in _candidates(c, s, stress, main, ta):
            c2 = c.copy()
            try:
                note = apply(c2)
            except (ValueError, KeyError, IndexError, StopIteration):
                continue
            if note is None:
                continue
            loop = any(it.what == "phase margin" for it in s.items)
            after = resolve(c2, loop) if solve else _rejudge(c2, s, main, ta)
            mine = next((x for x in after if x.ref == s.ref or x.ref == note.get("ref")), None)
            worse = {x.ref for x in after if x.status in ("over", "fails")} - before_over - {s.ref}
            if mine is None or mine.status in ("over", "fails") or worse:
                continue
            bad = [it.what for it in s.items if it.status in ("over", "fails", "warn", "near")]
            shown = [it for it in mine.items if it.what in bad] or ([mine.worst] if mine.worst else [])
            w = mine.worst
            said = " and ".join((it.text() if it.limit is not None else (it.note or it.text())) for it in shown[:2])
            done = (f" Solved again, {mine.ref} now has {said}." if said and solve else
                    f" Checked again, {mine.ref} now has {said}." if said else "")
            fixes.append({"kind": what, "title": title, "summary": note["why"] + "." + done,
                          "auto": False, "part": s.ref, "resolved": bool(solve),
                          "after": {"status": mine.status, "use": w.use if w else None}})
            break
    return fixes


def _rejudge(c2: Circuit, s: Stress, main, ta: float) -> list[Stress]:
    """A fix that only changes a rating needs no new solve: the same run judged against the new rating."""
    return evaluate(c2, main, ta)


def _series_resistor(c: Circuit, e: Element) -> Element | None:
    """The resistor in series with a part: alone with it on one of its nets."""
    nets = c.nets()
    for n in e.nodes:
        on = [x for x, _ in nets.get(n, []) if x is not e]
        if n != "0" and len(on) == 1 and on[0].letter == "r" and on[0].value:
            return on[0]
    for n in e.nodes:
        rs = [x for x, _ in nets.get(n, []) if x is not e and x.letter == "r" and x.value and x.kind == "resistor"]
        if n != "0" and len(rs) == 1:
            return rs[0]
    return None


def _candidates(c: Circuit, s: Stress, stress: list[Stress], main, ta: float = 25.0):
    e = c.element(s.ref) or next((x for x in c.elements if x.ref == s.ref), None)
    if e is None:
        return []
    w = s.worst
    items = {it.what: it for it in s.items}
    out = []
    # a current that is too high, held by a series resistor: raise it
    if s.kind in ("led", "diode", "zener", "schottky", "opto") or (s.kind == "mcu" and any(
            k.startswith("pin ") and it.status in ("over", "near") for k, it in items.items())):
        target_use = 0.5
        r = _series_resistor(c, e) if s.kind != "mcu" else None
        if s.kind == "mcu":
            pin = next(k for k, it in items.items() if k.startswith("pin ") and it.status in ("over", "near"))
            node = e.nodes[e.pins.index(pin.split()[1])]
            r = next((x for x, _ in c.nets().get(node, []) if x.letter == "r" and x.value), None)
            it = items[pin]
        else:
            it = items.get("forward current") or items.get("power") or items.get("LED current") or w
        if r is not None and it is not None and it.use:
            k = max(it.use / target_use, 1.05) if s.kind != "zener" else max(math.sqrt(it.use / target_use), 1.1)
            new = e_series(r.value * k)

            def apply(c2, r=r, new=new, it=it):
                c2.element(r.name).set_value(new)
                return {"why": f"Change {r.ref} from {fmt(r.value)} to {fmt(new)} ohm (E24) to bring {s.ref}'s "
                               f"{it.what} down to about half its rating"}
            out.append((f"Use {fmt(new)} ohm for {r.ref}", "resistor", apply, True))
    # a resistor running hot: the next standard power rating with margin at this ambient
    if s.kind == "resistor" and "power" in items and items["power"].use and items["power"].use >= 0.8:
        p = abs(items["power"].value)
        need = next((x for x in POWERS if x >= 2 * p), None)
        if need:
            def apply(c2, need=need):
                c2.element(e.name).part["ratings"]["power_W"] = need
                c2.element(e.name).part["generic"] = False
                return {"why": f"Fit a {fmt(need)} W resistor ({PKG_FOR_POWER.get(need, '')}) for {s.ref}, which "
                               f"dissipates {fmt(p)} W"}
            out.append((f"Use a {fmt(need)} W part for {s.ref}", "resistor rating", apply, False))
    # a capacitor near or over its voltage: the next standard rating
    if s.kind in ("capacitor", "electrolytic", "tantalum") and "voltage" in items:
        it = items["voltage"]
        k = 1.25 if s.kind != "capacitor" else 1.5
        need = next((x for x in CAP_VOLTS if x >= k * abs(it.value)), None)
        if need and (it.status in ("over", "near")):
            def apply(c2, need=need, it=it):
                c2.element(e.name).part["ratings"]["voltage_V"] = need
                return {"why": f"Use a {fmt(need)} V capacitor for {s.ref}, which sees {fmt(abs(it.value))} V"}
            out.append((f"Use a {fmt(need)} V part for {s.ref}", "capacitor rating", apply, False))
    # a diode past its current, surge or reverse voltage: a library diode that meets all three
    if s.kind in ("diode", "schottky") and e.letter == "d":
        ia = abs(items["forward current"].value) if "forward current" in items else 0
        vr = abs(items["reverse voltage"].value) if "reverse voltage" in items else 0
        pk = abs(items["peak current"].value) if "peak current" in items else 0
        for name in ("1N4148", "1N5819", "1N4001", "1N4004", "1N4007", "SS34", "1N5408", "SS54"):
            d = lookup(name)
            if d.ratings["if_A"] >= 1.5 * ia and d.ratings["vr_V"] >= 1.25 * vr and d.ratings["ifsm_A"] >= pk and \
                    d.key != s.part:
                out.append((f"Use a {name} for {s.ref}", "part", _swap(e, d, f"Use a {name} for {s.ref}: {d.source}"),
                            True))
                break
    # a MOSFET that is hot, over its current or only partly on: a better library part, then a heatsink
    if s.kind in ("nmos", "pmos"):
        out += _mosfet_swap(c, e, s, items)
    if s.kind in ("npn", "pnp", "nmos", "pmos") and "collector voltage" in items or "drain voltage" in items:
        it = items.get("collector voltage") or items.get("drain voltage")
        if it and it.status in ("over", "near"):
            fb = _flyback(c, e)
            if fb:
                out.append(fb)
    if s.kind in ("npn", "pnp") and items.get("temperature") and items["temperature"].status in ("over", "near"):
        ic = items["collector current"].value if "collector current" in items else 0
        vce = items["collector voltage"].value if "collector voltage" in items else 0
        for name in ("BC337", "BD139", "TIP31C", "TIP41C") if s.kind == "npn" else ("BC327", "BD140", "TIP32C",
                                                                                         "TIP42C"):
            b = lookup(name)
            if b.ratings["ic_A"] >= 1.5 * ic and b.ratings["vceo_V"] >= 1.25 * vce and b.key != s.part:
                out.append((f"Use a {name} for {s.ref}", "part", _swap(e, b, f"Use a {name} for {s.ref}: {b.source}"),
                            True))
                break
    if items.get("temperature") and items["temperature"].status in ("over", "near") and \
            (e.part.get("ratings") or {}).get("theta_jc"):
        out.append(_heatsink(e, s, items["temperature"], ta))
    if s.kind == "regulator":
        out += _regulator(c, e, s, items)
    if s.kind == "mcu" and "supply voltage" in items and items["supply voltage"].status in ("over", "warn"):
        out.append(_mcu_regulator(c, e, s))
    if s.kind == "opamp" and "phase margin" in items and items["phase margin"].status in ("fails", "warn"):
        iso = _isolation(c, e)
        if iso:
            out.append(iso)
    if s.kind == "fuse" and "current" in items:
        need = e_series(1.5 * abs(items["current"].value), [1, 1.25, 1.6, 2, 2.5, 3.15, 4, 5, 6.3, 8])

        def apply(c2, need=need):
            c2.element(e.name).part["ratings"]["ihold_A"] = need
            return {"why": f"Fit a {fmt(need)} A fuse for {s.ref}"}
        out.append((f"Use a {fmt(need)} A fuse", "fuse rating", apply, False))
    return out


def _swap(e: Element, p, why: str):
    def apply(c2):
        from .bind import _use_library
        me = c2.element(e.name)
        _use_library(c2, me, p)
        return {"why": why, "ref": me.ref}
    return apply


def _mosfet_swap(c, e, s, items) -> list:
    vds = abs(items["drain voltage"].value) if "drain voltage" in items else 0
    vgs = abs(items["gate voltage"].value) if "gate voltage" in items else 0
    idr = abs(items["drain current"].value) if "drain current" in items else 0
    drive = items["gate drive"].value if "gate drive" in items else vgs
    best = []
    for name, d in MOSFETS.items():
        pol, vmax, vgmax, idc, _, pts, *_ = d
        if pol != s.kind[0] or name == s.part or vmax < 1.25 * vds or vgmax < 1.1 * vgs or idc < 2 * idr:
            continue
        spec = min(v for v, _ in pts)
        if spec > max(drive, 0) + 0.05:
            continue                      # it needs more gate voltage than this circuit gives it
        rds = min(r for v, r in pts if v <= max(drive, spec) + 0.05)
        best.append((rds, name))
    out = []
    for _, name in sorted(best)[:2]:
        p = lookup(name)
        out.append((f"Use an {name} for {s.ref}", "part", _swap(e, p, f"Use an {name} for {s.ref}, which is fully "
                    f"on with {fmt(drive)} V on its gate: {p.source}"), True))
    return out


def _flyback(c: Circuit, e: Element):
    """A diode across the coil a transistor switches, so the coil's current has somewhere to go at switch off."""
    node = e.nodes[0]          # collector or drain
    coil = next((x for x, _ in c.nets().get(node, []) if x is not e and (x.kind in ("relay", "motor", "inductor")
                                                                       or x.part.get("params", {}).get("inductive"))),
                None)
    if coil is None:
        return None
    other = next(n for n in coil.nodes if n != node)
    amps = 0.15

    def apply(c2):
        from .netlist import Element as El
        name = "DFLY1"
        d = El(name, "d", [node, other], ["1N4148" if amps < 0.2 else "1N4007"], ["a", "k"], meta={"ref": "D_flyback"})
        c2.elements.append(d)
        from .bind import bind_new
        bind_new(c2, [d])
        return {"why": f"Put a 1N4148 across {coil.ref}, cathode to {c.label(other)}, so its current keeps "
                       f"flowing when {e.ref} switches off instead of forcing {e.ref}'s voltage up"}
    return (f"Add a flyback diode across {coil.ref}", "part", apply, True)


def _heatsink(e: Element, s: Stress, it, ta: float) -> tuple:
    r = e.part["ratings"]
    p = max((it.value - ta) / (r.get("theta_ja") or 1e9), s.p, 1e-6)     # the power at the worst supply end
    need = (r["tj_max_C"] - 25 - ta) / p - r["theta_jc"] - 0.5
    sink = max(round(need, 1), 0.5)

    def apply(c2):
        if need < 1.0:
            return None
        c2.element(e.name).part.setdefault("params", {})["heatsink"] = sink
        return {"why": f"Mount {s.ref} on a heatsink of {fmt(sink)} C/W or better (with a thermal pad), which keeps "
                       f"its junction 25 C under its limit at {fmt(p)} W"}
    return (f"Heatsink {s.ref} at {fmt(sink)} C/W", "heatsink", apply, False)


def _regulator(c, e, s, items) -> list:
    out = []
    prm = e.part.get("params", {})
    fails = items.get("regulation")
    iout = abs(items["output current"].value) if "output current" in items else 0
    if fails and fails.status == "fails" and prm.get("vout"):
        vin = abs(items["input voltage"].value) if "input voltage" in items else 0
        for name in ("AP2112K-3.3", "MCP1700-3302", "LM2940-5.0", "XC6206P332"):
            p = lookup(name)
            if p.params.get("vout") == prm["vout"] and p.params["irated"] >= 1.2 * iout and \
                    p.ratings.get("vin_V", 99) >= vin and p.params["dropout"] < prm["dropout"]:
                out.append((f"Use an {name} for {s.ref}", "part", _swap(e, p, f"Use an {name} for {s.ref}, which "
                            f"needs only {fmt(p.params['dropout'])} V of headroom: {p.source}"), True))
                break
    return out


def _mcu_regulator(c: Circuit, e: Element, s: Stress) -> tuple:
    from .ics import SUPPLY_PIN
    vmax = e.part["ratings"].get("vdd_op_V", 3.6)
    reg = "AMS1117-3.3" if vmax < 4 else "LM7805"

    def apply(c2):
        from .bind import bind_new
        from .netlist import Element as El
        me = c2.element(e.name)
        old = None
        for k, p in enumerate(me.pins):
            if SUPPLY_PIN.match(p):
                old = old or me.nodes[k]
                me.nodes[k] = "vdd_fix"
        if old is None:
            return None
        new = [El("XUREG1", "x", [old, "0", "vdd_fix"], [reg], ["in", "gnd", "out"], meta={"ref": "U_reg"}),
               El("CREG1", "c", ["vdd_fix", "0"], ["10u"], ["1", "2"], meta={"ref": "C_reg"})]
        c2.elements += new
        bind_new(c2, new)
        return {"why": f"Feed {s.ref} through an {reg} from {c.label(old)}, so it gets the voltage it is made for"}
    return (f"Add an {reg} in front of {s.ref}", "part", apply, True)


def _isolation(c: Circuit, e: Element):
    """An isolation resistor between an op amp's output and the capacitance it drives, feedback taken before it."""
    pin = dict(zip(e.pins, e.nodes))
    out_n, inn = pin["out"], pin["inn"]
    caps = [x for x, _ in c.nets().get(out_n, []) if x.letter == "c"]
    if not caps:
        return None

    def apply(c2):
        from .netlist import Element as El
        for x in list(c2.elements):
            if out_n in x.nodes and x.name != e.name and not (x.letter == "r" and inn in x.nodes):
                x.nodes = ["iso_out" if n == out_n else n for n in x.nodes]
        r = El("RISO1", "r", [out_n, "iso_out"], ["100"], ["1", "2"], meta={"ref": "R_iso"})
        c2.elements.append(r)
        from .bind import bind_new
        bind_new(c2, [r])
        return {"why": f"Put 100 ohm between {e.ref}'s output and {', '.join(x.ref for x in caps)}, with the feedback "
                       "taken before it, so the load's capacitance is outside the loop"}
    return (f"Isolate {e.ref} from its capacitive load with 100 ohm", "resistor", apply, True)


_ = DIODES
