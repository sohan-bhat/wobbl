"""A circuit described in words, built into a netlist with every guess said out loud.

"three red LEDs in parallel from a 9 V battery, each with 330 ohm", "a 555 blinking an LED at 2 Hz from 9 V",
"an AMS1117-3.3 from USB powering an ESP32 with an LED on GPIO2", "a non inverting LM358 amplifier with a
gain of 11 on 12 V", "a 2N2222 switching a 5 V relay from a 3.3 V GPIO without a flyback diode".

The words are read as a supply (a battery, USB, an adapter or a voltage), an optional regulator that makes a
rail, and blocks hung on the supply or the rail: LEDs, a divider, an RC filter, a 555 astable, an op amp
stage, a transistor or MOSFET switch with its load, an H bridge with a motor, a zener, a charger and a
rectifier. Values the words give are used; every value they leave out is chosen the way a designer would
(an LED resistor for about 15 mA, a 555's parts for the asked frequency) and reported through assume(text,
override) with the --set that changes it. Anything more involved is better given as a SPICE netlist or a
KiCad schematic, and the error says so.
"""
from __future__ import annotations

import math
import re

from ..units import parse_spice_value
from . import ics
from .netlist import Circuit, Element, NetlistError
from .parts import LED_VF, e_series

NUM = r"(\d+(?:\.\d+)?)"
COUNT = {"one": 1, "a": 1, "an": 1, "single": 1, "two": 2, "pair": 2, "three": 3, "four": 4, "five": 5, "six": 6,
         "seven": 7, "eight": 8, "nine": 9, "ten": 10, "twelve": 12, "sixteen": 16, "twenty": 20}


class _B:
    """Builds the netlist: numbered parts, fresh nodes, and the guesses with their overrides."""

    def __init__(self, text: str, params: dict, assume):
        self.t = " " + text.lower().replace("ω", " ohm").replace("µ", "u") + " "
        self.p = {k.lower(): v for k, v in params.items()}
        self.c = Circuit(title=text.strip()[:80], source="description")
        self.n: dict[str, int] = {}
        self.nodes = 0
        self.say = assume or (lambda text, override="": None)
        self.used: list[str] = []
        self.reg: list[Element] = []          # the regulator and its feedback resistors

    def name(self, letter: str) -> str:
        self.n[letter] = self.n.get(letter, 0) + 1
        return f"XU{self.n[letter]}" if letter == "x" else f"{letter.upper()}{self.n[letter]}"

    def node(self, base: str = "n") -> str:
        self.nodes += 1
        return f"{base}{self.nodes}"

    def add(self, letter: str, nodes: list[str], rest: list[str], pins: list[str], **meta) -> Element:
        from .netlist import PINS
        e = Element(self.name(letter), letter, nodes, rest, pins or PINS.get(letter, [])[:len(nodes)], meta=meta)
        self.c.elements.append(e)
        return e

    def load(self, nodes: list[str], ohms: float) -> Element:
        """A load the circuit drives, named RLOAD so it is reported rather than judged as a resistor part."""
        k = sum(1 for e in self.c.elements if e.name.upper().startswith("RLOAD")) + 1
        e = Element(f"RLOAD{k}", "r", nodes, [f"{ohms:.4g}"], ["1", "2"])
        self.c.elements.append(e)
        return e

    def guess(self, what: str, value: str, key: str, setv: str | None = None):
        shown = setv if setv is not None else value.split(" (")[0].replace(" ", "")
        self.say(f"{what} not given: assumed {value}", f"--set {key}={shown}")

    def val(self, key: str, pattern: str | None, default: str, what: str, unit: str = "") -> float:
        """A value from --set, else from the words, else the default with the guess said."""
        if key in self.p:
            return parse_spice_value(str(self.p[key]).lower().rstrip(unit.lower()))
        if pattern:
            m = re.search(pattern, self.t)
            if m:
                return parse_spice_value(m.group(1).replace(" ", ""))
        self.guess(what, default + unit, key)
        return parse_spice_value(default)

    def has(self, rx: str) -> bool:
        return re.search(rx, self.t) is not None


def build(text: str, params: dict, assume=None) -> Circuit:
    b = _B(text, params, assume)
    vs, sup = _supply(b)
    rail, vrail = _regulator(b, sup, vs)
    blocks = 0
    for fn in (_charger, _rectifier, _timer, _opamp, _hbridge, _switch, _zener, _filter, _divider, _mcu, _leds):
        if fn(b, rail, vrail, sup, vs):
            blocks += 1
    if not blocks and rail == sup:
        raise NetlistError("could not build a circuit from those words",
                           "Name the parts and how they connect (for example 'three red LEDs in parallel from 9 V, "
                           "each with 330 ohm'), or pass a SPICE netlist or a KiCad schematic")
    if rail != sup and not any(rail in e.nodes for e in b.c.elements if e.letter != "c" and e not in b.reg):
        load = b.val("load", rf"{NUM}\s*(?:m?a)\b(?! *h)", "100m", "regulator load current", "A")
        if b.has(rf"{NUM}\s*ma\b") and "load" not in b.p:
            load = load / 1000 if load > 5 else load
        b.load([rail, "0"], vrail / max(load, 1e-6))
    return b.c


# --- supply and rail ---------------------------------------------------------------------------------------------------
def _supply(b: _B) -> tuple[float, str]:
    sup = "vs"
    if "supply" in b.p:
        v = parse_spice_value(str(b.p["supply"]).lower().rstrip("v"))
        b.add("v", [sup, "0"], [f"{v:g}"], ["+", "-"], ref="supply")
        return v, sup
    m = re.search(r"(?:from|powered by|off|on|running on)\s+(?:an?\s+|the\s+)?(\d\s*x\s*aaa?|cr20\d\d|18650|li-?ion|"
                  r"lipo|lifepo4|9 ?v battery|usb-?c|usb|\d\s*s\s*lipo)\b", b.t)
    if m:
        bat = ics.battery(m.group(1).replace(" ", "").replace("battery", ""))
        if bat:
            b.c.subckts[bat.mname.lower()] = _sub(bat)
            from .bind import _as_dict
            b.add("x", [sup, "0"], [bat.mname], ["p", "n"], ref="BT1", battery=bat.key, **{"as": _as_dict(bat)})
            return bat.params["v"], sup
    bat = None
    for pat in (r"\b(\d)\s*x\s*(aa|aaa)\b", r"\b(cr2032|cr2025|18650|li-?ion|lipo|lifepo4|9 ?v battery|pp3|"
                                             r"nimh|usb-?c|usb)\b", r"\b(\d)\s*s\s*lipo\b", r"\b(\d) (aa|aaa)s?\b"):
        m = re.search(pat, b.t)
        if m:
            bat = ics.battery(m.group(0).replace(" ", "").replace("battery", ""))
            break
    if bat:
        b.c.subckts[bat.mname.lower()] = _sub(bat)
        from .bind import _as_dict
        b.add("x", [sup, "0"], [bat.mname], ["p", "n"], ref="BT1", battery=bat.key, **{"as": _as_dict(bat)})
        return bat.params["v"], sup
    m = re.search(rf"(?:from|on|at|with|powered by|supply of|off)\s+(?:an?\s+)?(?:\+/?-|±)?{NUM}\s*v\b", b.t) or \
        re.search(rf"\b{NUM}\s*v\s*(?:supply|battery|adapter|rail|source|dc|input|bench)", b.t) or \
        re.search(rf"\b{NUM}\s*v\b", b.t)
    if m:
        v = float(m.group(1))
    else:
        v = 5.0
        b.guess("supply voltage", "5 V", "supply")
    dual = b.has(r"±|\+/-|dual (?:supply|rail)|split supply")
    b.add("v", [sup, "0"], [f"{v:g}"], ["+", "-"], ref="supply")
    if dual:
        b.add("v", ["0", "vneg"], [f"{v:g}"], ["+", "-"], ref="negative supply")
    return v, sup


def _sub(p):
    from .netlist import Subckt
    return Subckt(p.mname, list(p.ports), p.text)


def _regulator(b: _B, sup: str, vs: float) -> tuple[str, float]:
    m = re.search(r"\b((?:lm|l|mc|ua)?78l?\d\d\w*|ams1117[-\w.]*|lm1117[-\w.]*|ld1117\w*|lm317\w*|mcp1700[-\w]*|"
                  r"ap2112[-\w.]*|xc6206\w*|ht73\d\d|lm2940[-\w.]*)", b.t)
    if not m:
        return sup, vs
    from .parts import lookup
    part = lookup(m.group(1).upper())
    if part is None or part.kind != "regulator":
        return sup, vs
    out = "vreg"
    if part.params.get("adjustable"):
        vt = b.val("vout", rf"(?:set to|for|at|to)\s+{NUM}\s*v\b", "5", "adjustable regulator output", "V")
        r1 = 240.0
        r2 = e_series(r1 * (vt / 1.25 - 1), up=False)
        adj = b.node("adj")
        b.reg += [b.add("x", [sup, adj, out], [part.key], ["in", "adj", "out"]),
                  b.add("r", [out, adj], [f"{r1:g}"], ["1", "2"]), b.add("r", [adj, "0"], [f"{r2:g}"], ["1", "2"])]
        vout = 1.25 * (1 + r2 / r1)
        b.say(f"{part.key} set with 240 ohm and {r2:g} ohm for about {vout:.3g} V", f"--set vout={vt:g}V")
    else:
        b.reg.append(b.add("x", [sup, "0", out], [part.key], ["in", "gnd", "out"]))
        vout = part.params["vout"]
    cin = "330n" if part.key.upper().startswith(("LM78", "L78", "78", "MC78", "UA78")) else "10u"
    b.add("c", [sup, "0"], [cin], ["1", "2"])
    b.add("c", [out, "0"], ["10u" if cin == "10u" else "100n"], ["1", "2"])
    b.say(f"{part.key} with the datasheet's {cin} input and output capacitors", "")
    return out, vout


# --- blocks --------------------------------------------------------------------------------------------------------
def _leds(b: _B, rail: str, vr: float, sup: str, vs: float) -> bool:
    m = re.search(r"\b(\d+|" + "|".join(COUNT) + r")?\s*(?:(?:red|green|blue|yellow|orange|white|amber|uv|ir|warm[- ]white)"
                  r"\s+)?leds?\b", b.t)
    if not m or b.has(r"555|op.?amp|amplifier") or "led" in " ".join(b.used):
        return False
    n = int(b.p.get("count", 0)) or (int(m.group(1)) if m.group(1) and m.group(1).isdigit() else
                                     COUNT.get(m.group(1) or "", 1))
    color = (re.search(r"\b(red|green|blue|yellow|orange|white|amber|uv|ir)\b", b.t) or [None, "red"])[1]
    color = str(b.p.get("led", color))
    vf = LED_VF.get(color, 2.0)
    series = b.has(r"in series|series string|daisy") and n > 1
    src = rail
    gpio = _gpio(b, rail, vr)
    v = gpio[1] if gpio else vr
    if gpio:
        src = gpio[0]
    r_given = re.search(rf"{NUM}\s*(k|m)?\s*(?:ohm|r\b|resistor)", b.t)
    if "resistor" in b.p:
        r = parse_spice_value(str(b.p["resistor"]))
    elif r_given:
        r = parse_spice_value(r_given.group(1) + (r_given.group(2) or ""))
    else:
        drop = v - (n * vf if series else vf)
        r = e_series(max(drop, 0.1) / 0.015)
        b.guess("LED resistor", f"{r:g} ohm (about 15 mA)", "resistor", f"{r:g}")
    model = f"LED_{color.upper()}"
    if series:
        a = b.node("a")
        b.add("r", [src, a], [f"{r:g}"], ["1", "2"])
        for k in range(n):
            k_node = "0" if k == n - 1 else b.node("a")
            b.add("d", [a, k_node], [model], ["a", "k"])
            a = k_node
    elif b.has(r"share|one resistor|single resistor|common resistor") and n > 1:
        a = b.node("a")
        b.add("r", [src, a], [f"{r:g}"], ["1", "2"])
        for _ in range(n):
            b.add("d", [a, "0"], [model], ["a", "k"])
    else:
        for _ in range(n):
            a = b.node("a")
            b.add("r", [src, a], [f"{r:g}"], ["1", "2"])
            b.add("d", [a, "0"], [model], ["a", "k"])
    b.used.append("led")
    return True


def _gpio(b: _B, rail: str, vr: float) -> tuple[str, float] | None:
    """An output pin of a microcontroller named in the words, as a node it drives, and its voltage."""
    if not b.has(r"gpio|\bpin\b|arduino|esp32|esp8266|microcontroller|\bmcu\b|atmega|attiny|rp2040|stm32|pico"):
        return None
    u = next((e for e in b.c.elements if e.kind == "mcu" or e.meta.get("mcu")), None)
    if u is None:
        _mcu(b, rail, vr, rail, vr, force=True)
        u = next((e for e in b.c.elements if e.meta.get("mcu")), None)
    if u is None:
        return None
    node = b.node("gpio")
    u.nodes.append(node)
    u.pins.append(f"io{len(u.pins) - 1}")
    return node, u.meta["vdd"]


def _mcu(b: _B, rail: str, vr: float, sup: str, vs: float, force: bool = False) -> bool:
    m = re.search(r"\b(esp32|esp8266|atmega328p?|arduino(?: uno| nano)?|attiny85|rp2040|pico|stm32f103|stm32)\b", b.t)
    if not m and not force:
        return False
    if any(e.meta.get("mcu") for e in b.c.elements):
        return False
    default = "esp32" if b.has(r"3\.3 ?v (?:gpio|pin|logic|micro)") else "atmega328p"
    name = (m.group(1) if m else default).replace("arduino uno", "atmega328p").replace("arduino nano", "atmega328p")
    name = {"arduino": "atmega328p", "pico": "rp2040", "stm32": "stm32f103"}.get(name, name).upper()
    if not m:
        b.guess("microcontroller", name, "mcu")
    three = name.startswith(("ESP", "RP", "STM"))
    node = rail if (abs(vr - 3.3) < 0.4) == three or rail == sup else sup
    vdd = vr if node == rail else vs
    if three and vdd > 3.7:
        b.say(f"{name} wired straight to {vdd:g} V as described", "")
    b.add("x", [node, "0"], [name], ["vdd", "gnd"], mcu=True, vdd=vdd, ref="U2" if rail != sup else "U1")
    return True


def _timer(b: _B, rail: str, vr: float, sup: str, vs: float) -> bool:
    if not b.has(r"\b555\b|ne555|lm555|tlc555|astable|timer"):
        return False
    part = "TLC555" if b.has(r"tlc555|lmc555|cmos 555") else "NE555"
    f = b.val("frequency", rf"{NUM}\s*(k?hz)", "", "blink rate", "") if b.has(r"\d\s*k?hz") or "frequency" in b.p \
        else None
    if f is not None and b.has(rf"{NUM}\s*khz") and "frequency" not in b.p:
        f *= 1000
    if f is None:
        f = 1.0 if b.has(r"blink|flash") else 1000.0
        b.guess("frequency", f"{f:g} Hz", "frequency")
    T = 1 / f
    c = e_series(T / (0.693 * 30e3), [1.0, 2.2, 4.7])
    ra = 1e3
    rb = e_series(max((T / (0.693 * c) - ra) / 2, 100))
    thr, dis, out = b.node("thr"), b.node("dis"), b.node("out")
    ctl = b.node("ctl")
    b.add("x", ["0", thr, out, rail, ctl, thr, dis, rail], [part],
          ["gnd", "trig", "out", "reset", "ctrl", "thres", "disch", "vcc"])
    b.add("r", [rail, dis], [f"{ra:g}"], ["1", "2"])
    b.add("r", [dis, thr], [f"{rb:g}"], ["1", "2"])
    b.add("c", [thr, "0"], [f"{c:.3g}"], ["1", "2"])
    b.add("c", [ctl, "0"], ["10n"], ["1", "2"])
    b.say(f"{part} astable with Ra 1 kohm, Rb {rb:g} ohm and C {c:.3g} F for about "
          f"{1 / (0.693 * (ra + 2 * rb) * c):.3g} Hz", f"--set frequency={f:g}")
    if b.has(r"\bleds?\b"):
        color = (re.search(r"\b(red|green|blue|yellow|white)\b", b.t) or [None, "red"])[1]
        a = b.node("a")
        r = e_series((vr - 1.7 - LED_VF.get(color, 2.0)) / 0.015) if vr > 5 else 150
        b.add("r", [out, a], [f"{r:g}"], ["1", "2"])
        b.add("d", [a, "0"], [f"LED_{color.upper()}"], ["a", "k"])
        b.used.append("led")
    elif b.has(r"buzzer|speaker"):
        b.add("r", [out, "0"], ["8" if b.has("speaker") else "170"], ["1", "2"])
    return True


def _opamp(b: _B, rail: str, vr: float, sup: str, vs: float) -> bool:
    m = re.search(r"\b(lm358|lm324|tl07[124]|tl08[124]|ne5532|lm741|ua741|mcp600[124]|opa2134|op07|lmv358)\b", b.t)
    if not (m or b.has(r"op.?amp|amplifier|comparator")):
        return False
    part = (m.group(1) if m else "lm358").upper()
    if not m:
        b.guess("op amp", "LM358", "opamp")
    g = b.val("gain", rf"gain (?:of )?{NUM}", "10", "gain")
    vneg = "vneg" if any("vneg" in e.nodes for e in b.c.elements) else "0"
    f = b.val("frequency", rf"{NUM}\s*(?:hz)\b", "1000", "test signal frequency", "")
    amp = b.val("amplitude", rf"{NUM}\s*(?:mv|v)\s*(?:peak|signal|input|sine)", "0.1", "input amplitude", "")
    if b.has(rf"{NUM}\s*mv") and "amplitude" not in b.p:
        amp = amp / 1000 if amp > 5 else amp
    inp, inn, out, src = b.node("inp"), b.node("inn"), b.node("out"), b.node("in")
    ref = "0" if vneg != "0" else b.node("vref")
    if ref != "0":
        b.add("r", [rail, ref], ["10k"], ["1", "2"])
        b.add("r", [ref, "0"], ["10k"], ["1", "2"])
        b.add("c", [ref, "0"], ["10u"], ["1", "2"])
        b.say("single supply: the signal and the gain network sit on a mid supply reference", "")
    bias = 0.0 if ref == "0" else vr / 2
    b.add("v", [src, "0"], [f"SIN({bias:g} {amp:g} {f:g})"], ["+", "-"], ref="input")
    rg = 10e3
    rf = rg * (abs(g) if b.has(r"\binverting\b") and not b.has(r"non.?inverting") else max(g - 1, 0.001))
    b.add("x", [inp, inn, rail, vneg, out], [part], ["inp", "inn", "vcc", "vee", "out"])
    if b.has(r"\binverting\b") and not b.has(r"non.?inverting"):
        b.add("r", [src, inn], [f"{rg:g}"], ["1", "2"])
        b.add("r", [inp, ref], ["0.001"], ["1", "2"])
    else:
        b.add("r", [src, inp], ["100"], ["1", "2"])
        b.add("r", [inn, ref], [f"{rg:g}"], ["1", "2"])
    b.add("r", [out, inn], [f"{rf:.4g}"], ["1", "2"])
    b.load([out, ref], 10e3)
    return True


def _switch(b: _B, rail: str, vr: float, sup: str, vs: float) -> bool:
    m = re.search(r"\b(2n2222a?|2n3904|bc547|bc337|tip120|tip122|s8050|irlz44n|irf540n?|irfz44n|irf520n?|2n7000|"
                  r"bs170|ao3400|fqp30n06l|irlb8721|irf3205)\b", b.t)
    if not (m or b.has(r"transistor|mosfet|\bnpn\b|\bfet\b")) or b.has(r"h.?bridge"):
        return False
    from .parts import lookup
    part = lookup(m.group(1).upper()) if m else None
    mos = (part.kind == "nmos") if part else b.has(r"mosfet|\bfet\b")
    name = part.key if part else ("IRLZ44N" if mos else "2N2222A")
    if not part:
        b.guess("transistor", name, "transistor")
    gp = _gpio(b, rail, vr)
    drive, vd = gp if gp else (b.node("drive"), vr)
    if not gp:
        b.add("v", [drive, "0"], [f"PULSE(0 {vd:g} 0 1u 1u 50m 100m)"], ["+", "-"], ref="drive")
        b.say(f"the switch is driven from a {vd:g} V signal that turns on for 50 ms and off for 50 ms", "")
    vload = b.val("load_voltage", rf"(?:a\s+)?{NUM}\s*v\s+(?:relay|motor|lamp|solenoid|load|led strip|fan)", f"{vs:g}",
                  "load voltage", "")
    top = sup
    if abs(vload - vs) > 0.05:
        top = "vload"
        b.add("v", [top, "0"], [f"{vload:g}"], ["+", "-"], ref="load supply")
    d = b.node("sw")
    if b.has(r"relay"):
        rly = ics.relay(vload)
        b.c.subckts[rly.mname.lower()] = _sub(rly)
        from .bind import _as_dict
        b.add("x", [top, d], [rly.mname], ["p", "n"], ref="K1", **{"as": _as_dict(rly)})
        inductive = True
    elif b.has(r"motor|fan"):
        mot = ics.motor(vload)
        b.c.subckts[mot.mname.lower()] = _sub(mot)
        from .bind import _as_dict
        b.add("x", [top, d], [mot.mname], ["p", "n"], ref="M1", **{"as": _as_dict(mot)})
        inductive = True
    else:
        il = b.val("load_current", rf"{NUM}\s*a\b", "1", "load current", "")
        b.load([top, d], vload / il)
        inductive = False
    if inductive and b.has(r"flyback|freewheel|snubber|diode across") and not b.has(r"without (?:a )?(?:flyback|diode)"
                                                                                       r"|no (?:flyback|diode)"):
        b.add("d", [d, top], ["1N4148" if b.has("relay") else "1N4007"], ["a", "k"])
    if mos:
        g = b.node("g")
        b.add("r", [drive, g], ["100"], ["1", "2"])
        b.add("r", [g, "0"], ["10k"], ["1", "2"])
        b.add("m", [d, g, "0"], [name], ["d", "g", "s"])
    else:
        base = b.node("b")
        rb = b.val("base_resistor", rf"{NUM}\s*(k)?\s*(?:ohm)?\s*base", "1k", "base resistor", "")
        b.add("r", [drive, base], [f"{rb:g}"], ["1", "2"])
        b.add("q", [d, base, "0"], [name], ["c", "b", "e"])
    return True


def _hbridge(b: _B, rail: str, vr: float, sup: str, vs: float) -> bool:
    if not b.has(r"h.?bridge"):
        return False
    big = b.has(r"\d\s*a\b") and float((re.search(rf"{NUM}\s*a\b", b.t) or [0, "1"])[1]) > 3
    pn, nn = ("IRF9540N", "IRFZ44N") if big or vs > 9 else ("AO3401", "AO3400")
    b.guess("bridge MOSFETs", f"{pn} high side and {nn} low side", "transistor")
    mot = ics.motor(vs, i_stall=b.val("stall", rf"{NUM}\s*a\s+stall", "2", "motor stall current", ""))
    b.c.subckts[mot.mname.lower()] = _sub(mot)
    from .bind import _as_dict
    a, c = b.node("ma"), b.node("mb")
    on_p, off_p, on_n, off_n = b.node("g"), b.node("g"), b.node("g"), b.node("g")
    b.add("v", [on_p, "0"], ["0"], ["+", "-"], ref="drive A high")
    b.add("v", [off_p, "0"], [f"{vs:g}"], ["+", "-"], ref="drive B high")
    b.add("v", [on_n, "0"], [f"{min(vs, 10):g}"], ["+", "-"], ref="drive B low")
    b.add("v", [off_n, "0"], ["0"], ["+", "-"], ref="drive A low")
    b.add("m", [a, on_p, sup], [pn], ["d", "g", "s"])
    b.add("m", [c, off_p, sup], [pn], ["d", "g", "s"])
    b.add("m", [a, off_n, "0"], [nn], ["d", "g", "s"])
    b.add("m", [c, on_n, "0"], [nn], ["d", "g", "s"])
    b.add("x", [a, c], [mot.mname], ["p", "n"], ref="MOT1", **{"as": _as_dict(mot)})
    b.say("the bridge drives the motor one way from standstill, so the start up current flows through one "
          "diagonal", "")
    return True


def _zener(b: _B, rail: str, vr: float, sup: str, vs: float) -> bool:
    m = re.search(rf"{NUM}\s*v\s*zener|zener of {NUM}\s*v", b.t)
    if not m and not b.has(r"zener"):
        return False
    vz = float(m.group(1) or m.group(2)) if m else b.val("vz", None, "5.1", "zener voltage")
    r = b.val("resistor", rf"{NUM}\s*(k)?\s*ohm", "220", "series resistor", "")
    z = b.node("z")
    b.add("r", [rail, z], [f"{r:g}"], ["1", "2"])
    b.add("d", ["0", z], [f"BZX55C{int(vz)}V{round((vz - int(vz)) * 10)}"], ["a", "k"])
    if b.has(r"load"):
        il = b.val("load", rf"{NUM}\s*ma\s+load", "10", "load current (mA)", "")
        b.load([z, "0"], vz / (il / 1000))
    return True


def _divider(b: _B, rail: str, vr: float, sup: str, vs: float) -> bool:
    if not b.has(r"divider"):
        return False
    vals = re.findall(rf"{NUM}\s*(k|m|meg)?\s*(?:ohm)?\b(?!\s*(?:v|hz|a|f)\b)", b.t)
    rs = [parse_spice_value(a + (k or "")) for a, k in vals if k or float(a) >= 10][:2]
    if len(rs) < 2:
        rs = [10e3, 10e3]
        b.guess("divider resistors", "10k and 10k", "r1")
    mid = b.node("mid")
    b.add("r", [rail, mid], [f"{rs[0]:g}"], ["1", "2"])
    b.add("r", [mid, "0"], [f"{rs[1]:g}"], ["1", "2"])
    return True


def _filter(b: _B, rail: str, vr: float, sup: str, vs: float) -> bool:
    if not b.has(r"(?:low|high|band).?pass|\bfilter\b"):
        return False
    high = b.has(r"high.?pass")
    fc = b.val("cutoff", rf"{NUM}\s*(k?hz)", "1000", "cutoff frequency", "") if not b.has(r"\d\s*k?\s*ohm") else None
    if fc is not None and b.has(rf"{NUM}\s*khz") and "cutoff" not in b.p:
        fc *= 1000
    rv = re.search(rf"{NUM}\s*(k)?\s*ohm|\b{NUM}(k)\b", b.t)
    cv = re.search(rf"{NUM}\s*([unp])\s*f\b", b.t)
    r = parse_spice_value((rv.group(1) or rv.group(3)) + ((rv.group(2) or rv.group(4)) or "")) if rv else 10e3
    c = parse_spice_value(cv.group(1) + cv.group(2)) if cv else e_series(1 / (2 * math.pi * r * (fc or 1e3)), [1, 2.2,
                                                                                                                 4.7])
    f0 = 1 / (2 * math.pi * r * c)
    src, out = b.node("in"), b.node("out")
    b.add("v", [src, "0"], [f"SIN(0 1 {f0:.4g}) AC 1"], ["+", "-"], ref="input")
    if high:
        b.add("c", [src, out], [f"{c:.3g}"], ["1", "2"])
        b.add("r", [out, "0"], [f"{r:g}"], ["1", "2"])
    else:
        b.add("r", [src, out], [f"{r:g}"], ["1", "2"])
        b.add("c", [out, "0"], [f"{c:.3g}"], ["1", "2"])
    b.say(f"{'high' if high else 'low'} pass RC of {r:g} ohm and {c:.3g} F (cutoff {f0:.4g} Hz), driven by a 1 V sine",
          "")
    return True


def _charger(b: _B, rail: str, vr: float, sup: str, vs: float) -> bool:
    m = re.search(r"\b(tp4056|mcp73831)\b", b.t)
    if not m and not b.has(r"charg(?:er|ing) (?:a |an )?(?:li|18650|cell|battery)"):
        return False
    from .parts import lookup
    part = lookup((m.group(1) if m else "tp4056").upper())
    ich = b.val("current", rf"{NUM}\s*(m?a)\b", "1" if part.key.upper().startswith("TP") else "0.5",
                "charge current", "")
    if b.has(rf"{NUM}\s*ma\b") and "current" not in b.p:
        ich = ich / 1000 if ich > 5 else ich
    vbat = b.val("vbat", rf"(?:cell|battery) at {NUM}\s*v", "3.7", "cell voltage while charging", "")
    k = part.params["k"]
    rprog = e_series(k / ich)
    bat, prog = b.node("bat"), b.node("prog")
    b.add("x", [rail, "0", bat, prog, b.node("stat")], [part.key], ["vin", "gnd", "bat", "prog", "stat"])
    b.add("r", [prog, "0"], [f"{rprog:g}"], ["1", "2"])
    b.add("v", [bat, "0"], [f"{vbat:g}"], ["+", "-"], ref="cell", note="liion")
    b.add("c", [rail, "0"], ["10u"], ["1", "2"])
    b.say(f"PROG resistor {rprog:g} ohm for about {k / rprog:.3g} A, with the cell held at {vbat:g} V", "")
    return True


def _rectifier(b: _B, rail: str, vr: float, sup: str, vs: float) -> bool:
    if not b.has(r"rectifier|\bv ?ac\b|transformer"):
        return False
    vac = b.val("vac", rf"{NUM}\s*v ?ac", "12", "AC voltage (RMS)", "")
    f = 60.0 if b.has(r"60 ?hz|us\b|america") else 50.0
    a, k = b.node("ac"), b.node("acn")
    # the supply source becomes the transformer secondary
    b.c.elements = [e for e in b.c.elements if e.meta.get("ref") != "supply"]
    w = b.node("w")
    b.add("v", [w, k], [f"SIN(0 {vac * math.sqrt(2):.4g} {f:g})"], ["+", "-"], ref="transformer")
    b.add("r", [w, a], ["1"], ["1", "2"], ref="winding", **{"as": {
        "key": "transformer winding", "kind": "load", "ratings": {}, "generic": False, "params": {},
        "source": "the transformer's winding resistance"}})
    b.add("r", [k, "0"], ["1meg"], ["1", "2"])
    b.say(f"a {vac:g} V AC {f:g} Hz transformer secondary with 1 ohm of winding resistance", "--set vac=12")
    out = b.node("dc")
    for x, y in ((a, out), (k, out), ("0", a), ("0", k)):
        b.add("d", [x, y], ["1N4007"], ["a", "k"])
    cap = b.val("capacitor", r"(\d+(?:\.\d+)?\s*[unm])\s*f\b", "1000u", "filter capacitor", "")
    b.add("c", [out, "0"], [f"{cap:.4g}"], ["1", "2"], note="polarized electrolytic 25V")
    il = b.val("load", rf"{NUM}\s*(?:m?a)\s+load", "0.5", "load current", "")
    b.load([out, "0"], vac * 1.4 / il)
    return True
