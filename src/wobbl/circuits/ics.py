"""Library entries for ICs and modules: linear regulators, op amps and comparators, the 555, chargers, the
TL431, optocouplers, microcontrollers, batteries, motors and relays.

Each is a behavioral subcircuit written from datasheet numbers. A regulator passes current to hold its output
until the headroom runs out (dropout) or it reaches its current limit. An op amp is a single pole gain stage
with a slew limit and an output that swings only as close to the rails as the datasheet says and draws its
load current from its own supply pins, so its dissipation is real. A microcontroller draws its supply current
and drives each output pin through its output resistance, with clamp diodes on every pin. Port orders follow
the usual SPICE conventions (op amps in+ in- v+ v- out, the 555 by its DIP pins), and every port has the pin
names KiCad symbols use, so schematics map by name.
"""
from __future__ import annotations

import math
import re

from .parts import Part, _g, builder

# port aliases: how symbol pin names map onto subcircuit ports
ALIASES = {
    "in": ("vi", "vin", "in", "input", "vin+", "ip"), "out": ("vo", "vout", "out", "output", "~", "o", "op", "q"),
    "vin": ("vin", "vcc", "vdd", "in", "v_in", "vi"),
    "gnd": ("gnd", "com", "adj/gnd", "gnd/adj", "0v", "vss", "gnd_pad", "ep", "pad"), "adj": ("adj", "fb"),
    "inp": ("+", "in+", "+in", "non-inv", "noninv", "v_in+", "inp", "vin+"),
    "inn": ("-", "in-", "-in", "inv", "v_in-", "inn", "vin-"),
    "vcc": ("v+", "vcc", "vdd", "vs+", "+vs", "+v", "vcc/vdd"), "vee": ("v-", "vee", "vss", "gnd", "-vs", "vs-", "-v"),
    "trig": ("tr", "trig", "trigger", "~{tr}"), "reset": ("r", "reset", "~{r}", "~{reset}", "rst"),
    "ctrl": ("cv", "ctrl", "cont", "control", "co"), "thres": ("thr", "thres", "threshold", "th"),
    "disch": ("dis", "disch", "discharge"), "bat": ("bat", "vbat", "batt", "bat+"),
    "prog": ("prog", "iset", "rprog"), "stat": ("stat", "chrg", "~{chrg}", "status"), "k": ("k", "cathode", "c"), "a": ("a", "anode"), "ref": ("ref", "r", "vref"),
    "c": ("c", "collector"), "e": ("e", "emitter"),
}


def port_of(pin_name: str, ports: tuple) -> str | None:
    n = pin_name.strip().lower()
    if n in ports:
        return n
    for p in ports:
        if n in ALIASES.get(p, ()):
            return p
    return None


# --- linear regulators ----------------------------------------------------------------------------------------
# family: (vout or 0 for adjustable, vin max, dropout V at rated current, rated A, limit A, quiescent A,
#          theta_ja, theta_jc, tj max, pass element, package, minimum load A)
REGS = {
    "78XX": (5, 35, 2.0, 1.5, 2.2, 5e-3, 50, 5, 125, "npn", "TO-220", 0),
    "78LXX": (5, 30, 1.7, 0.1, 0.14, 3.8e-3, 200, 0, 125, "npn", "TO-92", 0),
    "1117": (3.3, 15, 1.2, 0.8, 1.2, 5e-3, 90, 15, 125, "npn", "SOT-223", 0),
    "MCP1700": (3.3, 6.0, 0.35, 0.25, 0.55, 1.6e-6, 336, 0, 125, "pmos", "SOT-23", 0),
    "AP2112": (3.3, 6.0, 0.4, 0.6, 0.8, 55e-6, 250, 0, 125, "pmos", "SOT-23-5", 0),
    "XC6206": (3.3, 6.0, 0.5, 0.2, 0.3, 1e-6, 400, 0, 125, "pmos", "SOT-23", 0),
    "HT73XX": (3.3, 12, 0.9, 0.25, 0.35, 4e-6, 200, 0, 125, "pmos", "SOT-89", 0),
    "LM2940": (5, 26, 0.5, 1.0, 1.6, 10e-3, 53, 3, 125, "pnp", "TO-220", 0),
    "LM317": (0, 40, 2.0, 1.5, 2.2, 0.0, 50, 5, 125, "npn", "TO-220", 3.5e-3),
}


def _reg_name(u: str) -> tuple[str, float | None] | None:
    """The regulator family a part number belongs to and its fixed output, or None for an adjustable part."""
    m = re.match(r"^(?:LM|L|MC|UA|KA|NJM|CJ)?78(L)?(\d\d)", u)
    if m and int(m.group(2)) in (5, 6, 8, 9, 10, 12, 15, 18, 24, 33):
        v = 3.3 if m.group(2) == "33" else float(int(m.group(2)))
        return ("78LXX" if m.group(1) else "78XX"), v
    m = re.match(r"^(?:AMS|LM|LD|NCP|AZ|SPX|TLV|REG)1117[A-Z]{0,3}[-_]?(ADJ|\d\.\d+|\d\d)?", u)
    if m:
        s = m.group(1) or "ADJ"
        return "1117", (None if s == "ADJ" else float(s) if "." in s else int(s) / 10)
    m = re.match(r"^LD(33|50|18|25)", u)
    if m:
        return "1117", int(m.group(1)) / 10
    m = re.match(r"^MCP1700T?[-_]?(\d\d)", u)
    if m:
        return "MCP1700", int(m.group(1)) / 10
    m = re.match(r"^AP2112[A-Z]?[-_]?(\d\.\d)?", u)
    if m:
        return "AP2112", float(m.group(1) or 3.3)
    m = re.match(r"^XC6206P?(\d\d)", u)
    if m:
        return "XC6206", int(m.group(1)) / 10
    if u == "662K":
        return "XC6206", 3.3
    m = re.match(r"^HT73(\d\d)", u)
    if m:
        return "HT73XX", int(m.group(1)) / 10
    m = re.match(r"^LM2940[A-Z]*[-_]?(\d+\.?\d*)?", u)
    if m:
        return "LM2940", float(m.group(1) or 5)
    if re.match(r"^LM317", u):
        return "LM317", None
    return None


def regulator(name: str) -> Part | None:
    hit = _reg_name(name.upper())
    if not hit:
        return None
    fam, vout = hit
    v0, vin_max, vdo, irated, ilim, iq, thja, thjc, tj, kind, pkg, minload = REGS[fam]
    if fam == "78XX" and vout and vout >= 24:
        vin_max = 40
    adj = vout is None
    vref = 1.25
    gm = 50.0
    # headroom: a bipolar pass element needs a fixed drop plus a resistive part, a MOS pass element only the latter
    vdo0 = 0.6 * vdo if kind in ("npn", "pnp") else 0.0
    rdo = (vdo - vdo0) / irated
    m = "U_" + re.sub(r"[^A-Za-z0-9]", "_", name)
    ref = f"{vref} - V(out,adj)" if adj else f"{vout} - V(out,gnd)"
    third = "adj" if adj else "gnd"
    text = (f".subckt {m} in {third} out\n"
            f"B1 in out I = min(min({ilim}, max(0, {gm}*({ref}))), max(0, (V(in,out) - {_g(vdo0)})/{_g(rdo)}))\n"
            + (f"Iadj in adj {50e-6 if fam == 'LM317' else 55e-6}\n" if adj else f"Iq in gnd {_g(iq)}\n")
            + ".ends")
    r = {"vin_V": vin_max, "iout_A": irated, "theta_ja": thja, "theta_jc": thjc or None, "tj_max_C": tj}
    if fam == "LM317":
        r = {"vdiff_V": vin_max, **{k: v for k, v in r.items() if k != "vin_V"}}
    what = "adjustable" if adj else f"{vout:g} V"
    return Part(name, "regulator", r, "x", m, text, ("in", third, "out"),
                source=f"{name} datasheet: {what}, {vdo:g} V dropout at {irated:g} A, {ilim:g} A limit, "
                       f"{vin_max:g} V most in, {pkg}",
                params={"vout": vout, "vref": vref if adj else None, "dropout": vdo, "irated": irated,
                        "min_load": minload or (0.005 if adj else 0), "adjustable": adj, "package": pkg})


# --- op amps and comparators --------------------------------------------------------------------------------
# name: (channels, GBW Hz, open loop gain dB, slew V/us, offset V, output ohm, short circuit A, supply A per
#        channel, least supply V, most supply V, input range above V-, input range below V+, output swing below
#        V+, output swing above V-, theta_ja)
OPAMPS = {
    "LM358": (2, 0.7e6, 100, 0.3, 2e-3, 50, 0.040, 0.35e-3, 3, 32, -0.3, 1.5, 1.5, 0.02, 125),
    "LM324": (4, 0.7e6, 100, 0.3, 2e-3, 50, 0.040, 0.35e-3, 3, 32, -0.3, 1.5, 1.5, 0.02, 105),
    "LM741": (1, 1.0e6, 106, 0.5, 1e-3, 75, 0.025, 1.7e-3, 10, 36, 3.0, 3.0, 2.0, 2.0, 125),
    "TL071": (1, 3.0e6, 106, 13, 3e-3, 50, 0.040, 1.4e-3, 7, 36, 4.0, 0.0, 1.5, 1.5, 125),
    "TL072": (2, 3.0e6, 106, 13, 3e-3, 50, 0.040, 1.4e-3, 7, 36, 4.0, 0.0, 1.5, 1.5, 125),
    "TL074": (4, 3.0e6, 106, 13, 3e-3, 50, 0.040, 1.4e-3, 7, 36, 4.0, 0.0, 1.5, 1.5, 105),
    "NE5532": (2, 10e6, 100, 9, 0.5e-3, 50, 0.038, 4e-3, 6, 44, 3.0, 3.0, 2.0, 2.0, 97),
    "MCP6001": (1, 1.0e6, 112, 0.6, 4.5e-3, 100, 0.023, 0.1e-3, 1.8, 6.0, -0.3, -0.3, 0.025, 0.025, 230),
    "MCP6002": (2, 1.0e6, 112, 0.6, 4.5e-3, 100, 0.023, 0.1e-3, 1.8, 6.0, -0.3, -0.3, 0.025, 0.025, 163),
    "MCP6004": (4, 1.0e6, 112, 0.6, 4.5e-3, 100, 0.023, 0.1e-3, 1.8, 6.0, -0.3, -0.3, 0.025, 0.025, 120),
    "LMV321": (1, 1.0e6, 100, 1.0, 1.7e-3, 50, 0.060, 0.065e-3, 2.7, 5.5, -0.2, 0.8, 0.01, 0.01, 250),
    "LMV358": (2, 1.0e6, 100, 1.0, 1.7e-3, 50, 0.060, 0.065e-3, 2.7, 5.5, -0.2, 0.8, 0.01, 0.01, 190),
    "OPA2134": (2, 8.0e6, 120, 20, 0.5e-3, 40, 0.040, 4e-3, 5, 36, 2.5, 2.5, 1.0, 1.0, 150),
    "OP07": (1, 0.6e6, 112, 0.3, 75e-6, 60, 0.030, 2.5e-3, 6, 36, 2.0, 2.0, 2.0, 2.0, 150),
}
OPAMP_ALIAS = {"UA741": "LM741", "741": "LM741", "LM2904": "LM358", "LM358A": "LM358", "LM2902": "LM324",
               "TL082": "TL072", "TL081": "TL071", "TL084": "TL074", "NE5534": "NE5532", "LMV324": "LMV358",
               "OPA134": "OPA2134", "MCP6L02": "MCP6002"}
COMPARATORS = {"LM393": (2, 36, 2, 0.016, 0.4e-3, 1.5, 150), "LM339": (4, 36, 2, 0.016, 0.2e-3, 1.5, 120),
               "LM2903": (2, 36, 2, 0.016, 0.4e-3, 1.5, 150)}


def opamp(name: str) -> Part | None:
    key = OPAMP_ALIAS.get(name, name)
    if key not in OPAMPS:
        return comparator(name)
    ch, gbw, aol_db, sr, vos, rout, isc, iq, vsmin, vsmax, cml, cmh, voh, vol, thja = OPAMPS[key]
    aol = 10 ** (aol_db / 20)
    gm = aol / 1e6
    c1 = aol / (2 * math.pi * gbw * 1e6)
    isr = sr * 1e6 * c1
    m = f"U_{name}"
    text = (f".subckt {m} inp inn vcc vee out\n"
            f"Rin inp inn 10meg\n"
            f"B1 0 n1 I = max(-{_g(isr)}, min({_g(isr)}, {_g(gm)}*(V(inp,inn)+{_g(vos)})))\n"
            f"R1 n1 0 1meg\nC1 n1 0 {_g(c1)}\n"
            f"B2 n1 0 I = 10*max(0, V(n1)-V(vcc)+{_g(voh)}) + 10*min(0, V(n1)-V(vee)-{_g(vol)})\n"
            f"G2 0 n2 n1 0 1\nR2 n2 0 1\nC2 n2 0 {_g(1 / (2 * math.pi * 2 * gbw))}\n"
            f"Bsrc vcc out I = max(0, min({isc}, (V(n2)-V(out))/{rout}))\n"
            f"Bsnk out vee I = max(0, min({isc}, (V(out)-V(n2))/{rout}))\n"
            f"Iq vcc vee {_g(iq)}\n.ends")
    r = {"vsupply_V": vsmax, "iout_A": isc, "theta_ja": thja, "tj_max_C": 150}
    return Part(name, "opamp", r, "x", m, text, ("inp", "inn", "vcc", "vee", "out"),
                source=f"{name} datasheet: {gbw / 1e6:g} MHz, {sr:g} V/us, swing to {voh:g} V of V+ and {vol:g} V "
                       f"of V-, {isc * 1000:g} mA short circuit, {vsmin:g} to {vsmax:g} V supply",
                params={"gm": gm, "aol": aol, "gbw": gbw, "channels": ch, "vs_min": vsmin, "cm_low": cml,
                        "cm_high": cmh, "voh": voh, "vol": vol, "rr_node": "n1"})


def comparator(name: str) -> Part | None:
    if name not in COMPARATORS:
        return None
    ch, vsmax, vsmin, isink, iq, cmh, thja = COMPARATORS[name]
    m = f"U_{name}"
    text = (f".subckt {m} inp inn vcc vee out\nRin inp inn 10meg\n"
            f"B1 out vee I = 0.5*(1+tanh(2000*(V(inn)-V(inp)))) * min({isink}, max(0, V(out,vee))/15) + 1n\n"
            f"Iq vcc vee {_g(iq)}\n.ends")
    return Part(name, "comparator", {"vsupply_V": vsmax, "iout_A": isink, "theta_ja": thja, "tj_max_C": 150}, "x",
                m, text, ("inp", "inn", "vcc", "vee", "out"),
                source=f"{name} datasheet: open collector output sinking up to {isink * 1000:g} mA, {vsmin:g} to "
                       f"{vsmax:g} V supply", params={"channels": ch, "vs_min": vsmin, "cm_low": -0.3,
                                                      "cm_high": cmh, "open_collector": True})


# --- 555 timer ----------------------------------------------------------------------------------------------
TIMERS = {"NE555": ("bipolar", 4.5, 16, 18, 0.2, 0.2, 1.7, 3e-3, 100), "TLC555": ("cmos", 2, 15, 18, 0.01, 0.1,
                                                                                  0.1, 170e-6, 150)}
TIMER_ALIAS = {"LM555": "NE555", "NA555": "NE555", "SA555": "NE555", "SE555": "NE555", "555": "NE555",
               "LMC555": "TLC555", "ICM7555": "TLC555", "TLC555C": "TLC555"}


def timer(name: str) -> Part | None:
    key = TIMER_ALIAS.get(name, name)
    if key not in TIMERS:
        return None
    tech, vsmin, vsmax, vabs, isrc, isnk, vdrop, iq, thja = TIMERS[key]
    rq = 5.0 / iq
    rs, rk = (10, 10) if tech == "bipolar" else (60, 10)
    m = f"U_{name}"
    # comparators and the latch on their own nodes, so ngspice differentiates small expressions
    text = (f".subckt {m} gnd trig out reset ctrl thres disch vcc\n"
            f"R1 vcc ctrl 5k\nR2 ctrl lo 5k\nR3 lo gnd 5k\nRq vcc gnd {_g(rq)}\n"
            "Bs s 0 V = 0.5*(1+tanh(200*(V(lo,gnd)-V(trig,gnd))))\n"
            "Bt t 0 V = 0.5*(1+tanh(200*(V(thres,gnd)-V(ctrl,gnd))))\n"
            "Br r 0 V = 0.5*(1+tanh(20*(0.7-V(reset,gnd))))\n"
            "Bh h 0 V = 0.5*(1+tanh(40*(V(q)-0.5)))\n"
            "Bq 0 q I = 1m*(0.5*(1+tanh(40*((1-V(r))*(V(s)+(1-V(s))*(1-V(t))*V(h))-0.5)))-V(q))\nCq q 0 10p\n"
            "Bx x 0 V = 0.5*(1+tanh(60*(V(q)-0.5)))\n"
            f"Bsrc vcc out I = V(x)*max(0, min({isrc}, (V(vcc,gnd)-{vdrop}-V(out,gnd))/{rs}))\n"
            f"Bsnk out gnd I = (1-V(x))*max(0, min({isnk}, (V(out,gnd)-0.05)/{rk}))\n"
            f"Bdis disch gnd I = (1-V(x))*min(0.2, max(0, V(disch,gnd))/15) + 1n\n"
            f"Co out gnd 10p\nCd disch gnd 10p\n.ends")
    return Part(name, "timer", {"vsupply_V": vabs, "iout_A": max(isrc, isnk), "theta_ja": thja, "tj_max_C": 150},
                "x", m, text, ("gnd", "trig", "out", "reset", "ctrl", "thres", "disch", "vcc"),
                source=f"{name} datasheet: {tech}, {vsmin:g} to {vsmax:g} V, output {isrc * 1000:g} mA source and "
                       f"{isnk * 1000:g} mA sink, {iq * 1000:.3g} mA supply current",
                params={"vs_min": vsmin, "vs_max": vsmax, "tech": tech})


# --- chargers, references, optocouplers -----------------------------------------------------------------------
CHARGERS = {"TP4056": (1200, 4.2, 1.0, 4.0, 8.0, 0.3e-3, 90, 120, "SOP-8"),
            "MCP73831": (1000, 4.2, 0.5, 3.75, 7.0, 0.5e-3, 230, 110, "SOT-23-5")}


def charger(name: str) -> Part | None:
    key = next((k for k in CHARGERS if name.startswith(k)), None)
    if key is None:
        return None
    k, vfloat, imax, vinmin, vabs, iq, thja, treg, pkg = CHARGERS[key]
    m = f"U_{key}"
    # PROG is held at 1 V, and the charge current is k times the current out of it, tapering at the float voltage
    # STAT (CHRG on a TP4056) sinks to ground through about 30 ohm while it charges, so a status LED lights
    text = (f".subckt {m} vin gnd bat prog stat\nVp pi gnd 1\nVs pi prog 0\n"
            f"B1 vin bat I = min({imax}, {k}*max(0, I(Vs))) * max(0, min(1, ({vfloat}-V(bat,gnd))/0.02)) * "
            f"max(0, min(1, (V(vin,bat)-0.05)/0.05))\nIq vin gnd {_g(iq)}\n"
            f"Bst stat gnd I = max(0, min(0.025, V(stat,gnd)/30)) * max(0, min(1, ({vfloat}-V(bat,gnd))/0.02))\n.ends")
    return Part(name, "charger", {"vin_V": vabs, "iout_A": imax, "theta_ja": thja, "tj_max_C": 150}, "x", m, text,
                ("vin", "gnd", "bat", "prog", "stat"),
                source=f"{key} datasheet: charge current {k} V over the PROG resistor, {vfloat} V float, up to "
                       f"{imax:g} A, thermal regulation at {treg} C, {pkg}",
                params={"k": k, "vfloat": vfloat, "vin_min": vinmin, "t_reg": treg})


def tl431(name: str) -> Part | None:
    if not re.match(r"^(TL431|TL432|AZ431|KA431|LM431)", name):
        return None
    m = "U_TL431"
    text = (".subckt U_TL431 k a ref\nBk k a I = max(0, 5*(V(ref,a)-2.495))*max(0, min(1, V(k,a)-1.0)) + 1u\n"
            "Rref ref a 1meg\n.ends")
    return Part(name, "reference", {"vka_V": 37, "ika_A": 0.1, "theta_ja": 160, "tj_max_C": 150}, "x", m, text,
                ("k", "a", "ref"), source="TL431 datasheet: 2.495 V reference, 1 to 100 mA cathode current, 37 V",
                params={"vref": 2.495, "ik_min": 1e-3})


def opto(name: str) -> Part | None:
    if not re.match(r"^(PC817|EL817|LTV817|4N35|4N25|TLP521|PC123)", name):
        return None
    m = "U_" + name[:5]
    text = (f".subckt {m} a k c e\nD1 a k1 {m}_led\nVs k1 k 0\nF1 0 b Vs 0.01\nQ1 c b e {m}_q\nRb b e 1meg\n"
            f".model {m}_led D(IS=1e-18 N=1.6 RS=2)\n.model {m}_q NPN(IS=1e-14 BF=100 VAF=80 RC=10 CJC=8p "
            f"CJE=10p)\n.ends")
    return Part(name, "opto", {"if_A": 0.05, "vr_V": 6, "vceo_V": 35, "ic_A": 0.05, "pd_W": 0.2, "theta_ja": 500,
                               "tj_max_C": 125}, "x", m, text, ("a", "k", "c", "e"),
                source=f"{name} datasheet: 100% current transfer ratio, LED 50 mA most, 35 V 50 mA output",
                params={"ctr": 1.0})


# --- microcontrollers ---------------------------------------------------------------------------------------------
# name: (least VDD, most operating VDD, absolute VDD, supply current A, output ohm, pin abs A, total A, theta_ja)
MCUS = {
    "ATMEGA328P": (1.8, 5.5, 6.0, 0.012, 40, 0.040, 0.2, 80),
    "ATMEGA2560": (2.7, 5.5, 6.0, 0.030, 40, 0.040, 0.2, 50),
    "ATMEGA32U4": (2.7, 5.5, 6.0, 0.020, 40, 0.040, 0.2, 60),
    "ATTINY85": (2.7, 5.5, 6.0, 0.005, 40, 0.040, 0.06, 100),
    "ESP32": (3.0, 3.6, 3.6, 0.160, 30, 0.040, 1.0, 50),
    "ESP8266": (2.5, 3.6, 3.6, 0.080, 70, 0.012, 0.3, 60),
    "RP2040": (1.62, 3.63, 3.63, 0.025, 170, 0.012, 0.05, 50),
    "STM32F103": (2.0, 3.6, 4.0, 0.030, 50, 0.025, 0.15, 60),
    "STM32F4": (1.8, 3.6, 4.0, 0.050, 50, 0.025, 0.24, 50),
}
MCU_ALIAS = (("ESP32", "ESP32"), ("ESP-WROOM-32", "ESP32"), ("ESP8266", "ESP8266"), ("ESP-12", "ESP8266"),
             ("ESP-01", "ESP8266"), ("ATMEGA328", "ATMEGA328P"), ("ATMEGA2560", "ATMEGA2560"),
             ("ATMEGA32U4", "ATMEGA32U4"), ("ATTINY85", "ATTINY85"), ("ATTINY45", "ATTINY85"), ("RP2040", "RP2040"),
             ("STM32F103", "STM32F103"), ("STM32F4", "STM32F4"), ("STM32F0", "STM32F103"), ("STM32G0", "STM32F103"))
SUPPLY_PIN = re.compile(r"^(vcc|vdd|avcc|avdd|iovdd|vddio|vdd_?io|3v3|vdda|vdd\d*|vcc\d*|vbat|vdd_?3v3|vin)$", re.I)
GROUND_PIN = re.compile(r"^(gnd|vss|agnd|avss|vssa|gnd\d*|vss\d*|ep|pad|gnd_?pad|dgnd)$", re.I)


def mcu_key(name: str) -> str | None:
    u = name.upper()
    return next((k for p, k in MCU_ALIAS if u.startswith(p)), None)


def mcu(name: str) -> Part | None:
    key = mcu_key(name)
    if key is None:
        return None
    vmin, vop, vabs, iact, rout, ipin, itot, thja = MCUS[key]
    return Part(name, "mcu", {"vdd_V": vabs, "vdd_op_V": vop, "pin_A": ipin, "total_A": itot, "theta_ja": thja,
                              "tj_max_C": 125}, "x", f"U_{key}", "", (),
                source=f"{key} datasheet: {vmin:g} to {vop:g} V ({vabs:g} V absolute), about {iact * 1000:g} mA "
                       f"running, pins drive through about {rout} ohm and take {ipin * 1000:g} mA most",
                params={"vmin": vmin, "iact": iact, "rout": rout, "key": key})


def mcu_subckt(p: Part, pins: list[str], modes: dict[str, str], t_off: float | None = None,
               period: float | None = None) -> tuple[str, str]:
    """A subcircuit for one microcontroller as wired: its supply pins draw the running current, outputs drive
    high or low through the pin resistance (or switch on and then off at t_off, or run PWM), inputs are high
    impedance with clamp diodes to the rails. pins are the port names in order; modes say what each GPIO does."""
    vmin, iact, rout = p.params["vmin"], p.params["iact"], p.params["rout"]
    sup = [x for x in pins if SUPPLY_PIN.match(x)] or pins[:1]
    gnd = [x for x in pins if GROUND_PIN.match(x)]
    if not gnd:
        raise ValueError(f"{p.key} has no ground pin among {', '.join(pins)}")
    g0, v0 = gnd[0], sup[0]
    name = f"{p.mname}_{abs(hash((tuple(pins), tuple(sorted(modes.items())), t_off, period))) % 10 ** 8}"
    port = {x: f"p{k}" for k, x in enumerate(pins)}
    lines = [f".subckt {name} " + " ".join(port[x] for x in pins)]
    for k, s in enumerate(sup):
        lines.append(f"Bdd{k} {port[s]} {port[g0]} I = {_g(iact / len(sup))}*min(1, max(0, V({port[s]},{port[g0]})/"
                     f"{vmin}))")
    for k, g in enumerate(gnd[1:]):
        lines.append(f"Rg{k} {port[g]} {port[g0]} 0.01")
    for x in pins:
        if x in sup or x in gnd:
            continue
        px, mode = port[x], modes.get(x, "input")
        lines += [f"D{px}h {px} {port[v0]} DCLAMP", f"D{px}l {port[g0]} {px} DCLAMP"]
        if mode == "input":
            continue
        if mode == "high":
            st = "1"
        elif mode == "low":
            st = "0"
        else:
            if mode.startswith("pwm") and period:
                duty = float(mode.split(":")[1]) if ":" in mode else 0.5
                lines.append(f"Vs{px} s{px} 0 PULSE(0 1 0 {_g(period / 1000)} {_g(period / 1000)} "
                             f"{_g(duty * period)} {_g(period)})")
            else:
                toff = t_off or 1.0
                lines.append(f"Vs{px} s{px} 0 PULSE(1 0 {_g(toff)} {_g(toff / 1000)} {_g(toff / 1000)} 1e3 2e3)")
            st = f"V(s{px})"
        lines += [f"Bh{px} {port[v0]} {px} I = {st}*max(0, V({port[v0]},{px}))/{rout}",
                  f"Bl{px} {px} {port[g0]} I = (1-{st})*max(0, V({px},{port[g0]}))/{rout}"]
    lines += [".model DCLAMP D(IS=1e-14 N=1.0 RS=10)", ".ends"]
    return name, "\n".join(lines)


# --- batteries and sources ----------------------------------------------------------------------------------------
# name: (nominal V, full V, empty V, internal ohm, most continuous A)
BATTERIES = {
    "9V": (9.0, 9.6, 6.0, 1.7, 0.5), "AA": (1.5, 1.6, 1.0, 0.15, 2.0), "AAA": (1.5, 1.6, 1.0, 0.25, 1.0),
    "NIMH": (1.2, 1.4, 1.0, 0.03, 5.0), "CR2032": (3.0, 3.2, 2.0, 15.0, 0.003), "CR2025": (3.0, 3.2, 2.0, 20.0, 0.003),
    "LIION": (3.7, 4.2, 3.0, 0.05, 5.0), "LIPO": (3.7, 4.2, 3.3, 0.03, 10.0), "LIFEPO4": (3.2, 3.65, 2.5, 0.03, 10.0),
    "SLA": (12.0, 13.0, 10.5, 0.03, 10.0), "USB": (5.0, 5.25, 4.75, 0.15, 0.5), "USB-C": (5.0, 5.25, 4.75, 0.1, 3.0),
}


def battery_key(text: str) -> tuple[str, int] | None:
    """A battery and how many cells in series, from words like 9V, CR2032, 18650, 2S LiPo or 4xAA."""
    t = text.upper().replace(" ", "")
    m = re.match(r"^(\d)X?(AA|AAA|NIMH|18650|LIPO|LIION|CR2032)$", t) or re.match(r"^(\d)S(LIPO|LIION|LI-?ION)?$", t)
    n = int(m.group(1)) if m else 1
    body = (m.group(2) or "LIPO") if m else t
    for k, pat in (("CR2032", "CR2032"), ("CR2025", "CR2025"), ("9V", "9VBAT|^9V$|PP3|6LR61"), ("AAA", "AAA"),
                   ("AA", "AA|LR6"), ("NIMH", "NIMH"), ("LIFEPO4", "LIFEPO4|LFP"), ("LIPO", "LIPO"),
                   ("LIION", "18650|21700|LI-?ION|LIION|14500"), ("SLA", "SLA|LEAD"), ("USB-C", "USB-?C|TYPE-?C"),
                   ("USB", "USB|VBUS")):
        if re.search(pat, body):
            return k, n
    return None


def battery(text: str) -> Part | None:
    hit = battery_key(text)
    if not hit:
        return None
    k, n = hit
    vn, vf, ve, ri, imax = BATTERIES[k]
    m = f"BAT_{k.replace('-', '_')}_{n}"
    text_ = f".subckt {m} p n\nV1 pi n {vn * n:g}\nR1 p pi {ri * n:g}\n.ends"
    what = "USB port" if k.startswith("USB") else (f"{n} x {k}" if n > 1 else k)
    return Part(text, "battery", {"iout_A": imax}, "x", m, text_, ("p", "n"),
                source=f"{what}: {vn * n:g} V ({ve * n:g} to {vf * n:g} V), {ri * n:g} ohm inside, "
                       f"{imax:g} A most", params={"v": vn * n, "range": (ve * n, vf * n), "rint": ri * n})


# --- motors, relays, fuses ------------------------------------------------------------------------------------------
def motor(volts: float = 6.0, i_noload: float = 0.15, i_stall: float = 1.5, rpm: float = 6000.0, tau: float = 0.05,
          name: str = "motor") -> Part:
    r = volts / i_stall
    w = rpm * 2 * math.pi / 60
    ke = (volts - i_noload * r) / w
    b = ke * i_noload / w
    j = tau * ke * ke / r
    m = f"MOT_{int(volts * 10)}_{int(i_stall * 100)}"
    text = (f".subckt {m} p n\nRa p a {_g(r)}\nLa a b {_g(r * 1e-3)}\nBe b c V = {_g(ke)}*V(w)\nVs c n 0\n"
            f"Bt 0 w I = {_g(ke)}*I(Vs)\nCj w 0 {_g(j)}\nRf w 0 {_g(1 / b)}\n.ends")
    return Part(name, "motor", {"i_stall_A": i_stall}, "x", m, text, ("p", "n"),
                source=f"DC motor {volts:g} V, {i_noload:g} A free running and {i_stall:g} A stalled, {rpm:g} rpm",
                generic=True, params={"v": volts, "i_noload": i_noload})


def relay(volts: float = 5.0, ohms: float | None = None, name: str = "relay") -> Part:
    r = ohms or {3: 25, 5: 70, 6: 100, 9: 225, 12: 400, 24: 1600}.get(int(volts), 70 * (volts / 5) ** 2)
    m = f"RLY_{int(volts)}"
    text = f".subckt {m} p n\nRc p a {_g(r)}\nLc a n {_g(r * 3e-3)}\n.ends"
    return Part(name, "relay", {"vcoil_V": volts * 1.5}, "x", m, text, ("p", "n"),
                source=f"relay coil {volts:g} V, {r:g} ohm, 3 ms time constant", generic=ohms is None,
                params={"v": volts, "inductive": True})


def fuse(amps: float, ptc: bool = False, name: str = "fuse") -> Part:
    r = (0.6 / amps if ptc else 0.1 / amps)
    return Part(name, "fuse", {"ihold_A": amps}, "", "", "", source=f"{'resettable ' if ptc else ''}fuse rated "
                f"{amps:g} A, {r:.3g} ohm", params={"r": r, "ptc": ptc})


@builder
def _ics(name: str) -> Part | None:
    u = name.upper()
    for fn in (regulator, opamp, timer, charger, tl431, opto, mcu):
        p = fn(u)
        if p is not None:
            if fn is not regulator:
                p.key = name
            return p
    return None
