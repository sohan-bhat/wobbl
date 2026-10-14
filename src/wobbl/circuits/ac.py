"""Frequency response, and loop stability wherever there is feedback.

The response is an AC sweep from the circuit's signal source (one with an AC value, a sine or a pulse) to its
output (a probe, a net named out or vout, or an op amp's output), with the passband gain and the -3 dB
corners, checked against 1 / (2 pi R C) when the output is a single RC.

Stability is the loop gain of every op amp, found the way Bode defined it: the op amp's gain stage is cut for
the sweep only (it keeps its DC operating point through a node that no AC reaches), a 1 A AC test current goes
into the stage's node, and the loop gain is minus the current the stage would have returned, -gm (v+ - v-).
Where it falls through 1 is the crossover, the phase there gives the phase margin, and where the phase passes
-180 degrees the gain margin.
"""
from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np

from ..units import fmt
from ..verify.checks import Check
from .netlist import Circuit
from .solve import OPTIONS, materialize, ngspice

OUT_NAMES = ("out", "vout", "output", "amp_out", "o", "vo", "filter_out", "audio_out")


def _source(c: Circuit):
    for e in c.elements:
        if e.letter == "v" and re.search(r"\bac\b", " ".join(e.rest).lower()):
            return e, False
    for e in c.elements:
        if e.letter == "v" and e.kind == "signal":
            return e, True
    return None, False


def _output(c: Circuit, probes) -> str | None:
    for pr in probes or []:
        m = re.match(r"v\((\w+)\)", pr.lower())
        if m:
            return m.group(1)
    names = {n: c.label(n).lower().lstrip("/") for n in c.nodes()}
    for want in OUT_NAMES:
        for n, lab in names.items():
            if lab == want or n == want:
                return n
    amp = next((e for e in c.elements if e.kind == "opamp"), None)
    return amp.nodes[amp.pins.index("out")] if amp else None


def _sweep(c: Circuit, work: Path, fmin: float, fmax: float, tag: str) -> dict:
    cc = materialize(c, "op")
    net = cc.spice(instrument=False, extra=[OPTIONS])
    pts = 30
    return ngspice().run_raw(net, work, [f"ac dec {pts} {fmin:.4g} {fmax:.4g}"], tag)[0][0]


def analyze(c: Circuit, work: Path, op, question: str = "", probes=()) -> dict:
    checks, values = [], {}
    gbw = max([e.part.get("params", {}).get("gbw", 0) for e in c.elements if e.kind == "opamp"] or [0])
    fmax = max(10 * gbw, 1e7) if gbw else 1e8
    src, add_ac = _source(c)
    out = _output(c, probes)
    if src is not None and out is not None:
        cc = c.copy()
        s = cc.element(src.name)
        if add_ac:
            s.rest = [*s.rest, "AC", "1"]
        d = _sweep(cc, work, 0.1, fmax, "ac")
        f = np.real(d["frequency"])
        vin = d.get(f"v({src.nodes[0]})", np.ones_like(f)) - (d.get(f"v({src.nodes[1]})", 0) if src.nodes[1] != "0"
                                                              else 0)
        h = d.get(f"v({out})")
        if h is not None:
            H = h / np.where(np.abs(vin) > 0, vin, 1)
            values["response"] = _response(f, H)
            r = values["response"]
            text = (f"from {src.ref} to {c.label(out)}: passband gain {fmt(r['gain'])} ({fmt(r['gain_dB'])} dB)"
                    + (f", -3 dB at {', '.join(fmt(x) + ' Hz' for x in r['f3db'])}" if r["f3db"] else "")
                    + f", a {r['type']} response")
            checks.append(Check("frequency response", "info", text, r))
            hc = _rc_check(c, out, r)
            if hc:
                checks.append(hc)
            values["bode"] = {"f": f[::3].tolist(), "gain_dB": (20 * np.log10(np.abs(H[::3]) + 1e-30)).tolist()}
    for e in c.elements:
        if e.kind != "opamp" or not e.part.get("params", {}).get("gm"):
            continue
        try:
            st = loop_gain(c, e, work, fmax)
        except Exception as ex:  # noqa: BLE001
            checks.append(Check("loop stability", "skip", f"{e.ref}: {ex}"))
            continue
        values.setdefault("stability", {})[e.ref] = st
        checks.append(_stability_check(e.ref, st))
    return {"checks": checks, "values": values}


def _response(f, H) -> dict:
    mag = np.abs(H)
    k = int(np.argmax(mag))
    peak = float(mag[k])
    ref = peak / math.sqrt(2)
    crosses = []
    for i in range(len(f) - 1):
        a, b = mag[i] - ref, mag[i + 1] - ref
        if a == 0 or a * b < 0:
            t = a / (a - b) if a != b else 0
            crosses.append(float(10 ** (math.log10(f[i]) + t * (math.log10(f[i + 1]) - math.log10(f[i])))))
    low, high = mag[0] >= 0.9 * peak, mag[-1] >= 0.9 * peak
    kind = "flat" if low and high else "low pass" if low else "high pass" if high else "band pass"
    return {"gain": peak, "gain_dB": 20 * math.log10(peak + 1e-30), "f3db": crosses[:3], "type": kind,
            "f_peak": float(f[k]), "phase_deg_at_peak": float(np.degrees(np.angle(H[k])))}


def _rc_check(c: Circuit, out: str, r: dict) -> Check | None:
    """A single R and C at the output, one to the input side and one to ground: fc = 1 / (2 pi R C)."""
    on = [e for e, _ in c.nets().get(out, [])]
    rs = [e for e in on if e.letter == "r" and e.value]
    cs = [e for e in on if e.letter == "c" and e.value]
    if len(rs) != 1 or len(cs) != 1 or len(on) != 2 or len(r["f3db"]) != 1:
        return None
    fc = 1 / (2 * math.pi * rs[0].value * cs[0].value)
    err = abs(r["f3db"][0] - fc) / fc
    st = "pass" if err < 0.01 else "warn" if err < 0.05 else "fail"
    return Check("hand calculation", st, f"{rs[0].ref} and {cs[0].ref}: 1 / (2 pi R C) = {fmt(fc)} Hz by hand, the sweep "
                                         f"falls 3 dB at {fmt(r['f3db'][0])} Hz ({100 * err:.2g}% apart)",
                 {"shape": "rc corner"})


def loop_gain(c: Circuit, e, work: Path, fmax: float) -> dict:
    """Bode's return ratio of one op amp's gain stage: cut for AC only, driven with a 1 A test current."""
    cc = c.copy()
    me = cc.element(e.name)
    sub = cc.subckts[me.model.lower()]
    gm = e.part["params"]["gm"]
    name = sub.name + "_LOOP"
    text = sub.text.replace(f".subckt {sub.name} ", f".subckt {name} ", 1)
    text = re.sub(r"^(B1 0 n1 I = .*?)V\(inp,inn\)", r"\1V(vdc)", text, flags=re.M)
    text = text.replace("\n.ends", "\nEx vx 0 inp inn 1\nRx vx vdc 1\nCx vdc 0 1e12\nIt 0 n1 DC 0 AC 1\n.ends")
    from .netlist import Subckt
    cc.subckts[name.lower()] = Subckt(name, sub.ports, text)
    bare = [k for k, t in enumerate(me.rest) if "=" not in t and t.lower() != "params:"]
    me.rest[bare[-1]] = name
    for s in cc.elements:          # the loop gain is a small signal property: sources hold their DC values
        if s.letter in "vi" and re.search(r"\bac\b", " ".join(s.rest).lower()):
            s.rest = [re.sub(r"\bAC\s+\S+", "", " ".join(s.rest), flags=re.I).strip()]
    d = _sweep(cc, work, 0.1, fmax, f"loop_{e.name.lower()}")
    f = np.real(d["frequency"])
    pin = dict(zip(e.pins, e.nodes))
    vp = d.get(f"v({pin['inp']})", np.zeros_like(f)) if pin["inp"] != "0" else 0
    vn = d.get(f"v({pin['inn']})", np.zeros_like(f)) if pin["inn"] != "0" else 0
    T = -gm * (vp - vn)
    mag, ph = np.abs(T), np.unwrap(np.angle(T))
    if ph[0] > math.pi / 2:
        ph = ph - 2 * math.pi
    out = {"dc_loop_gain": float(mag[0]), "f": f[::4].tolist(), "T_dB": (20 * np.log10(mag[::4] + 1e-30)).tolist()}
    below = np.nonzero((mag[:-1] >= 1) & (mag[1:] < 1))[0]
    if mag[0] < 1 or not len(below):
        out.update(crossover_Hz=None, phase_margin_deg=None)
        return out
    i = int(below[0])
    t = math.log10(mag[i]) / (math.log10(mag[i]) - math.log10(mag[i + 1]))
    fc = 10 ** (math.log10(f[i]) + t * (math.log10(f[i + 1]) - math.log10(f[i])))
    phc = ph[i] + t * (ph[i + 1] - ph[i])
    out["crossover_Hz"] = float(fc)
    out["phase_margin_deg"] = float(180 + math.degrees(phc))
    n180 = np.nonzero((ph[:-1] > -math.pi) & (ph[1:] <= -math.pi))[0]
    out["gain_margin_dB"] = float(-20 * math.log10(mag[int(n180[0])] + 1e-30)) if len(n180) else None
    return out


def attach(stress, res: dict) -> None:
    """A loop's stability is a property of the circuit, so it joins the op amp's own items: a margin under 20
    degrees means it does not work as built, under 45 that it rings."""
    from .stress import Item
    for ref, st in (res.get("values", {}).get("stability") or {}).items():
        pm = st.get("phase_margin_deg")
        s = next((x for x in stress if x.ref == ref), None)
        if s is None or pm is None:
            continue
        status = "fails" if pm < 20 else "warn" if pm < 45 else "ok"
        note = ("it oscillates or rings hard" if pm < 20 else "it rings and overshoots on steps" if pm < 45 else
                "stable")
        s.items.append(Item("phase margin", pm, 45.0, "deg", None, status,
                            f"{pm:.0f} degrees of phase margin at {fmt(st['crossover_Hz'])} Hz: {note}"))
    for ch in res.get("checks", []):
        if ch.name == "loop stability" and ch.status in ("warn", "fail"):
            ch.status = "info"         # judged on the part above; the check itself solved fine


def _stability_check(ref: str, st: dict) -> Check:
    pm = st.get("phase_margin_deg")
    if pm is None:
        return Check("loop stability", "info", f"{ref}: the loop gain never crosses 1 (no negative feedback around it, "
                                               "or a comparator), so there is no margin to judge", st)
    gm = st.get("gain_margin_dB")
    text = (f"{ref}: loop gain {fmt(st['dc_loop_gain'])} at low frequency crosses 1 at {fmt(st['crossover_Hz'])} Hz "
            f"with {pm:.0f} degrees of phase margin" + (f" and {gm:.0f} dB of gain margin" if gm else ""))
    if pm < 20:
        return Check("loop stability", "fail", text + "; it will ring hard or oscillate", st)
    if pm < 45:
        return Check("loop stability", "warn", text + "; expect overshoot and ringing on steps", st)
    return Check("loop stability", "pass", text, st)
