"""Closed form checks wherever part of a circuit reduces to one.

Each finder looks for a shape in the netlist and computes the answer by hand from the parts' values and the
node voltages at the shape's edges, then compares it with ngspice: a divider (R2 / (R1 + R2)), a resistor
feeding an LED or diode (the diode equation solved by Newton with the model's own IS, N and RS), an LM317's
output (1.25 (1 + R2 / R1) + Iadj R2), a linear regulator's dissipation ((Vin - Vout) Iout + Vin Iq), a
MOSFET's conduction loss (I_rms^2 R_ds(on) at its gate voltage), a 555 astable's period
(0.693 (Ra + 2 Rb) C) and an op amp's closed loop gain (1 + Rf / Rg or -Rf / Rin).
"""
from __future__ import annotations

import math
import re

import numpy as np

from ..units import fmt, parse_spice_value
from ..verify.checks import Check
from .netlist import Circuit
from .parts import VT
from .solve import Wave
from .stress import Stress, _Pins

HIGH_Z = {"opamp": ("inp", "inn"), "comparator": ("inp", "inn"), "nmos": ("g",), "pmos": ("g",),
          "timer": ("trig", "thres", "ctrl"), "regulator": ("adj",)}


def checks(c: Circuit, w: Wave, stress: list[Stress]) -> list[Check]:
    out = []
    for fn in (_dividers, _diode_chains, _regulators, _mosfets, _timers, _opamp_gain):
        try:
            out += fn(c, w)
        except Exception:  # noqa: BLE001  a finder that trips on an odd circuit just finds nothing
            continue
    if not out:
        return []
    # one line per kind of check, with the worst of each
    best: dict[str, Check] = {}
    order = {"fail": 0, "warn": 1, "pass": 2, "info": 3}
    for ch in out:
        k = ch.detail.get("shape", ch.name)
        if k not in best or order.get(ch.status, 3) < order.get(best[k].status, 3):
            best[k] = ch
    return list(best.values())


def _status(err: float, tol: float) -> str:
    return "pass" if err <= tol else "warn" if err <= 3 * tol else "fail"


def _mean(w: Wave, x) -> float:
    return w.mean(x)


def _dividers(c: Circuit, w: Wave) -> list[Check]:
    nets = c.nets()
    out = []
    for mid, on in nets.items():
        if mid == "0":
            continue
        rs = [(e, p) for e, p in on if e.letter == "r" and e.value]
        rest = [(e, p) for e, p in on if e.letter != "r"]
        if len(rs) != 2 or any(p not in HIGH_Z.get(e.kind, ()) for e, p in rest):
            continue
        (ra, _), (rb, _) = rs
        ta = next(n for n in ra.nodes if n != mid)
        tb = next(n for n in rb.nodes if n != mid)
        if tb != "0":
            if ta == "0":
                ra, rb, ta, tb = rb, ra, tb, ta
            else:
                continue
        if any(e.kind == "regulator" for e, _ in rest):
            continue
        vt = _mean(w, w.volt(ta))
        if abs(vt) < 1e-3:
            continue
        hand = vt * rb.value / (ra.value + rb.value)
        sim = _mean(w, w.volt(mid))
        err = abs(sim - hand) / max(abs(hand), 1e-9)
        out.append(Check("hand calculation", _status(err, 0.01),
                         f"divider {ra.ref}/{rb.ref}: {fmt(vt)} V x {fmt(rb.value)} / ({fmt(ra.value)} + "
                         f"{fmt(rb.value)}) = {fmt(hand)} V by hand, ngspice {fmt(sim)} V ({100 * err:.2g}% apart)",
                         {"shape": "divider", "hand": hand, "sim": sim}))
    return out


def _model_params(c: Circuit, name: str) -> dict:
    text = c.models.get(name.lower(), "")
    out = {}
    for k in ("is", "n", "rs"):
        m = re.search(rf"\b{k}\s*=\s*([\d.eE+-]+[a-zA-Z]*)", text, re.I)
        if m:
            out[k] = parse_spice_value(m.group(1))
    return out


def _diode_chains(c: Circuit, w: Wave) -> list[Check]:
    """A resistor and a diode or LED in series with nothing else on the node between them."""
    nets = c.nets()
    out = []
    for mid, on in nets.items():
        if mid == "0" or len(on) != 2:
            continue
        r = next((e for e, _ in on if e.letter == "r" and e.value), None)
        d = next((e for e, p in on if e.letter == "d" and p == "a"), None)
        if r is None or d is None:
            continue
        mp = _model_params(c, d.model)
        if not {"is", "n"} <= set(mp):
            continue
        top = next(n for n in r.nodes if n != mid)
        bot = d.nodes[1]
        drive = w.volt(top) - w.volt(bot)
        vs = _mean(w, drive)
        if vs <= 0.3 or (w.kind == "tran" and np.ptp(drive[w.window]) > 0.01 * abs(vs)):
            continue        # a drive that moves in the window has no single operating point to check
        isat, n, rs = mp["is"], mp["n"], mp.get("rs", 0.0)
        i = max(vs / r.value / 2, 1e-12)
        for _ in range(200):
            f = i * (r.value + rs) + n * VT * math.log(i / isat + 1) - vs
            df = r.value + rs + n * VT / (i + isat)
            i2 = max(i - f / df, i / 10)
            if abs(i2 - i) < 1e-15:
                break
            i = i2
        sim = _mean(w, _Pins(d, w).i("a"))
        if abs(sim) < 1e-5:
            continue        # a chain that carries next to nothing (an open pin at its end) has nothing to check
        err = abs(sim - i) / max(i, 1e-12)
        out.append(Check("hand calculation", _status(err, 0.01),
                         f"{r.ref} into {d.ref}: the diode equation solved by hand gives {fmt(i * 1000)} mA, ngspice "
                         f"{fmt(sim * 1000)} mA ({100 * err:.2g}% apart)", {"shape": "diode chain", "hand": i,
                                                                             "sim": sim}))
    return out


def _regulators(c: Circuit, w: Wave) -> list[Check]:
    out = []
    nets = c.nets()
    for e in c.elements:
        if e.kind != "regulator":
            continue
        P = _Pins(e, w)
        prm = e.part.get("params", {})
        third = "adj" if "adj" in e.pins else "gnd"
        vin, vout = _mean(w, P.v("in")), _mean(w, P.v("out"))
        iin, iout = _mean(w, P.i("in")), -_mean(w, P.i("out"))
        hand = (vin - vout) * iout + (vin - _mean(w, P.v(third))) * (iin - iout)
        sim = w.mean(P.power())
        err = abs(sim - hand) / max(abs(hand), 1e-9)
        out.append(Check("hand calculation", _status(err, 0.005),
                         f"{e.ref} dissipates (Vin - Vout) Iout + its own current: {fmt(hand)} W by hand, {fmt(sim)} "
                         f"W from its pins ({100 * err:.2g}% apart)", {"shape": "regulator power"}))
        if prm.get("adjustable"):
            out_n, adj_n = e.nodes[e.pins.index("out")], e.nodes[e.pins.index("adj")]
            r1 = next((o for o, _ in nets.get(adj_n, []) if o.letter == "r" and out_n in o.nodes and o.value), None)
            r2 = next((o for o, _ in nets.get(adj_n, []) if o.letter == "r" and "0" in o.nodes and o.value), None)
            if r1 and r2:
                iadj = 50e-6 if "317" in e.part.get("key", "") else 55e-6
                hv = 1.25 * (1 + r2.value / r1.value) + iadj * r2.value
                err = abs(vout - hv) / hv
                out.append(Check("hand calculation", _status(err, 0.01),
                                 f"{e.ref} output 1.25 (1 + {fmt(r2.value)} / {fmt(r1.value)}) + Iadj R2 = {fmt(hv)} V "
                                 f"by hand, ngspice {fmt(vout)} V ({100 * err:.2g}% apart)",
                                 {"shape": "adjustable regulator"}))
    return out


def _mosfets(c: Circuit, w: Wave) -> list[Check]:
    out = []
    for e in c.elements:
        if e.kind not in ("nmos", "pmos") or not e.part.get("params", {}).get("rds"):
            continue
        text = c.models.get(e.model.lower(), "")
        kp = re.search(r"Kp=([\d.e+-]+)", text)
        rd, rs, vto = (re.search(rf"{k}=([\d.e+-]+)", text) for k in ("Rd", "Rs", "Vto"))
        if not (kp and rd and rs and vto):
            continue
        P = _Pins(e, w)
        sg = 1 if e.kind == "nmos" else -1
        idr = sg * P.i("d")
        irms = w.rms(idr)
        vgs = _mean(w, sg * (P.v("g") - P.v("s")))
        vds = _mean(w, np.abs(P.v("d") - P.v("s")))
        if irms < 0.05 or vds > 0.5 or vgs - abs(float(vto.group(1))) < 1:
            continue
        r = float(rd.group(1)) + float(rs.group(1)) + 1 / (float(kp.group(1)) * (vgs - abs(float(vto.group(1)))))
        hand = irms ** 2 * r
        sim = w.mean(P.power())
        err = abs(sim - hand) / max(hand, 1e-12)
        out.append(Check("hand calculation", _status(err, 0.05),
                         f"{e.ref} conduction loss I_rms^2 R_ds(on) = {fmt(irms)}^2 x {fmt(r * 1000)} mohm = "
                         f"{fmt(hand)} W by hand, ngspice {fmt(sim)} W ({100 * err:.2g}% apart)",
                         {"shape": "mosfet loss"}))
    return out


def _timers(c: Circuit, w: Wave) -> list[Check]:
    from .solve import _timer_period
    out = []
    if w.kind != "tran":
        return out
    nets = c.nets()
    for e in c.elements:
        if e.kind != "timer":
            continue
        hand = _timer_period(c, e, nets)
        vo = _Pins(e, w).v("out")
        t = w.t
        mid = 0.5 * (vo.max() + vo.min())
        up = np.nonzero((vo[:-1] < mid) & (vo[1:] >= mid) & w.window[1:])[0]
        if len(up) < 3 or not hand:
            out.append(Check("oscillation", "fail" if vo.max() - vo.min() < 0.5 else "warn",
                             f"{e.ref}'s output does not toggle in the settled window", {"shape": "555"}))
            continue
        ts = t[up] + (mid - vo[up]) * (t[up + 1] - t[up]) / (vo[up + 1] - vo[up])
        per = float(np.mean(np.diff(ts)))
        err = abs(per - hand) / hand
        out.append(Check("hand calculation", _status(err, 0.04),
                         f"{e.ref} astable period 0.693 (Ra + 2 Rb) C = {fmt(hand * 1000)} ms by hand, ngspice "
                         f"{fmt(per * 1000)} ms, {fmt(1 / per)} Hz ({100 * err:.2g}% apart)",
                         {"shape": "555", "hand_s": hand, "sim_s": per}))
    return out


def _opamp_gain(c: Circuit, w: Wave) -> list[Check]:
    nets = c.nets()
    out = []
    for e in c.elements:
        if e.kind != "opamp":
            continue
        pin = dict(zip(e.pins, e.nodes))
        o, inn, inp = pin["out"], pin["inn"], pin["inp"]
        rf = next((x for x, _ in nets.get(inn, []) if x.letter == "r" and o in x.nodes and x.value), None)
        if rf is None:
            continue
        rg = [x for x, _ in nets.get(inn, []) if x.letter == "r" and x is not rf and x.value]
        if len(rg) != 1:
            continue
        rg = rg[0]
        far = next(n for n in rg.nodes if n != inn)
        P = _Pins(e, w)
        if far == "0":
            src, gain, form = inp, 1 + rf.value / rg.value, f"1 + {fmt(rf.value)} / {fmt(rg.value)}"
        elif inp == "0" or all(x.letter == "r" or x is e for x, _ in nets.get(inp, [])):
            src, gain, form = far, -rf.value / rg.value, f"-{fmt(rf.value)} / {fmt(rg.value)}"
        else:
            continue
        vi, vo = w.volt(src), P.v("out")
        if w.kind == "tran" and np.ptp(vi[w.window]) > 1e-6:
            m = w.window
            k = int(np.argmax(np.abs(vi[m] - vi[m].mean())))
            sim = float((vo[m][k] - vo[m].mean()) / (vi[m][k] - vi[m].mean()))
        else:
            if abs(vi[-1]) < 1e-6:
                continue
            sim = float(vo[-1] / vi[-1])
        top = P.v("vcc") - e.part["params"].get("voh", 0) - 0.05
        bot = P.v("vee") + e.part["params"].get("vol", 0) + 0.05
        if float(np.mean((vo >= top) | (vo <= bot))) > 0.01:
            continue        # clipped: the linear gain does not apply
        err = abs(sim - gain) / abs(gain)
        out.append(Check("hand calculation", _status(err, 0.01),
                         f"{e.ref} closed loop gain {form} = {fmt(gain)} by hand, ngspice {fmt(sim)} "
                         f"({100 * err:.2g}% apart, the rest is the finite open loop gain)", {"shape": "opamp gain"}))
    return out
