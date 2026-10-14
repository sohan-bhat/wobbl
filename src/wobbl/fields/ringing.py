"""Switch node ringing of a buck converter once the board's parasitics are in the schematic.

The ideal buck schematic has no inductance in its hot loop (input capacitor,
switch, catch diode) and no capacitance at the switch node, so its switch node
is a clean square wave. A real board adds both: the loop inductance of the
copper (a board study gives it as loop_inductance) and the capacitance of the
switch node's copper, the diode and the switch (switch_capacitance). This
module simulates a few switching cycles with ngspice from the steady state,
with and without them, at a step fine enough for the ringing, and measures
the overshoot and the ringing frequency, which it checks against
1 / (2 pi sqrt(L C)). It also tries the usual fix, an RC snubber across the
diode sized from the measured ring, and reports what it buys.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from ..units import fmt
from ..verify.checks import Check

KEYS = ("loop_inductance", "switch_capacitance", "board")


def wanted(p: dict) -> bool:
    return any(p.get(k) for k in KEYS)


def parasitics(p: dict) -> tuple[float, float, float, str]:
    """Loop inductance, switch node capacitance and loop resistance, from the values or a board result folder."""
    from ..circuits.templates import val

    src = "given"
    lp = val(p, "loop_inductance", "0", "inductance")
    cs = val(p, "switch_capacitance", "0", "capacitance")
    if p.get("board"):
        d = Path(str(p["board"]))
        f = d / "outcome.json" if d.is_dir() else d
        if not f.exists():
            raise ValueError(f"no board result at {p['board']}. Run 'wobbl board microstrip --set length=20mm' first")
        v = json.loads(f.read_text()).get("values", {})
        if "loop_inductance_H" not in v:
            raise ValueError(f"{p['board']} has no loop inductance. Give the board study a length")
        lp = lp or float(v["loop_inductance_H"])
        cs = cs + float(v.get("switch_capacitance_F", 0.0))
        src = f"from the board study in {d}"
    cs = cs + val(p, "diode_capacitance", "100p", "capacitance")
    rl = val(p, "loop_resistance", "0.05", "resistance")
    return lp, cs, rl, src


def _netlist(p: dict, lp: float, cs: float, rl: float, snub: tuple[float, float] | None = None) -> str:
    from ..circuits.templates import val

    vin, vout, iout, f = p["vin"], p["vout"], p["current"], p["frequency"]
    period = 1 / f
    ton = p["duty"] * period
    rds = val(p, "rds", "0.05", "resistance")
    lines = ["* Buck power stage with the board's hot loop inductance and switch node capacitance",
             f"V1 vsrc 0 DC {vin}"]
    if lp:
        lines += [f"LP vsrc vl {lp:.6e} IC=0", f"RP vl vin {rl:.6e}"]
    else:
        lines += ["VP vsrc vin DC 0"]
    lines += [f"VG g 0 PULSE(0 5 0 {period / 200:.4e} {period / 200:.4e} {ton:.6e} {period:.6e})",
              "S1 vin sw g 0 SWMOD", f".model SWMOD SW(VT=2.5 VH=0.1 RON={rds} ROFF=1e6)",
              "D1 0 sw SCHOTTKY", ".model SCHOTTKY D(IS=5e-5 N=1.05 RS=0.02)",
              f"L1 sw out {p['inductor']} IC={iout}", f"C1 out cmid {p['capacitor']} IC={vout}",
              f"R2 cmid 0 {p['esr']}", f"R1 out 0 {vout / iout}"]
    if cs:
        lines += [f"CS sw 0 {cs:.6e} IC=0", f"CO vin sw {cs:.6e} IC={vin}"]
    if snub:
        lines += [f"RSN sw sn {snub[0]:.6e}", f"CSN sn 0 {snub[1]:.6e} IC=0"]
    return "\n".join(lines) + "\n.end\n"


def _run(p: dict, net: str, work: Path, name: str, step: float, cycles: int = 4) -> dict:
    from ..solvers import base

    period = 1 / p["frequency"]
    stop = cycles * period
    start = stop - 1.2 * period
    body = net.rstrip()[:-4].rstrip() + f"\n.tran {step:.4e} {stop:.6e} {start:.6e} {step:.4e} uic\n.end\n"
    sp = base.for_domain("circuit")
    data, _, _ = sp.run(body, work, ["v(sw)", "v(vin)"], name)
    return data


def _ring(t: np.ndarray, v: np.ndarray, vin: float, period: float) -> dict:
    """Peak of the switch node and the ringing frequency right after the turn on edge, where v(sw) rises."""
    peak = float(v.max())
    i_on = int(np.argmax(v > 0.5 * vin))
    seg_t, seg_v = t[i_on:], v[i_on:] - vin
    end = seg_t[0] + 0.3 * period
    m = seg_t < end
    seg_t, seg_v = seg_t[m], seg_v[m]
    zc = [seg_t[i] for i in range(1, len(seg_v)) if seg_v[i - 1] < 0 <= seg_v[i] or seg_v[i - 1] >= 0 > seg_v[i]]
    f = 0.0
    if len(zc) >= 4:
        f = (len(zc) - 1) / (2 * (zc[-1] - zc[0])) if len(zc) < 12 else 5 / (zc[10] - zc[0])
    return {"peak": peak, "freq": f, "t_on": float(t[i_on])}


def ringing(p: dict, work: Path, out: Path) -> dict:
    from ..report.render import line_plot

    lp, cs, rl, src = parasitics(p)
    period = 1 / p["frequency"]
    vin = p["vin"]
    f_cl = 1 / (2 * math.pi * math.sqrt(lp * cs)) if lp and cs else 0.0
    step = min(period / 4000, 1 / (60 * f_cl)) if f_cl else period / 4000
    ideal = _run(p, _netlist(p, 0.0, 0.0, 0.0), work, "ring_ideal", step)
    par = _run(p, _netlist(p, lp, cs, rl), work, "ring_board", step)
    par2 = _run(p, _netlist(p, lp, cs, rl), work, "ring_board_half", step / 2)
    r1 = _ring(par["time"], par["v(sw)"], vin, period)
    r2 = _ring(par2["time"], par2["v(sw)"], vin, period)
    ri = _ring(ideal["time"], ideal["v(sw)"], vin, period)
    over = r1["peak"] - vin
    checks = []
    d = abs(r1["peak"] - r2["peak"]) / max(abs(r2["peak"]), 1e-9)
    checks.append(Check("ringing time step", "pass" if d < 0.01 else ("warn" if d < 0.03 else "fail"),
                        f"the switch node peaks at {fmt(r1['peak'])} V with the fine step and {fmt(r2['peak'])} V "
                        f"with half of it ({100 * d:.2g}% apart)"))
    if f_cl and r1["freq"]:
        df = abs(r1["freq"] - f_cl) / f_cl
        checks.append(Check("ringing frequency", "pass" if df < 0.05 else ("warn" if df < 0.15 else "fail"),
                            f"the switch node rings at {fmt(r1['freq'] / 1e6)} MHz after turn on, and "
                            f"1 / (2 pi sqrt(L C)) with {fmt(lp * 1e9)} nH and {fmt(cs * 1e12)} pF gives "
                            f"{fmt(f_cl / 1e6)} MHz ({100 * df:.2g}% apart)", {"closed_form_Hz": f_cl}))
    vf = 0.35
    bound = 2 * vin + vf
    checks.append(Check("overshoot bound", "pass" if r1["peak"] <= bound * 1.01 else "warn",
                        f"the peak of {fmt(r1['peak'])} V stays under the lossless limit 2 Vin + Vf = {fmt(bound)} V"))
    snub_c = 3 * cs if cs else 0.0
    snub_r = math.sqrt(lp / cs) if lp and cs else 0.0
    sn = {}
    if snub_c:
        rs = _run(p, _netlist(p, lp, cs, rl, (snub_r, snub_c)), work, "ring_snubber", step)
        sn = _ring(rs["time"], rs["v(sw)"], vin, period)
    rating = float(p.get("switch_rating", 0) or 0)
    verdict = None
    if rating:
        verdict = "ok" if r1["peak"] <= 0.8 * rating else ("marginal" if r1["peak"] <= rating else "fails")
    head = (f"With {fmt(lp * 1e9)} nH of hot loop inductance and {fmt(cs * 1e12)} pF at the switch node the switch "
            f"node overshoots to {fmt(r1['peak'])} V ({fmt(over)} V above the {fmt(vin)} V input) and rings at "
            f"{fmt(r1['freq'] / 1e6)} MHz, where the ideal schematic peaks at {fmt(ri['peak'])} V.")
    if rating:
        head += f" That is {100 * r1['peak'] / rating:.0f}% of the {fmt(rating)} V switch rating."
    fixes = []
    if sn:
        p_loss = 0.5 * snub_c * vin ** 2 * p["frequency"] * 2
        fixes.append({"kind": "snubber", "title": f"Add an RC snubber of {fmt(snub_r)} ohm and {fmt(snub_c * 1e12)} pF "
                                                  "across the catch diode",
                      "summary": f"Simulated with it, the switch node peaks at {fmt(sn['peak'])} V instead of "
                                 f"{fmt(r1['peak'])} V, and the resistor burns about {fmt(p_loss * 1e3)} mW.",
                      "auto": False})
    fixes.append({"kind": "layout", "title": "Shrink the hot loop",
                  "summary": "Put the input capacitor right next to the switch and diode, with the return on the "
                             "layer just below, so the loop inductance and the overshoot both fall.", "auto": False})
    tw = ideal["time"]
    zoom_i = (tw >= ri["t_on"] - 0.02 * period) & (tw <= ri["t_on"] + 0.25 * period)
    tp = par["time"]
    zoom_p = (tp >= r1["t_on"] - 0.02 * period) & (tp <= r1["t_on"] + 0.25 * period)
    n = min(zoom_i.sum(), zoom_p.sum())
    t_ns = (tp[zoom_p][:n] - tp[zoom_p][0]) * 1e9
    ys = {"with board parasitics": par["v(sw)"][zoom_p][:n],
          "ideal schematic": np.interp(tp[zoom_p][:n] - r1["t_on"] + ri["t_on"], tw, ideal["v(sw)"])}
    img = line_plot(out / "images" / "ringing.png", t_ns, ys, "ns after turn on", "switch node, V")
    values = {"loop_inductance_H": lp, "switch_capacitance_F": cs, "loop_resistance_ohm": rl, "sw_peak_V": r1["peak"],
              "sw_peak_ideal_V": ri["peak"], "ring_Hz": r1["freq"], "ring_closed_form_Hz": f_cl,
              "snubber": {"R_ohm": snub_r, "C_F": snub_c, "sw_peak_V": sn.get("peak")} if sn else None}
    assumptions = [f"Board parasitics {src}, {fmt(lp * 1e9)} nH of loop inductance with {fmt(rl * 1e3)} mohm, and "
                   f"{fmt(cs * 1e12)} pF at the switch node (including 100 pF of diode junction unless "
                   "diode_capacitance says otherwise), the same across the open switch.",
                   "The diode has no reverse recovery and the switch turns on and off instantly, so the measured "
                   "ringing comes from the parasitics alone."]
    return {"checks": checks, "values": values, "images": {"ringing": img}, "fixes": fixes, "headline": head,
            "assumptions": assumptions, "verdict": verdict, "metric": ("switch node peak", f"{fmt(r1['peak'])} V")}


def attach(p, work, out, checks, values, imgs, fixes, assumptions, metrics, verdict, head):
    """Run the ringing study and add it to a buck result being built in circuits.run. Returns verdict, headline."""
    r = ringing(p, Path(work), Path(out))
    checks += r["checks"]
    values.update(r["values"])
    imgs.update(r["images"])
    fixes += r["fixes"]
    assumptions += r["assumptions"]
    metrics[-1] = r["metric"]
    order = {"ok": 0, "marginal": 1, "fails": 2}
    if r["verdict"] and order[r["verdict"]] > order.get(verdict, 0):
        verdict = r["verdict"]
    return verdict, head + " " + r["headline"]
