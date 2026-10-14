"""Bench drawings of a circuit run: each template's parts wired up on a bench, from the run's own numbers.

setup.svg shows a plausible bench for the run: the LED on a breadboard under its supply, the RC network
with a function generator and a scope, the buck converter board between a supply and an electronic load,
the MOSFET on a heatsink sized from its junction to air resistance. Component values, color bands,
instrument readings, scope traces, the moving current and the verdict all come from the run's
parameters and results, so changing a --set value changes the drawing.
"""
from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np

from ..units import fmt
from .bench import (BG, BLACK_LEAD, RED_LEAD, VERDICT, YEL, H, W, Breadboard, Svg, capacitor_top, clip, cubic,
                    diode_axial, dip, eng, flow, heatsink, inductor_axial, inductor_drum, instrument, jumper, lead,
                    led, mat, meter, mix, module, num, nums, pin, plug, poly, power_resistor, probe, readout,
                    resistor, scope, shimmer, terminal, to92, to220, to220_standing, toroid)
from .board import plan
from .schematic import schematic

SERIES = (7, 13)  # the series part runs between these columns in row c; column 12 is a gap in the rails


def _climb(final: float = 1.0, n: int = 25) -> list[float]:
    """Power on: n samples over the ramp, rising fast and settling on the final value."""
    return [final * (1 - (1 - k / (n - 1)) ** 3) for k in range(n)]


def _wire_board(s: Svg, bb: Breadboard, posts: dict, part) -> str:
    """The LED and RC boards: part(a, b) across the channel at the series part's far column, jumpers to the
    rails and leads to the source. Returns the path the current takes, from the + post round to the - post."""
    c0, c1 = SERIES
    hp, hm = bb.hole(3, "t+"), bb.hole(2, "b-")
    jumper(s, [bb.hole(c0, "t+"), bb.hole(c0, "a")], RED_LEAD)
    jumper(s, [bb.hole(c1, "j"), bb.hole(c1, "b-")], BLACK_LEAD)
    part(bb.hole(c1, "e"), bb.hole(c1, "f"))
    (px, py), (mx, my) = posts["+"], posts["-"]
    red = ((px, py), (px, py + 44), (hp[0], hp[1] - 50), hp)
    black = ((mx, my), (mx - 50, my + 90), (hm[0] - 96, hm[1]), hm)
    lead(s, cubic(*red), RED_LEAD)
    lead(s, cubic(*black), BLACK_LEAD)
    pin(s, *hp, 90)
    pin(s, *hm, 0)
    plug(s, px, py, RED_LEAD)
    plug(s, mx, my, BLACK_LEAD)
    loop = [bb.hole(c0, "t+"), bb.hole(c0, "c"), bb.hole(c1, "c"), bb.hole(c1, "b-"), hm]
    back = "C{} {} {} {} {} {}".format(*map(num, (*black[2], *black[1], *black[0])))
    return cubic(*red) + "L" + poly(loop)[1:] + back


def led_setup(p: dict, res: dict, data: dict, verdict: str) -> tuple[str, str]:
    i, vs, r = float(res["i_led_A"]), float(p["supply"]), float(p["resistor"])
    imax, color, rating = float(p["imax"]), str(p["led"]), float(p.get("rating", 0.25))
    p_r = i * i * r
    climb = _climb()
    s = Svg("led")
    mat(s)
    posts = instrument(s, 20, 24, 176, 124, "DC POWER SUPPLY",
                       [(f"{vs:.2f}", "V"), (f"{i:.3f}", "A", nums(i * k for k in climb), "3")])
    bb = Breadboard(s, 164, 170)
    iref = imax * 2 / 3
    glow = [min(1.0, math.sqrt(max(i * k, 0) / iref)) for k in climb]
    dome = {}
    resistor(s, bb.hole(SERIES[0], "c"), bb.hole(SERIES[1], "c"), r, bb.p)

    def part(a, b):
        dome["at"] = led(s, a, b, color, bb.p, glow, math.sqrt(i / iref))

    path = _wire_board(s, bb, posts, part)
    flow(s, path, min(max(80 * i / imax, 4), 240), data_k=nums(climb, 3))
    cx, cy, rd = dome["at"]
    hot = VERDICT["fails"] if i > imax else VERDICT["warm"]
    if i > 0.8 * imax:
        if i > imax:
            s.add("circle", cx=cx, cy=cy, r=rd * 2, fill="none", stroke=hot, stroke_width=1.5, stroke_dasharray="3 3",
                  data_after=True)
        shimmer(s, cx + rd * 2.6, cy - 4, hot)
    if p_r > 0.6 * rating:
        rx, ry = bb.hole(sum(SERIES) // 2, "c")
        shimmer(s, rx, ry - 12, VERDICT["fails"] if p_r > rating else VERDICT["warm"])
    readout(s, "LED CURRENT", i * 1000, "mA", [i * 1000 * k for k in climb], imax * 1000,
            f"{fmt(imax * 1000)} mA rating", verdict)
    label = (f"Bench setup drawn from this run. A {fmt(vs)} V supply drives a {color} LED through a {fmt(r)} ohm "
             f"resistor on a breadboard, and {fmt(i * 1000)} mA flows against its {fmt(imax * 1000)} mA rating.")
    return s.svg(label, live=1.8), label


def rc_setup(p: dict, res: dict, data: dict, verdict: str) -> tuple[str, str]:
    r, c, v, tau = float(p["r"]), float(p["c"]), float(p["v"]), float(p["tau"])
    tt, vo = np.asarray(data["time"], float), np.asarray(data["v(out)"], float)
    ramp = np.interp(np.linspace(0, tt[-1], 25), tt, vo)
    s = Svg("rc")
    mat(s)
    posts = instrument(s, 20, 24, 176, 124, "FUNCTION GENERATOR", [(f"{v:.2f}", "V")], lamp="STEP")
    ts = np.linspace(0, tt[-1], 160)
    bnc = scope(s, 212, 18, 200, 146, [(ts / tt[-1], 0.1 + 0.8 * np.interp(ts, tt, vo) / v, YEL)], "ramp",
                marks=[tau / tt[-1]])
    bb = Breadboard(s, 164, 170)
    resistor(s, bb.hole(SERIES[0], "c"), bb.hole(SERIES[1], "c"), r, bb.p)
    path = _wire_board(s, bb, posts, lambda a, b: capacitor_top(s, a, b, c, bb.p))
    tip = bb.hole(SERIES[1], "a")
    back = (tip[0] + 44 * 0.5, tip[1] - 44 * 0.866)
    gnd = bb.hole(16, "t-")
    lead(s, cubic(back, (back[0] + 30, back[1] - 10), (gnd[0] - 6, gnd[1] - 30), gnd), BLACK_LEAD, 2.5)
    pin(s, *gnd, 90)
    probe(s, cubic(bnc["1"], (bnc["1"][0], bnc["1"][1] + 30), (back[0] + 30, back[1] - 30), back), tip, 120)
    flow(s, path, 90, data_k=nums(np.clip((v - ramp) / v, 0, 1), 3))
    readout(s, "CAPACITOR", float(ramp[-1]), "V", ramp, v, f"τ = RC = {eng(tau, 's')}", verdict)
    label = (f"Bench setup drawn from this run. A {fmt(v)} V step charges a {eng(c, 'F')} capacitor through "
             f"{eng(r, 'ohm')} on a breadboard, with a time constant of {eng(tau, 's')}.")
    return s.svg(label, live=3), label


def buck_setup(p: dict, res: dict, data: dict, verdict: str) -> tuple[str, str]:
    vin, vout, iout, f = (float(p[k]) for k in ("vin", "vout", "current", "frequency"))
    duty, rip, vavg = float(p["duty"]), float(res["ripple_V"]), float(res["vout_avg"])
    spec = float(p.get("max_ripple", 0.01 * vout))
    spec = spec if spec > 0 else 0.01 * vout
    rload = vout / iout
    tt, vo, il = (np.asarray(data[k], float) for k in ("time", "v(out)", "i(l1)"))
    period = 1 / f
    t0 = (round(tt[-1] / period) - 3) * period  # three cycles, each starting as the switch turns on
    ts = np.linspace(t0, t0 + 3 * period, 181)
    v3, i3 = np.interp(ts, tt, vo), np.interp(ts, tt, il)
    run_pp = [1000 * float(np.ptp(v3[:max(2, round(u * 180) + 1)])) for u in np.linspace(0, 1, 25)]
    run_pp[-1] = rip * 1000

    def lane(y, mid):
        return mid + 0.13 * (y - y.mean()) / max(float(np.ptp(y)) / 2, 1e-15)

    s = Svg("buck")
    mat(s)
    supply = instrument(s, 20, 24, 176, 100, "DC POWER SUPPLY", [(f"{vin:.2f}", "V")])
    u = (ts - t0) / (3 * period)
    bnc = scope(s, 212, 18, 200, 146, [(u, lane(v3, 0.7), YEL), (u, lane(i3, 0.3), "#5aa9ff")], "3",
                inputs=("2", "1"))
    load = instrument(s, 436, 130, 184, 118, "ELECTRONIC LOAD",
                      [(f"{vavg:.2f}", "V"), (f"{vavg / rload:.2f}", "A")], lamp="CR", posts=("+", "-"))
    x0, y0, w, h = 30, 262, 580, 200
    s.add("rect", x=x0 + 4, y=y0 + 6, width=w, height=h, rx=6, fill="#000", opacity=0.55)
    s.add("rect", x=x0, y=y0, width=w, height=h, rx=6, fill="#1d4a33", stroke="#123322")
    for hx, hy in ((x0 + 14, y0 + 14), (x0 + w - 14, y0 + 14), (x0 + 14, y0 + h - 14), (x0 + w - 14, y0 + h - 14)):
        s.add("circle", cx=hx, cy=hy, r=8, fill="#b9975b")
        s.add("circle", cx=hx, cy=hy, r=4.5, fill=BG)
    cu = dict(fill="none", stroke="#2f7550", stroke_width=12, stroke_linejoin="round", stroke_linecap="round")
    sw = [(222, 330), (299, 330)], [(250, 330), (250, 362)]
    s.add("path", d=poly([(126, 300), (126, 330), (178, 330)]) + poly([(150, 330), (150, 362)]), **cu)
    s.add("path", d=poly(sw[0]) + poly(sw[1]), **cu)
    s.add("path", d=poly([(329, 330), (514, 330), (514, 300)]) + poly([(450, 330), (450, 434)]), **cu)
    s.add("path", d=poly([(94, 300), (94, 434), (546, 434), (546, 300)]) + poly([(150, 400), (150, 434)])
          + poly([(250, 396), (250, 434)]), **cu)
    s.add("path", d="".join(f"M{x} 434h0" for x in range(110, 540, 26)), stroke="#c9b27a", stroke_width=4,
          stroke_linecap="round")  # vias stitching the ground pour
    s.add("path", d=poly(sw[0]) + poly(sw[1]), fill="none", stroke=YEL, stroke_width=6, stroke_linecap="round",
          opacity=0.7, data_pwm=f"{duty:.4g} 1")
    vin_t, vout_t = terminal(s, 110, 290, ("-", "+")), terminal(s, 530, 290, ("+", "-"))
    capacitor_top(s, (150, 362), (150, 406), 1e-4, 18)
    s.add("rect", x=185, y=312, width=30, height=36, rx=2, fill="#16181b", stroke="#000")
    for k in range(4):
        for px in (178, 215):
            s.add("rect", x=px, y=316 + 8 * k, width=7, height=3.4, fill="#b8bdc3")
    s.add("circle", cx=191, cy=318, r=1.8, fill="#3a3f45")
    s.add("rect", x=241, y=362, width=18, height=34, rx=1.5, fill="#1b1d20", stroke="#000")
    s.add("rect", x=241, y=366, width=18, height=5, fill="#9aa1a8")  # the cathode band, toward the switch node
    i1 = i3[:61]
    leads = toroid(s, 314, 380, 44, wave=0.15 + 0.85 * (i1 - i1.min()) / max(float(np.ptp(i1)), 1e-15))
    s.add("path", d=poly([(299, 330), leads[1]]) + poly([(329, 330), leads[0]]), stroke="#c8773a", stroke_width=3)
    s.add("rect", x=441, y=336, width=18, height=32, rx=1.5, fill="#a9845a")
    for ty in (336, 361):
        s.add("rect", x=441, y=ty, width=18, height=7, rx=1, fill="#c6cbd0")
    s.add("rect", x=383, y=318, width=26, height=24, rx=5, fill="#24282d", stroke="#4a4f57")  # current probe jaw
    s.add("path", d="M386 330H406", stroke="#0a0b0c", stroke_width=2)
    s.open(font_size=8, fill="#e8ebe6", letter_spacing=0.6)
    for x, y, t in ((70, 294, "VIN"), (570, 294, "VOUT"), (200, 307, "S1"), (234, 384, "D1"), (346, 425, "L1"),
                    (466, 356, "C1")):
        s.add("text", t, x=x, y=y, text_anchor="end" if t in ("VIN", "D1") else None)
    s.close()
    (sp, sm), (lp, lm) = (supply["+"], supply["-"]), (load["+"], load["-"])
    red_in = (sp, (sp[0], sp[1] + 70), (vin_t["+"][0], 220), vin_t["+"])
    black_in = (sm, (sm[0], sm[1] + 60), (vin_t["-"][0], 220), vin_t["-"])
    red_out = (lp, (lp[0], lp[1] + 26), (vout_t["+"][0], 240), vout_t["+"])
    black_out = (lm, (lm[0], lm[1] + 26), (vout_t["-"][0], 240), vout_t["-"])
    for pts, col in ((red_in, RED_LEAD), (black_in, BLACK_LEAD), (red_out, RED_LEAD), (black_out, BLACK_LEAD)):
        lead(s, cubic(*pts), col)
        plug(s, *pts[0], col)
    vtip = (450, 338)
    vback = (vtip[0] + 22, vtip[1] - 38)
    probe(s, cubic(bnc["1"], (bnc["1"][0], bnc["1"][1] + 60), (vback[0] + 20, vback[1] - 50), vback), vtip, 120)
    lead(s, cubic(bnc["2"], (bnc["2"][0], bnc["2"][1] + 60), (396, 260), (396, 318)), "#4b5057", 3)

    def rev(pts):
        return "C{} {} {} {} {} {}".format(*map(num, (*pts[2], *pts[1], *pts[0])))

    speed = 70.0
    on, off = f"{duty:.4g} 1", f"{duty:.4g} 1 inverted"
    flow(s, poly([(329, 330), (514, 330), vout_t["+"]]) + rev(red_out) + cubic(*black_out)
         + "L" + poly([(546, 434), (250, 434)])[1:], speed)
    flow(s, cubic(*red_in) + "L" + poly([(126, 330), (299, 330)])[1:] + poly([(250, 434), (94, 434),
                                                                                 vin_t["-"]]) + rev(black_in),
         speed, data_pwm=on)
    flow(s, poly([(250, 434), (250, 330), (299, 330)]), speed, data_pwm=off)
    readout(s, "OUTPUT RIPPLE", rip * 1000, "mV", run_pp, spec * 1000, f"{fmt(spec * 1000)} mV limit", verdict)
    label = (f"Bench setup drawn from this run. A {fmt(vin)} V supply feeds a buck converter switching at "
             f"{eng(f, 'Hz')} into a {fmt(rload)} ohm load, and the scope shows {fmt(rip * 1000)} mV of ripple on "
             f"{fmt(vavg)} V.")
    return s.svg(label, live=3), label


def mosfet_setup(p: dict, res: dict, data: dict, verdict: str) -> tuple[str, str]:
    i, rds, vbus, f = (float(p[k]) for k in ("current", "rds_on", "vbus", "frequency"))
    duty = float(p.get("duty", 1.0 if f == 0 else 0.5))
    ta, theta = float(p.get("ambient", 25)), float(p.get("theta_ja", 62))
    tjmax, tc = float(p.get("tj_max", 150)), float(p.get("rds_tempco", 0.006))
    tj, psw, pcond = float(res["tj_C"]), float(res["p_sw_W"]), float(res["p_cond_W"])
    rl = max(vbus / i - rds, 1e-3)
    tt, idr = np.asarray(data["time"], float), np.asarray(data["i(vs)"], float)
    iavg = float(np.mean(idr[tt >= tt[-1] - (10 / f if f > 0 else tt[-1] / 2)]))
    # the run's thermal loop, replayed: the on resistance warms with the junction until the two agree
    pc25, seq = pcond / (1 + tc * (tj - 25)), [ta]
    for _ in range(50):
        seq.append(min(ta + (pc25 * (1 + tc * (seq[-1] - 25)) + psw) * theta, 1000.0))
        if abs(seq[-1] - seq[-2]) < 1e-3 or seq[-1] >= 1000:
            break
    seq[-1] = tj
    climb = np.interp(np.linspace(0, len(seq) - 1, 25), np.arange(len(seq)), seq)
    heat = min(max((tj - ta) / max(tjmax - ta, 1e-9), 0.0), 1.2)
    s = Svg("mosfet")
    mat(s)
    supply = instrument(s, 20, 24, 176, 124, "DC POWER SUPPLY", [(f"{vbus:.2f}", "V"), (f"{iavg:.2f}", "A")])
    rows = [(f"{f / 1000:.2f}", "kHz"), (f"{duty * 100:.1f}", "%")] if f > 0 else [("10.00", "V")]
    instrument(s, 214, 24, 184, 124, "PULSE GENERATOR" if f > 0 else "GATE SUPPLY", rows, lamp="OUT", posts=())
    bnc = (214 + 184 - 24, 24 + 124 - 15)
    s.add("circle", cx=bnc[0], cy=bnc[1], r=7, fill="#aab0b6", stroke="#5d636b")
    s.add("circle", cx=bnc[0], cy=bnc[1], r=3.4, fill="#16181b")
    cool_sink, cool_tab = "#8e959c", "#c3c8cd"
    sink, tab = mix(cool_sink, "#c08060", 0.25 * heat), mix(cool_tab, "#e0a080", 0.35 * heat)
    cx, cy = 440, 330
    s.add("circle", cx=cx + 60, cy=cy, r=150, fill=s.grad("radial", [(0, "#ff6a2b", 0.55), (1, "#ff6a2b", 0)]),
          opacity=0.6 * heat, data_series=nums(0.6 * min(max((t - ta) / max(tjmax - ta, 1e-9), 0), 1.2)
                                                  for t in climb), data_attr="opacity")
    if theta < 55:  # a bigger sink for a lower junction to air resistance
        hw, hh = min(220.0, 90 + 3000 / theta), min(190.0, 80 + 2400 / theta)
        heatsink(s, cx - 34, cy - hh / 2, hw, hh, sink, cool_sink)
    legs = to220(s, cx, cy, tab, cool_tab)
    tx, ty = legs["tab"]
    s.add("circle", cx=tx - 20, cy=ty, r=46, fill=s.grad("radial", [(0, "#ff5a1f", 0.8), (1, "#ff5a1f", 0)]),
          opacity=0.9 * min(heat, 1), data_series=nums(0.9 * min(max((t - ta) / max(tjmax - ta, 1e-9), 0), 1)
                                                          for t in climb), data_attr="opacity")
    s.add("circle", cx=tx + 4, cy=ty - 26, r=3.5, fill="#c9a645", stroke="#6e5824")  # the thermocouple bead
    s.add("rect", x=tx - 12, y=ty - 34, width=30, height=16, rx=1.5, fill="#d9922e", opacity=0.55,
          transform=f"rotate(-8 {num(tx)} {num(ty - 26)})")
    plug_at = (W - 60, 132)
    lead(s, cubic((tx + 4, ty - 26), (tx + 40, ty - 90), (plug_at[0], 230), plug_at), "#e8c21c", 2.4)
    s.add("rect", x=plug_at[0] - 9, y=plug_at[1] - 12, width=18, height=12, rx=2, fill="#f2cf1d", stroke="#8c7410")
    r_in, r_out = power_resistor(s, 170, 340, rl)
    (sp, sm) = supply["+"], supply["-"]
    red1 = (sp, (sp[0], sp[1] + 90), (r_in[0], r_in[1] - 110), (r_in[0], r_in[1] - 6))
    red2 = (r_out, (r_out[0] + 30, r_out[1]), (legs["d"][0] - 30, legs["d"][1]), legs["d"])
    black1 = (legs["s"], (legs["s"][0] - 20, 452), (70, 462), (46, 360))
    black2 = (black1[3], (36, 250), (sm[0], sm[1] + 70), sm)
    split = (legs["g"][0] + 26, legs["g"][1] - 44)
    coax = (bnc, (bnc[0], bnc[1] + 60), (split[0], split[1] - 50), split)
    gate = (split, (split[0] - 4, split[1] + 20), (legs["g"][0] + 12, legs["g"][1]), legs["g"])
    gnd_at = (legs["s"][0] + 30, legs["s"][1])
    gret = (split, (split[0] + 16, split[1] + 30), (gnd_at[0] + 4, gnd_at[1] - 30), gnd_at)
    lead(s, cubic(*coax), "#3a3f45", 4.5)
    lead(s, cubic(*gate), "#d0453a", 2.6)
    lead(s, cubic(*gret), BLACK_LEAD, 2.6)
    s.add("path", d=cubic(*gate), fill="none", stroke=YEL, stroke_width=3.2, stroke_linecap="round", opacity=0.85,
          data_pwm=f"{duty:.4g} 1" if f > 0 else None)
    clip(s, *legs["g"], 0, "#d0453a")
    clip(s, *gnd_at, 90, "#2b2e33")
    for pts, col in ((red1, RED_LEAD), (red2, RED_LEAD), (black1, BLACK_LEAD), (black2, BLACK_LEAD)):
        lead(s, cubic(*pts), col, 6)
    plug(s, *sp, RED_LEAD)
    plug(s, *sm, BLACK_LEAD)
    clip(s, *legs["d"], 0, RED_LEAD)
    clip(s, *legs["s"], 0, BLACK_LEAD)
    loop = (cubic(*red1) + "L" + poly([r_in, r_out])[1:] + cubic(*red2)[cubic(*red2).index("C"):]
            + "L" + poly([legs["d_in"], legs["s_in"], legs["s"]])[1:] + cubic(*black1)[cubic(*black1).index("C"):]
            + cubic(*black2)[cubic(*black2).index("C"):])
    flow(s, loop, 85, data_pwm=f"{duty:.4g} 1" if f > 0 else None)
    if verdict in ("warm", "hot"):
        shimmer(s, cx + 40, cy - 100, VERDICT["fails"] if verdict == "hot" else VERDICT["warm"], n=4, gap=14)
    readout(s, "JUNCTION", tj, "°C", climb, tjmax, f"{fmt(tjmax)} °C limit", verdict)
    sink_txt = f"a {fmt(theta)} C/W heatsink" if theta < 55 else "no heatsink"
    label = (f"Bench setup drawn from this run. A {fmt(vbus)} V supply drives {fmt(i)} A through a {fmt(rl)} ohm "
             f"load and a TO-220 MOSFET on {sink_txt}, switched at {eng(f, 'Hz') if f else 'DC'}, and the junction "
             f"settles at {fmt(tj)} C.")
    return s.svg(label, live=4), label


SETUPS = {"led": led_setup, "rc": rc_setup, "buck": buck_setup, "mosfet": mosfet_setup}
FIGURES = [["setup", "Bench setup, drawn from this run"], ["schematic", "Schematic, drawn from the netlist"]]


def _notes(template: str, p: dict, res: dict, data: dict) -> dict[str, str]:
    """Readings to print beside the schematic's nodes, named as in the template's netlist."""
    if template == "led":
        return {"vin": f"{fmt(float(p['supply']))} V", "a": f"{fmt(float(res['v_led_V']))} V"}
    if template == "rc":
        return {"out": f"{fmt(float(res['v_at_tau']))} V at τ"}
    if template == "buck":
        return {"vin": f"{fmt(float(p['vin']))} V", "out": f"{fmt(float(res['vout_avg']))} V"}
    vd = np.asarray(data["v(d)"], float)
    return {"vbus": f"{fmt(float(p['vbus']))} V", "d": f"{fmt(float(vd.min()))} V on"}


def _save(path: Path, make) -> str | None:
    """Write one drawing. A drawing that cannot be made is left out rather than failing the run."""
    try:
        path.write_text(make(), encoding="utf-8")
    except Exception:  # noqa: BLE001
        if os.environ.get("WOBBL_DEBUG"):
            raise
        return None
    return str(path)


def write_drawings(images: Path, template: str, params: dict, data: dict, values: dict, verdict: str,
                   netlist: str) -> dict[str, str]:
    """setup.svg and schematic.svg for a template run, as name -> path."""
    images.mkdir(parents=True, exist_ok=True)
    try:
        notes = _notes(template, params, values, data)
    except (KeyError, ValueError, TypeError):
        notes = {}
    out = {"setup": _save(images / "setup.svg", lambda: SETUPS[template](params, values, data, verdict)[0]),
           "schematic": _save(images / "schematic.svg", lambda: schematic(netlist, notes, ns=f"{template}-sch"))}
    return {k: v for k, v in out.items() if v}


def write_schematic(images: Path, netlist: str) -> dict[str, str]:
    """schematic.svg for a netlist of the user's own."""
    images.mkdir(parents=True, exist_ok=True)
    path = _save(images / "schematic.svg", lambda: schematic(netlist))
    return {"schematic": path} if path else {}


# ---- any circuit, from its board plan ----------------------------------------------------------

INSTRUMENT = {"supply": "DC POWER SUPPLY", "generator": "FUNCTION GENERATOR", "current": "CURRENT SOURCE"}


def _flight(a, z):
    """A flexible jumper's arc from hole to hole, lifted off the board."""
    lift = min(70.0, 10 + 0.28 * math.hypot(z[0] - a[0], z[1] - a[1]))
    return a, (a[0], a[1] - lift), (z[0], z[1] - lift), z


def _rev(c):
    return c[::-1]


def board_setup(pl: dict, verdict: str = "ok", key: dict | None = None, ns: str = "brd") -> str:
    """setup.svg for any circuit from its plan (board.plan): the board with its parts and wires, the bench
    instruments and their leads, the measured current moving along every wire, and one live reading.

    key is the reading to show live: {"label", "value", "unit", "series", "limit", "note"}. Without one the
    supply's current is shown."""
    s = Svg(ns)
    mat(s)
    ins = pl["instruments"]
    right = W - 236
    gap = 10
    iw = min(176.0, (right - 20 - gap * max(len(ins) - 1, 0)) / max(len(ins), 1))
    posts = {}
    for k, inst in enumerate(ins):
        x = 20 + k * (iw + gap)
        rows = [tuple(r) for r in inst["readings"]][:2] or [("", "")]
        ends = {ld["post"]: ld["x"] for ld in inst["leads"]}
        order = ("+", "-") if ends.get("+", 1e9) < ends.get("-", -1e9) else ("-", "+")  # leads that do not cross
        if inst["kind"] == "meter":
            posts[inst["name"]] = meter(s, x, 18, min(iw, 124), 118, *rows[-1], posts=order)
        else:
            name = INSTRUMENT[inst["kind"]]
            posts[inst["name"]] = instrument(s, x, 18, iw, 118, name if iw >= 150 else name.split()[-1], rows,
                                             lamp="CV" if inst["kind"] == "supply" else "OUT", posts=order)
    b = pl["board"]
    top = 156
    p = min(16.5, (W - 40) / b["width"], (H - top - 12) / b["height"])
    bx, by = (W - b["width"] * p) / 2, top + (H - top - 12 - b["height"] * p) / 2

    def at(x, y):
        return bx + x * p, by + y * p

    Breadboard(s, bx, by, p, b["cols"])
    paths = []  # (cubic or polyline points, current, k) for the moving dots
    for w in pl["wires"]:
        pts = [at(*q) for q in w["points"]]
        if w["kind"] == "jumper":
            c = _flight(pts[0], pts[-1])
            lead(s, cubic(*c), w["color"], 3.4)
            for x, y in (pts[0], pts[-1]):
                s.add("circle", cx=x, cy=y, r=2.2, fill="#b8bdc3", stroke="#5b574d", stroke_width=0.8)
            paths.append(("c", c, w["current"], w["k"]))
        else:
            jumper(s, pts, w["color"])
            paths.append(("p", pts, w["current"], w["k"]))
    peak = pl.get("peak") or 0.0
    for prt in pl["parts"]:
        hs = [at(pn["x"], pn["y"]) for pn in prt["pins"]]
        body, kd = prt["body"], prt["kind"]
        cx, cy = sum(x for x, _ in hs) / len(hs), sum(y for _, y in hs) / len(hs)
        heat = prt.get("heat") or 0
        if heat > 0.5:
            s.add("circle", cx=cx, cy=cy, r=2.2 * p, opacity=min(1.0, heat - 0.35),
                  fill=s.grad("radial", [(0, "#ff5a1f", 0.7), (1, "#ff5a1f", 0)]))
        if body == "axial-resistor":
            resistor(s, hs[0], hs[1], prt["value"] or 0, p)
        elif body == "radial-capacitor":
            capacitor_top(s, hs[0], hs[1], prt["value"] or 1e-7, p)
        elif body.startswith("inductor"):
            if body == "inductor-radial" or math.dist(hs[0], hs[1]) < 3 * p:
                inductor_drum(s, hs[0], hs[1], p)
            else:
                inductor_axial(s, hs[0], hs[1], prt["value"], p)
        elif body == "axial-diode":
            diode_axial(s, hs[0], hs[1], p)
        elif body == "led-5mm":
            cur = abs(prt["current"] or 0)
            ks = prt["k"] or [1.0]
            glow = [min(1.0, math.sqrt(k * (peak if prt["k"] else cur) / 0.015)) for k in ks]
            led(s, hs[0], hs[1], prt["color"] or "red", p, glow, math.sqrt(max(cur, 1e-6) / 0.015))
        elif body == "to92":
            to92(s, hs, p, prt["name"])
        elif body == "to220":
            to220_standing(s, hs, p, prt["name"])
        elif body == "dip":
            tops = [h for h, pn in zip(hs, prt["pins"]) if pn["hole"][1] in "abcde"]
            bots = [h for h, pn in zip(hs, prt["pins"]) if pn["hole"][1] not in "abcde"]
            dip(s, tops, bots, p, prt["model"] or prt["name"])
        else:
            module(s, hs, p, prt["name"])
        if len(hs) == 2:
            paths.append(("p", hs, prt["current"], prt["k"]))
        if heat > 1:
            shimmer(s, cx + 1.6 * p, cy - 0.8 * p, VERDICT["fails"])
    for inst in ins:
        for ld in inst["leads"]:
            px, py = posts[inst["name"]][ld["post"]]
            hx, hy = at(ld["x"], ld["y"])
            dy = max(hy - py, 30)
            c = ((px, py), (px, py + 0.45 * dy), (hx, hy - 0.45 * dy), (hx, hy))
            col = RED_LEAD if ld["post"] == "+" else BLACK_LEAD
            lead(s, cubic(*c), col)
            pin(s, hx, hy, math.degrees(math.atan2(hy - c[2][1], hx - c[2][0])))
            plug(s, px, py, col)
            paths.append(("c", c, ld["current"], ld["k"]))
    # the measured current along every wire and part, at a speed proportional to it
    imax = max([abs(cur) for _, _, cur, _ in paths if cur] + [0.0])
    live = 3.0 if pl.get("span") else 1.6
    for kind, pts, cur, k in paths:
        if not cur or abs(cur) < 0.02 * imax:
            continue
        pts = pts if cur > 0 else _rev(pts)
        d = cubic(*pts) if kind == "c" else poly(pts)
        if k:
            flow(s, d, min(240.0, 90 * peak / imax), data_k=nums(k, 3))
        else:
            flow(s, d, 90 * abs(cur) / imax, data_k=nums(_climb(), 3))
    if key is None:  # the supply's current, else the largest current through a part
        sup = next((i for i in ins if i["kind"] == "supply" and abs(i.get("current") or 0) > 1e-9), None)
        big = max(pl["parts"], key=lambda q: abs(q["current"] or 0), default=None)
        name, amps = (("SUPPLY", abs(sup["current"])) if sup else
                      (big["name"].upper(), abs(big["current"] or 0)) if big else ("SUPPLY", 0.0))
        milli = amps < 1
        key = {"label": f"{name} CURRENT", "value": amps * (1000 if milli else 1), "unit": "mA" if milli else "A",
               "limit": None, "note": f"from {sup['name']}" if sup else "the largest on the board"}
    series = key.get("series") or _climb(key["value"])
    readout(s, key["label"], key["value"], key["unit"], series, key.get("limit"), key.get("note", ""), verdict)
    n = len(pl["parts"])
    label = (f"Bench setup drawn from this run, with {n} part{'s' if n != 1 else ''} on a breadboard, "
             f"{len(ins)} instrument{'s' if len(ins) != 1 else ''} and the measured current on every wire.")
    return s.svg(label, live=live)


def write_circuit(images: Path, circuit, data: dict | None = None, verdict: str = "ok",
                  key: dict | None = None) -> dict[str, str]:
    """setup.svg and schematic.svg for any circuit: its parts placed and wired on a breadboard by board.plan,
    carrying the run's currents (the vectors of Circuit.spice(instrument=True)), and its schematic."""
    images.mkdir(parents=True, exist_ok=True)
    out = {"setup": _save(images / "setup.svg", lambda: board_setup(plan(circuit, data), verdict, key)),
           "schematic": _save(images / "schematic.svg", lambda: schematic(circuit.spice()))}
    return {k: v for k, v in out.items() if v}
