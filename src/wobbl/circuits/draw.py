"""Bench drawings of a circuit run: each template's parts wired up on a bench, from the run's own numbers.

setup.svg shows a plausible bench for the run: the LED on a breadboard under its supply, the RC network
with a function generator and a scope, the buck converter board between a supply and an electronic load,
the MOSFET on a heatsink sized from its junction to air resistance. Component values, color bands,
instrument readings, scope traces, the moving current and the verdict all come from the run's
parameters and results, so changing a --set value changes the drawing.
"""
from __future__ import annotations

import math

import numpy as np

from ..units import fmt
from .bench import (BLACK_LEAD, RED_LEAD, VERDICT, YEL, Breadboard, Svg, capacitor_top, cubic, eng, flow, instrument,
                    jumper, lead, led, mat, num, nums, pin, plug, poly, probe, readout, resistor, scope, shimmer)

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
