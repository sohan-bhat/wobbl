"""Lithium ion thermal runaway onset: when a cell's own decomposition heat outruns its cooling.

A charged cell slowly decomposes, its SEI first, then its anode and cathode reacting with the electrolyte and the
electrolyte itself, and each step releases heat at an Arrhenius rate. Semenov's criterion compares that heat with
what the surface sheds by convection and radiation. Below a critical surrounding temperature the two curves cross
and the cell settles a little above its surroundings, and above it they no longer meet and the cell runs away.
Wobbl finds that critical temperature two ways. The Semenov tangency uses fresh reactants and no consumption, the
classical and conservative estimate. A transient of the same lumped cell from room temperature in an oven at the
surrounding temperature, with each reactant used up as it reacts, is bisected for the lowest temperature that runs
away, which is the onset the kinetics were fitted to reproduce.

The kinetics are Hatchard and Dahn's for a fully charged LiCoO2 18650 (E-One Moli ICR18650, 1.65 Ah), with Kim,
Pesaran and Spotnitz's electrolyte decomposition, as tabulated by Hu, He, Rein and Restuccia (Fire Technology 2020,
Table 2), whose model and Hatchard's oven tests put a single cell's critical oven temperature at 155 C. Other
chemistries, LFP above all, run away at other temperatures, so the answer is indicative for them. The cell is
lumped, which holds while its Biot number stays small, and Wobbl checks that.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from ..outcome_generic import GenericOutcome
from ..units import fmt
from ..verify.checks import Check, confidence

R = 8.314462618      # J/(mol K)
SIGMA = 5.670374e-8  # W/(m2 K4)
# Feng et al. (Frontiers in Energy Research 2018) call the moment a cell heats itself faster than 10 C/min the
# trigger of thermal runaway
RUNAWAY_RATE = 10.0 / 60.0   # K/s
MARGIN = 20.0         # K the critical temperature should clear the surroundings by
HU = ("Hu, He, Rein and Restuccia, Numerical Study of Self-Heating Ignition of a Box of Lithium-Ion Batteries During "
      "Storage, Fire Technology 2020, Table 2, from Hatchard et al. 2001 and Kim et al. 2007")


@dataclass(frozen=True)
class Kinetics:
    A: tuple[float, float, float, float]       # 1/s, SEI, negative, positive, electrolyte
    E: tuple[float, float, float, float]       # J/mol
    H: tuple[float, float, float, float]       # J/kg
    W: tuple[float, float, float]              # kg/m3 of negative, positive and electrolyte material
    c0: tuple[float, float, float, float, float]   # c_sei, c_n, alpha, z, c_e at the start
    rho: float = 2580.0                        # kg/m3, the cell as a whole
    cp: float = 830.0                          # J/(kg K)
    k: float = 3.4                             # W/(m K)
    name: str = ""
    source: str = ""


LCO = Kinetics((2.08e15, 1.67e6, 6.67e13, 5.14e25), (1.35e5, 7.72e4, 1.40e5, 2.74e5),
               (2.57e5, 1.71e6, 3.14e5, 1.55e5), (363.0, 726.0, 407.0), (0.15, 0.75, 0.04, 0.033, 1.0),
               name="LiCoO2 cell at 100% charge (Moli ICR18650)", source=HU)
CELLS = {"18650": (0.018, 0.065), "21700": (0.021, 0.070), "26650": (0.026, 0.065), "14500": (0.014, 0.050)}


@dataclass
class Cell:
    diameter: float             # m
    length: float               # m
    h: float = 7.17             # W/(m2 K), Kim et al.'s value for a cell in still air
    emissivity: float = 0.8
    kin: Kinetics = LCO
    label: str = "18650"

    @property
    def volume(self) -> float:
        return math.pi / 4 * self.diameter ** 2 * self.length

    @property
    def area(self) -> float:
        return math.pi * self.diameter * self.length + math.pi / 2 * self.diameter ** 2

    @property
    def biot(self) -> float:
        return self.h * self.volume / self.area / self.kin.k


def _rates(k: Kinetics, T, c_sei, c_n, a, z, c_e):
    """Rates of the four reactions (1/s) and their heat (W/m3) at temperature T and state."""
    z0 = k.c0[3]
    r = np.array([k.A[0] * math.exp(-k.E[0] / (R * T)) * max(c_sei, 0.0),
                  k.A[1] * math.exp(-z / z0) * math.exp(-k.E[1] / (R * T)) * max(c_n, 0.0),
                  k.A[2] * max(a, 0.0) * max(1 - a, 0.0) * math.exp(-k.E[2] / (R * T)),
                  k.A[3] * math.exp(-k.E[3] / (R * T)) * max(c_e, 0.0)])
    W = (k.W[0], k.W[0], k.W[1], k.W[2])
    return r, np.array([k.H[i] * W[i] * r[i] for i in range(4)])


def generation(cell: Cell, T):
    """Heat the cell makes with fresh reactants, W, for a temperature or an array of them."""
    k = cell.kin
    c_sei, c_n, a, z, c_e = k.c0
    T = np.asarray(T, float)
    q = (k.H[0] * k.W[0] * k.A[0] * np.exp(-k.E[0] / (R * T)) * c_sei
         + k.H[1] * k.W[0] * k.A[1] * math.exp(-z / k.c0[3]) * np.exp(-k.E[1] / (R * T)) * c_n
         + k.H[2] * k.W[1] * k.A[2] * a * (1 - a) * np.exp(-k.E[2] / (R * T))
         + k.H[3] * k.W[2] * k.A[3] * np.exp(-k.E[3] / (R * T)) * c_e)
    return q * cell.volume


def loss(cell: Cell, T, Ta: float):
    """Heat the surface sheds by convection and radiation, W."""
    T = np.asarray(T, float)
    return cell.area * (cell.h * (T - Ta) + cell.emissivity * SIGMA * (T ** 4 - Ta ** 4))


def semenov(cell: Cell) -> tuple[float, float]:
    """The surrounding temperature at which loss just touches generation with fresh reactants, and the cell's
    temperature at that point, K. Below it the cell has a steady state, above it none."""
    def gap(Ta):
        Ts = np.linspace(Ta, Ta + 250.0, 5001)[1:]
        g = loss(cell, Ts, Ta) - generation(cell, Ts)
        i = int(np.argmax(g))
        return float(g[i]), float(Ts[i])

    lo, hi = 300.0, 700.0
    if gap(lo)[0] <= 0:
        return lo, lo
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if gap(mid)[0] > 0:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-3:
            break
    return lo, gap(lo)[1]


@dataclass
class Transient:
    Ta: float
    t: np.ndarray
    T: np.ndarray
    y: np.ndarray
    runaway: bool
    t_runaway: float | None
    energy_error: float


def transient(cell: Cell, Ta: float, hours: float = 24.0, T0: float = 301.15, rtol: float = 1e-7,
              stop: bool = True) -> Transient:
    """The lumped cell from T0 placed in surroundings at Ta, with its reactants used up as they react."""
    from scipy.integrate import solve_ivp

    k = cell.kin
    C = k.rho * k.cp * cell.volume

    def f(t, s):
        T = s[0]
        r, q = _rates(k, T, *s[1:6])
        gen = float(np.sum(q)) * cell.volume
        out = loss(cell, T, Ta)
        return [(gen - out) / C, -r[0], -r[1], r[2], r[1], -r[3], gen, out]

    def ran(t, s):
        # the cell heats itself faster than the trigger rate while above its surroundings
        return f(t, s)[0] - RUNAWAY_RATE if s[0] > Ta else -RUNAWAY_RATE
    ran.terminal = True
    ran.direction = 1
    s0 = [T0, *k.c0[:2], k.c0[2], k.c0[3], k.c0[4], 0.0, 0.0]
    sol = solve_ivp(f, (0.0, hours * 3600.0), s0, method="LSODA", rtol=rtol, atol=[1e-6, 1e-10, 1e-10, 1e-10, 1e-10,
                                                                                    1e-10, 1e-3, 1e-3],
                    events=ran if stop else None, dense_output=False, max_step=600.0 if stop else 30.0)
    T = sol.y[0]
    stored = C * (T[-1] - T0)
    err = abs(sol.y[6, -1] - sol.y[7, -1] - stored) / max(abs(sol.y[6, -1]), abs(sol.y[7, -1]), abs(stored), 1e-9)
    hit = stop and len(sol.t_events[0]) > 0
    return Transient(Ta, sol.t, T, sol.y[1:6], hit, float(sol.t_events[0][0]) if hit else None, err)


def critical(cell: Cell, start: float, tol: float = 0.25) -> tuple[float, Transient, Transient]:
    """The lowest surrounding temperature whose transient runs away, by stepping up from start then bisecting.
    Returns it with the last transients that stayed and that ran away."""
    lo_run = transient(cell, start)
    if lo_run.runaway:
        hi, hi_run = start, lo_run
        lo = start - 5.0
        while (r := transient(cell, lo)).runaway:
            hi, hi_run, lo = lo, r, lo - 5.0
        lo_run = r
    else:
        lo, hi = start, start + 5.0
        while not (r := transient(cell, hi)).runaway:
            lo, lo_run, hi = hi, r, hi + 5.0
            if hi > start + 200:
                raise RuntimeError("the cell never ran away within 200 K of the Semenov estimate")
        hi_run = r
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        r = transient(cell, mid)
        if r.runaway:
            hi, hi_run = mid, r
        else:
            lo, lo_run = mid, r
    return hi, lo_run, hi_run


COOLING_WORDS = {"sealed": "still air in a sealed box", "box": "still air in a sealed box",
                 "still": "still air in a sealed box", "open": "natural convection in open air",
                 "natural": "natural convection in open air", "fan": "a small fan", "forced": "strong forced air",
                 "water": "water cooling", "liquid": "water cooling"}


def cell_from(cell: str = "18650", film: str = "", emissivity: float = 0.8) -> Cell:
    """A cell from its format (18650) or size ('18x65' in mm), with its cooling as a film coefficient ('25 W/m2K')
    or in the heat domain's words ('a small fan', 'sealed box')."""
    from ..thermal.study import H_TYPICAL
    from ..units import parse_quantity

    key = cell.strip().lower().replace("mm", "")
    if key in CELLS:
        d, L = CELLS[key]
    else:
        try:
            d, L = (float(v) / 1000 for v in key.replace("by", "x").split("x"))
        except ValueError:
            raise ValueError(f"unknown cell '{cell}' (use {', '.join(CELLS)} or a size like 18x65 in mm)") from None
    h = 7.17
    if film:
        words = next((v for k, v in COOLING_WORDS.items() if k in film.lower()), None)
        if words:
            h = H_TYPICAL[words]
        else:
            h = parse_quantity(film.replace(" ", "") if "/" in film else film, "film").value * 1000
    if h <= 0 or not 0 <= emissivity <= 1:
        raise ValueError("the cooling needs a film coefficient above zero and an emissivity from 0 to 1")
    return Cell(d, L, h, emissivity, label=key if key in CELLS else f"{fmt(d * 1000)} by {fmt(L * 1000)} mm")


def surroundings_from(result_dir: str) -> tuple[float, str]:
    """The hottest temperature of a finished heat study (K), and how to say where it came from."""
    import json

    p = Path(result_dir) / "outcome.json"
    if not p.exists():
        raise ValueError(f"no Wobbl result in {result_dir}")
    d = json.loads(p.read_text())
    if d.get("domain") != "thermal" or "t_max_C" not in d.get("values", {}):
        raise ValueError(f"{result_dir} is not a heat result, so it has no temperature to take")
    return d["values"]["t_max_C"] + 273.15, f"the hottest point of the heat study in {str(result_dir).rstrip('/')}"


def run_runaway(cell: Cell, ambient: float, out_dir, question: str = "", source: str = "",
                render: bool = True) -> GenericOutcome:
    """Critical surrounding temperature of a cell, against the temperature it actually sits in (K)."""
    t0 = time.time()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ta_sem, t_touch = semenov(cell)
    ta_cr, stay, ran = critical(cell, ta_sem)
    fine = transient(cell, ta_cr, rtol=1e-9)
    checks = []
    bi = cell.biot
    checks.append(Check("lumped cell", "pass" if bi < 0.1 else "warn",
                        f"Biot number {bi:.3g} for {fmt(cell.h)} W/(m2 K) over the cell, so its inside stays within "
                        f"{fmt(100 * bi / 2)}% of one temperature" + ("" if bi < 0.1 else
                                                                      ", too uneven for a lumped cell")))
    ebal = max(stay.energy_error, ran.energy_error)
    checks.append(Check("energy balance", "pass" if ebal < 0.005 else "warn",
                        f"heat made equals heat stored plus heat shed to {100 * ebal:.2g}% over each transient"))
    dt_a, dt_b = ran.t_runaway, fine.t_runaway
    if dt_a and dt_b:
        d = abs(dt_a - dt_b) / dt_b
        checks.append(Check("time step convergence", "pass" if d < 0.01 else "warn",
                            f"at {fmt(ta_cr - 273.15)} C the cell runs away after {fmt(dt_a / 60)} min, and "
                            f"{fmt(dt_b / 60)} min with a tolerance 100 times tighter ({100 * d:.2g}% apart)"))
    else:
        checks.append(Check("time step convergence", "warn", "the tighter transient did not run away at the "
                                                             "bisected temperature, so the onset sits within the "
                                                             "solver's tolerance of the bracket"))
    _, q = _rates(cell.kin, t_touch, *cell.kin.c0)
    share = float(q.max() / q.sum()) if q.sum() > 0 else 0.0
    e_dom = cell.kin.E[int(np.argmax(q))]
    fk = R * ta_sem ** 2 / e_dom
    d = abs((t_touch - ta_sem) - fk) / fk
    checks.append(Check("closed form", "pass" if d < 0.25 else "info",
                        f"at the Semenov point the cell sits {fmt(t_touch - ta_sem)} K above its surroundings, against "
                        f"R Ta^2 / E = {fmt(fk)} K for one reaction carrying {100 * share:.0f}% of the heat, "
                        f"{100 * d:.0f}% apart"))
    checks.append(Check("Semenov bound", "pass" if ta_sem <= ta_cr + 0.5 else "warn",
                        f"with fresh reactants that never run out the cell would run away above "
                        f"{fmt(ta_sem - 273.15)} C, below the {fmt(ta_cr - 273.15)} C the transient needs, since the "
                        "small store of SEI burns out before it can carry the cell"))
    level, reasons = confidence(checks)
    margin = ta_cr - ambient
    fixes = []
    if margin <= 0:
        verdict = "hot"
        t_on = _onset(cell, ambient)
        head = (f"At {fmt(ambient - 273.15)} C the {cell.label} cell's own decomposition outruns its cooling and it "
                f"runs away" + (f" after about {fmt(t_on / 60)} min" if t_on else "") + f", since thermal runaway "
                f"starts above {fmt(ta_cr - 273.15)} C.")
    elif margin < MARGIN:
        verdict = "warm"
        head = (f"Thermal runaway starts above {fmt(ta_cr - 273.15)} C, only {fmt(margin)} K above the "
                f"{fmt(ambient - 273.15)} C the {cell.label} cell sits in.")
    else:
        verdict = "ok"
        head = (f"Thermal runaway starts above {fmt(ta_cr - 273.15)} C, {fmt(margin)} K above the "
                f"{fmt(ambient - 273.15)} C the {cell.label} cell sits in, so self heating cannot run away.")
    if verdict != "ok":
        # the transient onset sits a fixed step above Semenov's for this cell, so Semenov steers the cooling and one
        # more transient bisection confirms it
        target = ambient + MARGIN - (ta_cr - ta_sem)
        h_need = _cooling_for(cell, target)
        if h_need:
            better = replace(cell, h=h_need)
            ta_new = critical(better, semenov(better)[0])[0]
            short = ambient + MARGIN - ta_new
            if short > 0.5:      # the step between the two onsets shifts with the cooling, so correct it once
                h2 = _cooling_for(cell, target + short)
                if h2:
                    h_need, better = h2, replace(cell, h=h2)
                    ta_new = critical(better, semenov(better)[0])[0]
            fixes.append({"kind": "cooling", "title": f"Raise the cooling to about {fmt(h_need)} W/(m2 K)",
                          "summary": f"Simulated, the onset moves to {fmt(ta_new - 273.15)} C, "
                                     f"{fmt(ta_new - ambient)} K above the surroundings. That takes forced air or a "
                                     "thermal path to a metal plate rather than still air.", "auto": False,
                          "after": {"h_W_m2K": h_need, "critical_ambient_C": ta_new - 273.15}})
        fixes.append({"kind": "temperature", "title": f"Keep the surroundings under {fmt(ta_cr - MARGIN - 273.15)} C",
                      "summary": f"That keeps {fmt(MARGIN)} K between the cell and its {fmt(ta_cr - 273.15)} C onset.",
                      "auto": False})
    values = {"critical_ambient_C": ta_cr - 273.15, "semenov_ambient_C": ta_sem - 273.15,
              "semenov_cell_C": t_touch - 273.15, "ambient_C": ambient - 273.15, "margin_K": margin,
              "time_to_runaway_min": (ran.t_runaway or 0.0) / 60, "biot": bi, "h_W_m2K": cell.h,
              "emissivity": cell.emissivity, "cell_mm": [cell.diameter * 1000, cell.length * 1000]}
    imgs = {}
    if render:
        try:
            # ovens 10 K under and 5 K over the onset, run on past the trigger so the burst shows
            over = transient(cell, ta_cr + 5.0)
            end = 2.0 * over.t_runaway / 3600 if over.t_runaway else 3.0
            calm = transient(cell, ta_cr - 10.0, hours=end, stop=False)
            burst = transient(cell, ta_cr + 5.0, hours=end, stop=False)
            imgs["onset"] = _chart(out, calm, burst, over, ta_cr, end)
            from . import playback
            values["files"] = {"playback": playback.runaway(out, cell, calm, burst, over.t_runaway, ta_cr, ambient)}
        except Exception as e:  # noqa: BLE001
            imgs["error"] = str(e)
    values["_figures"] = [["onset", "Cell temperature in an oven just below and just above the onset"]]
    assumptions = [f"{cell.kin.name}, {fmt(cell.diameter * 1000)} by {fmt(cell.length * 1000)} mm, with the kinetics "
                   f"of {cell.kin.source}.",
                   f"Cooled by {fmt(cell.h)} W/(m2 K) of convection and radiation at an emissivity of "
                   f"{fmt(cell.emissivity)} over its whole surface" + (f", in surroundings taken from {source}"
                                                                       if source else "") + ".",
                   "The cell is lumped, open circuit and at rest, with no internal short, overcharge or outside fire. "
                   "Runaway is the moment the cell heats itself faster than 10 C/min, the trigger Feng et al. "
                   "(Frontiers in Energy Research 2018) use.",
                   "The kinetics describe a fully charged LiCoO2 cell. LFP cells run away at higher temperatures and "
                   "NMC cells at similar ones, so for other chemistries treat the answer as indicative.",
                   "The kinetics carry the cell up to the trigger. What follows it, the separator melting, venting and "
                   "fire, is beyond them, so the temperatures after the trigger are not a forecast."]
    metrics = [("runaway above", f"{fmt(ta_cr - 273.15)} C"), ("surroundings", f"{fmt(ambient - 273.15)} C"),
               ("margin", f"{fmt(margin)} K"), ("Semenov estimate", f"{fmt(ta_sem - 273.15)} C")]
    o = GenericOutcome("battery", f"{cell.label} cell", question, verdict, head, metrics, level, reasons, checks,
                       assumptions, values, imgs, fixes, solver="Wobbl lumped Arrhenius kinetics (scipy LSODA)",
                       t_total=time.time() - t0)
    return o.write(out)


def _onset(cell: Cell, Ta: float) -> float | None:
    return transient(cell, Ta).t_runaway


def _cooling_for(cell: Cell, Ta_target: float) -> float | None:
    """Film coefficient whose Semenov onset is Ta_target, by bisection, or None past 1000 W/(m2 K)."""
    lo, hi = cell.h, 1000.0
    if semenov(replace(cell, h=hi))[0] < Ta_target:
        return None
    for _ in range(40):
        mid = math.sqrt(lo * hi)
        if semenov(replace(cell, h=mid))[0] >= Ta_target:
            hi = mid
        else:
            lo = mid
        if hi / lo < 1.01:
            break
    return hi


def _chart(out: Path, calm: Transient, burst: Transient, ran: Transient, ta_cr: float, end: float) -> str:
    from .charts import BLUE, RED, _figure, _label, _save

    fig, ax = _figure()
    ax.plot(calm.t / 60, calm.T - 273.15, color=BLUE, lw=2, label=f"oven at {fmt(calm.Ta - 273.15)} C, settles")
    ax.plot(burst.t / 60, burst.T - 273.15, color=RED, lw=2, label=f"oven at {fmt(burst.Ta - 273.15)} C, runs away")
    if ran.t_runaway:
        ax.scatter([ran.t_runaway / 60], [ran.T[-1] - 273.15], s=40, color="#f2f2ee", zorder=5,
                   label="heating itself at 10 C/min")
    ax.axhline(ta_cr - 273.15, color="#9aa0a6", ls="--", lw=1)
    ax.legend(facecolor="#0e0f11", edgecolor="#3a3d42", labelcolor="#f2f2ee", fontsize=9)
    ax.set_xlim(0, end * 60)
    _label(ax, "minutes", "cell temperature, C")
    return _save(fig, out / "images" / "onset.png")
