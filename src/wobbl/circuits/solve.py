"""Solve an instrumented circuit with ngspice and hand back every node voltage and pin current.

The transient's stop time and step come from the circuit itself: the periods of its sources and timers and
the time constants of its capacitors and inductors with the resistors around them. A run starts from power
on, with supplies ramping up from zero, and is extended until the last two windows agree, so what is
reported is the settled behavior plus whatever the start up did on the way (inrush, the first long 555
cycle). A circuit with nothing that changes in time is solved as an operating point.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..solvers import base as solvers
from ..units import parse_spice_value
from .bind import dc_value
from .netlist import Circuit

OPTIONS = ".options rshunt=1e10 itl1=400 itl4=60 gmin=1e-12"
MAX_POINTS = 60000


@dataclass
class Wave:
    kind: str                                 # op or tran
    t: np.ndarray                             # time, or [0] for an operating point
    v: dict[str, np.ndarray]                  # node voltages
    i: dict[str, list[np.ndarray]]            # each part's current into each of its pins
    window: np.ndarray = field(default=None)  # the samples that count as settled
    notes: list[str] = field(default_factory=list)
    stop: float = 0.0
    step: float = 0.0

    def volt(self, node: str) -> np.ndarray:
        return self.v.get(node, np.zeros_like(self.t)) if node != "0" else np.zeros_like(self.t)

    def mean(self, x: np.ndarray, mask: np.ndarray | None = None) -> float:
        """Time average over the settled window (trapezoids, since the time steps vary)."""
        m = self.window if mask is None else mask
        if self.kind == "op" or m is None or m.sum() < 2:
            return float(x[-1])
        tt, xx = self.t[m], x[m]
        return float(np.trapezoid(xx, tt) / max(tt[-1] - tt[0], 1e-30)) if hasattr(np, "trapezoid") else \
            float(np.trapz(xx, tt) / max(tt[-1] - tt[0], 1e-30))

    def rms(self, x: np.ndarray) -> float:
        return math.sqrt(max(self.mean(x * x), 0.0))


def ngspice():
    a = solvers.for_domain("circuit")
    a.require()
    return a


def materialize(c: Circuit, mode: str = "op", t_off: float | None = None, period: float | None = None,
                ramp: float | None = None) -> Circuit:
    """A copy ready to solve: microcontroller subcircuits written for how their pins are driven in this run,
    and, for a power on transient, supplies that ramp up from zero."""
    from . import ics

    out = c.copy()
    for e in out.elements:
        if e.kind == "mcu":
            p = ics.Part(e.part["key"], "mcu", e.part["ratings"], "x", e.part.get("mname") or "U_MCU", "", (),
                         params=e.part["params"])
            modes = _mcu_modes(out, e, mode)
            name, text = ics.mcu_subckt(p, e.pins, modes, t_off, period)
            from .netlist import Subckt
            out.subckts[name.lower()] = Subckt(name, list(e.pins), text)
            if e.letter != "x":
                e.name, e.letter = "X" + e.name, "x"
            bare = [k for k, t in enumerate(e.rest) if "=" not in t and t.lower() != "params:"]
            if bare:
                e.rest[bare[-1]] = name
            else:
                e.rest = [name]
        if ramp and e.letter == "v" and e.kind == "supply":
            v = dc_value(e)
            if v is not None:
                e.rest = [f"PULSE(0 {v:.6g} 0 {ramp:.4g} {ramp:.4g} 1e6 2e6)"]
    if ramp:
        for s in out.subckts.values():
            if s.name.upper().startswith("BAT_"):
                s.text = re.sub(r"^(V1 pi n )(\S+)$", lambda m: f"{m.group(1)}PULSE(0 {m.group(2)} 0 {ramp:.4g} "
                                f"{ramp:.4g} 1e6 2e6)", s.text, flags=re.M)
    return out


def _mcu_modes(c: Circuit, e, mode: str) -> dict[str, str]:
    """How each GPIO is driven: as set (U1.PB5=high), else an output when it feeds a part that takes a drive
    signal (a resistor, an LED, a transistor), else an input. In a transient an output switches off partway."""
    from . import ics

    set_modes = e.meta.get("modes", {})
    nets = c.nets()
    out = {}
    for pin, node in zip(e.pins, e.nodes):
        if ics.SUPPLY_PIN.match(pin) or ics.GROUND_PIN.match(pin):
            continue
        if pin.lower() in set_modes:
            out[pin] = set_modes[pin.lower()]
            continue
        others = [(o, p) for o, p in nets.get(node, []) if o is not e]
        drives = any(o.kind in ("resistor", "led", "diode", "npn", "pnp", "nmos", "pmos", "opto", "relay")
                     for o, _ in others) and node != "0"
        pulled = any(o.kind == "resistor" and any(n2 in c.names and re.search(r"v|3v3|5v", c.names[n2], re.I)
                                                    for n2 in o.nodes if n2 != node) for o, _ in others)
        if e.letter == "x" and e.pins[0] == "vdd" and pin.startswith("io"):
            drives = True
        out[pin] = ("toggle" if mode == "tran" else "high") if drives and not pulled else "input"
    return out


def run(c: Circuit, commands: list[str], work: Path, tag: str = "run", extra: list[str] | None = None) -> list[Wave]:
    """Solve the instrumented circuit for each analysis command and read the voltages and pin currents."""
    net = c.spice(instrument=True, extra=[OPTIONS, *(extra or [])])
    data, log = ngspice().run_raw(net, work, commands, tag)
    out = []
    for cmd, d in zip(commands, data):
        kind = cmd.split()[0].lower()
        if kind == "ac":
            out.append(d)       # complex vectors, read by the AC analysis itself
            continue
        t = d.get("time", np.zeros(1))
        if kind == "op":
            t = np.zeros(1)
        v = {n: np.real(d[f"v({n})"]) for n in c.nodes() if f"v({n})" in d}
        cur: dict[str, list[np.ndarray]] = {}
        for e, k, vn in c.probes():
            cur.setdefault(e.name, [None] * len(e.nodes))[k] = np.real(d.get(f"i({vn})", np.zeros_like(t)))
        for name, pins in cur.items():
            if len(pins) == 2 and pins[1] is None:
                pins[1] = -pins[0]
        out.append(Wave("tran" if kind == "tran" else "op", np.real(t), v, cur))
    return out


# --- time scales ------------------------------------------------------------------------------------------------
def _spice_args(text: str) -> list[float]:
    vals = []
    for tok in re.split(r"[\s,()]+", text):
        try:
            vals.append(parse_spice_value(tok))
        except Exception:  # noqa: BLE001
            continue
    return vals


def time_scales(c: Circuit) -> tuple[list[float], list[float], list[str]]:
    """The periods (sources, timers, PWM) and time constants (RC, L/R, LC) the circuit has, with how each was
    found, for choosing the transient's length and step."""
    periods, taus, why = [], [], []
    for e in c.elements:
        low = " ".join(e.rest).lower()
        if e.letter in "vi":
            m = re.search(r"pulse\s*\(([^)]*)\)", low)
            if m:
                a = _spice_args(m.group(1))
                if len(a) >= 7 and a[6] > 0:
                    periods.append(a[6])
                    why.append(f"{e.ref} pulses every {a[6]:.3g} s")
                elif len(a) >= 5:
                    taus.append(max(a[2] + a[3] + a[4] + (a[5] if len(a) > 5 else 0), 1e-9))
            m = re.search(r"sin\s*\(([^)]*)\)", low)
            if m:
                a = _spice_args(m.group(1))
                if len(a) >= 3 and a[2] > 0:
                    periods.append(1 / a[2])
                    why.append(f"{e.ref} is a {a[2]:.3g} Hz sine")
    nets = c.nets()
    for e in c.elements:
        if e.kind == "timer":
            t = _timer_period(c, e, nets)
            if t:
                periods.append(t)
                why.append(f"{e.ref} oscillates with a period near {t:.3g} s from its timing parts")
    for e in c.elements:
        if e.letter != "c" or not e.value or e.kind == "parasitic":
            continue
        rs = []
        for n in e.nodes:
            if n == "0":
                continue
            for o, _ in nets.get(n, []):
                if o.letter == "r" and o.value:
                    rs.append(o.value)
        if rs:
            rpar = 1 / sum(1 / r for r in rs)
            taus.append(rpar * e.value)
    for e in c.elements:
        if e.letter == "l" and e.value:
            rs = [o.value for n in e.nodes for o, _ in nets.get(n, []) if o.letter == "r" and o.value]
            cs = [o.value for n in e.nodes for o, _ in nets.get(n, []) if o.letter == "c" and o.value]
            if rs:
                taus.append(e.value / max(sum(rs), 1e-3))
            if cs:
                periods_lc = 2 * math.pi * math.sqrt(e.value * max(cs))
                taus.append(periods_lc)
    return periods, taus, why


def _timer_period(c: Circuit, e, nets) -> float | None:
    """A 555 astable's period, 0.693 (Ra + 2 Rb) C, from the parts on its pins; or the RC on its trigger."""
    pin = dict(zip(e.pins, e.nodes))
    th, dis, vcc = pin.get("thres"), pin.get("disch"), pin.get("vcc")
    caps = [o for o, _ in nets.get(th, []) if o.letter == "c" and "0" in o.nodes and o.value] if th else []
    if not caps:
        return None
    cap = sum(o.value for o in caps)
    rb = [o for o, _ in nets.get(dis, []) if o.letter == "r" and th in o.nodes and o.value] if dis and dis != th else []
    ra = [o for o, _ in nets.get(dis, []) if o.letter == "r" and vcc in o.nodes and o.value] if dis else []
    if ra:
        return 0.693 * (ra[0].value + 2 * (rb[0].value if rb else 0)) * cap
    rs = [o.value for o, _ in nets.get(th, []) if o.letter == "r" and o.value]
    return 1.4 * (sum(rs) or 1e4) * cap


def plan_tran(c: Circuit) -> tuple[float, float, list[str]]:
    """Step and stop time for the transient, from the circuit's periods and time constants."""
    periods, taus, why = time_scales(c)
    tau = max(taus) if taus else 0.0
    if periods:
        T = max(periods)
        stop = max(12 * T, 4 * tau)
        step = min(min(periods) / 200, (min(taus) / 10) if taus else 1)
        why.append(f"run for {stop:.3g} s, about {stop / T:.0f} of the longest period")
    elif tau:
        stop = 6 * tau
        step = max(min(taus), tau / 2000) / 20
        why.append(f"run for {stop:.3g} s, six of the longest time constant ({tau:.3g} s)")
    else:
        stop, step = 1e-3, 1e-6
        why.append("no time scale found, so a 1 ms run")
    stop = min(max(stop, 1e-6), 30.0)
    step = max(step, stop / MAX_POINTS)
    return step, stop, why


def settled(w: Wave, period: float | None = None, frac: float = 0.15, tol: float = 0.005) -> tuple[bool, float]:
    """Whether the run has settled: every node's average over the last window matches the window before it,
    to tol of its swing (or 1 mV). Windows are whole periods when the circuit has one. Returns the verdict and
    the largest change."""
    t = w.t
    if len(t) < 10:
        return True, 0.0
    t1 = t[-1]
    span = frac * t1
    if period and period < t1 / 4:
        span = max(1, math.floor(frac * t1 / period)) * period
    a = (t >= t1 - span) & (t <= t1)
    b = (t >= t1 - 2 * span) & (t < t1 - span)
    worst = 0.0
    for v in w.v.values():
        if a.sum() < 2 or b.sum() < 2:
            continue
        ma, mb = w.mean(v, a), w.mean(v, b)
        swing = max(float(np.ptp(v)), 1e-3)
        worst = max(worst, abs(ma - mb) / swing if abs(ma - mb) > 1e-3 else 0.0)
    return worst <= tol, worst


def measured_period(w: Wave) -> float | None:
    """The period the run actually settled into, from the node that swings most crossing its middle in the last
    half of the run, or None when nothing repeats."""
    if len(w.t) < 20:
        return None
    half = w.t >= w.t[-1] / 2
    best = max(w.v.values(), key=lambda v: float(np.ptp(v[half])), default=None)
    if best is None or np.ptp(best[half]) < 1e-3:
        return None
    x, t = best[half], w.t[half]
    mid = 0.5 * (x.max() + x.min())
    up = np.nonzero((x[:-1] < mid) & (x[1:] >= mid))[0]
    if len(up) < 3:
        return None
    ts = t[up] + (mid - x[up]) * (t[up + 1] - t[up]) / (x[up + 1] - x[up])
    d = np.diff(ts)
    return float(np.mean(d)) if np.std(d) < 0.05 * np.mean(d) else None


def run_tran(c: Circuit, work: Path, tag: str = "tran", step: float | None = None, stop: float | None = None,
             tries: int = 3, factor: float = 1.0, extra: list[str] | None = None) -> Wave:
    """Power on transient, extended until the last two windows agree (up to tries runs)."""
    why = []
    if stop is None:
        step, stop, why = plan_tran(c)
    w = None
    for k in range(tries):
        ramp = min(stop / 50, 1e-3)
        s = step * factor
        cc = materialize(c, "tran", t_off=0.6 * stop if _toggles(c) else None, period=_pwm_period(c), ramp=ramp)
        w = run(cc, [f"tran {s:.4g} {stop:.4g} 0 {s:.4g} uic"], work, f"{tag}{k}", extra)[0]
        w.step, w.stop = s, stop
        w.notes = list(why)
        if _toggles(c):
            w.window = (w.t >= 0.4 * stop) & (w.t < 0.6 * stop)
            w.notes.append("outputs switch on at power up and off at 60% of the run, so both states and the "
                           "switch off are seen")
            return w
        T = measured_period(w) or max(time_scales(c)[0] or [0]) or None
        ok, chg = settled(w, T)
        span = max(3, math.floor(0.3 * stop / T)) * T if T and T < stop / 6 else 0.2 * stop
        w.window = w.t >= stop - span
        if ok or k == tries - 1:
            if not ok:
                w.notes.append(f"still moving by {100 * chg:.2g}% of its swing at the end of {stop:.3g} s")
            return w
        stop *= 3
        step *= 1.5
        why.append(f"not settled after {stop / 3:.3g} s, so run to {stop:.3g} s")
    return w


def _toggles(c: Circuit) -> bool:
    return any(e.kind == "mcu" for e in c.elements) and any(
        _mcu_modes(c, e, "tran").get(p) == "toggle" for e in c.elements if e.kind == "mcu" for p in e.pins)


def _pwm_period(c: Circuit) -> float | None:
    for e in c.elements:
        for m in e.meta.get("modes", {}).values():
            if str(m).startswith("pwm"):
                f = re.search(r"@(\S+)", str(m))
                return 1 / parse_spice_value(f.group(1)) if f else 1e-3
    return None


def run_op(c: Circuit, work: Path, tag: str = "op", extra: list[str] | None = None) -> Wave:
    w = run(materialize(c, "op"), ["op"], work, tag, extra)[0]
    return w
