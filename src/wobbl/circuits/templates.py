"""Ready made circuit checks: LED current, RC timing, buck regulator ripple, MOSFET heating.

Each template writes a SPICE netlist, runs ngspice, and also computes the
textbook answer so the two can be compared. Values use plain SI units.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..units import parse_spice_value, parse_quantity

VT = 8.617333e-5 * 300.15  # thermal voltage at 27 C as ngspice computes it, V

# 5 mm through hole LEDs: forward voltage at 20 mA, continuous current rating
LEDS = {
    "red": (2.0, 0.030), "orange": (2.05, 0.030), "yellow": (2.1, 0.030), "green": (3.1, 0.030),
    "blue": (3.2, 0.030), "white": (3.2, 0.030), "uv": (3.4, 0.030), "ir": (1.3, 0.050),
    "power-white": (3.0, 0.350), "power-red": (2.2, 0.350),
}
E24 = [1.0, 1.1, 1.2, 1.3, 1.5, 1.6, 1.8, 2.0, 2.2, 2.4, 2.7, 3.0, 3.3, 3.6, 3.9, 4.3, 4.7, 5.1, 5.6, 6.2, 6.8, 7.5,
       8.2, 9.1]


def e24_above(x: float) -> float:
    if x <= 0:
        return 1.0
    dec = 10 ** math.floor(math.log10(x))
    for m in E24 + [10.0]:
        if m * dec >= x * 0.999:
            return m * dec
    return 10 * dec


def val(params: dict, key: str, default=None, kind: str | None = None) -> float:
    v = params.get(key, default)
    if v is None:
        raise ValueError(f"missing parameter '{key}'")
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip()
    if kind:
        try:
            return parse_quantity(s, kind).value
        except Exception:  # noqa: BLE001
            pass
    return parse_spice_value(s)


@dataclass
class Template:
    name: str
    netlist: str
    vectors: list[str]
    params: dict
    analysis: str = "dc"           # dc or tran
    tran: tuple[float, float] | None = None   # (step, stop)
    expect: dict = field(default_factory=dict)  # closed form numbers


def led(p: dict) -> Template:
    vs = val(p, "supply", "5", "voltage")
    r = val(p, "resistor", "150", "resistance")
    color = str(p.get("led", "red")).lower()
    vf, imax = LEDS.get(color, LEDS["red"])
    vf = val(p, "vf", vf, "voltage")
    imax = val(p, "imax", imax, "current")
    n, rs = max(2.0, vf / 1.2), 1.5  # keeps IS above the ~1e-28 floor ngspice applies
    i_ref = 0.020 if imax < 0.2 else 0.350
    isat = i_ref / math.exp((vf - i_ref * rs) / (n * VT))
    net = f"""* LED with series resistor
V1 vin 0 DC {vs}
R1 vin a {r}
D1 a 0 LEDX
.model LEDX D(IS={isat:.6e} N={n} RS={rs})
.dc V1 {vs} {vs} 1
.end
"""
    # textbook: solve Vs = I R + I Rs + n Vt ln(I/Is + 1) by Newton
    i = max((vs - vf) / r, 1e-9)
    for _ in range(100):
        f = i * (r + rs) + n * VT * math.log(i / isat + 1) - vs
        df = (r + rs) + n * VT / (i + isat)
        i_new = max(i - f / df, 1e-15)
        if abs(i_new - i) < 1e-15:
            break
        i = i_new
    p_out = dict(p, supply=vs, resistor=r, vf=vf, imax=imax, led=color, n=n, rating=val(p, "rating", "0.25", "si_power"))
    return Template("led", net, ["i(v1)", "v(a)"], p_out, "dc",
                    expect={"i_led": i, "i_simple": max((vs - vf) / r, 0.0)})


def rc(p: dict) -> Template:
    r = val(p, "r", "10k", "resistance")
    c = val(p, "c", "10u", "capacitance")
    v = val(p, "v", "5", "voltage")
    tau = r * c
    net = f"""* RC charging
V1 in 0 PULSE(0 {v} 0 1n 1n 1 2)
R1 in out {r}
C1 out 0 {c} IC=0
.end
"""
    return Template("rc", net, ["v(out)"], dict(p, r=r, c=c, v=v, tau=tau), "tran", (tau / 400, 5 * tau),
                    expect={"v_at_tau": v * (1 - math.exp(-1)), "tau": tau, "t99": tau * math.log(100)})


def buck(p: dict) -> Template:
    vin = val(p, "vin", "12", "voltage")
    vout = val(p, "vout", "5", "voltage")
    iout = val(p, "current", "1", "current")
    f = val(p, "frequency", "500k", "frequency")
    L = val(p, "inductor", "10u", "inductance")
    C = val(p, "capacitor", "22u", "capacitance")
    esr = val(p, "esr", "0.01", "resistance")
    rds = val(p, "rds", "0.05", "resistance")
    vd = 0.35  # Schottky catch diode
    duty = (vout + vd + iout * 0.0) / (vin - iout * rds + vd)
    period = 1.0 / f
    ton = duty * period
    rload = vout / iout
    di = (vin - vout) * (vout / vin) / (L * f)
    dv = _ripple(di, vout / vin, f, C, esr)
    cycles = int(p.get("cycles", 300))
    net = f"""* Buck converter power stage, open loop at the steady state duty cycle
V1 vin 0 DC {vin}
VG g 0 PULSE(0 5 0 {period / 200:.4e} {period / 200:.4e} {ton:.6e} {period:.6e})
S1 vin sw g 0 SWMOD
.model SWMOD SW(VT=2.5 VH=0.1 RON={rds} ROFF=1e6)
D1 0 sw SCHOTTKY
.model SCHOTTKY D(IS=5e-5 N=1.05 RS=0.02)
L1 sw out {L} IC={iout}
C1 out cmid {C} IC={vout}
R2 cmid 0 {esr}
RLOAD out 0 {rload}
.end
"""
    return Template("buck", net, ["v(out)", "i(l1)"], dict(p, vin=vin, vout=vout, current=iout, frequency=f,
                                                          inductor=L, capacitor=C, esr=esr, duty=duty, cycles=cycles),
                    "tran", (period / 400, cycles * period),
                    expect={"ripple_v": dv, "ripple_i": di, "ccm": iout > di / 2})


def _ripple(di: float, duty: float, f: float, C: float, esr: float) -> float:
    """Exact peak to peak output ripple for a triangular capacitor current: ESR and charge terms out of phase."""
    T = 1.0 / f
    t = np.linspace(0, T, 4001)
    rise = duty * T
    ic = np.where(t < rise, -di / 2 + di * t / rise, di / 2 - di * (t - rise) / (T - rise))
    q = np.concatenate([[0.0], np.cumsum(0.5 * (ic[1:] + ic[:-1]) * np.diff(t))])
    v = esr * ic + q / C
    return float(v.max() - v.min())


def mosfet(p: dict) -> Template:
    i = val(p, "current", "10", "current")
    rds = val(p, "rds_on", "0.01", "resistance")
    vbus = val(p, "vbus", "12", "voltage")
    f = val(p, "frequency", "0", "frequency")
    duty = float(p.get("duty", 1.0 if f == 0 else 0.5))
    rload = max(vbus / i - rds, 1e-3)
    if f > 0:
        period = 1 / f
        drive = f"PULSE(0 10 0 {period / 500:.4e} {period / 500:.4e} {duty * period:.6e} {period:.6e})"
        tran = (period / 500, 40 * period)
    else:
        drive = "DC 10"
        tran = (1e-6, 1e-4)
    net = f"""* MOSFET conduction with a resistive load
V1 vbus 0 DC {vbus}
RL vbus d {rload}
VG g 0 {drive}
S1 d s g 0 SWMOD
.model SWMOD SW(VT=5 VH=0.1 RON={rds} ROFF=1e7)
VS s 0 DC 0
.end
"""
    return Template("mosfet", net, ["v(d)", "i(vs)"], dict(p, current=i, rds_on=rds, vbus=vbus, frequency=f,
                                                            duty=duty), "tran", tran)


TEMPLATES = {"led": led, "rc": rc, "buck": buck, "mosfet": mosfet}
