"""Every part against its ratings: voltage, current, power and the temperature its power gives it.

Each part's voltages and currents come from its own pins, so this works the same for a resistor, a
transistor model and a regulator subcircuit. Steady values (average current, RMS current, average power)
are taken over the settled window of the run; peaks (the most voltage a capacitor sees, the inrush through
a diode) over the whole run, start up included. Temperature is ambient plus average power times the part's
thermal resistance, with a MOSFET's on resistance rising as it heats. A use at or above 80% of a rating is
near the limit and above 100% is over it; for a part whose ratings are assumed (a generic stand in) going
over is reported as a warning to check the real part's datasheet.

Some checks are about whether the circuit works rather than whether it burns: a regulator in dropout, a
microcontroller below its minimum supply, a MOSFET only partly on, an op amp input outside its common mode
range, a fuse carrying more than it holds.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import numpy as np

from ..units import fmt
from .netlist import Circuit, Element
from .solve import Wave

NEAR = 0.8


@dataclass
class Item:
    what: str
    value: float
    limit: float | None
    unit: str
    use: float | None = None
    status: str = "ok"          # ok, near, over, unknown, info, fails (does not work), warn (works poorly)
    note: str = ""

    def text(self) -> str:
        v = f"{fmt(self.value)} {self.unit}".strip()
        if self.limit is not None and self.use is not None:
            return f"{self.what} {v} of {fmt(self.limit)} {self.unit} ({100 * self.use:.0f}%)".replace("  ", " ")
        return f"{self.what} {v}" + (f", {self.note}" if self.note else "")


@dataclass
class Stress:
    ref: str
    kind: str
    part: str
    items: list[Item] = field(default_factory=list)
    v: float = 0.0              # the voltage across it that matters most (peak)
    i: float = 0.0              # its main current (RMS over the settled window)
    p: float = 0.0              # average power it dissipates (negative when it supplies power)
    tj: float | None = None
    generic: bool = False
    source: str = ""

    @property
    def status(self) -> str:
        order = ["fails", "over", "warn", "near", "unknown", "ok", "info"]
        st = [it.status for it in self.items] or ["ok"]
        worst = min(st, key=lambda s: order.index(s) if s in order else 5)
        if self.generic and worst == "over":
            return "warn"
        return worst

    @property
    def worst(self) -> Item | None:
        rated = [it for it in self.items if it.use is not None]
        return max(rated, key=lambda it: it.use) if rated else None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["status"] = self.status
        w = self.worst
        d["use"] = w.use if w else None
        return d


def _rated(what: str, value: float, limit, unit: str, note: str = "") -> Item:
    if limit is None or limit <= 0:
        return Item(what, value, None, unit, None, "unknown", note)
    use = abs(value) / limit
    return Item(what, value, limit, unit, use, "over" if use > 1.0 else "near" if use >= NEAR else "ok", note)


class _Pins:
    """A part's pin voltages and currents in one run, by pin name."""

    def __init__(self, e: Element, w: Wave):
        self.e, self.w = e, w
        self.cur = w.i.get(e.name) or w.i.get("X" + e.name) or [np.zeros_like(w.t)] * len(e.nodes)
        self.idx = {p: k for k, p in enumerate(e.pins)}

    def v(self, pin: str) -> np.ndarray:
        return self.w.volt(self.e.nodes[self.idx[pin]])

    def i(self, pin: str) -> np.ndarray:
        x = self.cur[self.idx[pin]]
        return x if x is not None else np.zeros_like(self.w.t)

    def power(self) -> np.ndarray:
        p = np.zeros_like(self.w.t)
        for k, n in enumerate(self.e.nodes):
            if self.cur[k] is not None:
                p = p + self.w.volt(n) * self.cur[k]
        return p


def _theta(e: Element) -> float | None:
    r = e.part.get("ratings", {})
    hs = e.part.get("params", {}).get("heatsink")
    if hs is not None and r.get("theta_jc"):
        return r["theta_jc"] + 0.5 + hs
    return r.get("theta_ja")


def _temp(e: Element, p_avg: float, ta: float, tc: float = 0.0) -> Item | None:
    th, tj_max = _theta(e), e.part.get("ratings", {}).get("tj_max_C")
    if not th or not tj_max:
        return None
    p0 = max(p_avg, 0.0)
    den = 1 - th * p0 * tc
    if den <= 0.05:
        return Item("temperature", 999.0, tj_max, "C", 99.0, "over", "thermal runaway: it heats faster than it "
                    "can shed the heat")
    tj = (ta + th * p0 * (1 - 25 * tc)) / den if tc else ta + th * p0
    use = (tj - ta) / max(tj_max - ta, 1e-9)
    hs = e.part.get("params", {}).get("heatsink")
    note = f"{fmt(p0)} W at {fmt(th)} C/W" + (" with the heatsink" if hs is not None else "") + f" from {fmt(ta)} C"
    return Item("temperature", tj, tj_max, "C", use, "over" if use > 1 else "near" if use >= NEAR else "ok", note)


def evaluate(c: Circuit, w: Wave, ambient: float = 25.0, peaks: list[Wave] | None = None) -> list[Stress]:
    """The stress of every part in one solved run (peaks also looked for in the other runs given)."""
    out = []
    for e in c.elements:
        if e.kind in ("parasitic", "coupling", "line", "xspice") or not e.nodes:
            continue
        try:
            s = _one(e, w, ambient, peaks or [])
        except (KeyError, IndexError):
            continue
        if s is not None:
            out.append(s)
    return out


def _peak(e: Element, waves: list[Wave], fn) -> float:
    vals = []
    for w in waves:
        try:
            vals.append(fn(_Pins(e, w)))
        except (KeyError, IndexError):
            continue
    return max(vals) if vals else 0.0


def _one(e: Element, w: Wave, ta: float, more: list[Wave]) -> Stress | None:
    P = _Pins(e, w)
    every = [w, *more]
    r = e.part.get("ratings", {})
    prm = e.part.get("params", {})
    s = Stress(e.ref, e.kind, e.part.get("key", ""), generic=bool(e.part.get("generic")),
               source=e.part.get("source", ""))
    pw = P.power()
    s.p = w.mean(pw)
    k = e.kind
    pins = e.pins
    if len(pins) == 2:
        a, b = pins
        vab = P.v(a) - P.v(b)
        ia = P.i(a)
        s.v = _peak(e, every, lambda q: float(np.max(np.abs(q.v(a) - q.v(b)))))
        s.i = w.rms(ia)
    if k == "resistor":
        derate = 1.0 if ta <= r.get("derate_from_C", 70) else max(0.0, (r.get("zero_at_C", 155) - ta) /
                                                                      (r.get("zero_at_C", 155) - r.get("derate_from_C", 70)))
        allowed = r.get("power_W", 0.25) * derate
        hot = ta + (r.get("zero_at_C", 155) - r.get("derate_from_C", 70)) * s.p / max(r.get("power_W", 0.25), 1e-9)
        s.items.append(_rated("power", s.p, allowed, "W", f"surface near {fmt(hot)} C" +
                              (f", rating derated to {fmt(allowed)} W at {fmt(ta)} C" if derate < 1 else "")))
        s.items.append(_rated("voltage", s.v, r.get("voltage_V"), "V"))
    elif k in ("capacitor", "electrolytic", "tantalum"):
        s.items.append(_rated("voltage", s.v, r.get("voltage_V"), "V",
                              "" if r.get("voltage_V") else f"rating not given; a {fmt(_next_volts(s.v))} V part "
                              "covers it"))
        if k != "capacitor":
            vmin = -_peak(e, every, lambda q: float(np.max(-(q.v(a) - q.v(b)))))
            if vmin < -0.5:
                s.items.append(Item("reverse voltage", -vmin, 0.5, "V", -vmin / 0.5, "over",
                                    "a polarized capacitor is reverse biased"))
            if r.get("ripple_A"):
                s.items.append(_rated("ripple current", s.i, r["ripple_A"], "A"))
            elif w.kind == "tran":
                s.items.append(Item("ripple current", s.i, None, "A", None, "info", "RMS, for checking against the "
                                    "part's ripple rating"))
    elif k == "inductor":
        ipk = _peak(e, every, lambda q: float(np.max(np.abs(q.i(a)))))
        s.items.append(_rated("peak current", ipk, r.get("isat_A"), "A", "" if r.get("isat_A") else
                              "saturation current not given; pick one rated above this"))
        if r.get("irms_A"):
            s.items.append(_rated("RMS current", s.i, r["irms_A"], "A"))
    elif k in ("diode", "schottky", "zener", "led"):
        fwd = np.maximum(ia, 0)
        iavg = w.mean(fwd)
        s.i = iavg
        ipk = _peak(e, every, lambda q: float(np.max(q.i(a))))
        vrev = _peak(e, every, lambda q: float(np.max(q.v(b) - q.v(a))))
        if k == "zener":
            s.items.append(_rated("power", s.p, r.get("pd_W"), "W"))
        else:
            s.items.append(_rated("forward current", iavg, r.get("if_A"), "A"))
            if r.get("ifsm_A") and ipk > 1.5 * max(iavg, 1e-6):
                s.items.append(_rated("peak current", ipk, r.get("ifsm_A"), "A", "surge rating"))
            s.items.append(_rated("reverse voltage", max(vrev, 0.0), r.get("vr_V"), "V"))
        if k == "led" and iavg < 0.5e-3:
            s.items.append(Item("brightness", iavg, None, "A", None, "warn", "too little current to light"))
        t = _temp(e, s.p, ta)
        if t:
            s.items.append(t)
            s.tj = t.value
    elif k in ("npn", "pnp"):
        sg = 1 if k == "npn" else -1
        vce = _peak(e, every, lambda q: float(np.max(sg * (q.v("c") - q.v("e")))))
        veb = _peak(e, every, lambda q: float(np.max(sg * (q.v("e") - q.v("b")))))
        ic = sg * P.i("c")
        s.v, s.i = vce, w.mean(np.abs(ic))
        icpk = _peak(e, every, lambda q: float(np.max(np.abs(q.i("c")))))
        s.items += [_rated("collector voltage", vce, r.get("vceo_V"), "V"),
                    _rated("collector current", s.i, r.get("ic_A"), "A")]
        if r.get("ic_A") and icpk > 1.2 * r["ic_A"]:
            s.items.append(_rated("peak collector current", icpk, 2 * r["ic_A"], "A", "pulse limit taken as twice "
                                  "the continuous rating"))
        if veb > 0.5 * (r.get("vebo_V") or 99):
            s.items.append(_rated("reverse base voltage", veb, r.get("vebo_V"), "V"))
        t = _temp(e, s.p, ta)
        if t:
            s.items.append(t)
            s.tj = t.value
    elif k in ("nmos", "pmos"):
        _mosfet(e, P, w, every, s, r, prm, ta)
    elif k == "regulator":
        _regulator(e, P, w, every, s, r, prm, ta)
    elif k in ("opamp", "comparator", "timer"):
        _amp(e, P, w, every, s, r, prm, ta)
    elif k == "mcu":
        _mcu(e, P, w, every, s, r, prm, ta)
    elif k == "charger":
        vin = _peak(e, every, lambda q: float(np.max(q.v("vin") - q.v("gnd"))))
        ich = -w.mean(P.i("bat"))
        s.v, s.i = vin, ich
        s.items += [_rated("input voltage", vin, r.get("vin_V"), "V"), _rated("charge current", ich, r.get("iout_A"),
                                                                              "A")]
        if w.mean(P.v("vin") - P.v("gnd")) < prm.get("vin_min", 0):
            s.items.append(Item("input voltage", w.mean(P.v("vin") - P.v("gnd")), prm["vin_min"], "V", None,
                                "fails", f"below the {fmt(prm['vin_min'])} V it needs to charge"))
        t = _temp(e, s.p, ta)
        if t:
            if t.value > prm.get("t_reg", 999):
                t.note += f"; above {prm['t_reg']} C its thermal regulation cuts the charge current"
                t.status = "warn" if t.status == "ok" else t.status
            s.items.append(t)
            s.tj = t.value
    elif k == "reference":
        vka = _peak(e, every, lambda q: float(np.max(q.v("k") - q.v("a"))))
        ik = w.mean(P.i("k"))
        s.v, s.i = vka, ik
        s.items += [_rated("cathode voltage", vka, r.get("vka_V"), "V"), _rated("cathode current", ik, r.get("ika_A"),
                                                                                "A")]
        if ik < prm.get("ik_min", 0):
            s.items.append(Item("cathode current", ik, prm["ik_min"], "A", None, "warn",
                                f"below the {fmt(prm['ik_min'] * 1000)} mA it needs to regulate"))
        t = _temp(e, s.p, ta)
        if t:
            s.items.append(t)
    elif k == "opto":
        s.items += [_rated("LED current", w.mean(np.maximum(P.i("a"), 0)), r.get("if_A"), "A"),
                    _rated("output voltage", _peak(e, every, lambda q: float(np.max(q.v("c") - q.v("e")))),
                           r.get("vceo_V"), "V"),
                    _rated("output current", w.mean(np.abs(P.i("c"))), r.get("ic_A"), "A"),
                    _rated("power", s.p, r.get("pd_W"), "W")]
    elif k in ("battery", "supply", "signal", "isource"):
        s.p = -w.mean(pw)            # what it delivers
        if len(pins) == 2:
            idis = -w.mean(ia)
            s.i = idis
            if r.get("iout_A"):
                s.items.append(_rated("current drawn", idis, r["iout_A"], "A"))
            else:
                s.items.append(Item("current drawn", idis, None, "A", None, "info", f"delivers {fmt(s.p)} W"))
    elif k == "fuse":
        s.items.append(_rated("current", s.i, r.get("ihold_A"), "A"))
        if r.get("ihold_A") and s.i > r["ihold_A"]:
            s.items[-1].status, s.items[-1].note = "fails", "the fuse would open"
    elif k == "relay":
        s.items.append(_rated("coil voltage", s.v, r.get("vcoil_V"), "V"))
    elif k == "motor":
        ipk = _peak(e, every, lambda q: float(np.max(np.abs(q.i(pins[0])))))
        s.items.append(Item("current", s.i, None, "A", None, "info", f"{fmt(ipk)} A peak at start"))
    elif k == "load":
        s.items.append(Item("power", s.p, None, "W", None, "info", f"draws {fmt(s.i)} A"))
    else:
        s.items.append(Item("power", s.p, None, "W", None, "info", "no ratings known"))
    return s


def _next_volts(v: float) -> float:
    from .parts import CAP_VOLTS
    return next((x for x in CAP_VOLTS if x >= 1.5 * v), CAP_VOLTS[-1])


def _mosfet(e, P, w, every, s, r, prm, ta):
    sg = 1 if e.kind == "nmos" else -1
    vds = _peak(e, every, lambda q: float(np.max(sg * (q.v("d") - q.v("s")))))
    vgs = _peak(e, every, lambda q: float(np.max(np.abs(q.v("g") - q.v("s")))))
    idr = sg * P.i("d")
    s.v, s.i = vds, w.rms(idr)
    idpk = _peak(e, every, lambda q: float(np.max(np.abs(q.i("d")))))
    s.items += [_rated("drain voltage", vds, r.get("vds_V"), "V"), _rated("gate voltage", vgs, r.get("vgs_V"), "V"),
                _rated("drain current", s.i, r.get("id_A"), "A")]
    if r.get("idm_A") and idpk > r.get("id_A", 1e9):
        s.items.append(_rated("peak drain current", idpk, r["idm_A"], "A"))
    on = np.abs(idr) > 0.5 * max(float(np.max(np.abs(idr))), 1e-12)
    vds_on = float(np.mean(np.abs(P.v("d") - P.v("s"))[on])) if on.any() else 0.0
    vgs_on = float(np.mean(sg * (P.v("g") - P.v("s"))[on])) if on.any() else 0.0
    conducting = s.i > 0.1
    t = _temp(e, s.p, ta, prm.get("rds_tempco", 0.006) if conducting and vds_on < 1.0 else 0.0)
    if t:
        s.items.append(t)
        s.tj = t.value
    spec = prm.get("spec_vgs")
    if spec and s.i > 0.01 and vgs_on < spec - 0.2 and vds_on > 0.2:
        s.items.append(Item("gate drive", vgs_on, spec, "V", None, "warn",
                            f"only partly on: {fmt(vgs_on)} V on the gate, below the {fmt(spec)} V its on resistance "
                            f"is specified at, so it drops {fmt(vds_on)} V"))


def _regulator(e, P, w, every, s, r, prm, ta):
    third = "adj" if "adj" in e.pins else "gnd"
    if r.get("vdiff_V"):
        vin = _peak(e, every, lambda q: float(np.max(q.v("in") - q.v("out"))))
        s.items.append(_rated("input to output voltage", vin, r["vdiff_V"], "V"))
    else:
        vin = _peak(e, every, lambda q: float(np.max(q.v("in") - q.v(third))))
        s.items.append(_rated("input voltage", vin, r.get("vin_V"), "V"))
    iout = -w.mean(P.i("out"))
    s.v, s.i = vin, iout
    s.items.append(_rated("output current", iout, r.get("iout_A"), "A"))
    t = _temp(e, s.p, ta)
    if t:
        if t.value > 150:
            t.note += "; its thermal shutdown would cut out"
        s.items.append(t)
        s.tj = t.value
    vo = w.mean(P.v("out") - P.v(third))
    vi = w.mean(P.v("in") - P.v(third))
    want = prm.get("vref") if prm.get("adjustable") else prm.get("vout")
    if want and vi > 0.3 * want and vo < 0.97 * want:
        what = "reference" if prm.get("adjustable") else "output"
        s.items.append(Item("regulation", vo, want, "V", None, "fails",
                            f"in dropout: its {what} is {fmt(vo)} V instead of {fmt(want)} V, since {fmt(vi)} V in "
                            f"leaves less than its {fmt(prm.get('dropout', 0))} V of headroom"))
    if prm.get("min_load") and 0 <= iout < prm["min_load"]:
        s.items.append(Item("load", iout, prm["min_load"], "A", None, "warn",
                            f"below the {fmt(prm['min_load'] * 1000)} mA it needs to regulate"))


def _amp(e, P, w, every, s, r, prm, ta):
    neg, pos = ("gnd", "vcc") if e.kind == "timer" else ("vee", "vcc")
    vs = _peak(e, every, lambda q: float(np.max(q.v(pos) - q.v(neg))))
    vs_avg = w.mean(P.v(pos) - P.v(neg))
    iout = _peak(e, every, lambda q: float(np.max(np.abs(q.i("out")))))
    s.v, s.i = vs, w.rms(P.i("out"))
    s.items += [_rated("supply voltage", vs, r.get("vsupply_V"), "V"), _rated("output current", iout, r.get("iout_A"),
                                                                              "A")]
    if vs_avg < prm.get("vs_min", 0):
        s.items.append(Item("supply voltage", vs_avg, prm["vs_min"], "V", None, "fails",
                            f"below the {fmt(prm['vs_min'])} V it needs"))
    if e.kind in ("opamp", "comparator"):
        hi = np.maximum(P.v("inp"), P.v("inn")) - (P.v("vcc") - prm.get("cm_high", 0))
        lo = (P.v("vee") + prm.get("cm_low", 0)) - np.minimum(P.v("inp"), P.v("inn"))
        over = max(float(np.max(hi)), float(np.max(lo)))
        if over > 0.05:
            s.items.append(Item("input range", over, None, "V", None, "warn",
                                f"an input goes {fmt(over)} V outside its common mode range, where it misbehaves"))
    if e.kind == "opamp":
        top = P.v("vcc") - prm.get("voh", 0) - 0.05
        bot = P.v("vee") + prm.get("vol", 0) + 0.05
        vo = P.v("out")
        frac = w.mean(((vo >= top) | (vo <= bot)).astype(float))
        if frac > 0.02:
            s.items.append(Item("output swing", frac, None, "", None, "info",
                                f"the output sits at its swing limit {100 * frac:.0f}% of the time (clipping, or "
                                "used as a comparator)"))
    t = _temp(e, s.p, ta)
    if t:
        s.items.append(t)
        s.tj = t.value


def _mcu(e, P, w, every, s, r, prm, ta):
    from .ics import GROUND_PIN, SUPPLY_PIN

    gnd = next((p for p in e.pins if GROUND_PIN.match(p)), None)
    if gnd is None:
        return
    sup = [p for p in e.pins if SUPPLY_PIN.match(p)] or [e.pins[0]]
    vdd = max(_peak(e, every, lambda q, p=p: float(np.max(q.v(p) - q.v(gnd)))) for p in sup)
    vavg = min(w.mean(P.v(p) - P.v(gnd)) for p in sup)
    s.v = vdd
    it = _rated("supply voltage", vdd, r.get("vdd_V"), "V")
    if it.status in ("ok", "near") and vdd > r.get("vdd_op_V", 99):
        it.status, it.note = "warn", f"above the {fmt(r['vdd_op_V'])} V it is meant to run at"
    s.items.append(it)
    if vavg < prm.get("vmin", 0):
        s.items.append(Item("supply voltage", vavg, prm["vmin"], "V", None, "fails",
                            f"below the {fmt(prm['vmin'])} V it needs to run"))
    total = sum(w.mean(P.i(p)) for p in sup)
    s.i = total
    s.items.append(_rated("supply current", total, r.get("total_A"), "A"))
    worst_pin, worst = None, 0.0
    for p in e.pins:
        if p in sup or GROUND_PIN.match(p):
            continue
        ip = _peak(e, every, lambda q, p=p: float(np.max(np.abs(q.i(p)))))
        if ip > worst:
            worst_pin, worst = p, ip
    if worst_pin:
        s.items.append(_rated(f"pin {worst_pin} current", worst, r.get("pin_A"), "A"))
    t = _temp(e, s.p, ta)
    if t:
        s.items.append(t)
        s.tj = t.value
    _ = math
