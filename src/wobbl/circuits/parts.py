"""Parts library: common parts and part numbers, each with a SPICE model and its datasheet ratings.

An entry is a handful of datasheet numbers, and the model is written from them when a circuit uses the part,
so the library stays small. Diodes and LEDs are fitted to their forward curves, transistors carry Gummel Poon
or VDMOS parameters matched to their datasheet gain, threshold and on resistance, and ICs (regulators, op
amps, the 555, chargers, microcontrollers) are behavioral subcircuits with their dropout, swing, current
limits and supply current. Ratings are absolute maxima and thermal resistances from the same datasheets, and
the stress check reads them. Where a datasheet gives a maximum and a typical value, the model uses the one
that runs the part harder (the higher on resistance or forward drop), so heating is not understated.

A part that is not here gets a generic model of its kind with stated parameters, and the report says so.
A maker's own SPICE model can be used instead by saving its file in ~/.wobbl/parts (read by bind.py) or by
.include in a netlist; a model the netlist or that folder defines wins over the built in one, and the
library's ratings still apply to it.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

VT = 8.617333e-5 * 300.15        # thermal voltage at 27 C as ngspice computes it, V


@dataclass
class Part:
    key: str                       # library name
    kind: str                      # resistor, capacitor, inductor, diode, schottky, zener, led, npn, pnp, nmos,
                                   # pmos, regulator, opamp, comparator, timer, charger, mcu, battery, ...
    ratings: dict = field(default_factory=dict)
    letter: str = ""               # the SPICE letter the model needs: d, q, m or x ('' for passives)
    mname: str = ""                # model or subcircuit name on the element line
    text: str = ""                 # the .model line or .subckt block
    ports: tuple = ()              # subcircuit ports, in order
    source: str = ""               # where the numbers come from, for the report
    generic: bool = False          # stands in for a part the library does not have
    params: dict = field(default_factory=dict)   # electrical numbers the hand checks use (vf, vout, vref ...)


def _g(x: float) -> str:
    return f"{x:.4g}"


# --- passives ---------------------------------------------------------------------------------------------
# chip resistor sizes: rated power at 70 C and the most voltage across them; leaded parts by body (DIN code)
RES_PKG = {"0201": (0.05, 25), "0402": (0.0625, 50), "0603": (0.1, 75), "0805": (0.125, 150),
           "1206": (0.25, 200), "1210": (0.5, 200), "1812": (0.75, 200), "2010": (0.75, 200), "2512": (1.0, 200),
           "0204": (0.25, 200), "0207": (0.25, 250), "0309": (0.5, 350), "0411": (1.0, 350), "0414": (2.0, 500),
           "0516": (2.0, 500), "0614": (3.0, 500), "0617": (3.0, 750), "0918": (4.0, 750), "0922": (5.0, 750)}
CAP_VOLTS = [4, 6.3, 10, 16, 25, 35, 50, 63, 100, 160, 200, 250, 400, 450, 630, 1000]
POWERS = [0.0625, 0.1, 0.125, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0]
E12 = [1.0, 1.2, 1.5, 1.8, 2.2, 2.7, 3.3, 3.9, 4.7, 5.6, 6.8, 8.2]
E24 = [1.0, 1.1, 1.2, 1.3, 1.5, 1.6, 1.8, 2.0, 2.2, 2.4, 2.7, 3.0, 3.3, 3.6, 3.9, 4.3, 4.7, 5.1, 5.6, 6.2, 6.8, 7.5,
       8.2, 9.1]


def e_series(x: float, series: list[float] = E24, up: bool = True) -> float:
    """The nearest standard value at or above x (up) or at or below it."""
    if x <= 0:
        return series[0]
    dec = 10 ** math.floor(math.log10(x))
    vals = [m * dec for m in series] + [10 * dec] + [m * dec / 10 for m in series]
    vals.sort()
    if up:
        return min(v for v in vals if v >= x * 0.999)
    return max(v for v in vals if v <= x * 1.001)


def resistor(pkg: str = "", power: float | None = None, volts: float | None = None, tol: float = 0.05) -> Part:
    p, v = RES_PKG.get(pkg, (0.25, 250))
    told = f"a {pkg} resistor" if pkg in RES_PKG else "a 1/4 W leaded resistor, the usual hobby part"
    r = {"power_W": power or p, "voltage_V": volts or v, "derate_from_C": 70.0, "zero_at_C": 155.0,
         "tolerance": tol}
    src = "rating as marked" if power else f"rating of {told}"
    return Part("resistor", "resistor", r, source=src, generic=not (pkg in RES_PKG or power))


def capacitor(volts: float | None = None, dielectric: str = "", pkg: str = "", polarized: bool = False,
              ripple: float | None = None, tol: float | None = None) -> Part:
    d = dielectric.lower()
    kind = "electrolytic" if polarized and d not in ("tantalum", "polymer") else ("tantalum" if d == "tantalum"
                                                                                  else "capacitor")
    r = {"voltage_V": volts, "tolerance": tol if tol is not None else (0.2 if polarized or d in ("x5r", "x7r", "y5v")
                                                                        else 0.1)}
    if ripple:
        r["ripple_A"] = ripple
    if polarized:
        r["reverse_V"] = 0.0 if kind == "electrolytic" else 0.1 * (volts or 10)
    src = (f"{volts:g} V rating as marked" if volts else "voltage rating not given") + (f", {dielectric}"
                                                                                         if dielectric else "")
    return Part("capacitor", kind, r, source=src, generic=volts is None, params={"dielectric": d, "package": pkg})


def inductor(isat: float | None = None, irms: float | None = None, dcr: float | None = None) -> Part:
    r = {"isat_A": isat, "irms_A": irms or isat}
    src = "saturation and RMS current as marked" if isat else "current ratings not given"
    return Part("inductor", "inductor", r, source=src, generic=isat is None, params={"dcr": dcr})


# --- diodes -----------------------------------------------------------------------------------------------
def _fit_diode(v1: float, i1: float, v2: float, i2: float, n: float) -> tuple[float, float]:
    """IS and RS so the diode equation passes through two points of the forward curve at emission n."""
    nvt = n * VT
    # v = nvt ln(i/IS) + i RS  ->  v - nvt ln i = -nvt ln IS + i RS, two equations in ln IS and RS
    a1, a2 = v1 - nvt * math.log(i1), v2 - nvt * math.log(i2)
    rs = max((a2 - a1) / (i2 - i1), 1e-4)
    lnis = (rs * i1 - a1) / nvt
    return math.exp(lnis), rs


# name: (kind, vrrm V, if_avg A, ifsm A, (V, A), (V, A), n, cjo F, tt s, theta_ja C/W, tj_max C, package)
# forward points are typical curve readings from the maker's datasheet
DIODES = {
    "1N4148": ("diode", 100, 0.2, 1.0, (0.62, 1e-3), (0.85, 0.1), 1.75, 4e-12, 4e-9, 350, 175, "DO-35"),
    "1N4001": ("diode", 50, 1.0, 30, (0.68, 0.01), (0.93, 1.0), 1.85, 25e-12, 5.7e-6, 50, 175, "DO-41"),
    "1N5408": ("diode", 1000, 3.0, 200, (0.65, 0.01), (0.95, 3.0), 1.7, 60e-12, 6e-6, 20, 150, "DO-201AD"),
    "UF4007": ("diode", 1000, 1.0, 30, (0.95, 0.01), (1.45, 1.0), 2.0, 15e-12, 75e-9, 50, 150, "DO-41"),
    "1N5817": ("schottky", 20, 1.0, 25, (0.30, 0.1), (0.42, 1.0), 1.05, 110e-12, 0, 80, 125, "DO-41"),
    "1N5818": ("schottky", 30, 1.0, 25, (0.31, 0.1), (0.45, 1.0), 1.05, 110e-12, 0, 80, 125, "DO-41"),
    "1N5819": ("schottky", 40, 1.0, 25, (0.32, 0.1), (0.48, 1.0), 1.05, 110e-12, 0, 80, 125, "DO-41"),
    "B5819W": ("schottky", 40, 1.0, 25, (0.32, 0.1), (0.48, 1.0), 1.05, 110e-12, 0, 250, 125, "SOD-123"),
    "SS14": ("schottky", 40, 1.0, 30, (0.33, 0.1), (0.50, 1.0), 1.05, 110e-12, 0, 88, 125, "SMA"),
    "SS34": ("schottky", 40, 3.0, 80, (0.30, 0.1), (0.50, 3.0), 1.05, 250e-12, 0, 55, 125, "SMC"),
    "SS54": ("schottky", 40, 5.0, 150, (0.30, 0.1), (0.55, 5.0), 1.05, 400e-12, 0, 40, 125, "SMC"),
    "BAT54": ("schottky", 30, 0.2, 0.6, (0.32, 1e-3), (0.80, 0.1), 1.05, 10e-12, 0, 500, 125, "SOT-23"),
    "BAT43": ("schottky", 30, 0.2, 4.0, (0.33, 2e-3), (0.85, 0.2), 1.05, 7e-12, 0, 300, 125, "DO-35"),
}
DIODE_ALIAS = {"1N914": "1N4148", "LL4148": "1N4148", "1N4148W": "1N4148", "1N4002": "1N4001",
               "1N4003": "1N4001", "1N4004": "1N4001", "1N4005": "1N4001", "1N4006": "1N4001", "1N4007": "1N4001",
               "M7": "1N4001", "M4": "1N4001", "M1": "1N4001", "MBR140": "1N5819", "SS12": "SS14", "SS16": "SS14",
               "SS24": "SS14", "SS32": "SS34", "SS36": "SS34", "SB140": "1N5819", "SR160": "1N5819"}
RECT_VOLTS = {"1N4001": 50, "1N4002": 100, "1N4003": 200, "1N4004": 400, "1N4005": 600, "1N4006": 800,
              "1N4007": 1000, "M1": 50, "M4": 400, "M7": 1000, "SS12": 20, "SS16": 60, "SS24": 40, "SS32": 20,
              "SS36": 60, "SR160": 60}


def diode(name: str) -> Part:
    key = DIODE_ALIAS.get(name, name)
    kind, vr, ifav, ifsm, p1, p2, n, cjo, tt, th, tj, pkg = DIODES[key]
    vr = RECT_VOLTS.get(name, vr)
    isat, rs = _fit_diode(p1[0], p1[1], p2[0], p2[1], n)
    m = f"D_{name}".replace("-", "_")
    text = (f".model {m} D(IS={_g(isat)} N={n} RS={_g(rs)} CJO={_g(cjo)} M=0.4 TT={_g(tt)} BV={_g(vr * 1.2)} "
            f"IBV=10u)")
    r = {"vr_V": vr, "if_A": ifav, "ifsm_A": ifsm, "theta_ja": th, "tj_max_C": tj}
    return Part(name, kind, r, "d", m, text, source=f"{name} datasheet: forward curve, {vr:g} V, {ifav:g} A, {pkg}",
                params={"vf": p2[0], "at": p2[1], "package": pkg})


# indicator LEDs by color: forward voltage at 20 mA (as the LED template uses)
LED_VF = {"red": 2.0, "orange": 2.05, "yellow": 2.1, "green": 3.1, "blue": 3.2, "white": 3.2, "uv": 3.4,
          "ir": 1.3, "pink": 3.2, "purple": 3.3, "amber": 2.05, "warm-white": 3.2}
LED_PKG = {"0402": 0.020, "0603": 0.025, "0805": 0.030, "1206": 0.030, "3mm": 0.030, "5mm": 0.030, "plcc": 0.030}


def led_model(name: str, vf: float, i_ref: float, rs: float = 1.5) -> str:
    """The LED template's diode equation: n from the forward voltage, IS through vf at the reference current."""
    n = max(2.0, vf / 1.2)
    isat = i_ref / math.exp((vf - i_ref * rs) / (n * VT))
    return f".model {name} D(IS={isat:.6e} N={n:.4g} RS={rs} BV=6 IBV=10u CJO=20p)"


def led(color: str = "red", pkg: str = "", power: float = 0.0, vf: float | None = None) -> Part:
    c = color.lower() if color.lower() in LED_VF else "red"
    if power >= 0.9:          # 1 W and 3 W emitters
        imax, iref, rs = (0.35, 0.35, 1.5) if power < 2 else (0.7, 0.35, 0.8)
        th = 15.0
    else:
        imax, iref, rs, th = LED_PKG.get(pkg, 0.030), 0.020, 1.5, 0.0
    v = vf or (LED_VF[c] if power < 0.9 else {"red": 2.2, "orange": 2.2, "yellow": 2.2, "amber": 2.2}.get(c, 3.0))
    m = f"LED_{c.replace('-', '_')}_{int(iref * 1000)}"
    r = {"if_A": imax, "vr_V": 5.0}
    if th:
        r.update({"theta_ja": th, "tj_max_C": 125})
    what = f"{power:g} W emitter" if power >= 0.9 else (f"{pkg} indicator LED" if pkg in LED_PKG else
                                                         "indicator LED")
    return Part(f"LED-{c}", "led", r, "d", m, led_model(m, v, iref, rs),
                source=f"{c} {what}: {v:g} V at {iref * 1000:g} mA, {imax * 1000:g} mA most",
                generic=not pkg and power < 0.9, params={"vf": v, "at": iref, "color": c})


def zener(vz: float, pd: float = 0.5, name: str = "") -> Part:
    izt = 0.25 * pd / vz             # the datasheet test current is about a quarter of the rated power
    zz = 7.0 if vz <= 6.2 else 0.2 * vz ** 1.5
    tj = 175
    m = f"DZ_{str(vz).replace('.', 'V')}"
    rs = min(zz, 30) / 2
    text = f".model {m} D(IS=1n N=1.6 RS={_g(rs)} BV={_g(vz - izt * rs)} IBV={_g(izt)} CJO=100p)"
    r = {"pd_W": pd, "if_A": 0.2, "theta_ja": (tj - 25) / pd, "tj_max_C": tj}
    return Part(name or f"zener {vz:g} V", "zener", r, "d", m, text,
                source=f"{vz:g} V {pd:g} W zener: {zz:.3g} ohm at {izt * 1000:.3g} mA", params={"vz": vz, "izt": izt})


_Z1N47 = [3.3, 3.6, 3.9, 4.3, 4.7, 5.1, 5.6, 6.2, 6.8, 7.5, 8.2, 9.1, 10, 11, 12, 13, 15, 16, 18, 20, 22, 24, 27, 30,
          33, 36, 39, 43, 47, 51, 56, 62, 68, 75, 82, 91, 100]


def _zener_name(name: str) -> Part | None:
    u = name.upper()
    m = re.match(r"^1N(47\d\d)A?$", u)
    if m and 0 <= int(m.group(1)) - 4728 < len(_Z1N47):
        return zener(_Z1N47[int(m.group(1)) - 4728], 1.0, name)
    m = re.match(r"^(BZX55|BZX79|BZX84|BZT52|MMSZ|BZV55|ZPD|ZMM)[-_]?[A-Z]?(\d+)V(\d+)?", u)
    if m:
        pd = {"BZX84": 0.3, "BZT52": 0.5, "MMSZ": 0.5}.get(m.group(1), 0.5)
        return zener(float(f"{m.group(2)}.{m.group(3) or 0}"), pd, name)
    return None


# --- bipolar transistors ----------------------------------------------------------------------------------
# standard Gummel Poon parameter sets for the four classic small signal parts
_GP = {
    "3904": "IS=6.734f XTI=3 EG=1.11 VAF=74.03 BF=416.4 NE=1.259 ISE=6.734f IKF=66.78m XTB=1.5 BR=.7371 NC=2 "
            "RC=1 CJC=3.638p MJC=.3085 VJC=.75 FC=.5 CJE=4.493p MJE=.2593 VJE=.75 TR=239.5n TF=301.2p ITF=.4 VTF=4 "
            "XTF=2 RB=10",
    "3906": "IS=1.41f XTI=3 EG=1.11 VAF=18.7 BF=180.7 NE=1.5 ISE=0 IKF=80m XTB=1.5 BR=4.977 NC=2 RC=2.5 "
            "CJC=9.728p MJC=.5776 VJC=.75 FC=.5 CJE=8.063p MJE=.3677 VJE=.75 TR=33.42n TF=179.3p ITF=.4 VTF=4 XTF=6 "
            "RB=10",
    "2222": "IS=14.34f XTI=3 EG=1.11 VAF=74.03 BF=255.9 NE=1.307 ISE=14.34f IKF=.2847 XTB=1.5 BR=6.092 NC=2 RC=1 "
            "CJC=7.306p MJC=.3416 VJC=.75 FC=.5 CJE=22.01p MJE=.377 VJE=.75 TR=46.91n TF=411.1p ITF=.6 VTF=1.7 "
            "XTF=3 RB=10",
    "2907": "IS=650.6E-18 XTI=3 EG=1.11 VAF=115.7 BF=231.7 NE=1.829 ISE=54.81f IKF=1.079 XTB=1.5 BR=3.563 NC=2 "
            "RC=.715 CJC=14.76p MJC=.5383 VJC=.75 FC=.5 CJE=19.82p MJE=.3357 VJE=.75 TR=111.3n TF=603.7p ITF=.65 "
            "VTF=5 XTF=1.7 RB=10",
}
# name: (polarity, vceo V, vebo V, ic A, theta_ja, theta_jc or 0, tj_max, parameters or typical hFE, package)
BJTS = {
    "2N3904": ("npn", 40, 6, 0.2, 200, 83.3, 150, "3904", "TO-92"),
    "MMBT3904": ("npn", 40, 6, 0.2, 357, 0, 150, "3904", "SOT-23"),
    "2N3906": ("pnp", 40, 5, 0.2, 200, 83.3, 150, "3906", "TO-92"),
    "MMBT3906": ("pnp", 40, 5, 0.2, 357, 0, 150, "3906", "SOT-23"),
    "2N2222A": ("npn", 40, 6, 0.6, 200, 83.3, 150, "2222", "TO-92"),
    "MMBT2222A": ("npn", 40, 6, 0.6, 357, 0, 150, "2222", "SOT-23"),
    "2N2907A": ("pnp", 60, 5, 0.6, 200, 83.3, 150, "2907", "TO-92"),
    "MMBT2907A": ("pnp", 60, 5, 0.6, 357, 0, 150, "2907", "SOT-23"),
    "BC547": ("npn", 45, 6, 0.1, 250, 0, 150, 290, "TO-92"),
    "BC548": ("npn", 30, 5, 0.1, 250, 0, 150, 290, "TO-92"),
    "BC546": ("npn", 65, 6, 0.1, 250, 0, 150, 290, "TO-92"),
    "BC557": ("pnp", 45, 5, 0.1, 250, 0, 150, 250, "TO-92"),
    "BC558": ("pnp", 30, 5, 0.1, 250, 0, 150, 250, "TO-92"),
    "BC337": ("npn", 45, 5, 0.8, 200, 0, 150, 250, "TO-92"),
    "BC327": ("pnp", 45, 5, 0.8, 200, 0, 150, 250, "TO-92"),
    "S8050": ("npn", 25, 5, 0.5, 200, 0, 150, 200, "TO-92"),
    "S8550": ("pnp", 25, 5, 0.5, 200, 0, 150, 200, "TO-92"),
    "BD139": ("npn", 80, 5, 1.5, 100, 10, 150, 100, "TO-126"),
    "BD140": ("pnp", 80, 5, 1.5, 100, 10, 150, 100, "TO-126"),
    "TIP31C": ("npn", 100, 5, 3.0, 62.5, 3.125, 150, 50, "TO-220"),
    "TIP32C": ("pnp", 100, 5, 3.0, 62.5, 3.125, 150, 50, "TO-220"),
    "TIP41C": ("npn", 100, 5, 6.0, 62.5, 1.92, 150, 50, "TO-220"),
    "TIP42C": ("pnp", 100, 5, 6.0, 62.5, 1.92, 150, 50, "TO-220"),
    "2N3055": ("npn", 60, 7, 15.0, 35, 1.52, 200, 40, "TO-3"),
    "TIP120": ("npn", 60, 5, 5.0, 62.5, 1.92, 150, -1000, "TO-220"),
    "TIP121": ("npn", 80, 5, 5.0, 62.5, 1.92, 150, -1000, "TO-220"),
    "TIP122": ("npn", 100, 5, 5.0, 62.5, 1.92, 150, -1000, "TO-220"),
    "TIP125": ("pnp", 60, 5, 5.0, 62.5, 1.92, 150, -1000, "TO-220"),
    "TIP127": ("pnp", 100, 5, 5.0, 62.5, 1.92, 150, -1000, "TO-220"),
}
BJT_ALIAS = {"2N2222": "2N2222A", "PN2222A": "2N2222A", "PN2222": "2N2222A", "2N2907": "2N2907A",
             "PN2907A": "2N2907A", "BC549": "BC548", "BC559": "BC558", "BC550": "BC547", "BC560": "BC557",
             "SS8050": "S8050", "SS8550": "S8550", "MMBT2222": "MMBT2222A", "TIP31": "TIP31C", "TIP32": "TIP32C",
             "TIP41": "TIP41C", "TIP42": "TIP42C", "TIP126": "TIP125"}


# pin numbers by package, as the parts sit in their packages (TO-92 flat side facing, TO-220 tab at the back)
BJT_PINOUT = {"EBC": {"e": 1, "b": 2, "c": 3}, "CBE": {"c": 1, "b": 2, "e": 3}, "BCE": {"b": 1, "c": 2, "e": 3},
              "ECB": {"e": 1, "c": 2, "b": 3}, "BEC": {"b": 1, "e": 2, "c": 3}}


def bjt_pinout(name: str, pkg: str) -> dict:
    if pkg == "SOT-23":
        return BJT_PINOUT["BEC"]
    if name.startswith(("BC", "MPSA")):
        return BJT_PINOUT["CBE"]
    if name.startswith("BD"):
        return BJT_PINOUT["ECB"]
    if pkg in ("TO-220", "TO-3"):
        return BJT_PINOUT["BCE"]
    return BJT_PINOUT["EBC"]


def _gp(beta: float, ic: float, pol: str) -> str:
    """A Gummel Poon set scaled to a part's current rating and typical gain."""
    s = ic / 0.2
    return (f"IS={_g(1e-14 * s)} BF={beta:g} VAF={80 if pol == 'npn' else 60} IKF={_g(0.7 * ic)} ISE={_g(1e-14 * s)} "
            f"NE=1.4 BR=4 RC={_g(0.4 / s)} RB={_g(10 / math.sqrt(s))} RE={_g(0.1 / s)} CJE={_g(10e-12 * s)} "
            f"CJC={_g(4e-12 * s)} TF={_g(0.4e-9 * math.sqrt(s))} TR=50n XTB=1.5")


def bjt(name: str) -> Part:
    key = BJT_ALIAS.get(name, name)
    pol, vceo, vebo, ic, thja, thjc, tj, gp, pkg = BJTS[key]
    m = f"Q_{name}"
    r = {"vceo_V": vceo, "vebo_V": vebo, "ic_A": ic, "theta_ja": thja, "theta_jc": thjc or None, "tj_max_C": tj}
    typ = pol.upper()
    pk = {"package": pkg, "pinout": bjt_pinout(key, pkg)}
    if isinstance(gp, str):
        return Part(name, pol, r, "q", m, f".model {m} {typ}({_GP[gp]})", source=f"standard SPICE parameters for "
                    f"the 2N{gp}, {vceo} V {ic:g} A {pkg}", params={"beta": 150, **pk})
    if gp > 0:
        return Part(name, pol, r, "q", m, f".model {m} {typ}({_gp(gp, ic, pol)})",
                    source=f"{name} datasheet: hFE {gp}, {vceo} V {ic:g} A {pkg}", params={"beta": gp, **pk})
    # Darlington: a driver and an output transistor with the usual 8k and 120 ohm base resistors
    d1, d2 = (f"{_gp(100, ic / 20, pol)}", f"{_gp(30, ic, pol)}")
    cd = "e c" if pol == "npn" else "c e"
    text = (f".subckt {m} c b e\nQ1 c b m {m}_1\nQ2 c m e {m}_2\nR1 b m 8k\nR2 m e 120\nD1 {cd} {m}_d\n"
            f".model {m}_1 {typ}({d1})\n.model {m}_2 {typ}({d2})\n.model {m}_d D(IS=1e-10 N=1.5 RS=0.05)\n.ends")
    return Part(name, pol, r, "x", m, text, ("c", "b", "e"),
                source=f"{name} datasheet: Darlington, hFE {-gp}, {vceo} V {ic:g} A {pkg}", params={"beta": -gp, **pk})


# --- MOSFETs -------------------------------------------------------------------------------------------------
# name: (polarity, vds, vgs max, id A, idm A, [(vgs, rds max)...], vth typ, (vth min, vth max), ciss, crss,
#        coss F, theta_ja, theta_jc or 0, tj max, package); on resistances are datasheet maxima at 25 C
MOSFETS = {
    "2N7000": ("n", 60, 20, 0.2, 0.5, [(10, 5.0), (4.5, 5.3)], 2.1, (0.8, 3.0), 60e-12, 5e-12, 25e-12, 312.5, 0,
               150, "TO-92"),
    "2N7002": ("n", 60, 20, 0.115, 0.8, [(10, 5.0), (5, 7.5)], 2.1, (1.0, 2.5), 50e-12, 5e-12, 25e-12, 417, 0, 150,
               "SOT-23"),
    "BS170": ("n", 60, 20, 0.5, 1.2, [(10, 5.0)], 2.1, (0.8, 3.0), 60e-12, 6e-12, 30e-12, 150, 0, 150, "TO-92"),
    "BSS138": ("n", 50, 20, 0.22, 0.88, [(10, 3.5), (4.5, 6.0)], 1.3, (0.8, 1.5), 50e-12, 7e-12, 13e-12, 350, 0,
               150, "SOT-23"),
    "AO3400": ("n", 30, 12, 5.7, 30, [(10, 0.0265), (4.5, 0.032), (2.5, 0.048)], 1.05, (0.65, 1.45), 700e-12,
               50e-12, 80e-12, 100, 0, 150, "SOT-23"),
    "AO3401": ("p", 30, 12, 4.0, 30, [(10, 0.050), (4.5, 0.060), (2.5, 0.085)], 0.9, (0.5, 1.3), 700e-12, 60e-12,
               90e-12, 100, 0, 150, "SOT-23"),
    "SI2302": ("n", 20, 8, 2.9, 10, [(4.5, 0.057), (2.5, 0.075)], 0.65, (0.4, 0.85), 300e-12, 40e-12, 60e-12, 166,
               0, 150, "SOT-23"),
    "SI2301": ("p", 20, 8, 2.3, 10, [(4.5, 0.112), (2.5, 0.142)], 0.7, (0.4, 1.0), 400e-12, 60e-12, 80e-12, 166, 0,
               150, "SOT-23"),
    "IRLML2502": ("n", 20, 12, 4.2, 33, [(4.5, 0.045), (2.5, 0.080)], 0.9, (0.6, 1.2), 740e-12, 63e-12, 100e-12,
                  100, 0, 150, "SOT-23"),
    "IRLML6402": ("p", 20, 12, 3.7, 22, [(4.5, 0.065), (2.5, 0.135)], 0.55, (0.4, 1.2), 630e-12, 110e-12,
                  140e-12, 100, 0, 150, "SOT-23"),
    "IRLZ44N": ("n", 55, 16, 47, 160, [(10, 0.022), (5, 0.025), (4, 0.035)], 1.5, (1.0, 2.0), 1700e-12, 150e-12,
                400e-12, 62, 1.4, 175, "TO-220"),
    "IRLZ34N": ("n", 55, 16, 30, 110, [(10, 0.035), (5, 0.046), (4, 0.060)], 1.5, (1.0, 2.0), 880e-12, 100e-12,
                220e-12, 62, 2.2, 175, "TO-220"),
    "IRF540N": ("n", 100, 20, 33, 110, [(10, 0.044)], 3.0, (2.0, 4.0), 1960e-12, 40e-12, 250e-12, 62, 1.15, 175,
                "TO-220"),
    "IRF520N": ("n", 100, 20, 9.7, 38, [(10, 0.2)], 3.0, (2.0, 4.0), 330e-12, 25e-12, 92e-12, 62, 3.5, 175,
                "TO-220"),
    "IRFZ44N": ("n", 55, 20, 49, 160, [(10, 0.0175)], 3.0, (2.0, 4.0), 1470e-12, 88e-12, 360e-12, 62, 1.5, 175,
                "TO-220"),
    "IRF3205": ("n", 55, 20, 110, 390, [(10, 0.008)], 3.0, (2.0, 4.0), 3250e-12, 250e-12, 780e-12, 62, 0.75, 175,
                "TO-220"),
    "IRF840": ("n", 500, 20, 8, 32, [(10, 0.85)], 3.0, (2.0, 4.0), 1300e-12, 120e-12, 310e-12, 62, 1.0, 150,
               "TO-220"),
    "FQP30N06L": ("n", 60, 20, 32, 128, [(10, 0.035), (5, 0.045)], 1.75, (1.0, 2.5), 1040e-12, 50e-12, 270e-12,
                  62.5, 1.88, 175, "TO-220"),
    "IRLB8721": ("n", 30, 20, 62, 250, [(10, 0.0087), (4.5, 0.016)], 1.8, (1.35, 2.35), 1077e-12, 120e-12,
                 360e-12, 62, 2.0, 175, "TO-220"),
    "IRF9540N": ("p", 100, 20, 23, 76, [(10, 0.117)], 3.0, (2.0, 4.0), 1300e-12, 160e-12, 400e-12, 62, 1.6, 175,
                 "TO-220"),
    "IRF4905": ("p", 55, 20, 74, 260, [(10, 0.020)], 3.0, (2.0, 4.0), 3400e-12, 750e-12, 1400e-12, 62, 0.75, 175,
                "TO-220"),
    "IRF9Z34N": ("p", 55, 20, 19, 68, [(10, 0.10)], 3.0, (2.0, 4.0), 620e-12, 90e-12, 250e-12, 62, 2.2, 175,
                 "TO-220"),
}
MOSFET_ALIAS = {"AO3400A": "AO3400", "AO3401A": "AO3401", "SI2302CDS": "SI2302", "SI2302DS": "SI2302",
                "SI2301CDS": "SI2301", "SI2301DS": "SI2301", "IRF540": "IRF540N", "IRF520": "IRF520N",
                "IRFZ44": "IRFZ44N", "IRLZ44": "IRLZ44N", "IRF9540": "IRF9540N", "2N7002K": "2N7002",
                "BSS138K": "BSS138", "A2SHB": "SI2302", "A1SHB": "SI2301", "DMG2302U": "SI2302"}


def fit_vdmos(pts: list[tuple[float, float]], vth: float) -> tuple[float, float]:
    """KP and series resistance so Rds(Vgs) = Rser + 1 / (KP (Vgs - Vth)) meets the datasheet's on resistances."""
    pts = sorted(pts, reverse=True)
    v1, r1 = pts[0]
    if len(pts) > 1:
        v2, r2 = pts[-1]
        if r2 > r1 and v2 > vth:
            kp = (1 / (v2 - vth) - 1 / (v1 - vth)) / (r2 - r1)
            rser = r1 - 1 / (kp * (v1 - vth))
            if rser >= 0.1 * r1:
                return kp, rser
    rser = 0.5 * r1
    return 1 / ((r1 - rser) * (v1 - vth)), rser


def mosfet(name: str) -> Part:
    key = MOSFET_ALIAS.get(name, name)
    pol, vds, vgs, idc, idm, pts, vth, vrange, ciss, crss, coss, thja, thjc, tj, pkg = MOSFETS[key]
    kp, rser = fit_vdmos(pts, vth)
    isd = idc / math.exp(0.9 / (1.2 * VT))
    m = f"M_{name}"
    text = (f".model {m} VDMOS({'pchan ' if pol == 'p' else ''}Vto={_g(-vth if pol == 'p' else vth)} Kp={_g(kp)} "
            f"Rd={_g(0.8 * rser)} Rs={_g(0.2 * rser)} Rg=2 lambda=0.01 ksubthres=0.1 Cgs={_g(ciss - crss)} "
            f"Cgdmax={_g(5 * crss)} Cgdmin={_g(crss / 2)} A=1 Cjo={_g(max(coss - crss, crss))} Is={_g(isd)} N=1.2 "
            f"Rb={_g(0.5 / idc)} BV={_g(vds * 1.1)} TT=50n)")
    r = {"vds_V": vds, "vgs_V": vgs, "id_A": idc, "idm_A": idm, "theta_ja": thja, "theta_jc": thjc or None,
         "tj_max_C": tj}
    spec = ", ".join((f"{rr:.3g} ohm" if rr >= 1 else f"{rr * 1000:.3g} mohm") + f" at {v:g} V" for v, rr in pts)
    return Part(name, f"{pol}mos", r, "m", m, text,
                source=f"VDMOS fitted to the {name} datasheet: {spec}, threshold {vth:g} V, {vds} V {idc:g} A {pkg}",
                params={"vth": vth, "vth_range": vrange, "rds": pts, "spec_vgs": min(v for v, _ in pts),
                        "rds_tempco": 0.006, "package": pkg,
                        "pinout": {"TO-220": {"g": 1, "d": 2, "s": 3}, "SOT-23": {"g": 1, "s": 2, "d": 3}}.get(
                            pkg, {"d": 1, "g": 2, "s": 3} if key == "BS170" else {"s": 1, "g": 2, "d": 3})})


# --- lookup -----------------------------------------------------------------------------------------------
_BUILDERS: list = []        # functions name -> Part | None, added by the sections below


def builder(fn):
    _BUILDERS.append(fn)
    return fn


@builder
def _diodes(name: str) -> Part | None:
    u = name.upper()
    if u in DIODES or u in DIODE_ALIAS:
        return diode(u)
    return _zener_name(name)


@builder
def _leds(name: str) -> Part | None:
    m = re.match(r"^LED[-_ ]?(RED|GREEN|BLUE|YELLOW|ORANGE|WHITE|AMBER|UV|IR|PINK|PURPLE)?(?:[-_ ]?(0402|0603|0805|1206|"
                 r"3MM|5MM))?$", name.upper())
    if not m:
        return None
    return led((m.group(1) or "red").lower(), (m.group(2) or "").lower())


@builder
def _transistors(name: str) -> Part | None:
    u = name.upper()
    if u in BJTS or u in BJT_ALIAS:
        return bjt(u)
    if u in MOSFETS or u in MOSFET_ALIAS:
        return mosfet(u)
    return None


def lookup(name: str) -> Part | None:
    """The library part a name, a part number or a model name means, or None. Package and packing suffixes
    are tolerated, so IRLZ44NPBF finds IRLZ44N and LM7805CT finds LM7805."""
    if not name:
        return None
    raw = name.strip().strip("{}").split(":")[-1]
    if not raw or len(raw) < 2:
        return None
    for fn in _BUILDERS:
        p = fn(raw)
        if p is not None:
            return p
    u = raw.upper()
    if re.search(r"\s", u):
        return None
    for cut in range(len(u) - 1, 2, -1):      # longest known prefix: LM358DR -> LM358
        for fn in _BUILDERS:
            p = fn(u[:cut])
            if p is not None and _suffix_ok(u[cut:]):
                return p
    return None


def _suffix_ok(s: str) -> bool:
    """Package and packing suffixes such as PBF, CT, N, DR, -TR, A, but not a different part's digits."""
    return bool(re.match(r"^[-_/]?[A-Z]{0,5}\d?[A-Z]{0,4}(?:[-_/]?[A-Z0-9]{0,4})?$", s)) and not re.match(r"^\d", s)


from . import ics  # noqa: E402,F401  the IC, battery and module entries register their own builders
