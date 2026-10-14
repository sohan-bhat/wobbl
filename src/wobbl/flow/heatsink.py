"""Fan cooled heat sinks: the airflow decides the temperature.

"Does a 15 W chip stay under 85 C on this heat sink with a 60 mm fan?" becomes: the fan's curve meets the heat
sink's pressure drop at an operating flow, that flow through the gaps between the fins sets a heat transfer
coefficient, and that coefficient goes onto the fins in a CalculiX heat study, which works out how hot the chip
side gets with the fins' own conduction included.

The air side is one gap between two fins, solved by OpenFOAM as a 2D channel with the flow and the temperature
developing from the inlet, half a gap wide with a symmetry plane in the middle. The fins sit at one temperature
and the air enters at another, so the heat the air carries away and the log mean temperature difference give the
mean coefficient exactly as a heat sink datasheet defines it. Laminar gaps, below a Reynolds number of 2300 on the
hydraulic diameter, need no turbulence model, and the answer is checked against Stephan's correlation for
developing flow between plates and Teertstra's model for plate fin heat sinks.
"""
from __future__ import annotations

import math
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..units import fmt
from ..verify.checks import Check, confidence
from . import correlations as C
from . import foam, playback
from .fluids import get_fluid
from .verify import compare_check, gci, gci_check, mass_check, residual_check

GRIDS = ((135, 40), (90, 27), (60, 18))   # cells along and across half a gap, fine to coarse
WORKERS = 2


# --- one gap between two fins, without dimensions ----------------------------------------------------------------
def channel_run(re_: float, pr: float, l_dh: float, cells: tuple[int, int], work: Path,
                fields: bool = False) -> dict:
    """Developing laminar flow and heat between parallel plates: gap 1, mean velocity 1, Reynolds number on the
    hydraulic diameter (twice the gap). Returns the mean Nusselt number, the pressure drop over rho U^2, the flows
    in and out and the solver log, and with fields the cell centres, velocity and temperature."""
    nx, ny = cells
    case = foam.Case(Path(work) / f"gap{nx}x{ny}-re{re_:.6g}")
    dh, U = 2.0, 1.0
    L, nu = l_dh * dh, dh / re_
    foam.channel(case, L, 0.5, nx, ny, gx=6.0, gy=4.0, lead=1.0)      # cells crowd the leading edge and wall
    turb = foam.Turbulence()
    foam.write_fields(case, (U, 0, 0), turb, scalar={"inlet": 0.0, "wall": 1.0})
    foam.write_physics(case, nu, turb)
    fo = (foam.fo_scalar(pr) + foam.fo_patch_sum("qin", "inlet") + foam.fo_patch_sum("qout", "outlet")
          + foam.fo_patch_sum("tout", "outlet", "T", "weightedAverage", "phi")
          + foam.fo_patch_sum("pin", "inlet", "p", "areaAverage") + (foam.fo_cell_centres() if fields else ""))
    foam.write_steady(case, turb, 4000, fo, scalar=True, tol=1e-7, small=nx * ny < 5000)
    t0 = time.time()
    case.run("blockMesh")
    log = case.run("simpleFoam")
    qin = -float(foam.column(case, "qin", "sum(phi)")[-1])
    qout = float(foam.column(case, "qout", "sum(phi)")[-1])
    tout_hist = foam.column(case, "tout", "weightedAverage(T)")
    tout = float(tout_hist[-1])
    ntu = math.log(1 / (1 - tout))
    out = {"cells": nx * ny, "grid": cells, "nu": qout * ntu / (L * case.depth) * dh / (nu / pr),
           "dp": float(foam.column(case, "pin", "areaAverage(p)")[-1]) / U ** 2, "t_out": tout,
           "q_in": qin, "q_out": qout, "settled": float(abs(tout_hist[-1] - tout_hist[int(0.9 * len(tout_hist))]) /
                                                     max(tout, 1e-12)), "log": log, "seconds": time.time() - t0}
    if fields:
        last = foam.last_time(case)
        out["fields"] = {k: foam.read_field(last / k) for k in ("C", "U", "T")}
    case.clean()
    return out


def channel_levels(re_: float, pr: float, l_dh: float, work: Path, grids=GRIDS) -> list[dict]:
    with ThreadPoolExecutor(WORKERS) as pool:
        return list(pool.map(lambda g: channel_run(re_, pr, l_dh, g, work, fields=g == grids[0]), grids))


def shah_london_dp(re_: float, l_dh: float) -> float:
    """Pressure drop over rho U^2 for laminar flow developing between parallel plates from a uniform inlet, from
    Shah's apparent friction factor (Shah and London 1978), within about 2% of exact solutions."""
    xp = l_dh / re_
    fre = 3.44 / math.sqrt(xp) + (24 + 0.674 / (4 * xp) - 3.44 / math.sqrt(xp)) / (1 + 0.000029 * xp ** -2)
    return 4 * (fre / re_) * l_dh * 0.5


# --- the heat sink study ------------------------------------------------------------------------------------------
@dataclass
class FanSinkStudy:
    part: str                       # a heat sink, a parametric .py with base, size, fins, fin, fin_height in mm
    material: str = "6063-t5"
    power: float = 15.0             # W
    heat_faces: str = ""
    fan_flow: float = 0.0           # m3/s free delivery of the fan
    fan_pressure: float = 0.0       # Pa at shut off
    velocity: float = 0.0           # m/s through the gaps, instead of a fan
    ambient: float = 25.0           # C
    limit: float | None = None      # C
    emissivity: float = 0.0
    geometry: dict = field(default_factory=dict)   # overrides of the part's parameters, mm
    keep: bool = False
    question: str = ""


def sink_geometry(st: FanSinkStudy) -> dict:
    from ..geometry.parametric import is_parametric, load_params

    g = dict(load_params(st.part)) if is_parametric(st.part) else {}
    g.update({k: v for k, v in st.geometry.items() if v})
    need = ("base", "size", "fins", "fin", "fin_height")
    missing = [k for k in need if not g.get(k)]
    if missing:
        raise ValueError(f"the heat sink needs {', '.join(missing)} (mm), from the part's PARAMS or given directly")
    pitch = (g["size"] - g["fin"]) / (g["fins"] - 1)
    gap = pitch - g["fin"]
    if gap <= 0:
        raise ValueError("the fins touch: their thickness is more than their pitch")
    m = 1e-3
    return {"base": g["base"] * m, "width": g["size"] * m, "length": g.get("length", g["size"]) * m,
            "fins": int(g["fins"]), "fin": g["fin"] * m, "height": g["fin_height"] * m, "gap": gap * m,
            "pitch": pitch * m, "channels": int(g["fins"]) - 1}


def entry_exit_k(sigma: float) -> float:
    """Loss coefficients of the sudden contraction into the fins and the expansion out of them, on the gap
    velocity: 0.42 (1 - sigma^2) and (1 - sigma)^2 (Kays and London's limits for parallel plates)."""
    return 0.42 * (1 - sigma ** 2) + (1 - sigma) ** 2


def run_fansink(st: FanSinkStudy, out_dir: str | Path, progress=None):
    from ..outcome_generic import GenericOutcome
    from ..solvers.base import get
    from ..thermal.study import Film, ThermalStudy, run_thermal
    from .install import ensure
    from .render import chart, field2d

    t0 = time.time()
    say = progress or (lambda a, b: None)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    work = out / "cfd"
    ensure(lambda m: say("setup", m))
    g = sink_geometry(st)
    air = get_fluid("air", st.ambient)
    dh = 2 * g["gap"]
    l_dh = g["length"] / dh
    area = g["channels"] * g["gap"] * g["height"]          # open flow area between the fins
    sigma = area / (g["width"] * g["height"])
    k_ee = entry_exit_k(sigma)

    # the sink's pressure drop against flow, from the coarse grid at three speeds, fitted as a/Re + b
    def sink_dp(v: float, dp_star: float) -> float:
        return (dp_star + 0.5 * k_ee) * air.rho * v * v

    if st.velocity:
        v = st.velocity
        curve = None
    else:
        if not (st.fan_flow and st.fan_pressure):
            raise ValueError("give the fan's free flow and shut off pressure, like '12 CFM, 30 Pa', or a velocity")
        v_free = st.fan_flow / area
        trial = [0.25 * v_free, 0.5 * v_free, 0.8 * v_free]
        say("cfd", f"pressure drop of one gap at {', '.join(fmt(x) for x in trial)} m/s")
        res = [air.rho * x * dh / air.mu for x in trial]
        with ThreadPoolExecutor(WORKERS) as pool:
            pts = list(pool.map(lambda r: channel_run(r, air.Pr, l_dh, GRIDS[-1], work)["dp"], res))
        a, b = np.linalg.lstsq(np.c_[1 / np.array(res), np.ones(3)], np.array(pts), rcond=None)[0]

        def miss(x):
            return sink_dp(x, a / (air.rho * x * dh / air.mu) + b) - st.fan_pressure * (1 - x * area / st.fan_flow)

        lo, hi = 1e-4 * v_free, v_free
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if miss(mid) < 0 else (lo, mid)
        v = 0.5 * (lo + hi)
        curve = (a, b, trial, pts)
    re_ = air.rho * v * dh / air.mu
    if re_ > C.LAMINAR_RE:
        raise ValueError(f"the flow between the fins is turbulent at Re {fmt(re_)}, and Wobbl models laminar fin "
                         "gaps only, so try a slower fan or wider gaps")
    say("cfd", f"one gap at {fmt(v)} m/s (Re {fmt(re_)}) on three grids")
    levels = channel_levels(re_, air.Pr, l_dh, work)
    fine = levels[0]
    h = fine["nu"] * air.k / dh
    q = v * area                                            # m3/s through the sink
    dp = sink_dp(v, fine["dp"])
    rise = st.power / (air.rho * air.cp * q)
    t_ref = st.ambient + rise / 2

    checks: list[Check] = []
    gn = gci([L["nu"] for L in levels], [1 / L["grid"][0] for L in levels])
    checks.append(gci_check(gn, "Nusselt number", "", tol=0.02))
    checks.append(mass_check(fine["q_in"], fine["q_out"], "m3/s per unit depth", tol=1e-4))
    checks.append(residual_check(fine["log"], 1e-5))
    want = C.nu_plates_developing(re_, air.Pr, g["length"], dh)
    checks.append(compare_check("Nusselt correlation", fine["nu"], want, "Stephan's correlation for developing "
                                "flow between plates", "", 0.05))
    nu_t = C.nu_teertstra(v, g["gap"], g["length"], air.nu, air.Pr) * dh / g["gap"]
    checks.append(compare_check("heat sink model", fine["nu"], nu_t, "Teertstra's plate fin model", "", 0.15))
    checks.append(compare_check("pressure drop", fine["dp"], shah_london_dp(re_, l_dh), "Shah and London's apparent "
                                "friction", "", 0.15))
    say("heat", f"h {fmt(h)} W/(m2 K) on the fins in air at {fmt(t_ref)} C, CalculiX")
    base_mm = g["base"] * 1e3
    films = [Film(h * 1e-3, t_ref, f"z>{base_mm + 0.01:g} and normal {d}", f"{fmt(h)} W/m2K from CFD")
             for d in ("+x", "-x", "+y", "-y")]
    films += [Film(h * 1e-3, t_ref, f"z={base_mm:g}", ""), Film(5e-3, t_ref, f"z<{base_mm - 0.01:g}", ""),
              Film(5e-3, t_ref, "z=max", "")]
    th = ThermalStudy(str(Path(st.part).resolve()), st.material, st.power * 1e3, st.heat_faces, films,
                      ambient=t_ref, limit=st.limit, emissivity=st.emissivity, question=st.question)
    to = run_thermal(th, out / "thermal", lambda a, b: say(a, b))
    checks += to.checks
    # an effectiveness hand check: the sink at one temperature, with straight fin efficiency, gives up the power
    mat_k = _conductivity(st.material)
    mfin = math.sqrt(2 * h / (mat_k * g["fin"]))
    eta = math.tanh(mfin * g["height"]) / (mfin * g["height"])
    a_fin = g["fins"] * 2 * g["height"] * g["length"]
    a_base = g["channels"] * g["gap"] * g["length"]
    ua = h * (eta * a_fin + a_base)
    cap = air.rho * air.cp * q
    t_hand = st.ambient + st.power / (cap * (1 - math.exp(-ua / cap)))
    lo_, hi_ = to.values.get("levels", [{}])[-1].get("t_min", 0.0), to.values.get("t_max_C", 0.0)
    checks.append(Check("sink effectiveness", "info",
                        f"fins of efficiency {eta:.2f} shedding {fmt(st.power)} W into {fmt(q * 60000)} L/min of air "
                        f"put a sink at one temperature at {fmt(t_hand)} C, and the FEA spans {fmt(lo_)} to "
                        f"{fmt(hi_)} C" + (", with radiation adding to the cooling" if st.emissivity else ""),
                        {"fin_efficiency": eta, "uniform_sink_C": t_hand}))
    if curve:
        checks.append(Check("operating point", "pass", f"the fan curve meets the sink at {fmt(q * 60000)} L/min "
                                                       f"and {fmt(dp)} Pa, {100 * q / st.fan_flow:.0f}% of the "
                                                       "fan's free flow"))
    level, reasons = confidence(checks, [("medium", "the air is ducted through the fins with none going around "
                                                    "them, so a sink without a shroud gets less")])
    tmax = to.values["t_max_C"]
    limit = to.values["limit_C"]
    verdict = to.verdict
    head = (f"{fmt(q * 60000)} L/min through the fins gives {fmt(h)} W/(m2 K), and the hottest point reaches "
            f"{fmt(tmax)} C against a limit of {fmt(limit)} C.")
    fixes = []
    if verdict != "ok":
        fixes.append({"kind": "fan", "title": "Use a stronger fan",
                      "summary": f"The sink needs about {fmt((tmax - st.ambient) / max(limit - 5 - st.ambient, 1))}x "
                                 "the heat transfer, and h grows roughly with the square root of the flow in "
                                 "laminar gaps.", "auto": False})
        fixes.append({"kind": "fins", "title": "Add fin area",
                      "summary": "Taller fins or more of them add area, while gaps under about 3 mm choke the "
                                 "flow and lose more than they add.", "auto": False})
    imgs = {"temperature": to.images.get("temperature", "")}
    if imgs["temperature"]:
        dest = out / "images" / "temperature.png"
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(imgs["temperature"], dest)
        imgs["temperature"] = str(dest)
    curves = {}
    f = fine["fields"]
    x, y = f["C"][:, 0] * g["gap"] * 1e3, f["C"][:, 1] * g["gap"] * 1e3      # the gap is 1 in the CFD
    if curve:
        qq = np.linspace(0.02, 1.0, 40) * st.fan_flow
        sys_dp = [sink_dp(x_, curve[0] / (air.rho * x_ * dh / air.mu) + curve[1]) for x_ in qq / area]
        curves = {"fan": (qq * 60000, st.fan_pressure * (1 - qq / st.fan_flow)), "heat sink": (qq * 60000, sys_dp)}
    try:
        imgs["air"] = field2d(out / "images" / "air.png", x, y, f["T"], "air between two fins, warming from 0 at "
                              "the inlet toward 1 at the fin", "inferno", vectors=f["U"][:, :2], stretch=4)
        if curves:
            imgs["fan"] = chart(out / "images" / "fan.png", curves, "flow, L/min", "pressure, Pa",
                                {"operating point": ([q * 60000], [dp])}, "Where the fan curve meets the heat sink")
    except Exception as e:  # noqa: BLE001
        imgs["error"] = str(e)
    playback.write(out, "channel", "Air through one gap of a fan cooled heat sink",
                   "One gap between two fins seen from the side, the fin wall at y0 and the middle of the gap at y1, "
                   "so mirror it about y1 for the whole gap. u and v are the air velocity, theta is how far the air "
                   "has warmed from the inlet temperature toward the fin's, 0 to 1. Air enters at x0 and the fin begins at x 0, "
                   "one gap downstream.",
                   [playback.grid_field("air", x / 1e3, y / 1e3, {"u": f["U"][:, 0] * v, "v": f["U"][:, 1] * v,
                                                                  "theta": f["T"]},
                                        {"u": "m/s", "v": "m/s", "theta": ""}, nx=200, ny=24)],
                   curves={k: playback.series(cx, cy, "L/min", "Pa") for k, (cx, cy) in curves.items()},
                   scalars={"airflow": (q * 60000, "L/min"), "gap_velocity": (v, "m/s"), "h": (h, "W/m2K"),
                            "inlet_air": (st.ambient, "C"), "air_rise": (rise, "C"),
                            "hottest": (to.values["t_max_C"], "C"), "power": (st.power, "W")})
    if not st.keep:      # the heat study's numbers and picture are in this result, so its own folder goes
        shutil.rmtree(work, ignore_errors=True)
        shutil.rmtree(out / "thermal", ignore_errors=True)
    metrics = [("max temperature", f"{fmt(tmax)} C"), ("limit", f"{fmt(limit)} C"),
               ("airflow", f"{fmt(q * 60000)} L/min"), ("h on the fins", f"{fmt(h)} W/m2K")]
    assumptions = [
        f"{g['fins']} fins {fmt(g['fin'] * 1e3)} mm thick and {fmt(g['height'] * 1e3)} mm tall, gaps of "
        f"{fmt(g['gap'] * 1e3)} mm, {fmt(g['length'] * 1e3)} mm along the flow.",
        "The fan's air is ducted through the fins with none going around them, as with a shroud or a fan "
        "mounted on the end of the sink.",
        "The fan curve is a straight line from its free flow to its shut off pressure." if curve else
        f"The air moves at {fmt(v)} m/s between the fins, as given.",
        f"Air at {fmt(st.ambient)} C warms by {fmt(rise)} C through the sink, so the fins see air at {fmt(t_ref)} C "
        "on average.",
        "One heat transfer coefficient from the gap's mean covers every fin and the base between them, the fin "
        "tips under the shroud and the outside of the base get 5 W/(m2 K).",
        "Entry and exit losses at the fins come from Kays and London's contraction and expansion coefficients.",
    ] + [a for a in to.assumptions if "fixed film coefficient" not in a]
    values = {"t_max_C": tmax, "limit_C": limit, "h_W_m2K": h, "flow_m3_s": q, "velocity_m_s": v, "re": re_,
              "dp_Pa": dp, "air_rise_C": rise, "nusselt": fine["nu"], "levels": [{k: L[k] for k in
                                                                                  ("grid", "nu", "dp", "seconds")}
                                                                                 for L in levels],
              "_figures": [["temperature", "Heat sink temperature with the hottest point marked"],
                           ["air", "Air between two fins, warming from the inlet on the left"],
                           ["fan", "Where the fan curve meets the heat sink"]]}
    o = GenericOutcome("flow", str(Path(st.part).resolve()), st.question, verdict, head, metrics, level, reasons,
                       checks, assumptions, values, imgs, fixes,
                       solver=f"OpenFOAM {get('openfoam').version()} and {to.solver}", t_total=time.time() - t0)
    return o.write(out)


def _conductivity(material: str) -> float:
    from ..materials import resolve_material
    return resolve_material(material).conductivity
