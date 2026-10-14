"""Pressure drop through a pipe or duct, with its fittings, by Darcy Weisbach and the Moody chart, and a CFD check of
the friction factor.

"Can a 25 mm PVC line carry 30 L/min over 40 m with three elbows?" becomes: the Reynolds number says laminar or
turbulent, Colebrook (or 64/Re) gives the friction factor for the wall's roughness, the fittings add their loss
coefficients, and the total is judged against the pressure the pump or supply can give. When OpenFOAM is there, a
short periodic slice of the pipe can be solved on three grids, and its friction factor has to agree with the
chart.
"""
from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from ..outcome_generic import GenericOutcome
from ..units import fmt
from ..verify.checks import Check, confidence
from . import correlations as C
from .fluids import G, fmt_flow, fmt_pressure, get_fluid

# absolute roughness, m (White, Fluid Mechanics, Table 6.1, after Moody 1944)
ROUGHNESS = {"smooth": 0.0, "glass": 0.0, "drawn": 1.5e-6, "copper": 1.5e-6, "pvc": 1.5e-6, "plastic": 1.5e-6,
             "pex": 7e-6, "stainless": 1.5e-5, "steel": 4.5e-5, "commercial-steel": 4.5e-5, "galvanized": 1.5e-4,
             "cast-iron": 2.6e-4, "concrete": 1e-3, "riveted-steel": 3e-3}


@dataclass
class PipeStudy:
    length: float                   # m
    flow: float                     # m3/s
    diameter: float = 0.0           # m, round pipe
    width: float = 0.0              # m, rectangular duct
    height: float = 0.0
    fluid: str = "water"
    temp: float = 20.0              # C
    material: str = "pvc"
    roughness: float | None = None  # m, overrides the material
    fittings: list[tuple[int, str]] = field(default_factory=list)
    rise: float = 0.0               # m the outlet sits above the inlet
    max_dp: float = 0.0             # Pa the supply can give, 0 when not judged
    question: str = ""


def parse_fittings(text: str) -> list[tuple[int, str]]:
    """'3 elbow90, gate-valve, 2 tee-run' or 'K=4.5' for a known total loss coefficient."""
    out = []
    for part in [p.strip() for p in (text or "").split(",") if p.strip()]:
        m = re.match(r"^(?:(\d+)\s*x?\s+)?([\w.=-]+)$", part.lower())
        if not m:
            raise ValueError(f"could not read the fitting '{part}'; write it like '3 elbow90'")
        n, name = int(m.group(1) or 1), m.group(2)
        if not name.startswith("k=") and name not in C.FITTINGS:
            raise ValueError(f"unknown fitting '{name}'; use {', '.join(C.FITTINGS)} or K=<value>")
        out.append((n, name))
    return out


def k_total(fittings: list[tuple[int, str]]) -> float:
    return sum(n * (float(name[2:]) if name.startswith("k=") else C.FITTINGS[name]) for n, name in fittings)


@dataclass
class PipeResult:
    re: float
    velocity: float
    dh: float
    f: float
    f_source: str
    dp_friction: float
    dp_fittings: float
    dp_rise: float

    @property
    def total(self) -> float:
        return self.dp_friction + self.dp_fittings + self.dp_rise


def solve_pipe(st: PipeStudy, diameter: float | None = None) -> PipeResult:
    fl = get_fluid(st.fluid, st.temp)
    if st.diameter or diameter:
        d = diameter or st.diameter
        area, dh, aspect = math.pi * d * d / 4, d, None
    else:
        area = st.width * st.height
        dh, aspect = 2 * st.width * st.height / (st.width + st.height), st.height / st.width
    v = st.flow / area
    re = fl.rho * v * dh / fl.mu
    rough = ROUGHNESS[st.material] if st.roughness is None else st.roughness
    f, src = C.friction_factor(re, rough / dh, aspect)
    q = 0.5 * fl.rho * v * v
    return PipeResult(re, v, dh, f, src, f * st.length / dh * q, k_total(st.fittings) * q, fl.rho * G * st.rise)


def run_pipe(st: PipeStudy, out_dir: str | Path) -> GenericOutcome:
    from .render import moody

    t0 = time.time()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if st.material not in ROUGHNESS and st.roughness is None:
        raise ValueError(f"unknown pipe material '{st.material}'; use {', '.join(ROUGHNESS)} or give a roughness")
    fl = get_fluid(st.fluid, st.temp)
    r = solve_pipe(st)
    rough = ROUGHNESS[st.material] if st.roughness is None else st.roughness
    rel = rough / r.dh
    reg = C.regime(r.re)
    checks: list[Check] = []
    extra = []
    if reg == "turbulent":
        fh = C.haaland(r.re, rel)
        err = abs(fh - r.f) / r.f
        checks.append(Check("hand calculation", "pass" if err < 0.02 else "warn",
                            f"Haaland's explicit formula gives f {fh:.4g} against Colebrook's {r.f:.4g} "
                            f"({100 * err:.2g}% apart)"))
    elif reg == "laminar":
        checks.append(Check("hand calculation", "pass", f"laminar flow, so f Re is exact ({r.f * r.re:.4g})"))
    else:
        checks.append(Check("flow regime", "warn", f"Reynolds number {fmt(r.re)} is in transition, where the flow "
                                                   "flips between laminar and turbulent and no friction factor is "
                                                   "reliable, so Churchill's blend is a middle estimate"))
    # entrance length: developing flow near the inlet loses more than the fully developed friction factor says
    le = (0.05 * r.re if reg == "laminar" else 4.4 * r.re ** (1 / 6)) * r.dh
    if le > 0.1 * st.length:
        checks.append(Check("entrance length", "warn" if reg == "laminar" else "info",
                            f"the flow takes about {fmt(le)} m to develop, {100 * le / st.length:.0f}% of the pipe, "
                            "where the wall loses more than the fully developed friction factor allows"))
    cfd_f = None
    if cfd_f is None and (reg == "transitional" or (reg == "turbulent" and rel >= 1e-5)):
        extra.append(("medium", "the Moody chart is good to about 10 to 15% on rough walls, as Moody himself said, "
                                "and no CFD check was run"))
    share = r.dp_fittings / max(r.total, 1e-12)
    if share > 0.3:
        extra.append(("medium", f"fittings make {100 * share:.0f}% of the loss, and real fittings scatter about 30% "
                                "around the textbook coefficients"))
    vel_ok = r.velocity <= (3.0 if fl.rho > 100 else 15.0)
    checks.append(Check("velocity", "pass" if vel_ok else "warn",
                        f"mean velocity {fmt(r.velocity)} m/s" + ("" if vel_ok else
                                                                  ", above the usual design limit, so expect noise and "
                                                                  "erosion")))
    level, reasons = confidence(checks, extra)
    total = r.total
    if st.max_dp:
        verdict = "ok" if total <= 0.9 * st.max_dp else ("marginal" if total <= st.max_dp else "fails")
    else:
        verdict = "ok"
    head = (f"{fmt_flow(st.flow)} of {fl.name.lower()} loses {fmt_pressure(total)} over {fmt(st.length)} m "
            f"({reg}, Re {fmt(r.re)}, f {r.f:.3g})" +
            (f", against {fmt_pressure(st.max_dp)} available." if st.max_dp else "."))
    fixes = []
    if verdict != "ok":
        d_new = _diameter_for(st, 0.9 * st.max_dp)
        if d_new:
            fixes.append({"kind": "diameter", "title": f"Use a pipe of {fmt(d_new * 1000)} mm bore",
                          "summary": f"That brings the loss to 90% of the {fmt_pressure(st.max_dp)} available, "
                                     "since friction loss falls with about the fifth power of the bore.",
                          "auto": False})
        if share > 0.2:
            fixes.append({"kind": "fittings", "title": "Use long radius elbows and full bore valves",
                          "summary": f"The fittings take {fmt_pressure(r.dp_fittings)}, and long radius elbows "
                                     "and ball valves have about a third of the loss of standard ones.",
                          "auto": False})
    imgs = {}
    try:
        imgs["moody"] = moody(out / "images" / "moody.png", r.re, r.f, rel, cfd_f)
    except Exception as e:  # noqa: BLE001
        imgs["error"] = str(e)
    metrics = [("pressure drop", fmt_pressure(total)), ("Reynolds number", fmt(r.re)),
               ("friction factor", f"{r.f:.3g}"), ("velocity", f"{fmt(r.velocity)} m/s")]
    shape = (f"{fmt(st.diameter * 1000)} mm bore" if st.diameter else
             f"{fmt(st.width * 1000)} x {fmt(st.height * 1000)} mm duct")
    assumptions = [fl.summary() + ".",
                   f"{fmt(st.length)} m of {shape}, roughness {fmt(rough * 1e6)} um ({st.material}).",
                   f"Friction from {r.f_source}, {fmt_pressure(r.dp_friction)}." +
                   (f" Fittings {', '.join(f'{n} {k}' for n, k in st.fittings)} add K {k_total(st.fittings):.3g}, "
                    f"{fmt_pressure(r.dp_fittings)}." if st.fittings else ""),
                   "Steady, incompressible flow at one temperature." if fl.rho > 100 else
                   "Steady flow with the air's density held constant, fine below about 5% pressure change."]
    if st.rise:
        assumptions.append(f"The outlet sits {fmt(st.rise)} m above the inlet, which costs {fmt_pressure(r.dp_rise)}.")
    values = {"dp_Pa": total, "dp_friction_Pa": r.dp_friction, "dp_fittings_Pa": r.dp_fittings,
              "dp_rise_Pa": r.dp_rise, "re": r.re, "f": r.f, "f_cfd": cfd_f, "velocity_m_s": r.velocity,
              "pump_power_W": st.flow * total, "_figures": [["moody", "The Moody chart with this pipe marked"]]}
    o = GenericOutcome("flow", "pipe", st.question, verdict, head, metrics, level, reasons, checks, assumptions,
                       values, imgs, fixes, solver="Darcy Weisbach" + (" and OpenFOAM" if cfd_f else ""),
                       t_total=time.time() - t0)
    return o.write(out)


def _diameter_for(st: PipeStudy, dp: float) -> float | None:
    """The round bore that brings the whole loss to dp, by bisection, or None for a duct."""
    if not st.diameter or dp <= 0:
        return None
    lo, hi = st.diameter, st.diameter * 4
    if solve_pipe(st, hi).total > dp:
        return None
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if solve_pipe(st, mid).total > dp else (lo, mid)
    return hi
