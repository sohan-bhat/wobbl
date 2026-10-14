"""Electric field studies: capacitance and field strength, with a breakdown margin.

"Will 5 kV arc across these plates?" becomes an electrostatic model solved
with GetDP: the conductors are held at their voltages, the charge on each one
comes back from the solve, and the peak field is compared with the dielectric
strength of whatever sits between them. Wobbl checks that the charges add up
to zero, that the capacitance from charge equals the one from stored energy,
refines the mesh until the peak field settles and compares with a closed form.
"""
from __future__ import annotations

import math
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..materials import get_dielectric
from ..outcome_generic import GenericOutcome
from ..units import fmt
from ..verify.checks import Check, confidence, convergence_check
from . import plots, solve
from .forms import EPS0, ElecProblem, electrostatic
from .mesh2d import Model2D, Shape, mesh
from .params import ParamError, check_known, text, value

MM = 1e-3
LEVELS = (1.4, 1.0, 0.7)


@dataclass
class ESpec:
    template: str
    volts: float
    dielectric: str
    target: float                       # breakdown margin wanted
    dims: dict = field(default_factory=dict)


def paschen_air(gap: float) -> float:
    """Uniform field breakdown strength of air at sea level for a gap in meters, V/m (Paschen with A = 15 /(cm Torr),
    B = 365 V/(cm Torr), secondary emission 0.01). About 3.5 kV/mm at 10 mm and 5 kV/mm at 1 mm."""
    pd = 760.0 * max(gap, 20e-6) * 100.0
    denom = math.log(15.0 * pd) - math.log(math.log(1.0 + 1.0 / 0.01))
    return 365.0 * pd / max(denom, 0.5) / max(gap, 20e-6)


def peek_sphere(radius: float) -> float:
    """Peek's law for the onset of corona on a sphere in air at sea level, V/m."""
    r_cm = radius * 100.0
    return 27.2e5 * (1.0 + 0.54 / math.sqrt(max(r_cm, 1e-3)))


# --- templates ----------------------------------------------------------------------
PLATES_KEYS = {"diameter", "gap", "thickness", "voltage", "dielectric", "edge", "margin"}
COAX_KEYS = {"inner_d", "outer_d", "offset", "voltage", "dielectric", "margin"}
SPHERE_KEYS = {"diameter", "gap", "voltage", "margin"}


def plates_spec(p: dict) -> ESpec:
    check_known(p, PLATES_KEYS, "plates")
    d = {"R": value(p, "diameter", 100, "length") / 2, "d": value(p, "gap", 2, "length"),
         "t": value(p, "thickness", 1, "length"), "edge": text(p, "edge", "round")}
    if d["edge"] not in ("round", "square"):
        raise ParamError("edge is round or square")
    if d["R"] <= d["t"]:
        raise ParamError("the plates must be wider than they are thick")
    return ESpec("plates", value(p, "voltage", 1000, "voltage"), text(p, "dielectric", "air"),
                 value(p, "margin", 2), d)


def plates_geometry(s: ESpec, k: float):
    R, d, t = s.dims["R"], s.dims["d"], s.dims["t"]
    shapes = []
    if s.dielectric != "air":
        shapes.append(Shape("dielectric", "rect", (0, -d / 2, R, d / 2)))
    fine = min(d, t) / 10                    # at the edges, where the field peaks
    face = min(d, t) / 2                     # along the flat faces, where it is uniform
    spots = []
    for name, z0 in (("top", d / 2), ("bottom", -d / 2 - t)):
        if s.dims["edge"] == "round":
            shapes += [Shape(name, "rect", (0, z0, R - t / 2, z0 + t), hole=True, size=face),
                       Shape(name, "disk", (R - t / 2, z0 + t / 2, t / 2), hole=True, size=face)]
            spots.append((R - t / 2, z0 + t / 2, fine + t / 2 * 0.2))
        else:
            shapes.append(Shape(name, "rect", (0, z0, R, z0 + t), hole=True, size=face))
            spots += [(R, z0, fine), (R, z0 + t, fine)]
    extent = max(R, d / 2 + t)
    model = Model2D("axi", shapes, 2.5 * extent, clip="x>=0", shell=3.0 * extent, size=min(d / 2, R / 15),
                    far=extent / 4, growth=0.25, lines={"axis": "x=0", "outer": "outer"}, spots=spots).scaled(k)
    prob = ElecProblem("axi", {"dielectric": get_dielectric(s.dielectric).epsr},
                       {"top": s.volts / 2, "bottom": -s.volts / 2, "outer": 0.0},
                       shell=(model.radius, model.shell, 0, 0), probes=[(0.0005 * R, 0.0)])
    a = 3 * (d + 2 * t)
    view = (R - 1.6 * a, R + a, -a / 1.6, a / 1.6)       # one edge, where the field peaks
    return model, prob, view


def kirchhoff(R: float, d: float, t: float, epsr: float = 1.0) -> float:
    """Kirchhoff's capacitance of two coaxial discs of radius R, gap d and thickness t (square edges), F.
    Asymptotically exact as d / R goes to zero; the dielectric factor only holds when it fills all space."""
    return epsr * EPS0 * (math.pi * R ** 2 / d + R * math.log(16 * math.pi * R * (d + t) / (math.e * d ** 2)) +
                          R * (t / d) * math.log((d + t) / t))


def coax_spec(p: dict) -> ESpec:
    check_known(p, COAX_KEYS, "coax")
    d = {"a": value(p, "inner_d", 1, "length") / 2, "b": value(p, "outer_d", 4.6, "length") / 2,
         "e": value(p, "offset", 0, "length")}
    if d["a"] + d["e"] >= d["b"]:
        raise ParamError("the inner conductor must fit inside the outer one")
    return ESpec("coax", value(p, "voltage", 1000, "voltage"), text(p, "dielectric", "pe"), value(p, "margin", 2), d)


def coax_geometry(s: ESpec, k: float):
    a, b, e = s.dims["a"], s.dims["b"], s.dims["e"]
    fine = a / 10
    shapes = [Shape("inner", "disk", (e, 0.0, a), hole=True, size=fine)]
    model = Model2D("planar", shapes, b, size=min((b - a) / 8, a / 2), far=(b - a) / 6, growth=0.2,
                    lines={"outer": "outer"}, refine=[("inner", fine, b - a)]).scaled(k)
    prob = ElecProblem("planar", {"air": get_dielectric(s.dielectric).epsr}, {"inner": s.volts, "outer": 0.0},
                       probes=[(e + a + 0.5 * (b - a - e), 0.0)])
    return model, prob, (-b * 1.1, b * 1.1, -b * 1.1, b * 1.1)


def coax_capacitance(a: float, b: float, e: float, epsr: float) -> float:
    """Exact capacitance per meter of two (possibly offset) circular cylinders, F/m."""
    return 2 * math.pi * EPS0 * epsr / math.acosh((a * a + b * b - e * e) / (2 * a * b))


def sphere_spec(p: dict) -> ESpec:
    check_known(p, SPHERE_KEYS, "sphere")
    return ESpec("sphere", value(p, "voltage", 10000, "voltage"), "air", value(p, "margin", 2),
                 {"R": value(p, "diameter", 20, "length") / 2, "g": value(p, "gap", 5, "length")})


def sphere_geometry(s: ESpec, k: float):
    R, g = s.dims["R"], s.dims["g"]
    fine = min(R, g) / 15
    shapes = [Shape("sphere", "disk", (0.0, g + R, R), hole=True, size=fine)]
    extent = g + 2 * R
    # the plane and the far boundary are one grounded conductor, so its charge is counted once
    model = Model2D("axi", shapes, 2.5 * extent, clip="x>=0,y>=0", shell=3.0 * extent, size=min(g, R) / 4,
                    far=extent / 4, growth=0.2, lines={"axis": "x=0", "ground": "y=0|outer"},
                    refine=[("sphere", fine, 2 * g)]).scaled(k)
    prob = ElecProblem("axi", {}, {"sphere": s.volts, "ground": 0.0},
                       shell=(model.radius, model.shell, 0, 0), probes=[(0.0005 * R, g * 0.999)])
    return model, prob, (-1.6 * R - g, 1.6 * R + g, -0.2 * g, g + 2.4 * R)


def sphere_images(R: float, g: float, V: float, n: int = 200) -> tuple[float, float]:
    """Capacitance and peak field of a sphere above a grounded plane by image charges (exact series), F and V/m."""
    h = g + R
    q, z = 4 * math.pi * EPS0 * R * V, h
    C, E = 0.0, 0.0
    zp = g                                   # the sphere's lowest point
    for _ in range(n):
        C += q / V
        E += q / (4 * math.pi * EPS0) * (1 / (z - zp) ** 2 + 1 / (z + zp) ** 2)
        q, z = q * R / (h + z), h - R * R / (h + z)
        if q < 1e-14 * C * V:
            break
    return C, E


TEMPLATES = {"plates": (plates_spec, plates_geometry), "coax": (coax_spec, coax_geometry),
             "sphere": (sphere_spec, sphere_geometry)}


def _solve(model: Model2D, prob: ElecProblem, work: Path, tag: str, fields: bool = False) -> dict:
    d = work / tag
    d.mkdir(parents=True, exist_ok=True)
    msh = mesh(model, d / "m.msh")
    prob.fields = fields
    solve.run(electrostatic(prob, msh.tags), msh.path, d, resolution="E")
    out = {"W": solve.read_global(d, "W.txt")[0], "nodes": msh.n_nodes,
           "Q": {c: float(solve.read_global(d, f"Q_{c}.txt")[0]) for c in prob.potentials if c in msh.tags}}
    out["Epk"], out["where"] = solve.peak_element(d, "epeak.txt")
    out["probes"] = [solve.read_point(d, f"probe{i}.txt") for i in range(len(prob.probes))]
    if fields:
        out["mesh"] = msh
        out["v"] = solve.read_nodes(d, "v.txt", msh.node_ids)
        out["en"] = solve.element_map(d, "en.txt", msh.X, msh.tris)
    return out


def run_electric(template: str, params: dict, out_dir: str | Path, question: str = "", keep: bool = False,
                 log=lambda stage, detail="": None) -> GenericOutcome:
    t_start = time.time()
    if template not in TEMPLATES:
        raise ParamError(f"unknown electric template '{template}' (use {', '.join(TEMPLATES)})")
    spec_fn, geom_fn = TEMPLATES[template]
    s = spec_fn(dict(params))
    diel = get_dielectric(s.dielectric)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    work = out / "work"
    shutil.rmtree(work, ignore_errors=True)
    levels = []
    for j, k in enumerate(LEVELS):
        log("solve", f"mesh {j + 1} of {len(LEVELS)}")
        m, p, view = geom_fn(s, k)
        levels.append(_solve(m, p, work, f"l{j}", fields=(j == len(LEVELS) - 1)))
    fine = levels[-1]
    drive = {"plates": "top", "coax": "inner", "sphere": "sphere"}[template]
    c_q = [abs(r["Q"][drive]) / s.volts for r in levels]     # the plates sit at plus and minus half the voltage
    epk = [r["Epk"] for r in levels]
    checks = [convergence_check(epk, 0.02, "peak field", "V/m", stopped_by="three meshes"),
              convergence_check(c_q, 0.005, "capacitance", "F", stopped_by="three meshes")]
    checks[1].name = "capacitance convergence"
    qs = fine["Q"]
    qbal = abs(sum(qs.values())) / max(abs(qs[drive]), 1e-30)
    checks.append(Check("charge balance", "pass" if qbal < 0.005 else ("warn" if qbal < 0.02 else "fail"),
                        "the charges on all conductors, including the far boundary, add up to "
                        f"{100 * qbal:.2g}% of the driven conductor's charge", {"charges_C": qs}))
    c_w = 2 * fine["W"] / s.volts ** 2
    cw_err = abs(c_w - c_q[-1]) / c_q[-1]
    checks.append(Check("energy balance", "pass" if cw_err < 0.005 else ("warn" if cw_err < 0.02 else "fail"),
                        f"capacitance from charge {fmt(c_q[-1] * 1e12)} pF and from stored energy "
                        f"{fmt(c_w * 1e12)} pF ({100 * cw_err:.2g}% apart)"))
    unit_c = "pF/m" if template == "coax" else "pF"
    C = c_q[-1]
    e_max = epk[-1]
    exact_note = ""
    if template == "plates":
        R, d, t = s.dims["R"], s.dims["d"], s.dims["t"]
        ck = kirchhoff(R, d, t, diel.epsr if s.dielectric == "air" else 1.0)
        par = EPS0 * diel.epsr * math.pi * R ** 2 / d
        if s.dielectric == "air":
            err = abs(C - ck) / ck
            st = ("pass" if err < 0.01 else "warn") if s.dims["edge"] == "square" else ("pass" if err < 0.03 else "info")
            checks.append(Check("closed form", st,
                                f"Kirchhoff's formula for square edged discs gives {fmt(ck * 1e12)} pF, the field solve "
                                f"{fmt(C * 1e12)} pF ({100 * err:.2g}% apart, rounded edges sit a little lower); the "
                                f"ideal parallel plate value without fringing is {fmt(par * 1e12)} pF",
                                {"kirchhoff_F": ck, "parallel_plate_F": par}))
        else:
            checks.append(Check("closed form", "info",
                                f"the ideal parallel plate value is {fmt(par * 1e12)} pF and the field solve gives "
                                f"{fmt(C * 1e12)} pF; fringing through the air around the dielectric adds the rest"))
        e_uni = s.volts / d
        strength = paschen_air(d) if s.dielectric == "air" else diel.strength * 1e6
        exact_note = f"A uniform field would be {fmt(e_uni / 1e6)} kV/mm."
    elif template == "coax":
        a, b, e = s.dims["a"], s.dims["b"], s.dims["e"]
        cx = coax_capacitance(a, b, e, diel.epsr)
        err = abs(C - cx) / cx
        checks.append(Check("closed form", "pass" if err < 0.005 else ("warn" if err < 0.02 else "fail"),
                            f"the exact capacitance of the cylinders is {fmt(cx * 1e12)} pF/m, the field solve "
                            f"{fmt(C * 1e12)} pF/m ({100 * err:.2g}% apart)", {"exact_F_per_m": cx}))
        if not e:
            ex = s.volts / (a * math.log(b / a))
            e_err = abs(e_max - ex) / ex
            checks.append(Check("peak field closed form", "pass" if e_err < 0.03 else "warn",
                                f"V / (a ln(b/a)) gives {fmt(ex / 1e6)} kV/mm at the inner conductor, the field solve "
                                f"{fmt(e_max / 1e6)} kV/mm ({100 * e_err:.2g}% apart). The solve samples the field at "
                                "element corners on a faceted circle, which errs a little high"))
        strength = diel.strength * 1e6 if s.dielectric != "air" else paschen_air(b - a)
    else:
        R, g = s.dims["R"], s.dims["g"]
        ci, ei = sphere_images(R, g, s.volts)
        err_c, err_e = abs(C - ci) / ci, abs(e_max - ei) / ei
        worst = max(err_c, err_e)
        checks.append(Check("closed form", "pass" if worst < 0.02 else ("warn" if worst < 0.05 else "fail"),
                            f"image charges give {fmt(ci * 1e12)} pF and a peak field of {fmt(ei / 1e6)} kV/mm, the field "
                            f"solve {fmt(C * 1e12)} pF and {fmt(e_max / 1e6)} kV/mm ({100 * err_c:.2g}% and "
                            f"{100 * err_e:.2g}% apart)", {"images_F": ci, "images_V_per_m": ei}))
        strength = peek_sphere(R)
    margin = strength / max(e_max, 1e-30)
    sane = 0 < C < 1 and 0 < e_max < 1e10
    checks.append(Check("units and sanity", "pass" if sane else "warn",
                        f"capacitance {fmt(C * 1e12)} {unit_c}, peak field {fmt(e_max / 1e6)} kV/mm"))
    lvl, reasons = confidence(checks)
    if margin >= s.target:
        verdict = "ok"
        head = (f"The peak field is {fmt(e_max / 1e6)} kV/mm at {fmt(s.volts)} V, {margin:.2g}x below the "
                f"{fmt(strength / 1e6)} kV/mm the {diel.name.lower()} withstands. Capacitance {fmt(C * 1e12)} {unit_c}.")
    elif margin >= 1:
        verdict = "marginal"
        head = (f"The peak field is {fmt(e_max / 1e6)} kV/mm at {fmt(s.volts)} V, only {margin:.2g}x below the "
                f"{fmt(strength / 1e6)} kV/mm breakdown strength, under the {s.target:g}x margin wanted.")
    else:
        verdict = "fails"
        head = (f"It breaks down. The peak field is {fmt(e_max / 1e6)} kV/mm at {fmt(s.volts)} V, above the "
                f"{fmt(strength / 1e6)} kV/mm the {diel.name.lower()} withstands.")
    if exact_note:
        head += " " + exact_note
    fixes = []
    if verdict != "ok":
        vmax = s.volts * margin / s.target
        fixes.append({"kind": "voltage", "title": f"Keep the voltage under {fmt(vmax)} V",
                      "summary": f"The field scales with voltage, so {fmt(vmax)} V gives the {s.target:g}x margin.",
                      "auto": False})
        if template == "plates":
            fixes.append({"kind": "gap", "title": f"Widen the gap to about {fmt(s.dims['d'] * s.target / margin * 1e3)} mm",
                          "summary": "The field between flat plates falls with the gap. Round the plate edges if they "
                                     "are square, since the field peaks there.", "auto": False})
        elif template == "sphere":
            fixes.append({"kind": "radius", "title": "Use a larger electrode radius",
                          "summary": "A bigger sphere spreads the field and raises the corona onset field too.",
                          "auto": False})
        else:
            fixes.append({"kind": "insulation", "title": "Use a thicker or stronger dielectric",
                          "summary": "The peak field sits on the inner conductor and falls with ln(b/a) times a.",
                          "auto": False})
    msh = fine["mesh"]
    view = geom_fn(s, LEVELS[-1])[2]
    images = {"field": plots.field_map(out / "images" / "field.png", msh.X, msh.tris, msh.region, fine["en"] / 1e6,
                                       "|E|, kV/mm", f"field strength at {fmt(s.volts)} V",
                                       mirror=template == "sphere", lines=fine["v"], skip_outline={msh.tags["air"]},
                                       view=view, mask=~msh.mask("shell"),
                                       marks=[(fine["where"][0], fine["where"][1], f"peak {fmt(e_max / 1e6)} kV/mm")])}
    metrics = [("peak field", f"{fmt(e_max / 1e6)} kV/mm"), ("breakdown", f"{fmt(strength / 1e6)} kV/mm"),
               ("margin", f"{margin:.2g}x"), ("capacitance", f"{fmt(C * 1e12)} {unit_c}")]
    assumptions = [f"Electrostatic, {('axisymmetric' if template != 'coax' else 'a cross section of a long cable')}, "
                   "conductors are perfect and their surfaces are equipotential.",
                   f"Between the conductors is {diel.name} (relative permittivity {diel.epsr:g}, {diel.source}).",
                   "Breakdown strength is a short term value at sea level; humidity, surface contamination, pressure "
                   "and long term aging lower it, and gaps across a surface (creepage) break down sooner."]
    if template == "plates" and s.dielectric == "air":
        assumptions.append("Air breakdown uses Paschen's law for a uniform field across this gap.")
    if template == "sphere":
        assumptions.append("Air breakdown uses Peek's law for corona onset on a sphere of this radius.")
    o = GenericOutcome("electric", template, question, verdict, head, metrics, lvl, reasons, checks, assumptions,
                       {"capacitance_F": C, "peak_field_V_per_m": e_max, "peak_at_m": list(fine["where"]),
                        "breakdown_V_per_m": strength, "margin": margin, "energy_J": fine["W"],
                        "levels": [{"nodes": r["nodes"], "peak_V_per_m": r["Epk"]} for r in levels],
                        "params": {k: str(v) for k, v in params.items()},
                        "_figures": [["field", "Field strength with equipotential lines"]]},
                       images, fixes, solver=f"GetDP {solve.getdp().version()} with gmsh",
                       t_total=time.time() - t_start)
    if not keep:
        shutil.rmtree(work, ignore_errors=True)
    return o.write(out)
