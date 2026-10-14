"""Magnetic studies: solenoid actuators, lifting electromagnets and permanent magnet assemblies.

"Does this solenoid pull 8 N over a 6 mm stroke?" becomes an axisymmetric
magnetostatic model with nonlinear steel, solved with GetDP at every gap of
the stroke. The force comes from the Maxwell stress tensor around the moving
part, and Wobbl checks it against virtual work (the change of co-energy over a
small move), checks that energy plus co-energy equals flux linkage times
current, refines the mesh until the force settles and compares with a
magnetic circuit estimate.
"""
from __future__ import annotations

import math
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..materials import get_magnetic
from ..outcome_generic import GenericOutcome
from ..units import fmt
from ..verify.checks import Check, confidence, convergence_check
from . import plots, solve
from .forms import MU0, Coil, MagProblem, MagRegion, magnetostatic
from .mesh2d import Model2D, Shape, mesh
from .params import ParamError, check_known, text, value

MM = 1e-3
LEVELS = (1.4, 1.0, 0.7)        # element size factors: the sweep runs on the first, convergence adds the others


@dataclass
class Spec:
    template: str
    params: dict
    gaps: list[float]                       # m, the stroke points
    need: float                             # N, 0 when not given
    turns: float = 0.0
    current: float = 0.0
    coil_area: float = 0.0
    steel: str = "1010-steel"
    magnet: str = ""
    pull: float = 1.0                       # +1 when attraction is +y on the moving part
    dims: dict = field(default_factory=dict)
    air_core: bool = False


def _steel(name: str) -> MagRegion:
    m = get_magnetic(name)
    return MagRegion("", m.mur, m.bh)


def _region(name: str, mat: str, **kw) -> MagRegion:
    r = _steel(mat)
    return MagRegion(name, r.mur, r.bh, **kw)


def _outer(extent: float, shapes, refine, size, lines=None) -> Model2D:
    R = 2.5 * extent
    return Model2D("axi", shapes, R, clip="x>=0", shell=1.2 * R, size=size, far=R / 12, growth=0.25,
                   lines=lines or {"axis": "x=0", "outer": "outer"}, refine=refine)


# --- solenoid ------------------------------------------------------------------------
SOLENOID_KEYS = {"turns", "current", "coil_id", "coil_od", "length", "plunger_d", "frame", "clearance", "stroke",
                 "min_gap", "points", "material", "need", "stop", "core"}


def solenoid_spec(p: dict) -> Spec:
    check_known(p, SOLENOID_KEYS, "solenoid")
    d = {"ri": value(p, "coil_id", 12, "length") / 2, "ro": value(p, "coil_od", 30, "length") / 2,
         "lc": value(p, "length", 30, "length"), "rp": value(p, "plunger_d", 10, "length") / 2,
         "t": value(p, "frame", 2, "length"), "c": value(p, "clearance", 0.25, "length")}
    d["ls"] = value(p, "stop", d["lc"] / 3 * 1e3, "length")
    if d["ri"] < d["rp"] + 0.2 * MM:
        raise ParamError("the coil's inside diameter must clear the plunger by at least 0.2 mm on the radius")
    if d["ro"] <= d["ri"]:
        raise ParamError("coil_od must be larger than coil_id")
    stroke = value(p, "stroke", 6, "length")
    if stroke >= d["lc"] - d["ls"]:
        raise ParamError(f"a {stroke * 1e3:g} mm stroke would pull the plunger out of the coil. Keep it under "
                         f"{(d['lc'] - d['ls']) * 1e3:g} mm or shorten the stop")
    air = text(p, "core", "steel") in ("none", "air", "0", "no")
    s = Spec("solenoid", p, _gaps(p, stroke), value(p, "need", 0, "force"), value(p, "turns", 1000),
             value(p, "current", 0.5, "current"), (d["ro"] - d["ri"]) * d["lc"], text(p, "material", "1010-steel"),
             pull=1.0, dims=d, air_core=air)
    get_magnetic(s.steel)
    return s


def _gaps(p: dict, stroke: float) -> list[float]:
    n = int(value(p, "points", 6))
    g0 = value(p, "min_gap", max(stroke / 12, 0.2e-3) * 1e3, "length")
    if stroke <= g0:
        return [stroke]
    return [float(g) for g in np.linspace(g0, stroke, max(n, 2))]


def solenoid_geometry(s: Spec, gap: float, k: float = 1.0) -> tuple[Model2D, MagProblem, tuple]:
    d = s.dims
    ri, ro, lc, rp, t, c, ls = d["ri"], d["ro"], d["lc"], d["rp"], d["t"], d["c"], d["ls"]
    zs = lc / 2 - ls                         # stop face
    lp = lc - ls + t + lc / 3                # plunger length, sticking lc/3 out of the frame when closed
    coil = Shape("coil", "rect", (ri, -lc / 2, ro, lc / 2))
    size = min(t / 2, (ro - ri) / 5, lc / 16)
    regions = [MagRegion("air"), MagRegion("coil", js=s.turns * s.current / s.coil_area)]
    if s.air_core:
        # no steel and no gaps: the field is smooth, so elements follow the coil's size, not its winding thickness
        size = max((ro - ri) / 2, min(ri / 4, lc / 60))
        shapes = [coil]
        refine = []
        body = []
        probes = [(0.3 * ri, 0.0)]
    else:
        shapes = [coil,
                  Shape("frame", "poly", ((ro, -lc / 2 - t), (ro + t, -lc / 2 - t), (ro + t, lc / 2 + t),
                                          (0, lc / 2 + t), (0, lc / 2), (ro, lc / 2))),
                  Shape("frame", "rect", (rp + c, -lc / 2 - t, ro, -lc / 2)),
                  Shape("frame", "rect", (0, zs, rp, lc / 2)),
                  Shape("plunger", "rect", (0, zs - gap - lp, rp, zs - gap))]
        fine = max(min(gap, c, t) / 3, 0.04 * MM)
        refine = [("plunger", fine, 6 * size)]
        regions += [_region("frame", s.steel), _region("plunger", s.steel)]
        body = ["plunger"]
        probes = [(0.3 * rp, zs - gap / 2)]
    extent = max(ro + t, lc / 2 + t + lp)
    model = _outer(extent, shapes, refine, size).scaled(k)
    prob = MagProblem("axi", regions, ["axis", "outer"], shell=(model.radius, model.shell, 0, 0),
                      coils=[Coil("coil", s.turns, s.coil_area)], body=body, probes=probes)
    view = (-(ro + t) * 1.6, (ro + t) * 1.6, -lc / 2 - t - 0.6 * lp, lc / 2 + t + 0.25 * lc)
    return model, prob, view


def solenoid_estimate(s: Spec, gap: float) -> float:
    """Magnetic circuit: the working gap with fringing, plus the radial clearance where the plunger enters."""
    d = s.dims
    rp, c, t = d["rp"], d["c"], d["t"]

    def perm(g):
        a = math.pi * (rp + g / 2) ** 2
        r = g / (MU0 * a) + c / (MU0 * 2 * math.pi * (rp + c / 2) * t)
        return 1 / r

    dg = 1e-3 * max(gap, 1e-4)
    ni = s.turns * s.current
    return 0.5 * ni ** 2 * -(perm(gap + dg) - perm(gap - dg)) / (2 * dg)


# --- lifting electromagnet (pot core and armature plate) ---------------------------------
POT_KEYS = {"turns", "current", "diameter", "height", "pole_d", "wall", "base", "plate", "plate_d", "stroke",
            "min_gap", "points", "material", "need"}


def pot_spec(p: dict) -> Spec:
    check_known(p, POT_KEYS, "electromagnet")
    r3 = value(p, "diameter", 50, "length") / 2
    d = {"r3": r3, "h": value(p, "height", 25, "length"), "r1": value(p, "pole_d", 20, "length") / 2,
         "w": value(p, "wall", 4, "length"), "tb": value(p, "base", 5, "length"),
         "ta": value(p, "plate", 5, "length")}
    d["rpl"] = value(p, "plate_d", 2 * r3 * 1e3 + 10, "length") / 2
    d["r2"] = r3 - d["w"]
    if d["r2"] <= d["r1"] + 1 * MM:
        raise ParamError("the pole and the wall leave no room for the coil. Widen the magnet or narrow the pole")
    stroke = value(p, "stroke", 3, "length")
    area = (d["r2"] - d["r1"] - 0.6 * MM) * (d["h"] - 0.5 * MM)
    s = Spec("electromagnet", p, _gaps(p, stroke), value(p, "need", 0, "force"), value(p, "turns", 500),
             value(p, "current", 1, "current"), area, text(p, "material", "1010-steel"), pull=-1.0, dims=d)
    get_magnetic(s.steel)
    return s


def pot_geometry(s: Spec, gap: float, k: float = 1.0):
    d = s.dims
    r1, r2, r3, h, tb, ta, rpl = d["r1"], d["r2"], d["r3"], d["h"], d["tb"], d["ta"], d["rpl"]
    shapes = [Shape("core", "poly", ((0, -tb), (r3, -tb), (r3, h), (r2, h), (r2, 0), (r1, 0), (r1, h), (0, h))),
              Shape("coil", "rect", (r1 + 0.3 * MM, 0, r2 - 0.3 * MM, h - 0.5 * MM)),
              Shape("plate", "rect", (0, h + gap, rpl, h + gap + ta))]
    size = min(d["w"] / 3, (r2 - r1) / 6, h / 12)
    fine = max(gap / 3, 0.04 * MM)
    model = _outer(max(r3, rpl, h + tb + ta + gap), shapes, [("plate", fine, 6 * size)], size).scaled(k)
    regions = [MagRegion("air"), MagRegion("coil", js=s.turns * s.current / s.coil_area),
               _region("core", s.steel), _region("plate", s.steel)]
    prob = MagProblem("axi", regions, ["axis", "outer"], shell=(model.radius, model.shell, 0, 0),
                      coils=[Coil("coil", s.turns, s.coil_area)], body=["plate"],
                      probes=[((r1 + 0) * 0.4, h + gap / 2)])
    view = (-max(r3, rpl) * 1.3, max(r3, rpl) * 1.3, -tb - 0.3 * h, h + gap + ta + 0.4 * h)
    return model, prob, view


def pot_estimate(s: Spec, gap: float) -> float:
    """Two gaps in series, the center pole face and the ring under the wall, each with fringing."""
    d = s.dims
    a1 = math.pi * (d["r1"] + gap / 2) ** 2
    a2 = math.pi * (d["r3"] ** 2 - d["r2"] ** 2) + 2 * math.pi * d["r3"] * gap / 2
    ni = s.turns * s.current
    b1 = MU0 * ni / (gap * (1 + a1 / a2))
    phi = b1 * a1
    return phi ** 2 / (2 * MU0) * (1 / a1 + 1 / a2)


# --- permanent magnet holding a steel plate --------------------------------------------------
MAGNET_KEYS = {"diameter", "height", "grade", "cup", "cup_gap", "plate", "plate_d", "stroke", "min_gap", "points",
               "material", "need"}


def magnet_spec(p: dict) -> Spec:
    check_known(p, MAGNET_KEYS, "magnet")
    rm = value(p, "diameter", 20, "length") / 2
    d = {"rm": rm, "hm": value(p, "height", 10, "length"), "tc": value(p, "cup", 0, "length"),
         "gc": value(p, "cup_gap", 1, "length"), "ta": value(p, "plate", 5, "length")}
    d["rpl"] = value(p, "plate_d", 2 * (rm + (d["tc"] + d["gc"] if d["tc"] else 0)) * 1e3 + 20, "length") / 2
    stroke = value(p, "stroke", 5, "length")
    grade = text(p, "grade", "n42")
    mag = get_magnetic(grade)
    if not mag.br:
        raise ParamError(f"'{grade}' is not a magnet grade. Use n35, n42, n52 or ferrite")
    s = Spec("magnet", p, _gaps(p, stroke), value(p, "need", 0, "force"), steel=text(p, "material", "1010-steel"),
             magnet=mag.key, pull=-1.0, dims=d)
    get_magnetic(s.steel)
    return s


def magnet_geometry(s: Spec, gap: float, k: float = 1.0, plate: bool = True):
    d = s.dims
    rm, hm, tc, gc, ta, rpl = d["rm"], d["hm"], d["tc"], d["gc"], d["ta"], d["rpl"]
    mag = get_magnetic(s.magnet)
    shapes = [Shape("magnet", "rect", (0, 0, rm, hm))]
    regions = [MagRegion("air"), MagRegion("magnet", mag.mur, br=(0.0, mag.br))]
    if tc:
        shapes.append(Shape("cup", "poly", ((0, -tc), (rm + gc + tc, -tc), (rm + gc + tc, hm), (rm + gc, hm),
                                            (rm + gc, 0), (0, 0))))
        regions.append(_region("cup", s.steel))
    body = []
    size = min(hm / 8, rm / 6, (tc / 3 if tc else 1.0), ta / 3)
    refine = [("magnet", size / 2, 4 * size)]
    if plate:
        shapes.append(Shape("plate", "rect", (0, hm + gap, rpl, hm + gap + ta)))
        regions.append(_region("plate", s.steel))
        body = ["plate"]
        refine = [("plate", max(gap / 3, 0.04 * MM), 6 * size)]
    extent = max(rm + tc + gc, rpl, hm + tc + gap + ta)
    model = _outer(extent, shapes, refine, size).scaled(k)
    prob = MagProblem("axi", regions, ["axis", "outer"], shell=(model.radius, model.shell, 0, 0), body=body,
                      probes=[(0.3 * rm, hm + max(gap, 1 * MM) / 2)])
    view = (-extent * 1.3, extent * 1.3, -tc - 0.4 * hm, hm + gap + ta + 0.5 * hm)
    return model, prob, view


def magnet_estimate(s: Spec, gap: float) -> float:
    """Load line of the magnet against two air gaps through the plate (the face and the return path)."""
    d = s.dims
    mag = get_magnetic(s.magnet)
    a = math.pi * (d["rm"] + gap / 2) ** 2
    g_eff = 2 * gap + 0.5 * d["rm"] if not d["tc"] else gap * 1.5   # open magnets return through a long air path
    bg = mag.br * d["hm"] / (d["hm"] + mag.mur * g_eff)
    return bg ** 2 * a / (2 * MU0)


def magnet_axis_field(br: float, r: float, h: float, z: float) -> float:
    """Closed form flux density on the axis of an axially magnetized cylinder, z above its face."""
    return br / 2 * ((z + h) / math.hypot(z + h, r) - z / math.hypot(z, r))


TEMPLATES = {"solenoid": (solenoid_spec, solenoid_geometry, solenoid_estimate),
             "electromagnet": (pot_spec, pot_geometry, pot_estimate),
             "magnet": (magnet_spec, magnet_geometry, magnet_estimate)}


# --- solving --------------------------------------------------------------------------------
def _solve(model: Model2D, prob: MagProblem, work: Path, tag: str, fields: bool = False) -> dict:
    d = work / tag
    d.mkdir(parents=True, exist_ok=True)
    msh = mesh(model, d / "m.msh")
    prob.fields = fields
    t0 = time.time()
    log = solve.run(magnetostatic(prob, msh.tags), msh.path, d)
    it, rel = solve.residual(log)
    out = {"W": solve.read_global(d, "W.txt")[0], "Wc": solve.read_global(d, "Wc.txt")[0], "iters": it,
           "rel": rel, "nodes": msh.n_nodes, "t": time.time() - t0}
    if prob.coils:
        out["psi"] = solve.read_global(d, f"Psi_{prob.coils[0].name}.txt")[0]
    if prob.body:
        out["F"] = float(solve.read_global(d, "F.txt")[1])     # axial component (y is the axis)
    out["probes"] = [solve.read_point(d, f"probe{i}.txt") for i in range(len(prob.probes))]
    if (d / "bnl.txt").exists():
        out["bnl"] = solve.read_elements(d, "bnl.txt").max(axis=1)
    if (d / "hm.txt").exists():
        out["hm"] = solve.read_elements(d, "hm.txt").min(axis=1)
    if fields:
        out["mesh"] = msh
        out["az"] = solve.read_nodes(d, "az.txt", msh.node_ids)
        out["bt"] = solve.element_map(d, "bn.txt", msh.X, msh.tris)
    return out


def run_magnetic(template: str, params: dict, out_dir: str | Path, question: str = "", keep: bool = False,
                 log=lambda stage, detail="": None) -> GenericOutcome:
    t_start = time.time()
    if template not in TEMPLATES:
        raise ParamError(f"unknown magnetic template '{template}' (use {', '.join(TEMPLATES)})")
    spec_fn, geom_fn, est_fn = TEMPLATES[template]
    s = spec_fn(dict(params))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    work = out / "work"
    shutil.rmtree(work, ignore_errors=True)
    if s.air_core:
        return _air_core(s, out, work, question, keep, t_start, log)
    sweep = []
    for i, g in enumerate(s.gaps):
        log("solve", f"gap {g * 1e3:.3g} mm ({i + 1} of {len(s.gaps)})")
        m, p, view = geom_fn(s, g, LEVELS[0])
        sweep.append(_solve(m, p, work, f"g{i}", fields=True))      # fields feed the playback frames
    forces = np.array([s.pull * r["F"] for r in sweep])
    gi = int(np.argmin(forces))                 # the weakest point of the stroke governs
    g = s.gaps[gi]
    log("verify", f"refining the mesh at the {g * 1e3:.3g} mm gap")
    levels = [sweep[gi]]
    for j, k in enumerate(LEVELS[1:], start=1):
        m, p, view = geom_fn(s, g, k)
        levels.append(_solve(m, p, work, f"conv{j}", fields=(j == 1)))
    fine = levels[-1]
    f_gov = s.pull * fine["F"]
    checks = [convergence_check([s.pull * r["F"] for r in levels], 0.02, "pull force", "N",
                                stopped_by="three meshes")]
    log("verify", "virtual work at the same gap")
    dg = max(0.04 * g, 0.03 * MM)
    wc = []
    for sgn, tag in ((-1, "vw_minus"), (1, "vw_plus")):
        m, p, _ = geom_fn(s, g + sgn * dg, LEVELS[1])
        wc.append(_solve(m, p, work, tag)["Wc"])
    f_vw = (wc[0] - wc[1]) / (2 * dg)
    f_mid = s.pull * levels[1]["F"]
    dv = abs(f_vw - f_mid) / max(abs(f_mid), 1e-12)
    checks.append(Check("force cross check", "pass" if dv < 0.03 else ("warn" if dv < 0.08 else "fail"),
                        f"Maxwell stress gives {fmt(f_mid)} N and virtual work (co-energy change over a "
                        f"{fmt(2 * dg * 1e3)} mm move) gives {fmt(f_vw)} N, {100 * dv:.2g}% apart",
                        {"mst_N": f_mid, "virtual_work_N": f_vw}))
    if "psi" in fine:
        lhs, rhs = fine["W"] + fine["Wc"], fine["psi"] * s.current
        eb = abs(lhs - rhs) / max(abs(rhs), 1e-30)
        checks.append(Check("energy balance", "pass" if eb < 0.005 else ("warn" if eb < 0.02 else "fail"),
                            f"stored energy plus co-energy is {fmt(lhs * 1e3)} mJ and flux linkage times current is "
                            f"{fmt(rhs * 1e3)} mJ ({100 * eb:.2g}% apart)", {"W_J": fine["W"], "Wc_J": fine["Wc"]}))
    rels = [r["rel"] for r in sweep + levels if r["iters"]]
    if rels:
        worst = max(rels)
        checks.append(Check("nonlinear solve", "pass" if worst < 1e-6 else ("warn" if worst < 1e-3 else "fail"),
                            f"Newton iterations on the steel's BH curve converged to a relative residual of "
                            f"{worst:.1e} or better (at most {max(r['iters'] for r in sweep + levels)} iterations)"))
    # the circuit estimate holds best where the gap is small next to the pole, which is the closed end of the stroke
    g0 = s.gaps[0]
    est = est_fn(s, g0)
    f0 = float(forces[0])
    ratio = f0 / max(est, 1e-12)
    st = "pass" if 0.7 < ratio < 1.4 else ("info" if 0.4 < ratio < 2.5 else "warn")
    checks.append(Check("hand calculation", st,
                        f"a magnetic circuit estimate with fringing gives {fmt(est)} N at the {fmt(g0 * 1e3)} mm gap "
                        f"and the field solve {fmt(f0)} N ({ratio:.2g}x). The estimate ignores leakage and saturation, "
                        "so only a large mismatch points to a setup problem", {"estimate_N": est, "gap_mm": g0 * 1e3}))
    if s.template == "magnet" and not s.dims["tc"]:
        checks.append(_bare_magnet_check(s, work, geom_fn))
    steel = get_magnetic(s.steel)
    knee = steel.knee()
    bmax = float(np.max(fine["bnl"])) if "bnl" in fine else 0.0
    frac = float(np.mean(fine["bnl"] > knee)) if "bnl" in fine else 0.0
    where = f"over {100 * frac:.0f}% of its area" if frac >= 0.01 else "in a few spots"
    sat_word = (f"the steel peaks at {fmt(bmax)} T, past its {fmt(knee)} T knee {where}, so it is saturating"
                if bmax > knee else f"the steel peaks at {fmt(bmax)} T, below its {fmt(knee)} T knee")
    checks.append(Check("saturation", "info", sat_word, {"b_max_T": bmax, "knee_T": knee, "fraction": frac}))
    demag = None
    if s.magnet and "hm" in fine:
        mag = get_magnetic(s.magnet)
        hmin = float(np.min(fine["hm"]))
        demag = -hmin / mag.hcj
        # a linear magnet model holds while the reverse field stays clear of the coercivity
        checks.append(Check("demagnetization", "info" if demag < 0.8 else ("warn" if demag < 1.0 else "fail"),
                            f"the worst reverse field in the magnet is {fmt(-hmin / 1e3)} kA/m, {100 * demag:.0f}% "
                            f"of the {fmt(mag.hcj / 1e3)} kA/m coercivity at 20 C"
                            + (", and coercivity falls about 0.6% per degree as NdFeB warms" if demag > 0.5 else "")))
    bgap = float(np.linalg.norm(fine["probes"][0][:2])) if fine["probes"] else 0.0
    sane = 0 < f_gov < 1e5 and bgap < 3.0
    checks.append(Check("units and sanity", "pass" if sane else "warn",
                        f"attracting force {fmt(f_gov)} N, gap flux density {fmt(bgap)} T"))
    lvl, reasons = confidence(checks)
    # verdict on the weakest point of the stroke
    fmin, fmax = float(forces.min()), float(forces.max())
    f_sweep = forces * (f_gov / forces[gi])              # carry the converged correction across the stroke
    need = s.need
    word = {"solenoid": "The plunger pulls", "electromagnet": "The magnet pulls the plate with",
            "magnet": "The magnet holds the plate with"}[s.template]
    sat_tail = (f" The steel saturates at {fmt(bmax)} T, so more current buys little force." if bmax > knee * 1.05
                else "")
    if need:
        if f_gov >= 1.2 * need:
            verdict = "ok"
            head = (f"{word} {fmt(f_gov)} N at the {fmt(g * 1e3)} mm gap, the weakest point of the stroke, "
                    f"{f_gov / need:.2g}x the {fmt(need)} N needed.")
        elif f_gov >= need:
            verdict = "marginal"
            head = (f"{word} {fmt(f_gov)} N at the {fmt(g * 1e3)} mm gap, just above the {fmt(need)} N needed, "
                    "with under 20% margin.")
        else:
            verdict = "fails"
            head = f"{word} only {fmt(f_gov)} N at the {fmt(g * 1e3)} mm gap, short of the {fmt(need)} N needed."
    else:
        verdict = "ok"
        head = (f"{word} {fmt(f_gov)} N at the {fmt(g * 1e3)} mm gap and up to {fmt(float(f_sweep.max()))} N "
                f"at {fmt(s.gaps[int(np.argmax(f_sweep))] * 1e3)} mm.")
    head += sat_tail
    fixes = _fixes(s, f_sweep, f_gov, need, bmax > knee * 1.05) if need and verdict != "ok" else []
    if demag is not None and demag > 0.5:
        fixes.append({"kind": "grade", "title": "Use a high coercivity grade if the magnet runs warm",
                      "summary": f"The reverse field reaches {100 * demag:.0f}% of the coercivity at 20 C. An SH or UH "
                                 "grade keeps its coercivity to 150 C and above, at a few percent less remanence.",
                      "auto": False})
    images = {}
    view = geom_fn(s, g, LEVELS[1])[2]
    mid = levels[1]
    msh = mid["mesh"]
    images["field"] = plots.field_map(out / "images" / "field.png", msh.X, msh.tris, msh.region, mid["bt"], "|B|, T",
                                      f"flux density at the {fmt(g * 1e3)} mm gap", mirror=True,
                                      lines=mid["az"] * msh.X[:, 0], skip_outline={msh.tags["air"]}, view=view,
                                      mask=~msh.mask("shell"))
    gx = np.array(s.gaps) * 1e3
    images["force"] = plots.curve(out / "images" / "force.png", gx, {"pull force, N": f_sweep}, "gap, mm",
                                  "force, N", hline=need or None, hlabel=f"needed {fmt(need)} N" if need else "")
    metrics = [("force at weakest gap", f"{fmt(f_gov)} N"), ("needed", f"{fmt(need)} N" if need else "not given"),
               ("peak steel field", f"{fmt(bmax)} T"), ("gap", f"{fmt(g * 1e3)} mm")]
    values = {"gaps_mm": list(gx), "force_N": [float(v) for v in f_sweep], "force_raw_N": [float(v) for v in forces],
              "force_weakest_N": f_gov, "weakest_gap_mm": g * 1e3, "virtual_work_N": f_vw, "b_steel_max_T": bmax,
              "steel_knee_T": knee, "gap_flux_density_T": bgap,
              "levels": [{"nodes": r["nodes"], "force_N": s.pull * r["F"]} for r in levels]}
    if "psi" in fine:
        L = [r["psi"] / s.current for r in sweep]
        values["inductance_H"] = [float(v) for v in L]
        metrics[3] = ("inductance", f"{fmt(min(L) * 1e3)} to {fmt(max(L) * 1e3)} mH")
    if demag is not None:
        values["demag_fraction"] = demag
    assumptions = _assumptions(s, steel)
    o = GenericOutcome("magnetic", template, question, verdict, head, metrics, lvl, reasons, checks, assumptions,
                       {**values, "params": {k: str(v) for k, v in params.items()},
                        "_figures": [["force", "Pull force over the stroke"],
                                     ["field", "Flux density and flux lines at the weakest gap"]]},
                       images, fixes, solver=f"GetDP {solve.getdp().version()} with gmsh", t_total=time.time() - t_start)
    _playback(s, sweep, f_sweep, values.get("inductance_H"), geom_fn, out)
    if not keep:
        shutil.rmtree(work, ignore_errors=True)
    return o.write(out)


def _playback(s: Spec, sweep: list[dict], f_sweep, L, geom_fn, out: Path) -> str:
    """One frame per gap of the stroke: |B| and the flux function r a on the r-z half plane, the parts at that
    gap and the pull force there."""
    from . import playback as pb

    views = [geom_fn(s, g, LEVELS[0])[2] for g in s.gaps]
    win = (0.0, max(v[1] for v in views), min(v[2] for v in views), max(v[3] for v in views))
    nx = 40
    ny = int(min(96, max(24, round(nx * (win[3] - win[2]) / (win[1] - win[0])))))
    frames = []
    for i, (g, r) in enumerate(zip(s.gaps, sweep)):
        frames.append({"gap_mm": round(g * 1e3, 4), "force_N": round(float(f_sweep[i]), 5) if len(f_sweep) else None,
                       "inductance_mH": round(L[i] * 1e3, 5) if L else None,
                       **_frame(r, win, (nx, ny)), "parts": pb.shapes_mm(geom_fn(s, g, LEVELS[0])[0].shapes)})
    manifest = {"frames": "one per gap of the stroke, from closed to open",
                "frames[].gap_mm": "air gap between the moving part and the pole, mm",
                "frames[].force_N": "pull force on the moving part after the mesh refinement correction, N",
                "frames[].inductance_mH": "coil inductance, flux linkage over current, mH (null without a coil)",
                "frames[].b": "flux density magnitude |B| on the grid, T, packed",
                "frames[].flux": "flux function r a_phi on the grid, Wb/rad, packed; its contours are flux lines",
                "frames[].parts": "the model's parts in mm (x is the radius, y the axis), drawn over the grid",
                "grid": "x0, x1, y0, y1 in mm and nx by ny points; the model is axisymmetric about x = 0, so "
                        "mirror it for the full section"}
    return pb.write(out, "magnetic-stroke", manifest,
                    {"template": s.template, "grid": {"x0": 0.0, "x1": round(win[1] * 1e3, 4),
                                                      "y0": round(win[2] * 1e3, 4), "y1": round(win[3] * 1e3, 4),
                                                      "nx": nx, "ny": ny, "mirror": True},
                     "frames": frames, "need_N": s.need or None, "units": {"length": "mm", "b": "T", "force": "N"}})


def _frame(r: dict, win: tuple, n: tuple) -> dict:
    from . import playback as pb

    msh = r["mesh"]
    b = pb.grid(msh.X, msh.tris, win, n, tri_vals=np.where(msh.mask("shell"), np.nan, r["bt"]))
    flux = pb.grid(msh.X, msh.tris, win, n, node_vals=r["az"] * msh.X[:, 0])
    return {"b": pb.pack(b, 0.0, clip=99.5), "flux": pb.pack_signed(flux)}


def _bare_magnet_check(s: Spec, work: Path, geom_fn) -> Check:
    m, p, _ = geom_fn(s, 0.0, LEVELS[1], plate=False)
    rm, hm = s.dims["rm"], s.dims["hm"]
    z = 0.5 * rm
    p.probes = [(0.0005 * rm, hm + z)]
    r = _solve(m, p, work, "bare")
    got = float(abs(r["probes"][0][1]))
    mag = get_magnetic(s.magnet)
    want = magnet_axis_field(mag.br, rm, hm, z)
    err = abs(got - want) / want
    return Check("closed form", "pass" if err < 0.04 else ("warn" if err < 0.1 else "fail"),
                 f"the bare magnet's field on its axis {fmt(z * 1e3)} mm above the face is {fmt(got * 1e3)} mT, the "
                 f"closed form for a uniformly magnetized cylinder gives {fmt(want * 1e3)} mT ({100 * err:.2g}% apart, "
                 f"the closed form ignores the {mag.mur:g} recoil permeability)", {"fea_T": got, "closed_form_T": want})


def _air_core(s: Spec, out: Path, work: Path, question, keep, t_start, log) -> GenericOutcome:
    d = s.dims
    vals = []
    for j, k in enumerate(LEVELS):
        log("solve", f"mesh {j + 1} of {len(LEVELS)}")
        m, p, view = solenoid_geometry(s, 0.0, k)
        vals.append(_solve(m, p, work, f"air{j}", fields=(j == len(LEVELS) - 1)))
    fine = vals[-1]
    bc = [float(abs(r["probes"][0][1])) for r in vals]
    n = s.turns / d["lc"]
    ni = s.turns * s.current
    rs = np.linspace(d["ri"], d["ro"], 401)
    exact = float(np.mean(MU0 * n * s.current * d["lc"] / np.sqrt(d["lc"] ** 2 + 4 * rs ** 2)))
    err = abs(bc[-1] - exact) / exact
    checks = [convergence_check(bc, 0.01, "center flux density", "T", stopped_by="three meshes"),
              Check("closed form", "pass" if err < 0.01 else ("warn" if err < 0.03 else "fail"),
                    f"the center field is {fmt(bc[-1] * 1e3)} mT, the finite solenoid formula averaged over the "
                    f"winding gives {fmt(exact * 1e3)} mT and mu0 n I {fmt(MU0 * n * s.current * 1e3)} mT "
                    f"({100 * err:.2g}% from the finite formula)", {"exact_T": exact})]
    lhs, rhs = 2 * fine["W"], fine["psi"] * s.current
    eb = abs(lhs - rhs) / rhs
    checks.append(Check("energy balance", "pass" if eb < 0.005 else "warn",
                        f"twice the stored energy is {fmt(lhs * 1e3)} mJ and flux linkage times current "
                        f"{fmt(rhs * 1e3)} mJ ({100 * eb:.2g}% apart)"))
    lvl, reasons = confidence(checks)
    L = fine["psi"] / s.current
    head = (f"The air core coil makes {fmt(bc[-1] * 1e3)} mT at its center from {fmt(ni)} ampere turns, with an "
            f"inductance of {fmt(L * 1e3)} mH.")
    msh = fine["mesh"]
    images = {"field": plots.field_map(out / "images" / "field.png", msh.X, msh.tris, msh.region, fine["bt"], "|B|, T",
                                       "flux density of the air core coil", mirror=True,
                                       lines=fine["az"] * msh.X[:, 0], skip_outline={msh.tags["air"]},
                                       view=(-d["ro"] * 2.5, d["ro"] * 2.5, -d["lc"] * 0.8, d["lc"] * 0.8),
                                       mask=~msh.mask("shell"))}
    o = GenericOutcome("magnetic", "solenoid", question, "ok", head,
                       [("center field", f"{fmt(bc[-1] * 1e3)} mT"), ("inductance", f"{fmt(L * 1e3)} mH"),
                        ("ampere turns", fmt(ni)), ("mesh nodes", f"{fine['nodes']:,}")], lvl, reasons, checks,
                       ["Air core coil, uniform current over the winding cross section, axisymmetric.",
                        "Open space is modeled by a shell mapped to infinity."],
                       {"b_center_T": bc[-1], "b_levels_T": bc, "b_exact_T": exact, "inductance_H": L,
                        "_figures": [["field", "Flux density and flux lines"]]}, images,
                       solver=f"GetDP {solve.getdp().version()} with gmsh", t_total=time.time() - t_start)
    from . import playback as pb
    win = (0.0, d["ro"] * 2.5, -d["lc"] * 0.8, d["lc"] * 0.8)
    n = (32, int(min(96, max(24, round(32 * 1.6 * d["lc"] / (2.5 * d["ro"]))))))
    pb.write(out, "magnetic-stroke", {"frames": "a single frame, the air core coil",
                                      "frames[].b": "flux density magnitude |B| on the grid, T, packed",
                                      "frames[].flux": "flux function r a_phi, Wb/rad, packed; contours are flux lines",
                                      "grid": "x0, x1, y0, y1 in mm and nx by ny points, axisymmetric about x = 0"},
             {"template": "solenoid", "grid": {"x0": 0.0, "x1": round(win[1] * 1e3, 4), "y0": round(win[2] * 1e3, 4),
                                               "y1": round(win[3] * 1e3, 4), "nx": n[0], "ny": n[1], "mirror": True},
              "frames": [{"gap_mm": None, "force_N": None, "inductance_mH": round(L * 1e3, 5), **_frame(fine, win, n),
                          "parts": pb.shapes_mm(solenoid_geometry(s, 0.0, 1.0)[0].shapes)}],
              "units": {"length": "mm", "b": "T"}})
    if not keep:
        shutil.rmtree(work, ignore_errors=True)
    return o.write(out)


def _fixes(s: Spec, f_sweep, f_gov, need, saturated: bool) -> list[dict]:
    fixes = []
    target = 1.25 * need
    if s.turns and not saturated:
        k = math.sqrt(target / max(f_gov, 1e-9))
        fixes.append({"kind": "current", "title": f"Raise the ampere turns to about {fmt(s.turns * s.current * k)}",
                      "summary": f"Force grows with the square of the ampere turns while the steel is unsaturated, so "
                                 f"{k:.2g}x the current (or turns) gives about {fmt(target)} N. Check the coil's heating, "
                                 "which grows with the square of the current.", "auto": False})
    gaps = np.array(s.gaps)
    ok = gaps[f_sweep >= target]
    if len(ok) and ok.max() < gaps.max():
        fixes.append({"kind": "stroke", "title": f"Shorten the stroke to {fmt(ok.max() * 1e3)} mm",
                      "summary": f"The force reaches {fmt(target)} N (the need plus 25%) once the gap is "
                                 f"{fmt(ok.max() * 1e3)} mm or less.", "auto": False})
    if saturated:
        fixes.append({"kind": "steel", "title": "Thicken the steel where it saturates",
                      "summary": "The flux path is past the knee of the BH curve, so a thicker frame or a larger "
                                 "plunger raises the force more than extra current does.", "auto": False})
    elif s.template == "solenoid":
        fixes.append({"kind": "plunger", "title": "Use a larger plunger and stop",
                      "summary": "Force at a given gap scales with the pole face area while the steel carries the "
                                 "flux, so a wider plunger helps until the frame saturates.", "auto": False})
    return fixes


def _assumptions(s: Spec, steel) -> list[str]:
    a = [f"Axisymmetric model, open space modeled by a shell mapped to infinity.",
         f"Steel is {steel.name} with its DC magnetization curve ({steel.source}), no hysteresis.",
         "Static force at a constant current, with no eddy currents and no motion effects."]
    if s.turns:
        a.append(f"{fmt(s.turns)} turns at {fmt(s.current)} A ({fmt(s.turns * s.current)} ampere turns), current "
                 "spread evenly over the winding window.")
    if s.magnet:
        m = get_magnetic(s.magnet)
        a.append(f"Magnet {m.name}, remanence {fmt(m.br)} T with recoil permeability {m.mur:g} ({m.source}), "
                 "magnetized along the axis, at 20 C.")
    if s.template == "solenoid":
        a.append(f"Radial clearance of {fmt(s.dims['c'] * 1e3)} mm where the plunger passes the frame, and the stop "
                 "is part of the frame.")
    return a
