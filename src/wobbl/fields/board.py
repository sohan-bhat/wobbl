"""Board level effects: trace impedance, inductance and capacitance per length, and crosstalk.

A trace's characteristic impedance comes from two electrostatic solves of its
cross section with GetDP, one with the board's dielectric and one in air. The
air solve gives the inductance per length (L = 1 / (c^2 C_air), exact for TEM
lines), the dielectric one the capacitance, and Z0 = sqrt(L / C). Two traces
side by side give the capacitance and inductance matrices, from which the
even and odd mode impedances and the near and far end crosstalk follow.

The inductance and capacitance of a trace of given length are what a circuit
run needs to see ringing: pass them to 'wobbl circuit buck' as
loop_inductance and switch_capacitance, or point it at this result with
board=<folder>.
"""
from __future__ import annotations

import math
import shutil
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from ..materials import get_dielectric
from ..outcome_generic import GenericOutcome
from ..units import fmt
from ..verify.checks import Check, confidence, convergence_check
from . import plots, solve
from .forms import C0, EPS0, MU0, ElecProblem, electrostatic
from .mesh2d import Model2D, Shape, mesh
from .params import ParamError, check_known, text, value

MM = 1e-3
UM = 1e-6
ETA0 = MU0 * C0
LEVELS = (1.4, 1.0, 0.7)
SIGMA_CU = 5.8e7
KEYS = {"width", "height", "thickness", "dielectric", "length", "frequency", "spacing", "gap", "ground_width",
        "inner_d", "outer_d", "target", "tolerance", "rise", "swing", "max_crosstalk", "er"}


@dataclass
class BSpec:
    template: str
    w: float
    h: float
    t: float
    epsr: float
    dielectric: str
    length: float
    freq: float
    s: float = 0.0                       # spacing (coupled) or gap (coplanar)
    wg: float = 0.0                      # coplanar side ground width
    a: float = 0.0                       # coax radii
    b: float = 0.0
    target: float = 0.0
    tol: float = 0.1
    rise: float = 1e-9
    swing: float = 3.3
    max_xt: float = 0.05
    conductors: list[str] = field(default_factory=list)


def spec(template: str, p: dict) -> BSpec:
    check_known(p, KEYS, template)
    diel = text(p, "dielectric", "fr4")
    epsr = value(p, "er", 0) or get_dielectric(diel).epsr
    s = BSpec(template, value(p, "width", 0.3, "length"), value(p, "height", 0.2, "length"),
              value(p, "thickness", 35e-3, "length"), epsr, diel if not p.get("er") else f"er={epsr:g}",
              value(p, "length", 0, "length"), value(p, "frequency", 100e6, "frequency"),
              target=value(p, "target", 50 if template != "coupled" else 0), tol=value(p, "tolerance", 0.1),
              rise=value(p, "rise", 1e-9, "fast_time"), swing=value(p, "swing", 3.3, "voltage"),
              max_xt=value(p, "max_crosstalk", 0.05))
    if s.tol > 1:
        s.tol /= 100.0                   # 10 means 10%
    if s.max_xt > 1:
        s.max_xt /= 100.0
    if template == "coupled":
        s.s = value(p, "spacing", 0.2, "length")
        s.conductors = ["trace1", "trace2"]
    elif template == "coplanar":
        s.s = value(p, "gap", 0.15, "length")
        s.wg = value(p, "ground_width", 5 * s.w * 1e3, "length")
        s.conductors = ["trace"]
    elif template == "coax":
        s.a = value(p, "inner_d", 0.91, "length") / 2
        s.b = value(p, "outer_d", 2.95, "length") / 2
        s.epsr = value(p, "er", 0) or get_dielectric(text(p, "dielectric", "pe")).epsr
        s.dielectric = text(p, "dielectric", "pe") if not p.get("er") else f"er={s.epsr:g}"
        if s.a >= s.b:
            raise ParamError("inner_d must be smaller than outer_d")
        s.conductors = ["inner"]
    else:
        s.conductors = ["trace"]
    if template in ("microstrip", "coupled", "coplanar", "stripline") and (s.w <= 0 or s.h <= 0 or s.t <= 0):
        raise ParamError("width, height and thickness must be positive")
    return s


def geometry(s: BSpec, k: float, epsr: float, drive: str) -> tuple[Model2D, ElecProblem, tuple]:
    """Cross section in meters, y up from the ground plane. drive is the conductor held at 1 V."""
    w, h, t = s.w, s.h, s.t
    edge = min(w, h, s.s or w) / 8          # along the conductors
    corner = max(t / 2, 1 * UM)             # at their corners, where the field is singular
    if s.template == "coax":
        a, b = s.a, s.b
        shapes = [Shape("inner", "disk", (0.0, 0.0, a), hole=True, size=a / 12)]
        model = Model2D("planar", shapes, b, size=(b - a) / 8, far=(b - a) / 5, growth=0.25,
                        lines={"outer": "outer"}, refine=[("inner", a / 12, b - a)]).scaled(k)
        prob = ElecProblem("planar", {"air": epsr}, {"inner": 1.0, "outer": 0.0}, fields=False)
        return model, prob, (-b, b, -b, b)
    span = {"microstrip": w, "coupled": 2 * w + s.s, "coplanar": w + 2 * s.s + 2 * s.wg, "stripline": w}[s.template]
    R = 30 * max(span, h)
    lines = {"ground": "y=0", "outer": "outer"}
    clip = "y>=0"
    shapes = []
    if s.template == "stripline":
        bsep = 2 * h + t                    # ground to ground, trace centered
        clip = f"y>=0,y<={bsep!r}"
        lines = {"ground": "y=0", "top": f"y={bsep!r}", "outer": "outer"}
        y0 = h
        regions_eps = {"air": epsr}
    else:
        shapes.append(Shape("substrate", "rect", (-1.05 * R, 0.0, 1.05 * R, h)))   # cut by the domain
        y0 = h
        regions_eps = {"substrate": epsr}
    if s.template in ("microstrip", "stripline"):
        shapes.append(Shape("trace", "rect", (-w / 2, y0, w / 2, y0 + t), hole=True, size=edge))
    elif s.template == "coupled":
        x1 = s.s / 2 + w / 2
        shapes += [Shape("trace1", "rect", (-x1 - w / 2, y0, -x1 + w / 2, y0 + t), hole=True, size=edge),
                   Shape("trace2", "rect", (x1 - w / 2, y0, x1 + w / 2, y0 + t), hole=True, size=edge)]
    else:
        shapes += [Shape("trace", "rect", (-w / 2, y0, w / 2, y0 + t), hole=True, size=edge),
                   Shape("side", "rect", (w / 2 + s.s, y0, w / 2 + s.s + s.wg, y0 + t), hole=True, size=edge),
                   Shape("side", "rect", (-w / 2 - s.s - s.wg, y0, -w / 2 - s.s, y0 + t), hole=True, size=edge)]
    names = sorted({sh.name for sh in shapes if sh.hole})
    spots = [(x, y, corner) for sh in shapes if sh.hole
             for x in (sh.data[0], sh.data[2]) for y in (sh.data[1], sh.data[3])]
    size = min(h / 4, w / 4, (s.s / 3 if s.s else 1.0))
    model = Model2D("planar", shapes, R, clip=clip, size=size, far=R / 10, growth=0.25, lines=lines,
                    spots=spots).scaled(k)
    pots = {n: (1.0 if n == drive else 0.0) for n in names}
    pots.update({b: 0.0 for b in lines})
    prob = ElecProblem("planar", regions_eps, pots, fields=False)
    view = (-1.6 * span - 2 * h, 1.6 * span + 2 * h, 0.0, (2 * h + t) * 1.05 if s.template == "stripline" else 4 * h)
    return model, prob, view


def hammerstad_jensen(w: float, h: float, t: float, er: float) -> tuple[float, float]:
    """Microstrip Z0 and effective permittivity (Hammerstad and Jensen, IEEE MTT-S 1980), with their thickness
    correction. Better than 0.03% in Z0 and 0.2% in permittivity for zero thickness."""
    def z01(u):
        f = 6 + (2 * math.pi - 6) * math.exp(-(30.666 / u) ** 0.7528)
        return ETA0 / (2 * math.pi) * math.log(f / u + math.sqrt(1 + 4 / u ** 2))

    def eeff(u):
        a = 1 + math.log((u ** 4 + (u / 52) ** 2) / (u ** 4 + 0.432)) / 49 + math.log(1 + (u / 18.1) ** 3) / 18.7
        b = 0.564 * ((er - 0.9) / (er + 3)) ** 0.053
        return (er + 1) / 2 + (er - 1) / 2 * (1 + 10 / u) ** (-a * b)

    u = w / h
    if t > 0:
        th = t / h
        du1 = th / math.pi * math.log(1 + 4 * math.e / (th / math.tanh(math.sqrt(6.517 * u)) ** 2))
        dur = 0.5 * (1 + 1 / math.cosh(math.sqrt(er - 1))) * du1 if er > 1 else du1
        u1, ur = u + du1, u + dur
    else:
        u1 = ur = u
    e = eeff(ur) * (z01(u1) / z01(ur)) ** 2
    return z01(ur) / math.sqrt(eeff(ur)), e


def cohn_stripline(w: float, b: float, er: float) -> float:
    """Exact Z0 of a zero thickness stripline centered between planes b apart (Cohn, conformal mapping)."""
    from scipy.special import ellipk

    k = 1 / math.cosh(math.pi * w / (2 * b))
    kp = math.tanh(math.pi * w / (2 * b))
    return ETA0 / (4 * math.sqrt(er)) * ellipk(k ** 2) / ellipk(kp ** 2)


def wheeler_stripline(w: float, b: float, t: float, er: float) -> float:
    """Z0 of a centered stripline of finite thickness (Wheeler, IEEE MTT 1978, as given in Wadell's Transmission
    Line Design Handbook), within about 0.5%."""
    if t <= 0:
        return cohn_stripline(w, b, er)
    x = t / b
    m = 2 / (1 + 2 / 3 * x / (1 - x))
    dw = x / (math.pi * (1 - x)) * (1 - 0.5 * math.log((x / (2 - x)) ** 2 + (0.0796 * x / (w / b + 1.1 * x)) ** m))
    wp = (w / (b - t) + dw) * (b - t)
    q = 4 / math.pi * (b - t) / wp
    return 30 / math.sqrt(er) * math.log(1 + q * (2 * q + math.sqrt((2 * q) ** 2 + 6.27)))


def gcpw(w: float, s: float, h: float, er: float) -> tuple[float, float]:
    """Grounded coplanar waveguide, zero thickness, infinite side grounds (conformal mapping, as in Wadell's
    Transmission Line Design Handbook). Returns Z0 and effective permittivity."""
    from scipy.special import ellipk

    def ratio(k):
        return ellipk(k * k) / ellipk(1 - k * k)

    k = w / (w + 2 * s)
    k3 = math.tanh(math.pi * w / (4 * h)) / math.tanh(math.pi * (w + 2 * s) / (4 * h))
    r1, r3 = 1 / ratio(k), ratio(k3)          # K(k')/K(k) and K(k3)/K(k3')
    e = (1 + er * r1 * r3) / (1 + r1 * r3)
    z = 60 * math.pi / math.sqrt(e) / (ratio(k) + ratio(k3))
    return z, e


def coax_z(a: float, b: float, er: float) -> float:
    return ETA0 / (2 * math.pi * math.sqrt(er)) * math.log(b / a)


def _charges(s: BSpec, k: float, epsr: float, work: Path, tag: str, drive: str, fields: bool = False) -> dict:
    model, prob, view = geometry(s, k, epsr, drive)
    prob.fields = fields
    d = work / tag
    d.mkdir(parents=True, exist_ok=True)
    msh = mesh(model, d / "m.msh")
    solve.run(electrostatic(prob, msh.tags), msh.path, d, resolution="E")
    q = {c: float(solve.read_global(d, f"Q_{c}.txt")[0]) for c in prob.potentials if c in msh.tags}
    out = {"Q": q, "W": solve.read_global(d, "W.txt")[0], "nodes": msh.n_nodes}
    if fields:
        out.update({"msh": msh, "v": solve.read_nodes(d, "v.txt", msh.node_ids),
                    "en": solve.element_map(d, "en.txt", msh.X, msh.tris), "view": view})
    return out


def _level(s: BSpec, k: float, work: Path, j: int, fields: bool = False) -> dict:
    """Capacitance matrices with the dielectric and in air for one mesh level."""
    out = {"nodes": 0}
    for name, er in (("C", s.epsr), ("C0", 1.0)):
        drive = s.conductors[0]
        r = _charges(s, k, er, work, f"l{j}_{name}", drive, fields=fields and name == "C")
        if "msh" in r:
            out["fields"] = r
        out["nodes"] = max(out["nodes"], r["nodes"])
        c11 = r["Q"][drive]
        out[name] = c11
        out[name + "_bal"] = abs(sum(r["Q"].values())) / abs(c11)
        out[name + "_W"] = 2 * r["W"]
        if s.template == "coupled":
            out[name + "12"] = r["Q"]["trace2"]
    return out


def run_board(template: str, params: dict, out_dir: str | Path, question: str = "", keep: bool = False,
              log=lambda stage, detail="": None) -> GenericOutcome:
    t_start = time.time()
    if template not in ("microstrip", "stripline", "coplanar", "coupled", "coax"):
        raise ParamError(f"unknown board template '{template}' (use microstrip, stripline, coplanar, coupled or coax)")
    s = spec(template, dict(params))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    work = out / "work"
    shutil.rmtree(work, ignore_errors=True)
    levels = []
    for j, k in enumerate(LEVELS):
        log("solve", f"mesh {j + 1} of {len(LEVELS)}, with the dielectric and in air")
        levels.append(_level(s, k, work, j, fields=(j == len(LEVELS) - 1)))
    z = [1 / (C0 * math.sqrt(r["C"] * r["C0"])) for r in levels]
    fine = levels[-1]
    C, Ca = fine["C"], fine["C0"]
    L = 1 / (C0 ** 2 * Ca)
    Z0 = z[-1]
    eeff = C / Ca
    delay = math.sqrt(L * C)
    checks = [convergence_check(z, 0.005, "impedance", "ohm", stopped_by="three meshes")]
    bal = max(fine["C_bal"], fine["C0_bal"])
    checks.append(Check("charge balance", "pass" if bal < 0.005 else ("warn" if bal < 0.02 else "fail"),
                        f"the charge on the driven conductor is matched by the grounds to {100 * bal:.2g}%"))
    ew = max(abs(fine["C_W"] - C) / C, abs(fine["C0_W"] - Ca) / Ca)
    checks.append(Check("energy balance", "pass" if ew < 0.005 else ("warn" if ew < 0.02 else "fail"),
                        f"capacitance from charge and from stored energy agree to {100 * ew:.2g}%"))
    ref, ref_name, ref_e = None, "", None
    if template == "microstrip":
        ref, ref_e = hammerstad_jensen(s.w, s.h, s.t, s.epsr)
        ref_name = "Hammerstad and Jensen's formula with its thickness correction"
    elif template == "stripline":
        ref = wheeler_stripline(s.w, 2 * s.h + s.t, s.t, s.epsr)
        ref_name = "Wheeler's formula for a strip of this thickness"
    elif template == "coplanar":
        ref, ref_e = gcpw(s.w, s.s, s.h, s.epsr)
        ref_name = "the conformal mapping formula for zero thickness and infinite side grounds"
    elif template == "coax":
        ref = coax_z(s.a, s.b, s.epsr)
        ref_name = "the exact coax formula 60 ln(b/a) / sqrt(er)"
    if ref:
        err = abs(Z0 - ref) / ref
        tol_pass = {"coax": 0.005, "microstrip": 0.02, "coplanar": 0.04, "stripline": 0.04}[template]
        thick = template in ("stripline", "coplanar") and s.t > 0.05 * s.w
        st = "pass" if err < tol_pass else ("info" if thick else ("warn" if err < 3 * tol_pass else "fail"))
        extra = f" and an effective permittivity of {ref_e:.4g} (field solve {eeff:.4g})" if ref_e else ""
        checks.append(Check("closed form", st, f"{ref_name} gives {fmt(ref)} ohm{extra}, the field solve "
                                               f"{fmt(Z0)} ohm ({100 * err:.2g}% apart)", {"closed_form_ohm": ref}))
    values = {"z0_ohm": Z0, "eps_eff": eeff, "L_per_m_H": L, "C_per_m_F": C, "delay_per_m_s": delay,
              "levels": [{"nodes": r["nodes"], "z0_ohm": zz} for r, zz in zip(levels, z)]}
    metrics = [("impedance", f"{fmt(Z0)} ohm"), ("effective er", f"{eeff:.3g}"),
               ("inductance", f"{fmt(L * 1e6)} nH/mm"), ("capacitance", f"{fmt(C * 1e9)} pF/mm")]
    rdc = 1 / (SIGMA_CU * s.w * s.t) if template != "coax" else 1 / (SIGMA_CU * math.pi * s.a ** 2)
    skin = math.sqrt(2 / (2 * math.pi * s.freq * MU0 * SIGMA_CU))
    values.update({"r_dc_per_m_ohm": rdc, "skin_depth_m": skin})
    fixes = []
    if s.length:
        values.update({"length_m": s.length, "inductance_H": L * s.length, "capacitance_F": C * s.length,
                       "delay_s": delay * s.length, "resistance_dc_ohm": rdc * s.length,
                       "loop_inductance_H": L * s.length, "switch_capacitance_F": C * s.length})
    head_tail = ""
    if template == "coupled":
        C12, Ca12 = -fine["C12"], -fine["C012"]
        Lm = MU0 * EPS0 * (Ca12 / (Ca ** 2 - Ca12 ** 2))        # from inverting the 2x2 air capacitance matrix
        Ls = MU0 * EPS0 * (Ca / (Ca ** 2 - Ca12 ** 2))
        z_even = math.sqrt((Ls + Lm) / (C - C12))
        z_odd = math.sqrt((Ls - Lm) / (C + C12))
        kb = 0.25 * (Lm / Ls + C12 / C)
        td = s.length * math.sqrt(Ls * C) if s.length else 0.0
        next_ = kb * (min(1.0, 2 * td / s.rise) if s.length else 1.0)
        kf = 0.5 * (C12 / C - Lm / Ls)
        fext = abs(kf) * td / s.rise if s.length else 0.0
        worst = max(next_, fext)
        values.update({"z_even_ohm": z_even, "z_odd_ohm": z_odd, "z_diff_ohm": 2 * z_odd, "next": next_,
                       "fext": fext, "kb": kb, "C12_per_m_F": C12, "Lm_per_m_H": Lm})
        metrics = [("differential", f"{fmt(2 * z_odd)} ohm"), ("near end", f"{100 * next_:.2g}%"),
                   ("far end", f"{100 * fext:.2g}%"), ("odd, even", f"{fmt(z_odd)}, {fmt(z_even)} ohm")]
        checks.append(Check("coupling", "info",
                            f"mutual capacitance {fmt(C12 * 1e9)} pF/mm against {fmt(C * 1e9)} pF/mm to ground, mutual "
                            f"inductance {fmt(Lm * 1e6)} nH/mm against {fmt(Ls * 1e6)} nH/mm"))
        verdict = "ok" if worst <= s.max_xt else ("marginal" if worst <= 1.5 * s.max_xt else "fails")
        head = (f"Crosstalk reaches {100 * next_:.2g}% at the near end and {100 * fext:.2g}% at the far end "
                f"({fmt(worst * s.swing * 1e3)} mV on a {fmt(s.swing)} V swing), "
                f"{'within' if verdict == 'ok' else 'over'} the {100 * s.max_xt:.2g}% limit. The pair has "
                f"{fmt(2 * z_odd)} ohm differential and {fmt(z_even)} ohm even mode impedance.")
        if verdict != "ok":
            fixes.append({"kind": "spacing", "title": f"Space the traces at least {fmt(3 * s.h * 1e3)} mm apart",
                          "summary": "Coupling falls roughly with the square of spacing over height, and three times "
                                     "the height to the plane is the usual rule.", "auto": False})
            fixes.append({"kind": "height", "title": "Route closer to the ground plane",
                          "summary": "A thinner dielectric under the traces ties their fields to the plane.",
                          "auto": False})
        if not s.length:
            head_tail = " Give length= and rise= to size the far end crosstalk."
    else:
        if s.target:
            dz = abs(Z0 - s.target) / s.target
            verdict = "ok" if dz <= s.tol else ("marginal" if dz <= 2 * s.tol else "fails")
            head = (f"The trace is {fmt(Z0)} ohm, {100 * dz:.2g}% from the {fmt(s.target)} ohm target "
                    f"({'within' if verdict == 'ok' else 'outside'} the {100 * s.tol:.2g}% tolerance), with an "
                    f"effective permittivity of {eeff:.3g} and {fmt(delay * 1e9)} ps/mm delay.")
            if verdict != "ok" and template != "coax":
                wn = _width_for(s, s.target)
                if wn:
                    fixes.append({"kind": "width", "title": f"Make the trace about {fmt(wn * 1e3)} mm wide",
                                  "summary": f"Hammerstad and Jensen put a {fmt(wn * 1e3)} mm trace at "
                                             f"{fmt(s.target)} ohm on this stackup, so check it with another run.",
                                  "auto": False})
        else:
            verdict = "ok"
            head = f"The trace is {fmt(Z0)} ohm with an effective permittivity of {eeff:.3g}."
        if s.length:
            head_tail = (f" A {fmt(s.length * 1e3)} mm run has {fmt(L * s.length * 1e9)} nH, {fmt(C * s.length * 1e12)} "
                         f"pF and {fmt(delay * s.length * 1e12)} ps of delay.")
    head += head_tail
    lvl, reasons = confidence(checks)
    f = fine["fields"]
    msh = f["msh"]
    images = {"field": plots.field_map(out / "images" / "field.png", msh.X, msh.tris, msh.region, f["en"] / 1e3,
                                       "|E| at 1 V, V/mm", "field of the driven conductor at 1 V", lines=f["v"],
                                       skip_outline={msh.tags[n] for n in ("air", "substrate") if n in msh.tags},
                                       view=f["view"])}
    assumptions = [f"Quasi static TEM cross section of a long, uniform line. {_diel_text(s)}",
                   f"Copper {fmt(s.t * 1e6)} um thick, perfect conductor for the fields, and the ground plane is infinite.",
                   "Loss, dispersion and surface roughness are not in the impedance, and they matter above a few GHz."]
    if template == "microstrip":
        assumptions.append("No solder mask over the trace, which would lower Z0 by a few percent.")
    o = GenericOutcome("board", template, question, verdict, head, metrics, lvl, reasons, checks, assumptions,
                       {**values, "params": {k: str(v) for k, v in params.items()},
                        "_figures": [["field", "Electric field around the trace"]]}, images, fixes,
                       solver=f"GetDP {solve.getdp().version()} with gmsh", t_total=time.time() - t_start)
    from .playback import static_field
    parts = geometry(s, LEVELS[-1], s.epsr, s.conductors[0])[0].shapes
    static_field(out, f"board-{template}", msh, f["view"], f["en"], f["v"], parts,
                 {k: v for k, v in values.items() if isinstance(v, (int, float))}, mirror=False, unit_e="V/m at 1 V")
    if not keep:
        shutil.rmtree(work, ignore_errors=True)
    return o.write(out)


def _diel_text(s: BSpec) -> str:
    if s.dielectric.startswith("er="):
        return f"Dielectric with relative permittivity {s.epsr:g}."
    d = get_dielectric(s.dielectric)
    return f"{d.name}, relative permittivity {s.epsr:g} ({d.source})."


def _width_for(s: BSpec, target: float) -> float | None:
    """Trace width for a target impedance on this stackup, from the closed form."""
    f = {"microstrip": lambda w: hammerstad_jensen(w, s.h, s.t, s.epsr)[0],
         "stripline": lambda w: cohn_stripline(w, 2 * s.h + s.t, s.epsr),
         "coplanar": lambda w: gcpw(w, s.s, s.h, s.epsr)[0]}.get(s.template)
    if not f:
        return None
    lo, hi = s.h * 0.01, s.h * 30
    if not (f(hi) < target < f(lo)):
        return None
    for _ in range(80):
        mid = math.sqrt(lo * hi)
        lo, hi = (mid, hi) if f(mid) > target else (lo, mid)
    return math.sqrt(lo * hi)
