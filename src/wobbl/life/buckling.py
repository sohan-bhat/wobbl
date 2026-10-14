"""Buckling: the load factor at which a part snaps sideways, and how much of it real imperfections take away.

CalculiX's *BUCKLE step solves the part under its loads, then finds the factors lambda at which the stiffness the
loads take away cancels the elastic stiffness. The study refines the mesh until the first factor settles, then
matches the part to a shape class and applies that class's imperfection knockdown and the yield interaction
(knockdown.py), checks the answer against Euler where the part is a textbook column, and checks that the loads
balance the supports, how far apart the first modes are and whether the part yields before it buckles.
"""
from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from ..fe import StaticResult
from ..mesh.mesher import build_mesh
from ..solvers import base as solvers
from ..structural.build import build_static
from ..structural.outcome import process_caveat
from ..structural.run import failure_measure, support_reactions
from ..structural.spec import StructuralStudy
from ..units import fmt
from ..verify.checks import Check, balance_check, confidence, convergence_check
from . import ccx
from .common import QUALITY, change, excluded_nodes, governing, level_sizes, next_sizing, prepare
from .knockdown import (Column, classify, column_curve, column_info, euler_check, johnson, knockdown,
                        separation_check)
from .outcome import LifeOutcome, clean_work, save_study, verdict_for

SHAPES = ("auto", "column", "plate", "cylinder", "dome", "general")


@dataclass
class BucklingStudy:
    base: StructuralStudy            # part, material, the reference loads and the supports
    target_sf: float | None = None
    shape: str = "auto"
    modes: int = 4
    keep: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


def run_buckling(study: BucklingStudy, out_dir: str | Path, progress=None, pre=None) -> LifeOutcome:
    t0 = time.time()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    work = out / "work"
    base = study.base
    say = progress or (lambda a, b: None)
    if study.shape not in SHAPES:
        raise ValueError(f"unknown shape '{study.shape}' (use {', '.join(SHAPES)})")
    adapter = solvers.for_domain("structural")
    exe = adapter.require()
    p = prepare(base, work, progress, pre)
    if not p.loads:
        raise ValueError("a buckling study needs the loads that compress the part, for example "
                         "--load '1000 N -x @ x=max'")
    mat = p.material
    q = QUALITY.get(base.quality, QUALITY["normal"])
    sizes = level_sizes(p, base.quality, base.mesh_size)
    n_modes = max(2, int(study.modes))
    levels, last, stopped_by = [], None, "level limit"
    for k, planned in enumerate(sizes):
        sz = planned if last is None else next_sizing(planned, last[5], levels[-1]["n_nodes"], q["nodes"])
        if sz is None:
            stopped_by = "node limit"
            break
        say("mesh", f"level {k}: element size {sz.h:.3g} mm")
        mesh = build_mesh(p.prep, sz)
        built = build_static(mesh, p.loads, p.fixtures, mat, p.features.faces)
        wl = work / f"level{k}"
        deck = ccx.make_deck(built.model, wl / "static.inp")
        say("solve", f"level {k}: buckling of {mesh.n_nodes:,} nodes with {adapter.name}")
        ccx.run(exe, wl, "buckle", ccx.buckle_inp(deck, n_modes))
        factors = ccx.read_buckling_factors(wl / "buckle.dat")
        blocks = ccx.read_blocks(wl / "buckle.frd", mesh.n_nodes + 2 * len(deck.extra), ("DISP", "STRESS", "FORC"))
        (wl / "buckle.frd").unlink(missing_ok=True)
        pos = [f for f in factors if f > 0]
        if not pos:
            raise ValueError("these loads do not buckle the part in the modes found, because every buckling factor is "
                             f"negative ({', '.join(fmt(f) for f in factors)}), so the part would buckle only "
                             "with the loads reversed. Check the load directions")
        levels.append({"level": k, "n_nodes": mesh.n_nodes, "h": sz.h, "factor": pos[0], "factors": factors,
                       "t_solve": 0.0})
        last = (mesh, built, deck, factors, blocks, sz)
        say("check", f"level {k}: first buckling factor {pos[0]:.4g}"
                     + (f" ({100 * change([L['factor'] for L in levels]):.1f}% change)" if k else ""))
        if k >= 1 and change([L["factor"] for L in levels]) <= q["tol"]:
            stopped_by = "converged"
            break
    mesh, built, deck, factors, blocks, sz = last
    n = mesh.n_nodes
    disp = blocks["DISP"]
    U0 = disp[0][1][:n, :3]
    S0 = blocks["STRESS"][0][1][:n, :6]
    RF0 = blocks["FORC"][0][1][:n, :3] if "FORC" in blocks else np.zeros((n, 3))
    shapes = [a[:n, :3] for _, a in disp[1:]]
    order = [i for i, f in enumerate(factors) if f > 0]
    lam = factors[order[0]]
    mode1 = shapes[order[0]] if order[0] < len(shapes) else shapes[0]
    res0 = StaticResult(U0, S0, RF0)
    measure, mname = failure_measure(res0, mat)
    excluded, r_ex = excluded_nodes(mesh, built, p.features, sz, measure)
    gi = governing(measure, excluded)
    peak = float(measure[gi])
    F, M = built.model.applied_total()
    load = float(np.linalg.norm(F))

    # the shape class, its knockdown and the yield interaction
    col = column_info(mesh, built, p.prep)
    shape, why_shape = classify(study.shape, col, p.features, built, p.loads)
    target = study.target_sf or mat.recommended_sf
    yield_factor = mat.allowable / peak if peak > 0 else math.inf
    if shape == "column" and col is not None:
        design, gamma, kd_why, curve = column_curve(lam, col, mat)
        interaction = curve
    else:
        gamma, kd_why = knockdown(shape, p.features, built, p.loads)
        design, interaction = johnson(gamma * lam, yield_factor, mat.ductile)
    verdict = verdict_for(design, target)

    checks = [convergence_check([L["factor"] for L in levels], q["tol"], "first buckling factor", "times the load",
                                stopped_by,
                                min_levels=2)]
    rf, rm = support_reactions(built, res0, mesh)
    checks.append(balance_check(F, rf, M, rm, p.prep.diag))
    checks.append(euler_check(lam, col, mat))
    checks.append(separation_check(factors))
    yfirst = yield_factor < gamma * lam
    checks.append(Check("yield before buckling", "info" if yfirst else "pass",
                        f"the most stressed point ({fmt(peak)} MPa {mname} at the reference load, away from the "
                        f"supports) reaches {fmt(mat.allowable)} MPa at {fmt(yield_factor)} times the load, "
                        + ("before it buckles, so material strength governs and the design factor includes it"
                           if yfirst else "after it buckles, so buckling governs"),
                        {"yield_factor": yield_factor, "peak_MPa": peak}))
    checks.append(_sanity(lam, U0, p.prep.diag))
    extra = []
    if shape == "general":
        extra.append(("medium", "no published knockdown fits this shape, so the 0.5 imperfection knockdown is a "
                                "judgment, so pass --shape if the part is a column, plate, cylinder or dome"))
    level, reasons = confidence(checks, extra)

    crit = lam * load
    head = (f"Buckles at {fmt(lam)} times the applied load ({fmt(crit)} N) in a {shape} mode. With a knockdown "
            f"of {gamma:.2f} for imperfections" + (" and yielding" if interaction else "") +
            f" the design factor is {fmt(design)} against a target of {target:g}.")
    if level != "high":
        head += f" Confidence is {level}."
    say("render", "drawing the buckling modes")
    images = _images(out / "images", mesh, shapes, factors, levels)
    o = LifeOutcome(
        "buckling", str(base.part), base.question, verdict, head,
        [("buckling factor", fmt(lam)), ("knockdown", f"{gamma:.2f}"), ("design factor", fmt(design)),
         ("target", f"{target:g}")],
        level, reasons, checks,
        _assumptions(mat, shape, why_shape, kd_why, interaction, base),
        {"analysis": "buckling", "load_factor": lam, "load_factors": factors, "critical_load_N": crit,
         "applied_load_N": load, "shape": shape, "shape_reason": why_shape, "knockdown": gamma,
         "knockdown_reason": kd_why, "yield_factor": yield_factor, "design_factor": design, "target_sf": target,
         "interaction": interaction, "levels": levels, "exclusion_radius_mm": r_ex,
         "column": None if col is None else {"length_mm": col.length, "area_mm2": col.area, "i_min_mm4": col.i_min,
                                             "slenderness": col.slenderness, "k": col.k, "ends": col.ends},
         "_figures": [["mode1", "First buckling mode, deflection exaggerated"],
                      ["modes", "The next buckling modes"], ["convergence", "First buckling factor on each mesh"]],
         "_tables": [["Buckling factors", ["mode", "factor", "load N"],
                      [[i + 1, fmt(f), fmt(f * load)] for i, f in enumerate(factors)],
                      "Negative factors buckle the part only with every load reversed."]]},
        images, fixes(shape, design, target, col, mat, p.features, lam), deck.notes + p.prep.notes,
        solver=f"CalculiX {adapter.version()}", t_total=time.time() - t0)
    if _playback(out / "playback.json", mesh, mode1, lam, design, yield_factor, col, p):
        o.values["playback"] = "playback.json"
    o.write(out)
    save_study(study, out)
    clean_work(work, study.keep)
    return o


def _sanity(lam: float, U0: np.ndarray, diag: float) -> Check:
    d = float(np.linalg.norm(U0, axis=1).max())
    s = f"buckling factor {fmt(lam)}, deflection {fmt(d)} mm under the reference load ({100 * d / diag:.2g}% of size)"
    if not 1e-4 < lam < 1e6:
        return Check("units and sanity", "warn", s + ". A factor this far from one usually means a unit slip in the "
                                                     "load", {"factor": lam})
    if d > 0.1 * diag:
        return Check("units and sanity", "warn", s + ". The part bends a lot before buckling, so a linear buckling "
                                                     "answer is rough", {"factor": lam})
    return Check("units and sanity", "pass", s + ": plausible", {"factor": lam})


def _assumptions(mat, shape, why_shape, kd_why, interaction, base) -> list[str]:
    a = ["Linear eigenvalue buckling, where the stiffness the loads take away is found from the stress they cause in the "
         "perfect, undeformed part, and the loads keep their direction as it deflects. It overpredicts the collapse "
         "load of real parts, most of all for thin shells, which is what the knockdown is for.",
         f"The shape class is {shape}, {why_shape}, and the knockdown comes from {kd_why}.",
         f"{mat.name}: E {fmt(mat.E)} MPa, Poisson {mat.nu:g}, {mat.criterion} {fmt(mat.allowable)} MPa ({mat.source})."]
    if interaction:
        a.append(f"The design factor also counts {interaction}.")
    a.append("Fixed faces are perfectly rigid. A support that gives a little, or turns, lowers the buckling load, and "
             "for a column that is clamped on paper it can lower it by half or more.")
    made = process_caveat(mat, base.print_direction)
    if made:
        a.append(made)
    return a + list(base.assumptions)


def fixes(shape: str, design: float, target: float, col: Column | None, mat, features, lam: float) -> list[dict]:
    if design >= target or design <= 0:
        return []
    need = target / design
    out = []
    if shape == "column" and col is not None:
        k = _column_stiffening(lam, col, mat, target)
        if k is None:
            out.append({"kind": "section", "title": "Use a larger section",
                        "summary": f"Even a column too stiff to buckle would squash at {fmt(mat.allowable * col.area)} "
                                   f"N, short of {target:g} times the load, so it needs more area as well as more "
                                   "stiffness.", "auto": False})
        else:
            out.append({"kind": "stiffen", "title": f"Raise the column's second moment of area {k:.2g} times",
                        "summary": f"The elastic buckling load scales with I, now {fmt(col.i_min)} mm4 about the "
                                   f"weak axis. On the column curve the design factor reaches {target:g} at {k:.2g} "
                                   f"times that, a section {k ** 0.25:.2f} times as wide in every direction, or a "
                                   "larger tube of the same weight, which puts the material farther from the axis.",
                        "params": {"i_min": col.i_min * k}, "auto": False})
            out.append({"kind": "brace", "title": f"Shorten the unbraced length to {fmt(col.length / math.sqrt(k))} mm",
                        "summary": f"The elastic buckling load scales with 1/L^2, so {fmt(col.length / math.sqrt(k))} "
                                   f"mm of the {fmt(col.length)} mm column reaches the target. A brace that stops "
                                   "sideways motion at mid length halves the buckling length and does far more.",
                        "params": {"length": col.length / math.sqrt(k)}, "auto": False})
        need = k or need
    else:
        p = {"plate": 1 / 3, "cylinder": 1 / 2, "dome": 1 / 2}.get(shape, 1 / 3)
        t = features.thickness_typical
        law = {"plate": "t^3", "cylinder": "t^2", "dome": "t^2"}.get(shape, "between t^2 and t^3")
        out.append({"kind": "thicken", "title": f"Thicken the walls that move in the first mode to about "
                                                 f"{fmt(t * need ** p)} mm",
                    "summary": f"For a {shape} the buckling load under a fixed load grows about as {law}, so the "
                               f"{fmt(t)} mm walls need about {need ** p:.2f} times the thickness for a design "
                               f"factor of {target:g}. A rib or a flange across the buckle does the same with less "
                               "material.", "params": {"thickness": t * need ** p}, "auto": False})
    if mat.category != "metal" or mat.E < 150000:
        out.append({"kind": "material", "title": "Use a stiffer material",
                    "summary": f"Elastic buckling scales with Young's modulus, {fmt(mat.E)} MPa now, so a material "
                               f"{need:.2g} times as stiff reaches the target if it is also strong enough not to "
                               "yield first.", "params": {"E": mat.E * need}, "auto": False})
    return out


def _column_stiffening(lam: float, col, mat, target: float) -> float | None:
    """The factor on the elastic buckling load (on I, E or 1/L^2) that brings the column curve's design factor to
    the target, or None when even an infinitely stiff column would squash first."""
    if column_curve(1e9 * lam, col, mat)[0] < target:
        return None
    lo, hi = 1.0, 1e9
    for _ in range(200):
        mid = math.sqrt(lo * hi)
        if column_curve(lam * mid, col, mat)[0] >= target:
            hi = mid
        else:
            lo = mid
        if hi / lo < 1.0001:
            break
    return hi


def _playback(path: Path, mesh, mode1, lam, design, yield_factor, col, p) -> str | None:
    """The first buckling shape growing with load, from a column's straightness tolerance of L/1000 or, for other
    shapes, a dent half a wall deep."""
    from .playback import buckling_doc, write

    try:
        if col is not None:
            e0, why, length = col.length / 1000, "a straightness tolerance of L/1000", col.length
        else:
            e0, why, length = 0.5 * p.features.thickness_typical, "half the typical wall", p.prep.diag
        return write(path, buckling_doc(mesh, mode1, lam, design, yield_factor, length, e0, why))
    except Exception:  # noqa: BLE001  the playback file is a convenience, never a reason to fail
        return None


def _images(img: Path, mesh, shapes, factors, levels) -> dict:
    from .render import convergence, render_modes

    out = {}
    try:
        pos = [i for i, f in enumerate(factors) if f > 0 and i < len(shapes)]
        out["mode1"] = render_modes(img / "mode1.png", mesh.X, mesh.tets, [shapes[pos[0]]],
                                    [f"mode 1, factor {fmt(factors[pos[0]])}"])
        rest = pos[1:3]
        if rest:
            out["modes"] = render_modes(img / "modes.png", mesh.X, mesh.tets, [shapes[i] for i in rest],
                                        [f"mode {i + 1}, factor {fmt(factors[i])}" for i in rest])
        if len(levels) > 1:
            out["convergence"] = convergence(img / "convergence.png", [L["n_nodes"] for L in levels],
                                             {"first buckling factor": [L["factor"] for L in levels]}, "factor")
    except Exception as e:  # noqa: BLE001  pictures are a convenience, never a reason to fail
        out["error"] = str(e)[:200]
    return out
