"""The flow study on a CAD part from start to finish: plan the case, find a fan or pump's operating point, solve
on two or three grids, check the answer, and write the outcome, its pictures and the playback file."""
from __future__ import annotations

import re
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from ..units import fmt
from ..verify.checks import Check, confidence
from . import correlations as C
from .fluids import fmt_flow, fmt_pressure, get_fluid
from .partflow import QUALITY, WORKERS, PartFlowStudy, Plan, build_setup, plan_case, run_level
from .shape import PartGeometry, load
from .verify import gci, gci_check, mass_check, residual_check, yplus_check


def run_partflow(st: PartFlowStudy, out_dir: str | Path, progress=None, geometry: PartGeometry | None = None,
                 heat: bool = False):
    """Mesh, solve and check the flow around or through a part, and write its outcome, report and playback. heat
    carries the walls' temperature into the fluid as a passive scalar, for the heat transfer coefficients."""
    from ..solvers.base import get
    from .install import ensure

    t0 = time.time()
    say = progress or (lambda a, b: None)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    work = out / "work"
    if st.quality not in QUALITY:
        raise ValueError(f"quality is fast, normal or fine, not '{st.quality}'")
    q = QUALITY[st.quality]
    fl = get_fluid(st.fluid, st.temp)
    say("geometry", f"reading {Path(st.part).name}")
    g = geometry or load(st.part, work / "geometry", st.params)
    p = plan_case(st, g)
    ensure(lambda m: say("setup", m))
    op = operating_point(st, g, p, q, work, say) if st.curve else None
    s = build_setup(st, g, p, q, op["flow"] if op else None)
    s.heat = heat
    scales = [q["ratio"] ** k for k in range(q["levels"])]
    say("cfd", f"{p.mode} flow, {s.turb.model}, on {len(scales)} grids, {WORKERS} at a time")
    with ThreadPoolExecutor(WORKERS) as pool:
        levels = list(pool.map(lambda k: run_level(s, work, scales[k], f"grid{k}", fields=k == 0),
                               range(len(scales))))
    checks, extra = _checks(st, g, p, s, levels, fl, op)
    level, reasons = confidence(checks, extra)
    o = _outcome(st, g, p, s, levels, fl, op, checks, level, reasons, out)
    o.solver = f"OpenFOAM {get('openfoam').version()}, snappyHexMesh and simpleFoam"
    o.t_total = time.time() - t0
    o.geometry, o.plan, o.setup, o.fine = g, p, s, levels[0]
    try:
        _pictures(o, g, p, s, levels[0], out)
    except Exception as e:  # noqa: BLE001  a picture must never cost the answer
        o.images["error"] = str(e)
    try:
        _playback(o, g, p, s, levels[0], out, fl)
    except Exception as e:  # noqa: BLE001
        o.notes.append(f"No playback file: {e}")
    if not st.keep:
        shutil.rmtree(work, ignore_errors=True)
    return o.write(out)


def operating_point(st: PartFlowStudy, g: PartGeometry, p: Plan, q: dict, work: Path, say) -> dict:
    """Where the fan or pump curve meets the part's own resistance: the coarse grid at two flows fits
    dp = a Q + b Q^2, which the falling curve crosses once between no flow and its free flow."""
    pts = sorted((float(a), float(b)) for a, b in st.curve)
    qmax = max(a for a, _ in pts)

    def supply(x):
        return float(np.interp(x, [a for a, _ in pts], [b for _, b in pts]))

    trial = (0.35 * qmax, 0.75 * qmax)
    say("cfd", f"the part's resistance at {fmt_flow(trial[0])} and {fmt_flow(trial[1])} on the coarse grid")
    coarse = q["ratio"] ** (q["levels"] - 1)

    def one(k):
        return run_level(build_setup(st, g, p, q, trial[k]), work, coarse, f"curve{k}")["dp_total"]
    with ThreadPoolExecutor(WORKERS) as pool:
        dps = list(pool.map(one, range(2)))
    a, b = np.linalg.lstsq(np.array([[x, x * x] for x in trial]), np.array(dps), rcond=None)[0]
    lo, hi = 0.0, qmax
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if a * mid + b * mid * mid < supply(mid) else (lo, mid)
    flow = 0.5 * (lo + hi)
    return {"flow": flow, "dp": supply(flow), "a": float(a), "b": float(b),
            "trial": [[float(x), float(y)] for x, y in zip(trial, dps)], "curve": pts}


# --- checks ------------------------------------------------------------------------------------------------------
def _checks(st, g, p, s, levels, fl, op):
    fine = levels[0]
    checks, extra = [], []
    grid = fine["grid"]
    bad = grid.failed_checks or grid.max_non_ortho >= 70 or grid.max_skewness >= 4
    checks.append(Check("mesh quality", "warn" if bad else "pass",
                        f"{fine['cells']:,} cells, worst non orthogonality {grid.max_non_ortho:.0f} degrees and "
                        f"skewness {grid.max_skewness:.2g}" + (f", {grid.failed_checks} checkMesh checks failed"
                                                                if grid.failed_checks else ""),
                        {"cells": fine["cells"], "non_ortho": grid.max_non_ortho, "skewness": grid.max_skewness}))
    label, unit = ("drag", "N") if s.mode == "external" else ("pressure drop", "Pa")
    vals = [L["value"] for L in levels]
    if len(levels) >= 3:
        checks.append(gci_check(gci(vals, [L["cells"] ** (-1 / 3) for L in levels]), label, unit, tol=0.05))
    else:
        ch = abs(vals[0] - vals[1]) / max(abs(vals[0]), 1e-300)
        checks.append(Check("grid convergence", "pass" if ch < 0.03 else ("warn" if ch < 0.1 else "fail"),
                            f"{label} {fmt(vals[1])} -> {fmt(vals[0])} {unit} from {levels[1]['cells']:,} to "
                            f"{fine['cells']:,} cells, {100 * ch:.2g}% apart, on two grids only (fast quality)",
                            {"values": vals, "cells": [L["cells"] for L in levels]}))
    sw = fine["swing"]
    checks.append(Check("iterative convergence", "pass" if sw < 0.01 else ("warn" if sw < 0.05 else "fail"),
                        f"{label} swings {100 * sw:.2g}% over the last fifth of {fine['iterations']} iterations" +
                        ("" if sw < 0.01 else ", so the flow does not settle into one steady state, as behind a "
                                              "bluff body that sheds vortices, and the mean is reported")))
    if s.mode != "external":
        checks.append(mass_check(fine["q_in"], fine["q_out"], "m3/s", tol=1e-3))
    checks.append(residual_check(fine["log"], 1e-3))
    if s.turb.model != "laminar":
        checks.append(yplus_check(fine.get("yplus", {}), "wall functions"))
        from .verify import model_reason
        extra.append(model_reason("rans", "separated" if s.mode == "external" else "internal"))
    checks.append(_hand(g, p, s, fine))
    if s.mode == "external":
        checks.append(_blockage(s, fine))
    if op:
        mine = fine["dp_total"]
        err = abs(mine - op["dp"]) / max(op["dp"], 1e-300)
        checks.append(Check("operating point", "pass" if err < 0.1 else "warn",
                            f"the curve gives {fmt_pressure(op['dp'])} at {fmt_flow(op['flow'])}, and the fine grid "
                            f"needs {fmt_pressure(mine)} there ({100 * err:.2g}% apart)" +
                            ("" if err < 0.1 else ", so the coarse grid's fit set the flow a little off"), op))
    return checks, [e for e in extra if e]


def _blockage(s, fine) -> Check:
    """How much of the box's cross section the part fills, and the drag Maskell's correction would remove."""
    box_area = float((s.hi[1] - s.lo[1]) * (s.hi[2] - s.lo[2]))
    ratio = s.ref_area / box_area
    cd = fine["cd"]
    corrected = cd / (1 + 2.5 * cd * ratio)
    st = "pass" if ratio < 0.03 else ("warn" if ratio < 0.08 else "fail")
    return Check("blockage", st, f"the part fills {100 * ratio:.2g}% of the box's cross section, which Maskell's "
                                 f"correction says raises the drag coefficient from about {corrected:.3g} to "
                                 f"{cd:.3g}" + ("" if st == "pass" else ", so the box is tight for this body"),
                 {"ratio": ratio, "cd_corrected": corrected})


def _hand(g, p, s, fine) -> Check:
    if s.mode == "external":
        cd = fine["cd"]
        ok = 0.02 <= cd <= 2.5
        return Check("hand calculation", "pass" if ok else "warn",
                     f"drag coefficient {cd:.3g} on the frontal area, " + (
                         "inside the 0.02 to 2.5 that bodies from a streamlined shape to a flat plate span" if ok else
                         "outside the 0.02 to 2.5 that real bodies span, so check the speed and size"), {"cd": cd})
    if s.mode == "internal":
        length = p.passage.volume * 1e-9 / s.ref_area
    elif s.mode == "volume":
        length = g.prep.volume * 1e-9 / s.ref_area
    else:
        length = float((g.hi - g.lo).max()) * 1e-3
    re_ = s.speed * s.ref_length / s.nu
    f, src = C.friction_factor(re_, 0.0)
    straight = f * length / s.ref_length * 0.5 * s.rho * s.speed ** 2
    got = fine["dp_total"]
    ratio = got / max(straight, 1e-300)
    return Check("hand calculation", "info" if ratio >= 0.6 else "warn",
                 f"a straight smooth duct of the same {fmt(s.ref_length * 1e3)} mm hydraulic diameter and "
                 f"{fmt(length * 1e3)} mm length loses {fmt_pressure(straight)} by {src}, and the part loses "
                 f"{fmt_pressure(got)}, {ratio:.2f} times that" +
                 (", the rest from turns, junctions and changes of section" if ratio >= 1.1 else "") +
                 ("" if ratio >= 0.6 else ", which is less than a plain duct can lose, so check the openings"),
                 {"straight_Pa": straight, "length_m": length, "ratio": ratio})


# --- the answer --------------------------------------------------------------------------------------------------
def _list(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def _opening_name(patch: str) -> str:
    m = re.match(r"outlet(\d+)$", patch)
    return f"opening {m.group(1)}" if m else patch


def _outcome(st, g, p, s, levels, fl, op, checks, level, reasons, out):
    from ..outcome_generic import GenericOutcome

    fine = levels[0]
    re_ = s.speed * s.ref_length / s.nu
    regime = "laminar" if s.turb.model == "laminar" else "turbulent"
    values = {"mode": p.mode, "re": re_, "model": s.turb.model, "cells": fine["cells"], "fluid": fl.key,
              "levels": [{k: L[k] for k in ("cells", "value", "iterations", "t_mesh", "t_solve")} for L in levels]}
    fixes = []
    if s.mode == "external":
        drag, lift, side = fine["drag"], fine["lift"], fine["side"]
        head = (f"At {fmt(s.speed)} m/s in {fl.name.lower()} the part feels {fmt(drag)} N of drag (drag coefficient "
                f"{fine['cd']:.3g} on {fmt(s.ref_area * 1e4)} cm2 of frontal area), {fmt(lift)} N of lift and "
                f"{fmt(side)} N to the side, {regime} at Re {fmt(re_)}.")
        metrics = [("drag", f"{fmt(drag)} N"), ("lift", f"{fmt(lift)} N"), ("side force", f"{fmt(side)} N"),
                   ("drag coefficient", f"{fine['cd']:.3g}"), ("frontal area", f"{fmt(s.ref_area * 1e4)} cm2")]
        values.update(drag_N=drag, lift_N=lift, side_N=side, cd=fine["cd"], cl=fine["cl"], cs=fine["cs"],
                      frontal_area_m2=s.ref_area, speed_m_s=s.speed, length_m=s.ref_length,
                      direction=p.R[0].tolist(), up=p.R[2].tolist())
        limit, got, unit = st.max_force, drag, "N"
    else:
        dp = fine["dp_total"]
        flows = fine["flows"]
        tot = sum(flows.values()) or 1.0
        split = {_opening_name(k): v / tot for k, v in flows.items()}
        where = "past the part in its duct" if s.mode == "ducted" else "through the part"
        head = f"{fmt_flow(s.flow)} of {fl.name.lower()} loses {fmt_pressure(dp)} {where} ({regime}, Re {fmt(re_)})"
        if len(split) > 1:
            head += ", with " + _list([f"{100 * v:.0f}% leaving by {k}" for k, v in split.items()])
        head += "."
        if op:
            head = (f"The {'pump' if fl.rho > 100 else 'fan'} curve meets the part at {fmt_flow(op['flow'])} and "
                    f"{fmt_pressure(op['dp'])}. " + head)
        metrics = [("pressure drop", fmt_pressure(dp)), ("flow", fmt_flow(s.flow)),
                   ("inlet velocity", f"{fmt(s.speed)} m/s"), ("Reynolds number", fmt(re_)),
                   ("pumping power", f"{fmt(s.flow * dp)} W")]
        values.update(dp_Pa=dp, dp_static_Pa=fine["dp_static"], flow_m3_s=s.flow, velocity_m_s=s.speed,
                      split=split, pump_power_W=s.flow * dp, hydraulic_diameter_m=s.ref_length)
        if op:
            values["operating_point"] = {"flow_m3_s": op["flow"], "dp_Pa": op["dp"]}
        limit, got, unit = st.max_dp, dp, "Pa"
    verdict = "ok"
    if limit:
        verdict = "ok" if got <= 0.9 * limit else ("marginal" if got <= limit else "fails")
        head = head[:-1] + f", against a limit of {fmt(limit)} {unit}."
        if verdict != "ok":
            fixes.append({"kind": "flow", "title": "Lower the speed or open up the narrowest passage",
                          "summary": "Drag and pressure drop grow with about the square of the speed in turbulent "
                                     "flow, so 10% less flow gives about 19% less.", "auto": False})
    assumptions = [fl.summary() + "."] + p.why + [
        "Steady, incompressible flow" + (" with the k omega SST turbulence model and blended wall functions" if
                                         regime == "turbulent" else ", laminar") + ".",
        f"Meshed by snappyHexMesh with {fine['cells']:,} cells on the finest of {len(levels)} grids, without "
        "boundary layer cells."]
    if s.mode == "external":
        assumptions.append("The part is held still in a uniform stream with low turbulence, far from the ground, in "
                           "a box whose sides slip.")
    o = GenericOutcome("flow", str(Path(st.part).resolve()), st.question, verdict, head, metrics, level, reasons,
                       checks, assumptions, values, {}, fixes, notes=list(g.notes))
    o.workdir = str(out)
    o.criteria = [{"name": "drag" if s.mode == "external" else "pressure drop",
                   "value": "drag_N" if s.mode == "external" else "dp_Pa", "limit": limit, "sense": "max"}] \
        if limit else []
    return o


def surface_values(o, name: str = "p"):
    """A wall field at each surface vertex of the part, from the nearest wall face of the finest grid."""
    from scipy.spatial import cKDTree

    g, p, s, f = o.geometry, o.plan, o.setup, o.fine["fields"]
    vals = f.get(f"wall_{name}")
    if "wall_centres" not in f or vals is None or not len(vals):
        return None
    pts = f["wall_centres"] * 1e3
    if s.mode in ("external", "ducted"):
        pts = pts @ p.R + (g.lo + g.hi) / 2
    _, i = cKDTree(pts).query(g.X)
    return np.asarray(vals)[i]


def _slice(g, p, s, fine):
    """Cells cut by a plane through the middle of the flow, as 2D points in mm with their in plane velocity. The
    plane holds the stream and the up direction for external flow (a side view), the stream and the side direction
    for ducted flow (a view down between fins), and the passage's two longest extents for internal flow. A cell is
    cut when the plane passes within about half its own size of its centre, so every refinement level shows.
    Returns x, y, u, v, speed and the plane (origin, u axis, v axis) in the part's coordinates."""
    from scipy.spatial import cKDTree

    f = fine["fields"]
    pts = f["C"] * 1e3
    U = f["U"]
    if s.mode == "external":
        u_ax, v_ax, n_ax = np.eye(3)[0], np.eye(3)[2], np.eye(3)[1]
        mid = np.zeros(3)
    elif s.mode == "ducted":
        u_ax, v_ax, n_ax = np.eye(3)[0], np.eye(3)[1], np.eye(3)[2]
        mid = np.zeros(3)
    else:
        order = np.argsort(pts.max(0) - pts.min(0))
        n_ax, v_ax, u_ax = (np.eye(3)[k] for k in order)
        mid = 0.5 * (pts.max(0) + pts.min(0))
    near, _ = cKDTree(pts).query(pts, k=2)
    size = near[:, 1]
    d = (pts - mid) @ n_ax
    sel = np.abs(d) <= 0.55 * size
    x, y = pts[sel] @ u_ax, pts[sel] @ v_ax
    uu, vv = U[sel] @ u_ax, U[sel] @ v_ax
    if s.mode in ("external", "ducted"):     # back to the part's coordinates
        R, c = p.R, (g.lo + g.hi) / 2
        plane = {"origin": c.tolist(), "u": (R.T @ u_ax).tolist(), "v": (R.T @ v_ax).tolist()}
    else:
        plane = {"origin": (mid @ n_ax * n_ax).tolist(), "u": u_ax.tolist(), "v": v_ax.tolist()}
    return x, y, uu, vv, np.linalg.norm(U[sel], axis=1), plane, size[sel]


def _pictures(o, g, p, s, fine, out):
    from .render import field2d, surface_scalar

    pw = surface_values(o)
    if pw is not None:
        o.images["pressure"] = surface_scalar(out / "images" / "pressure.png", g.X, g.tris, pw * s.rho,
                                              "wall pressure, Pa", arrow=p.R[0] if s.mode != "internal" else None)
    x, y, uu, vv, mag, _, h = _slice(g, p, s, fine)
    o.images["velocity"] = field2d(out / "images" / "velocity.png", x, y, mag, "speed in the middle plane, m/s",
                                   "viridis", vectors=np.c_[uu, vv], spacing=h)
    o.values["_figures"] = [["pressure", "Pressure on the part's walls"],
                            ["velocity", "Speed through the middle of the flow"]]


def _playback(o, g, p, s, fine, out, fl):
    from . import playback

    pw = surface_values(o)
    mesh_b, surf = playback.surface_block(g.X, g.tris, {"pressure": (pw if pw is not None else
                                                                     np.zeros(len(g.X))) * s.rho},
                                          {"pressure": "Pa"})
    x, y, uu, vv, mag, plane, h = _slice(g, p, s, fine)
    ny = max(24, min(120, int(160 * np.ptp(y) / max(np.ptp(x), 1e-9))))
    field = playback.grid_field("flow", x, y, {"u": uu, "v": vv, "speed": mag},
                                {"u": "m/s", "v": "m/s", "speed": "m/s"}, nx=160, ny=ny, unit="mm", plane=plane,
                                spacing=h)
    hist = np.asarray(fine["history"])
    step = max(1, len(hist) // 200)
    curves = {"convergence": playback.series(np.arange(len(hist))[::step], hist[::step], "iteration",
                                             "N" if s.mode == "external" else "Pa")}
    scal = {"reynolds": (o.values["re"], "")}
    if s.mode == "external":
        scal.update(speed=(s.speed, "m/s"), drag=(o.values["drag_N"], "N"), lift=(o.values["lift_N"], "N"))
    else:
        scal.update(flow=(s.flow, "m3/s"), pressure_drop=(o.values["dp_Pa"], "Pa"), inlet_speed=(s.speed, "m/s"))
    about = ("The part's surface in mm, coloured by wall pressure, and the flow in a plane through the middle: the "
             "plane's grid runs along u and v from its origin, all in the part's coordinates, and the u and v "
             "components are the velocity along those axes. Move particles through the plane to show the stream "
             + ("passing the part." if s.mode == "external" else "through the part."))
    playback.write(out, "partflow", f"{fl.name} {'around' if s.mode == 'external' else 'through'} "
                                    f"{Path(o.part).stem}", about, [field], curves=curves, scalars=scal,
                   mesh=mesh_b, surface=surf, units={"length": "mm", "velocity": "m/s", "pressure": "Pa"})
