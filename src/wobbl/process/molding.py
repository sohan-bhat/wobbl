"""Injection molding fill: does a thin walled part fill before the melt freezes or the machine runs out of pressure?

"Will this enclosure lid short shot in polystyrene?" becomes: the melt enters at
the gate at 230 C into a 50 C mold over about a second, and flows through the
walls as a shear thinning Cross-WLF fluid (process.hele_shaw). Wobbl fills the
part twice: once with the melt at its injection temperature, which is the least
pressure it can take, and once with a skin freezing on the mold walls behind the
front as fast as conduction into a cold wall allows, which is the most. The
truth sits between them. It reports the fill time, the pressure at the switch to
packing, the flow front over time, weld lines and the last places to fill, and
judges the pressure against the machine's limit.
"""
from __future__ import annotations

import math
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import cKDTree

from ..units import fmt
from ..verify.checks import Check, confidence
from . import budget as bud
from . import hele_shaw as H
from .data import Resin, parse_resin

SWITCH = 0.98            # share full at the switch to packing, where the fill pressure is read
FRONT_SPEED = 100.0      # mm/s of flow front for the default fill time
PICARD_TOL = 2e-3        # change in gate pressure that settles the shear thinning within a step
VERIFY = ("mesh convergence", "time step", "volume balance", "hand calculation", "thin walls", "units and sanity")


@dataclass
class Gate:
    at: tuple[float, float, float]
    diameter: float = 0.0              # mm, 0 for the default

    def text(self) -> str:
        return f"gate at {','.join(fmt(v) for v in self.at)}" + (f" d={fmt(self.diameter)}" if self.diameter else "")


def parse_gate(text: str) -> Gate:
    """'40,0,2' or 'at 40,0,2 d=1.5' (mm)."""
    import re

    num = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"
    m = re.search(rf"({num})\s*,\s*({num})\s*,\s*({num})", text)
    if not m:
        raise ValueError(f"could not read the gate '{text}'; write it like 'at 40,0,2 d=1.5' (mm, a point on the "
                         "part's surface)")
    d = re.search(rf"\bd\s*=\s*({num})", text)
    return Gate(tuple(float(v) for v in m.groups()), float(d.group(1)) if d else 0.0)


@dataclass
class MoldingStudy:
    part: str
    resin: str = "gpps"
    melt: float | None = None          # C
    mold: float | None = None          # C
    fill_time: float | None = None     # s
    gates: list[Gate] = field(default_factory=list)   # none: Wobbl picks one
    p_max: float = 150.0               # MPa, the machine's injection pressure limit
    question: str = ""
    quality: str = "normal"
    scale: float = 1.0
    params: dict = field(default_factory=dict)
    mesh_size: float | None = None
    budget: float = bud.DEFAULT_BUDGET   # s, what the mesh levels may be predicted to take (process.budget)


# element size of each level against the base size, coarse to fine. The first two always run, so the mesh
# convergence check always has two meshes to compare. A third is meshed only when the pair has not converged and
# the budget allows it
LEVELS = {"fast": [1.0, 0.72], "normal": [1.0, 0.72, 0.52]}
CONVERGED = 0.03        # change in the switch pressure between meshes that ends the refinement


def element_thickness(mesh, feats) -> np.ndarray:
    """The wall each element sits in: the smallest wall thickness among the faces next to it, so an element at a
    plate's edge takes the plate's thickness and not the plate's width across its narrow end face."""
    _, _, cen = mesh.tri_geometry()
    th = np.array([feats.thickness_by_face.get(int(s), feats.thickness_typical) for s in mesh.tri_surf])
    th = np.where(np.isfinite(th) & (th > 0), th, feats.thickness_typical)
    tree = cKDTree(cen)
    ec = mesh.X[mesh.tets[:, :4]].mean(axis=1)
    k = min(12, len(cen))
    d, i = tree.query(ec, k=k)
    near = d <= d[:, :1] + 0.75 * feats.thickness_typical + 1e-9
    return np.where(near, th[i], np.inf).min(axis=1)


def gate_nodes(mesh, X_surface_nodes: np.ndarray, at, radius: float) -> np.ndarray:
    p = np.asarray(at, float)
    s = X_surface_nodes[int(np.argmin(np.linalg.norm(mesh.X[X_surface_nodes] - p, axis=1)))]
    near = np.nonzero(np.linalg.norm(mesh.X - mesh.X[s], axis=1) <= radius + 1e-9)[0]
    return near if len(near) else np.array([s])


def hand_estimates(visc: H.Viscosity, flow: float, b: float, length: float, volume: float) -> dict:
    """The same Hele Shaw law along one dimension: a center gated disk out to the flow length, where the melt
    spreads radially, and a strip as wide as the part's volume allows over that length, where it does not."""
    def gradient(q_per_width):
        lo, hi = 1e-8, 1e4
        for _ in range(80):
            mid = math.sqrt(lo * hi)
            qm = float(H.fluidity(visc, np.array([b]), np.array([mid]), np.array([0.0]))[0]) * mid
            lo, hi = (mid, hi) if qm < q_per_width else (lo, mid)
        return math.sqrt(lo * hi)

    r0 = max(b, 0.5)
    rs = np.geomspace(r0, max(length, 2 * r0), 60)
    g = np.array([gradient(flow / (2 * math.pi * r)) for r in rs])
    disk = float(np.trapezoid(g, rs)) if hasattr(np, "trapezoid") else float(np.trapz(g, rs))
    width = volume / (b * max(length, 1e-9))
    strip = gradient(flow / width) * length
    return {"disk_MPa": disk, "strip_MPa": strip, "width_mm": width}


def _share_pressure(res: H.FillResult, share: float = SWITCH) -> float:
    if not len(res.share):
        return float("nan")
    i = int(min(np.searchsorted(res.share, share), len(res.share) - 1))
    return float(res.gate_pressure[i])


def _clusters(mesh, nodes: np.ndarray, min_nodes: int = 2) -> list[np.ndarray]:
    if not len(nodes):
        return []
    adj = H.edge_graph(H.Mesh(mesh.X, mesh.tets[:, :4], np.ones(len(mesh.tets))))
    sub = adj[nodes][:, nodes]
    ncomp, lab = connected_components(sub, directed=False)
    out = [nodes[lab == c] for c in range(ncomp)]
    return sorted([c for c in out if len(c) >= min_nodes], key=len, reverse=True)


def run_molding(study: MoldingStudy, out_dir: str | Path, progress=None, keep: bool = False):
    from ..mesh.mesher import Sizing, build_mesh
    from ..outcome_generic import GenericOutcome
    from ..structural.run import describe_part
    from ..verify.checks import convergence_check

    t0 = time.time()
    limit = bud.deadline(study.budget)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    say = progress or (lambda a, b: None)
    resin: Resin = parse_resin(study.resin)
    melt = float(study.melt if study.melt is not None else resin.melt)
    mold = float(study.mold if study.mold is not None else resin.mold)
    if melt <= resin.no_flow:
        raise ValueError(f"the melt at {fmt(melt)} C is below the resin's no flow temperature {fmt(resin.no_flow)} C")
    say("geometry", f"reading {Path(study.part).name}")
    prep, feats = describe_part(study.part, out / "work", study.scale, None, study.params)
    t_wall = feats.thickness_typical
    size = float(np.max(np.asarray(feats.bbox[1]) - np.asarray(feats.bbox[0])))
    h0 = float(study.mesh_size) if study.mesh_size else max(1.5 * t_wall, size / 60.0)
    visc = H.Viscosity(resin, melt)
    freeze = {"alpha": resin.diffusivity * 1e6, "ratio": H.frozen_skin_ratio(resin, melt, mold)}
    scales = LEVELS["fast" if study.quality == "fast" else "normal"]
    say("budget", "meshing coarsely to predict the cost of the mesh levels")
    found = bud.pilots(lambda f_p, c_p: build_mesh(prep, Sizing(h=h0 * scales[0] * f_p, order=1, curvature=c_p)).X)
    bfit = bud.fit(bud.MOLDING, found, h0, scales, study.budget,
                   bool(study.mesh_size), extra=lambda n: 2 * bud.MOLDING.solve(n))
    say("budget", bud.note(bfit).split(". The budget")[0])
    f_r, c_r = bfit.factor, bfit.curvature
    levels, gates_xyz, flow, t_fill_set, L_flow = [], [], None, None, None
    t_levels = time.time()
    for k, sc in enumerate(scales):
        f = sc * f_r
        if k >= bud.MIN_LEVELS and not bud.room_for(bud.MOLDING, levels[-1]["mesh"].n_nodes,
                                                    h0 * scales[k - 1] * f_r, c_r, h0 * f,
                                                    time.time() - t_levels, study.budget, bfit.power):
            break
        say("mesh", f"level {k}: element size {h0 * f:.3g} mm")
        mesh = build_mesh(prep, Sizing(h=h0 * f, order=1, curvature=c_r))
        th = element_thickness(mesh, feats)
        hm = H.Mesh(mesh.X, mesh.tets[:, :4], th)
        surf = np.unique(mesh.tris[:, :3])
        if not gates_xyz:
            if study.gates:
                gates_xyz = [(np.asarray(g.at, float), g.diameter) for g in study.gates]
            else:
                say("gate", "finding the gate with the shortest flow paths")
                g = H.infer_gate(hm, surf)
                gates_xyz = [(mesh.X[g].copy(), 0.0)]
        gsets = [gate_nodes(mesh, surf, xyz, max(d / 2 if d else 0.8 * t_wall, 0.75 * h0 * f))
                 for xyz, d in gates_xyz]
        if L_flow is None:
            dist = dijkstra(H.edge_graph(hm), indices=np.concatenate(gsets), min_only=True)
            L_flow = float(np.nanmax(np.where(np.isfinite(dist), dist, np.nan)))
            t_fill_set = float(study.fill_time) if study.fill_time else float(np.clip(L_flow / FRONT_SPEED, 0.3, 5.0))
            flow = feats.volume / t_fill_set
        settings = H.FillSettings(gsets, flow, study.p_max, 5.0 * t_fill_set, picard_tol=PICARD_TOL,
                                  deadline=limit, budget=study.budget)
        say("fill", f"level {k}: {mesh.n_nodes:,} nodes, melt at {fmt(melt)} C")
        iso = H.fill(hm, visc, settings, say)
        levels.append({"mesh": mesh, "hm": hm, "gsets": gsets, "iso": iso, "p_switch": _share_pressure(iso)})
        if k >= 1 and abs(levels[-1]["p_switch"] / max(levels[-2]["p_switch"], 1e-12) - 1) < CONVERGED:
            break
    bfit.measured = {"levels": len(levels), "nodes": [int(L["mesh"].n_nodes) for L in levels],
                     "seconds": round(time.time() - t_levels, 1)}
    lv = levels[-1]
    mesh, hm, iso = lv["mesh"], lv["hm"], lv["iso"]
    say("fill", "again with a skin freezing on the mold walls")
    cold = H.fill(hm, visc, H.FillSettings(lv["gsets"], flow, study.p_max, 5.0 * t_fill_set, freeze=freeze,
                                           picard_tol=PICARD_TOL, deadline=limit, budget=study.budget), say)
    say("check", "time step: level 0 again with twice the fill per step")
    half = H.fill(levels[0]["hm"], visc, H.FillSettings(levels[0]["gsets"], flow, study.p_max, 5.0 * t_fill_set,
                                                       step_fill=0.02, picard_tol=PICARD_TOL, deadline=limit, budget=study.budget))
    checks: list[Check] = []
    p_iso, p_cold = lv["p_switch"], _share_pressure(cold)
    if len(levels) > 1:
        checks.append(convergence_check([L["p_switch"] for L in levels], 0.05, "pressure at the switch to packing",
                                        "MPa", "", min_levels=2))
    else:
        checks.append(convergence_check([lv["p_switch"]], 0.05, "pressure at the switch to packing", "MPa"))
    pa, pb = levels[0]["p_switch"], _share_pressure(half)
    ch = abs(pb - pa) / max(abs(pa), 1e-12)
    st = "pass" if ch < 0.03 else ("warn" if ch < 0.08 else "fail")
    checks.append(Check("time step", st, f"filling twice as much per step moves the switch pressure from {fmt(pa)} to "
                                         f"{fmt(pb)} MPa ({100 * ch:.2g}%)", {"change": ch}))
    cav = float(mesh.volume())
    err = abs(iso.injected - iso.in_cavity) / cav
    mesh_err = abs(cav - feats.volume) / feats.volume
    st = "pass" if err < 1e-6 and mesh_err < 0.01 else ("warn" if err < 1e-3 else "fail")
    checks.append(Check("volume balance", st, f"melt injected {fmt(iso.injected / 1000)} cm3 and melt in the cavity "
                                              f"{fmt(iso.in_cavity / 1000)} cm3 agree to {err:.1e}, and the mesh holds "
                                              f"{100 * (1 - mesh_err):.4g}% of the CAD volume",
                        {"injected_mm3": iso.injected, "in_cavity_mm3": iso.in_cavity, "cad_mm3": feats.volume}))
    b_med = float(np.median(hm.thickness))
    hand = hand_estimates(visc, flow, b_med, L_flow, feats.volume)
    lo_h, hi_h = sorted([hand["disk_MPa"], hand["strip_MPa"]])
    st = "pass" if 0.5 * lo_h <= p_iso <= 2.0 * hi_h else "info"
    checks.append(Check("hand calculation", st, f"the same melt along one dimension, over the {fmt(L_flow)} mm flow "
                                                f"length in a {fmt(b_med)} mm wall, needs {fmt(hand['disk_MPa'])} MPa "
                                                f"spreading from a center gate and {fmt(hand['strip_MPa'])} MPa down a "
                                                f"{fmt(hand['width_mm'])} mm wide strip; the fill takes {fmt(p_iso)} MPa",
                        hand))
    ratio = b_med / max(L_flow, 1e-9)
    st = "pass" if ratio <= 0.1 else ("warn" if ratio <= 0.25 else "fail")
    checks.append(Check("thin walls", st, f"walls of {fmt(b_med)} mm against a {fmt(L_flow)} mm flow length "
                                          f"(ratio {ratio:.2g}); the gap flow model needs walls thin next to how far "
                                          "the melt travels"))
    checks.append(Check("units and sanity", "pass" if 0 < p_iso < 1e4 else "fail",
                        f"{fmt(feats.volume / 1000)} cm3 shot filled at {fmt(flow / 1000)} cm3/s"))
    # the answer: does it fill, and with how much of the machine
    if iso.short:
        checks.append(Check("fill", "fail", f"even with the melt at {fmt(melt)} C it stops at {100 * iso.filled:.3g}% "
                                            f"full: {iso.stalled_by}"))
    elif cold.short:
        checks.append(Check("fill", "warn", f"it fills with the melt hot, but with the frozen skin it stops at "
                                            f"{100 * cold.filled:.3g}% full ({cold.stalled_by})"))
    else:
        checks.append(Check("fill", "pass", "it fills with the melt hot and with the frozen skin"))
    worst = p_cold if np.isfinite(p_cold) else float(np.max(cold.gate_pressure, initial=0.0))
    st = "pass" if worst <= 0.8 * study.p_max else ("warn" if p_iso <= study.p_max else "fail")
    checks.append(Check("pressure", st, f"{fmt(p_iso)} to {fmt(worst)} MPa at the switch to packing against the "
                                        f"machine's {fmt(study.p_max)} MPa", {"low_MPa": p_iso, "high_MPa": worst}))
    weld = H.weld_lines(hm, iso.t_fill)
    weld = np.setdiff1d(weld, np.concatenate(lv["gsets"]))
    lines = _clusters(mesh, weld)
    traps = air_traps(mesh, iso.t_fill, np.concatenate(lv["gsets"]))
    extra = [("medium", "process estimate: the fill pressure lies between a hot melt and a frozen skin bound, and "
                        "rests on one published viscosity fit for the resin")]
    level, reasons = confidence([c for c in checks if c.name in VERIFY], extra)
    fails = iso.short or p_iso > study.p_max
    weak = cold.short or worst > 0.8 * study.p_max
    verdict = "fails" if fails else ("marginal" if weak else "ok")
    t_end = float(iso.times[-1]) if len(iso.times) else 0.0
    gtxt = "; ".join(f"({', '.join(fmt(float(v)) for v in xyz)})" for xyz, _ in gates_xyz)
    if iso.short:
        head = (f"Short shot: the melt stops at {100 * iso.filled:.3g}% full ({iso.stalled_by}) with the gate at "
                f"{gtxt} mm.")
    elif fails:
        head = f"Needs {fmt(p_iso)} MPa to fill, more than the machine's {fmt(study.p_max)} MPa."
    else:
        head = (f"Fills in {fmt(t_end)} s from the gate at {gtxt} mm at {fmt(p_iso)} to {fmt(worst)} MPa against a "
                f"{fmt(study.p_max)} MPa machine" + (f", with {len(lines)} weld line{'s' if len(lines) != 1 else ''}."
                                                    if lines else "."))
    fixes = molding_fixes(study, visc, resin, melt, flow, b_med, L_flow, feats, p_iso, worst, verdict, hm,
                          np.concatenate(lv["gsets"]))
    assumptions = [
        f"{resin.name}: Cross-WLF n {resin.n:g}, tau* {fmt(resin.tau)} Pa, D1 {resin.d1:.3g} Pa s, D2 {fmt(resin.d2)} "
        f"K, D3 {resin.d3:g} K/Pa, A1 {resin.a1:g}, A2 {resin.a2:g} K; melt conductivity {resin.conductivity:g} "
        f"W/(m K), specific heat {fmt(resin.specific_heat)} J/(kg K). Sources: {resin.source}.",
        f"Melt at {fmt(melt)} C into a mold at {fmt(mold)} C; the melt stops flowing at {fmt(resin.no_flow)} C.",
        (f"Fill time {fmt(t_fill_set)} s as given." if study.fill_time else
         f"Fill time {fmt(t_fill_set)} s, a flow front of about {fmt(FRONT_SPEED)} mm/s over the {fmt(L_flow)} mm "
         "flow length, since none was given."),
        ("The gate is where it was given." if study.gates else
         f"No gate was given, so Wobbl put it at {gtxt} mm, where the melt has the shortest way to the farthest "
         "corner with thin walls counted as longer."),
        "Hele Shaw gap flow: pressure is the same across a wall, and fountain flow at the front and the gate's own "
        "runner are not modeled.",
        f"The hot bound keeps the melt at {fmt(melt)} C, and the cold bound grows a skin on each wall as deep as "
        "conduction into a wall at the mold temperature brings the melt to its no flow temperature, without the "
        "heat the moving melt brings back. The real fill sits between them.",
        f"The machine holds at most {fmt(study.p_max)} MPa at the gate; past it the flow falls instead.",
        bud.note(bfit),
    ]
    imgs = molding_images(out, mesh, iso, cold, lv["gsets"], study.p_max, gates_xyz)
    notes = list(prep.notes)
    try:
        molding_scene(out, mesh, iso, gates_xyz, lines, traps, study.p_max, verdict, head, study.question)
    except Exception as e:  # noqa: BLE001  the answer stands without its animation
        notes.append(f"the 3D view was not written: {e}")
    metrics = [("fill time", f"{fmt(t_end)} s"), ("fill pressure", f"{fmt(p_iso)} to {fmt(worst)} MPa"),
               ("machine limit", f"{fmt(study.p_max)} MPa"), ("weld lines", str(len(lines)))]
    values = {"process": "molding", "resin": resin.key, "melt_C": melt, "mold_C": mold, "fill_time_s": t_end,
              "fill_time_set_s": t_fill_set, "flow_mm3_s": flow, "pressure_hot_MPa": p_iso,
              "pressure_cold_MPa": worst, "p_max_MPa": study.p_max, "filled_hot": iso.filled,
              "filled_cold": cold.filled, "short_shot": bool(iso.short), "flow_length_mm": L_flow,
              "gates": [[float(v) for v in xyz] for xyz, _ in gates_xyz], "gate_placed": not study.gates,
              "weld_lines": [{"xyz": mesh.X[c].mean(axis=0).tolist(), "length_mm": float(np.ptp(mesh.X[c], axis=0).max()),
                              "nodes": len(c)} for c in lines[:8]],
              "air_traps": [mesh.X[i].tolist() for i in traps[:8]],
              "levels": [{"nodes": int(L["mesh"].n_nodes), "p_switch_MPa": L["p_switch"]} for L in levels],
              "budget": bfit.to_dict(),
              "_figures": [["fill", "Time for the melt to reach each point, gate marked"],
                           ["pressure", "Gate pressure over the fill, hot and frozen skin bounds"]]}
    np.savez_compressed(out / "fields.npz", X=mesh.X.astype(np.float32), t_fill=iso.t_fill.astype(np.float32),
                        t_fill_cold=cold.t_fill.astype(np.float32))
    o = GenericOutcome("molding", str(study.part), study.question, verdict, head, metrics, level, reasons, checks,
                       assumptions, values, imgs, fixes, notes,
                       solver=f"Wobbl Hele Shaw fill (scipy), {mesh.n_nodes:,} nodes", t_total=time.time() - t0)
    o.write(out)
    if not keep:
        shutil.rmtree(out / "work", ignore_errors=True)
    return o


def air_traps(mesh, t_fill: np.ndarray, gates: np.ndarray) -> np.ndarray:
    """The last places to fill: nodes that fill after every neighbor, where air is squeezed unless it can vent."""
    adj = H.edge_graph(H.Mesh(mesh.X, mesh.tets[:, :4], np.ones(len(mesh.tets)))).tocsr()
    t = np.nan_to_num(t_fill, nan=-1.0)
    later = np.full(len(t), True)
    for i in range(len(t)):
        nb = adj.indices[adj.indptr[i]:adj.indptr[i + 1]]
        if len(nb) and t[nb].max() > t[i]:
            later[i] = False
    later[gates] = False
    idx = np.nonzero(later & (t > 0))[0]
    return idx[np.argsort(-t[idx])]


def molding_fixes(study, visc, resin, melt, flow, b, L_flow, feats, p_iso, p_cold, verdict, hm, gates) -> list[dict]:
    fixes = []
    if verdict == "ok":
        return fixes
    n = max(resin.n, 0.2)
    k = 1.25
    fixes.append({"kind": "thickness", "title": f"Thicken the walls about {int(round(100 * (k - 1)))}%",
                  "summary": f"For a shear thinning melt the pressure goes as the wall thickness to the power "
                             f"-(1 + 2n), so walls of {fmt(b * k)} mm instead of {fmt(b)} mm need about "
                             f"{fmt(p_iso * k ** -(1 + 2 * n))} MPa instead of {fmt(p_iso)} MPa with the melt hot.",
                  "auto": False})
    hotter = melt + 20.0
    lo = hand_estimates(visc, flow, b, L_flow, feats.volume)
    hi = hand_estimates(H.Viscosity(resin, hotter), flow, b, L_flow, feats.volume)
    r = hi["disk_MPa"] / max(lo["disk_MPa"], 1e-12)
    fixes.append({"kind": "melt", "title": f"Run the melt at {fmt(hotter)} C",
                  "summary": f"20 C hotter lowers the viscosity enough to cut the fill pressure to about "
                             f"{fmt(p_iso * r)} MPa (the one dimensional estimate scales by {r:.2g}), and keeps the "
                             "skin thinner. Stay inside the resin supplier's window.", "auto": False})
    fixes.append({"kind": "gate", "title": "Add a second gate or move the gate toward the middle",
                  "summary": f"The farthest flow path is {fmt(L_flow)} mm. Pressure grows about as that length to the "
                             f"power 1 + n, so halving it with a second gate cuts the fill pressure to roughly "
                             f"{fmt(p_iso * 0.5 ** (1 + n))} MPa. Pass gates with --gate \"at X,Y,Z\".", "auto": False})
    return fixes


def molding_images(out: Path, mesh, iso, cold, gsets, p_max, gates_xyz) -> dict:
    from ..report.render import line_plot, render_scalar

    imgs = {}
    try:
        g = np.asarray(gates_xyz[0][0], float)
        tf = np.nan_to_num(iso.t_fill, nan=float(np.nanmax(iso.t_fill)) if np.isfinite(iso.t_fill).any() else 0.0)
        imgs["fill"] = render_scalar(out / "images" / "fill.png", mesh.X, mesh.tets[:, :4], mesh.tris[:, :3], tf,
                                     "melt arrives", "s", "plasma", g, "gate")
        curves = {"melt hot": iso.gate_pressure}
        x = iso.times
        if len(cold.times):
            cp = np.interp(x, cold.times, cold.gate_pressure, right=np.nan)
            curves["frozen skin"] = cp
        imgs["pressure"] = line_plot(out / "images" / "pressure.png", x, curves, "seconds", "gate pressure (MPa)",
                                     p_max, f"machine limit {fmt(p_max)} MPa")
    except Exception as e:  # noqa: BLE001
        imgs["error"] = str(e)
    return imgs


def molding_scene(out: Path, mesh, iso, gates_xyz, lines, traps, p_max, verdict, head, question) -> Path:
    """The part filling from its gate, front by front, with the gate pressure ticking along."""
    from ..viz import Scene

    tf = np.where(np.isfinite(iso.t_fill), iso.t_fill, np.inf)
    t_end = float(iso.times[-1]) if len(iso.times) else 0.0
    t = np.linspace(0.0, t_end, 80)
    s = Scene("Injection molding fill", domain="molding", question=question)
    s.timeline(t, unit="s", label="time after the screw moves", seconds=8, between="interpolated")
    part = s.mesh(mesh.X, mesh.tris[:, :3], name="melt", color="#ff8a3d")
    part.reveal(np.where(np.isfinite(tf), tf, 1e9), t)
    part.color_by("melt arrives", np.nan_to_num(iso.t_fill, nan=t_end), unit="s", colormap="plasma")
    s.mesh(mesh.X, mesh.tris[:, :3], name="cavity", color="#9aa0a6", opacity=0.15, prop=True)
    s.readout("gate pressure", np.interp(t, np.r_[0.0, iso.times], np.r_[0.0, iso.gate_pressure]), unit="MPa",
              limit=p_max)
    for xyz, _ in gates_xyz:
        s.label("gate", xyz, kind="note")
    for c in lines[:3]:
        at = mesh.X[c].mean(axis=0)
        when = float(np.nanmax(iso.t_fill[c]))
        s.label("weld line", at, t0=when, kind="note")
    for i in traps[:2]:
        s.label("last to fill", mesh.X[i], t0=float(iso.t_fill[i]), kind="note")
    if iso.short:
        s.event(t_end, "short shot", iso.stalled_by, kind="fail")
    else:
        s.event(t_end, "full", "the cavity is full and packing takes over")
    s.verdict(verdict.upper(), head, {"ok": "pass", "marginal": "warn", "fails": "fail"}.get(verdict, "info"))
    return s.save(out, budget=450_000)
