"""Plasticity: what a metal part does past first yield.

"Will this steel hook bend for good under 3 kN, and what load pulls it open?" becomes three questions. Where does it
first yield? How far does it stay bent once the load comes off? And what load collapses it? Wobbl answers the first
with a linear solve, the second by applying and removing the load with the metal's true stress strain curve and
large deflection theory, and the third by ramping an elastic perfectly plastic copy of the part until CalculiX can no
longer find equilibrium, which is the limit load of plasticity theory. The checks compare that collapse load over two
meshes and with a fully plastic section from beam theory, balance the reactions against the loads at collapse, make
sure nothing is left on the supports once the load is off, and hold the peak plastic strain against how far the metal
stretches before it breaks.
"""
from __future__ import annotations

import re
import time
from dataclasses import asdict, dataclass, field, replace
from dataclasses import replace as dc_replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from ..geometry.faces import find_sharp_corners
from ..materials import LIBRARY, Hardening, Material, hardening_curve, resolve_material
from ..mesh.mesher import Sizing, auto_sizing, build_mesh, estimate_nodes
from ..solvers import base as solvers
from ..solvers import calculix_nl as nl
from ..structural.build import build_static
from ..structural.hotspots import ZoneMap
from ..structural.run import (describe_part, exclusion_radius, feature_refinements, resolve_selections,
                              support_reactions)
from ..structural.spec import SpecError, StructuralStudy, parse_fixture, parse_load
from ..units import fmt
from ..verify.checks import Check, balance_check, confidence, convergence_check
from .common import NonlinearOutcome, clean_work, light_png, load_size, where

QUALITY = {"fast": 1500, "normal": 5000, "fine": 10000}   # nodes on the first of the two meshes
REFINE = 0.75          # element size of the second mesh against the first, about 2.4 times the nodes
RAMP = 4.0             # the collapse ramp runs to this many times the first yield load
COLLAPSE_TOL = 0.03    # collapse loads on the two meshes agree within this


@dataclass
class PlasticStudy:
    part: str
    material: str
    loads: list[str] = field(default_factory=list)
    fixtures: list[str] = field(default_factory=list)
    target: float | None = None      # margin wanted on the collapse load (default the material's safety factor)
    curve: str = "auto"              # auto: the material's true curve; perfect: flat at yield
    quality: str = "normal"
    question: str = ""
    scale: float = 1.0
    body: int | None = None
    params: dict = field(default_factory=dict)
    keep: bool = False
    playback: bool = False           # also write playback.json, every increment of the run for a player

    def to_dict(self) -> dict:
        return asdict(self)


def check_metal(mat: Material) -> str:
    """Brittle materials break near their elastic limit, so they have no plastic reserve to find. A ductile plastic
    yields, and von Mises plasticity describes it only roughly, which the answer says. Returns that caveat."""
    if not mat.ductile:
        raise SpecError(f"{mat.name} is brittle: it breaks near its tensile strength instead of yielding, so it has "
                        "no plastic reserve to find. Run 'wobbl run' for the elastic answer")
    if mat.category == "metal" or mat.key == "custom":
        return ""
    made = " A printed part also tears between its layers well before it yields along them." \
        if getattr(mat, "process", "") == "printed" else ""
    return (f"{mat.name} is a {mat.category}. Von Mises plasticity describes it only roughly: a plastic yields "
            "higher in compression than in tension, more slowly loaded it yields lower, and under a load held for "
            "hours it creeps on, so the collapse load here is a short term one." + made)


def curve_for(mat: Material, which: str = "auto") -> Hardening:
    """The curve past yield: 'auto' for the material's own, 'perfect' for a flat line at yield, or the user's, as
    'true 250@0, 300@0.02, 420@0.15' (true stress MPa at plastic strain), 'engineering 250@0.00125, 400@0.2'
    (engineering stress at total engineering strain, as a tensile test reads), or a CSV file of either."""
    w = (which or "auto").strip()
    if w.lower() == "auto":
        return hardening_curve(mat)
    if w.lower() == "perfect":
        return Hardening(((mat.yield_strength, 0.0),), "perfect", "elastic perfectly plastic at the yield stress")
    return parse_curve(w, mat)


_PAIR = re.compile(r"([-+]?\d*\.?\d+(?:e[-+]?\d+)?)\s*[@:,;\s]\s*([-+]?\d*\.?\d+(?:e[-+]?\d+)?)", re.I)


def parse_curve(text: str, mat: Material) -> Hardening:
    t = text.strip()
    engineering = t.lower().startswith(("engineering", "eng "))
    if engineering or t.lower().startswith("true "):
        t = t.split(None, 1)[1] if " " in t else ""
    path = Path(t).expanduser()
    if path.suffix.lower() in (".csv", ".txt", ".tsv", ".dat") or (len(t) < 400 and path.is_file()):
        if not path.is_file():
            raise SpecError(f"cannot find the stress strain curve file {t}")
        raw = path.read_text(errors="replace")
        engineering = engineering or "eng" in raw.splitlines()[0].lower() if raw.strip() else engineering
        rows = []
        for ln in raw.splitlines():
            vals = re.findall(r"[-+]?\d*\.?\d+(?:e[-+]?\d+)?", ln, re.I)
            if len(vals) >= 2 and not re.search(r"[a-df-z]", ln.replace("e-", "").replace("e+", ""), re.I):
                rows.append((float(vals[0]), float(vals[1])))
        where = path.name
    else:
        rows = [(float(a), float(b)) for a, b in _PAIR.findall(t.replace(",", " ").replace(";", " "))]
        where = "the text given"
    if not rows:
        raise SpecError(f"could not read a stress strain curve from '{text[:60]}'; give pairs of stress MPa and strain, "
                        "as in 'true 250@0, 300@0.02, 420@0.15' or 'engineering 250@0.00125, 400@0.2'")
    sig, eps = np.array(rows, float).T
    E = mat.E
    if engineering:
        sig, eps = sig * (1 + eps), np.log1p(eps)          # true stress and true total strain
        plastic = eps - sig / E
    elif eps[0] > 0 and abs(eps[0] - sig[0] / E) < 0.5 * sig[0] / E:
        plastic = eps - sig / E                              # a true curve with total strains
    else:
        plastic = eps
    keep = plastic >= -0.05 * sig / E          # points on the elastic line, within reading error
    if not keep.any():
        raise SpecError("every point of the curve lies on the elastic line, so it gives no yield")
    sig, plastic = sig[keep], np.maximum(plastic[keep], 0.0)
    order = np.argsort(plastic, kind="stable")
    sig, plastic = sig[order], plastic[order]
    plastic[0] = 0.0
    pts = [(float(sig[0]), 0.0)]
    for sg, pe in zip(sig[1:], plastic[1:]):
        if pe > pts[-1][1] + 1e-9 and sg >= pts[-1][0]:      # past the peak the metal necks, so it stays flat
            pts.append((float(sg), float(pe)))
    return Hardening(tuple(pts), "user", f"the {'engineering' if engineering else 'true'} curve from {where}, "
                                         f"{len(pts)} points from {fmt(pts[0][0])} MPa at yield")


def run_plastic(study: PlasticStudy, out_dir: str | Path, progress=None, pre=None,
                render_images: bool = True) -> NonlinearOutcome:
    t0 = time.time()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    work = out / "work"
    say = progress or (lambda a, b: None)
    mat = resolve_material(study.material)
    caveat = check_metal(mat)
    curve = curve_for(mat, study.curve)
    if curve.kind == "user":
        # the collapse run and first yield follow the yield stress of the curve given
        mat = dc_replace(mat, yield_strength=curve.points[0][0])
    adapter = solvers.for_domain("structural")
    adapter.require()
    base = StructuralStudy(str(study.part), study.material, [parse_load(x) for x in study.loads],
                           [parse_fixture(x) for x in study.fixtures], quality=study.quality, scale=study.scale,
                           body=study.body, question=study.question)
    base.params = dict(study.params)
    if pre is None:
        say("geometry", f"reading {Path(study.part).name}")
        pre = describe_part(base.part, work, base.scale, base.body, base.params)
    prep, feats = pre
    loads, fixtures, selections = resolve_selections(base, feats)
    target = QUALITY.get(study.quality, QUALITY["normal"])
    s0 = plastic_sizing(prep, feats, target)
    levels, last = [], None
    for k in range(2):
        sizing = s0 if k == 0 else _scaled(s0, REFINE)
        say("mesh", f"level {k}: element size {sizing.h:.3g} mm")
        mesh = build_mesh(prep, sizing)
        # many small holes and fillets can still multiply the nodes well past the estimate: coarsen them and mesh
        # again, since each increment of the ramp solves the whole mesh
        for _ in range(3 if k == 0 else 0):
            if mesh.n_nodes <= 4 * target:
                break
            s0 = sizing = _scaled(s0, 1.0, (mesh.n_nodes / (4 * target)) ** (1 / 3))
            say("mesh", f"level 0 again: {mesh.n_nodes:,} nodes is far over {target:,}, refinements coarsened")
            mesh = build_mesh(prep, sizing)
        built = build_static(mesh, loads, fixtures, mat, feats.faces)
        say("solve", f"level {k}: elastic solve on {mesh.n_nodes:,} nodes")
        lin = adapter.solve_static(built.model, work / f"level{k}", job="elastic")
        y = first_yield(mesh, built, feats, sizing, lin, mat)
        say("solve", f"level {k}: first yield at {y['factor']:.3g}x the load; ramping toward collapse")
        lim = limit_run(built, mat, y["factor"], lin, work / f"level{k}")
        levels.append({"level": k, "n_nodes": mesh.n_nodes, "h": sizing.h, "first_yield": y["factor"],
                       "collapse": lim["factor"], "collapsed": lim["collapsed"], "increments": lim["increments"],
                       "iterations": lim["iterations"], "t_solve": lim["t_solve"]})
        say("check", f"level {k}: collapse at {lim['factor']:.4g}x the load" if lim["collapsed"] else
            f"level {k}: no collapse up to {lim['factor']:.4g}x the load")
        last = (mesh, built, lin, y, lim)
        if not lim["collapsed"]:
            # nothing to converge: a second ramp on a finer mesh would only find no collapse again, slower
            break
    mesh, built, lin, y, lim = last
    service = None
    if y["factor"] < 1.0 and lim["factor"] > 1.0:
        say("solve", "applying and removing the load with the true stress strain curve and large deflection theory")
        try:
            service = service_run(built, mat, curve, y["factor"], work / "service", every=study.playback)
        except Exception as e:  # noqa: BLE001  the collapse answer stands without it
            service = {"error": str(e)[:200]}
    o = evaluate(study, mat, curve, prep, feats, loads, mesh, built, lin, y, lim, levels, service, caveat)
    o.t_total = time.time() - t0
    if render_images:
        say("render", "drawing the plastic strain and the load path")
        o.images = render(out / "images", mesh, feats, y, lim, service)
    o.notes = list(prep.notes)
    if study.playback:
        o.images["playback"] = playback(out / "playback.json", study, mesh, lim, service)
    try:
        from .scene import write_scene
        write_scene(out, study, o, mesh, base, selections, lim, service)
    except Exception as e:  # noqa: BLE001  a view that cannot be built never costs the result
        o.notes.append(f"The 3D view could not be built ({str(e)[:120]}).")
    o.write(out)
    clean_work(work, study.keep)
    return o


def _scaled(s: Sizing, h: float, refine: float | None = None) -> Sizing:
    """A sizing with the global size times h and the local refinements times refine (h when not given)."""
    refine = h if refine is None else refine
    return Sizing(h=s.h * h, order=s.order, curvature=s.curvature,
                  refinements=[replace(r, size=min(r.size * refine, s.h * h)) for r in s.refinements])


def plastic_sizing(prep, feats, nodes: int) -> Sizing:
    """The first mesh: holes and fillets resolved as the strength study resolves them, but no refinement at sharp
    inside corners, whose singular peak a collapse load does not depend on, and coarsened until it is within three
    times the node target, since every increment of a plastic ramp solves the whole mesh again."""
    s = auto_sizing(prep, thickness=feats.thickness_typical, max_nodes=nodes)
    s.curvature = 0
    s.refinements = feature_refinements(feats, s.h)
    for _ in range(8):
        if estimate_nodes(prep, s) <= 3 * nodes:
            break
        s = Sizing(h=s.h * 1.25, order=s.order, curvature=0,
                   refinements=[replace(r, size=r.size * 1.25) for r in s.refinements])
    return s


def first_yield(mesh, built, feats, sizing, lin, mat) -> dict:
    """The load multiple at which the part first yields, read away from sharp corners and supports, where linear
    stress is singular and depends on the mesh rather than the part."""
    vm = lin.von_mises()
    corners = find_sharp_corners(mesh, faces=feats.faces, tangent=feats.tangent_pairs)
    zmap = ZoneMap(mesh, built.zones, built.zone_kind, corners)
    r = exclusion_radius(feats, sizing, mesh, mesh.X[vm > 0.5 * vm.max()])
    excluded = zmap.excluded_mask(mesh.X, r)
    valid = ~excluded if (~excluded).any() else np.ones(len(vm), bool)
    i = int(np.nonzero(valid)[0][np.argmax(vm[valid])])
    return {"factor": mat.yield_strength / max(vm[i], 1e-12), "stress": float(vm[i]), "xyz": mesh.X[i].tolist(),
            "raw_factor": mat.yield_strength / max(float(vm.max()), 1e-12), "excluded_radius": r,
            "excluded": excluded}


def limit_run(built, mat, lam_y: float, lin, work: Path) -> dict:
    """Ramp an elastic perfectly plastic copy of the part with small deflection theory until CalculiX can no longer
    find equilibrium. The last converged load multiple is the collapse (limit) load, a lower bound within the last
    increment. The point that moved most in the elastic solve is followed for the load path."""
    lam_max = RAMP * lam_y
    umag = lin.displacement_magnitude()
    mon = int(np.argmax(umag))
    d = lin.U[mon] / max(float(umag[mon]), 1e-30)
    step = nl.Step(lam_max, first=0.8 / RAMP, largest=0.05, smallest=5e-4, max_increments=400, may_stop=True,
                   every_increment=True)
    p = mat.internal()
    model = nl.NonlinearModel(built.model, [nl.Material(mat.name, p["E"], p["nu"], p["density"],
                                                        [(mat.yield_strength, 0.0)])],
                              [step], monitor={"MON": np.array([mon])}, fields=("U", "RF", "PEEQ"))
    res = nl.solve_nonlinear(model, work, "limit")
    lam = np.asarray(res.times, float) * lam_max
    u = res.monitor["MON"][:, 0, :] @ d if len(res.monitor["MON"]) else np.zeros(len(lam))
    n = min(len(lam), len(u))
    lam, u = lam[:n], u[:n]
    k0 = lam[0] / u[0] if n and u[0] > 0 else float("inf")
    k_end = (lam[-1] - lam[-2]) / max(u[-1] - u[-2], 1e-30) if n > 1 else k0
    tes = None
    if n > 1 and np.isfinite(k0):
        # ASME's twice elastic slope: where the load path crosses a line twice as compliant as the elastic one
        g = lam - 0.5 * k0 * u
        j = np.nonzero(g < 0)[0]
        if len(j) and j[0] > 0:
            i = j[0]
            w = g[i - 1] / (g[i - 1] - g[i])
            tes = float(lam[i - 1] + w * (lam[i] - lam[i - 1]))
    frame = res.frames[-1] if res.frames else {}
    return {"factor": float(lam[-1]) if n else 0.0, "collapsed": not res.finished and not res.timed_out,
            "timed_out": res.timed_out, "ramp": lam_max,
            "lam": lam.tolist(), "u": u.tolist(), "monitor_xyz": built.model.mesh.X[mon].tolist(),
            "stiffness_ratio": float(k_end / k0) if np.isfinite(k0) and k0 > 0 else None, "tes": tes,
            "bracket": float(res.last_attempt * lam_max) if not res.finished else 0.0, "frame": frame,
            "frames": res.frames,
            "increments": res.increments, "iterations": res.iterations, "t_solve": res.t_solve, "solver": res.solver}


def service_run(built, mat, curve: Hardening, lam_y: float, work: Path, every: bool = False) -> dict:
    """Apply the load with the true stress strain curve and large deflection theory, then take it off. What stays
    bent afterwards is the permanent set."""
    p = mat.internal()
    first = float(np.clip(0.8 * lam_y, 0.05, 0.5))
    # a recorded run takes tenths of the load, so a player sees the plastic zone grow
    steps = [nl.Step(1.0, nlgeom=True, first=min(first, 0.1) if every else first, largest=0.1 if every else 0.25,
                     smallest=1e-5, every_increment=every),
             nl.Step(0.0, nlgeom=True, first=0.5, largest=0.5, smallest=1e-4, every_increment=every)]
    model = nl.NonlinearModel(built.model, [nl.Material(mat.name, p["E"], p["nu"], p["density"],
                                                        list(curve.points))], steps)
    res = nl.solve_nonlinear(model, work, "service")
    on = next(fr for fr in res.frames if abs(fr["time"] - 1.0) < 1e-6)
    off = next(fr for fr in res.frames if abs(fr["time"] - 2.0) < 1e-6)
    X = built.model.mesh.X
    i, j = int(np.argmax(on["PEEQ"])), int(np.argmax(np.linalg.norm(off["U"], axis=1)))
    k = int(np.argmax(np.linalg.norm(on["U"], axis=1)))
    return {"frames": res.frames if every else [], "peeq": float(on["PEEQ"][i]), "peeq_xyz": X[i].tolist(), "disp": float(np.linalg.norm(on["U"][k])),
            "disp_xyz": X[k].tolist(), "set": float(np.linalg.norm(off["U"][j])), "set_xyz": X[j].tolist(),
            "residual_stress": float(nl.von_mises(off["S"]).max()), "on": on, "off": off,
            "increments": res.increments, "iterations": res.iterations, "t_solve": res.t_solve}


def verdict_for(lam_y: float, lam_c: float, target: float) -> str:
    if lam_c < 1.0:
        return "collapses"
    if lam_c < target:
        return "marginal"
    return "yields" if lam_y < 1.0 else "holds"


def evaluate(study, mat, curve, prep, feats, loads, mesh, built, lin, y, lim, levels, service,
             caveat: str = "") -> NonlinearOutcome:
    from .section import plastic_section

    sm = built.model
    lam_y, lam_c = y["factor"], lim["factor"]
    target = study.target or mat.recommended_sf
    checks: list[Check] = []
    extra: list[tuple[str, str]] = []
    vals = [L["collapse"] for L in levels]
    if all(L["collapsed"] for L in levels):
        checks.append(convergence_check(vals, COLLAPSE_TOL, "collapse load", "x the load", min_levels=2))
    elif lim.get("timed_out"):
        checks.append(Check("mesh convergence", "warn", f"the ramp ran out of time at {fmt(lim['factor'])}x the load "
                            "without collapsing, so the collapse load is only known to be above that; one mesh was "
                            "ramped"))
    else:
        checks.append(Check("mesh convergence", "info", f"no collapse up to {fmt(lim['ramp'])}x the load, "
                            f"{RAMP:g} times first yield, so the collapse load is only known to be above that and "
                            "one mesh was enough to show it"))
    # the ramp ends either at a mechanism (the part keeps moving for almost no extra load) or at a numerical stop
    ratio = lim["stiffness_ratio"]
    if lim["collapsed"] and ratio is not None:
        s = (f"the ramp ended where the part had lost {100 * (1 - ratio):.1f}% of its elastic stiffness, the last "
             f"increments moving it for almost no extra load")
        if ratio <= 0.1:
            checks.append(Check("collapse", "pass", s + ": a plastic mechanism", {"stiffness_ratio": ratio}))
        else:
            checks.append(Check("collapse", "warn", f"the solver stopped while the part still had {100 * ratio:.0f}% "
                                "of its elastic stiffness, so this may be a numerical stop rather than a mechanism; "
                                "the collapse load is at least this", {"stiffness_ratio": ratio}))
    hinge = np.asarray(y["xyz"])
    if "PEEQ" in lim["frame"]:
        # the mechanism is where plastic strain piles up at collapse, away from the spots next to supports and rigid
        # attachments that yield first because the idealization is stiffer than the real thing
        pe = np.where(y["excluded"], -1.0, lim["frame"]["PEEQ"]) if (~y["excluded"]).any() else lim["frame"]["PEEQ"]
        hinge = np.asarray(mesh.X[int(np.argmax(pe))])
    section, why = plastic_section(mesh, hinge, sm, mat.yield_strength, carrying_supports(sm, lim["frame"]))
    if section is None:
        checks.append(Check("hand calculation", "skip", f"no fully plastic beam section applies ({why})"))
    else:
        r = lam_c / section["factor"]
        s = (f"a fully plastic section through the hinge (Z {fmt(section['Z_mm3'])} mm3, {fmt(section['area_mm2'])} "
             f"mm2) collapses at {fmt(section['factor'])}x the load by beam theory; the FEA collapse is {r:.2f}x that")
        if 0.85 <= r <= 1.2:
            checks.append(Check("hand calculation", "pass", s, section))
        elif r > 1.2:
            checks.append(Check("hand calculation", "info", s + ". The part carries more than one section can, "
                                "because the hinge spreads along it or the supports constrain it", section))
        else:
            checks.append(Check("hand calculation", "warn", s + ". The part gives way below the beam theory section, "
                                "so check the supports and how the load enters", section))
    F, M = sm.applied_total()
    if "RF" in lim["frame"]:
        R, RM = support_reactions(built, SimpleNamespace(RF=lim["frame"]["RF"]), mesh)
        partial = any(np.count_nonzero(np.abs(nc.normal) > 1e-6) > 1 for nc in sm.normal_constraints)
        c = balance_check(lam_c * F, R, lam_c * M, RM, prep.diag, partial)
        c.summary = f"at the collapse load, {c.summary}"
        checks.append(c)
    if service and "error" not in service:
        R, _ = support_reactions(built, SimpleNamespace(RF=service["off"]["RF"]), mesh)
        # a pure moment has no net force, so the leftover is judged against the moment's force scale too
        left = float(np.linalg.norm(R)) / max(float(np.linalg.norm(F)), float(np.linalg.norm(M)) / prep.diag, 1e-12)
        st = "pass" if left < 0.005 else ("warn" if left < 0.02 else "fail")
        checks.append(Check("unloading balance", st, f"once the load is off the supports carry {100 * left:.2g}% of "
                            "it, so the stress left in the part balances itself as it must", {"residual": left}))
        fracture = float(np.log1p(mat.elongation / 100.0))
        use = service["peeq"] / fracture if fracture > 0 else float("inf")
        s = (f"peak plastic strain {100 * service['peeq']:.2g}% against the {100 * fracture:.2g}% true strain at "
             f"which {mat.name} breaks in a tensile test ({fmt(mat.elongation)}% elongation)")
        if use < 0.25:
            checks.append(Check("ductility", "pass", s, {"peeq": service["peeq"], "fracture_strain": fracture}))
        elif use < 0.6:
            checks.append(Check("ductility", "warn", s + "; that is a large share of it, and the peak strain grows "
                                "with mesh refinement at a notch", {"peeq": service["peeq"]}))
        else:
            checks.append(Check("ductility", "fail", s + "; the metal may tear there before the part collapses",
                                {"peeq": service["peeq"]}))
    elif service and "error" in service:
        checks.append(Check("unloading balance", "info", f"the load and unload run did not finish ({service['error']}), "
                            "so the permanent set is not known"))
    bracket = lim["bracket"] / max(lam_c, 1e-12) if lim["collapsed"] else 0.0
    steps = (f"the collapse ramp took {lim['increments']} increments and {lim['iterations']} iterations, and the "
             f"attempt that failed was {100 * bracket:.2g}% past the last increment that converged, so the collapse "
             "load is known that closely")
    if service and "error" not in service:
        steps += f"; the load and unload run took {service['increments']} increments"
    checks.append(Check("load steps", "pass" if bracket <= 0.01 else "warn", steps,
                        {"bracket": bracket, "increments": lim["increments"]}))
    checks.append(_sanity(prep, mat, lam_y, lam_c, service))
    if caveat:
        checks.append(Check("material model", "warn", caveat))
    level, reasons = confidence(checks, extra)
    verdict = verdict_for(lam_y, lam_c, target)
    where_y = where(mesh, feats.faces, y["xyz"])
    where_h = where(mesh, feats.faces, hinge)
    size = load_size(loads, lam_c)
    at_c = f"{lam_c:.2f}x the load" + (f" ({size})" if size else "")
    has_set = service and "error" not in service
    if verdict == "collapses":
        head = f"Collapses at {at_c}: a plastic hinge forms {where_h}."
    elif verdict == "marginal":
        head = f"Carries the load, but it collapses at {at_c}, below the {target:g} target. The hinge forms {where_h}."
    elif verdict == "yields":
        head = (f"Yields at {lam_y:.2f}x the load {where_y} but carries it, collapsing only at {at_c} "
                f"(target {target:g}).")
        if has_set:
            head += f" Once the load comes off it stays bent by {fmt(service['set'])} mm."
    else:
        head = f"Stays elastic: first yield at {lam_y:.2f}x the load {where_y}, collapse at {at_c}."
    if not lim["collapsed"]:
        head = head.replace(f"{lam_c:.2f}x", f"more than {lam_c:.2f}x")
    if level != "high":
        head += f" Confidence is {level}."
    metrics = [("first yield", f"{lam_y:.3g}x the load"),
               ("collapse", ("more than " if not lim["collapsed"] else "") + f"{lam_c:.3g}x the load"),
               ("reserve past first yield", f"{lam_c / lam_y:.3g}x"), ("target", f"{target:g}")]
    if has_set:
        metrics += [("permanent set", f"{fmt(service['set'])} mm"),
                    ("peak plastic strain", f"{100 * service['peeq']:.2g}%")]
    o = NonlinearOutcome("plastic", str(study.part), study.question, verdict, head, metrics, level, reasons, checks,
                         assumptions(mat, curve, study, service) + ([caveat] if caveat else []),
                         solver=lim["solver"])
    o.values = {"first_yield_factor": lam_y, "first_yield_xyz": y["xyz"], "first_yield_MPa": y["stress"],
                "collapse_factor": lam_c, "collapsed": lim["collapsed"], "ramp_factor": lim["ramp"],
                "twice_elastic_slope_factor": lim["tes"], "hinge_xyz": hinge.tolist(), "target": target,
                "levels": levels, "plastic_section": section,
                "load_path": {"factor": lim["lam"], "deflection_mm": lim["u"], "point": lim["monitor_xyz"]},
                "_figures": [["plastic", "Plastic strain under the load, with the peak marked"],
                             ["collapse", "Plastic strain at collapse: the mechanism"],
                             ["set", "Permanent set once the load comes off"],
                             ["path", "Load against deflection on the way to collapse"]]}
    if has_set:
        o.values.update({"peak_plastic_strain": service["peeq"], "peak_plastic_strain_xyz": service["peeq_xyz"],
                         "permanent_set_mm": service["set"], "permanent_set_xyz": service["set_xyz"],
                         "deflection_at_load_mm": service["disp"], "residual_stress_MPa": service["residual_stress"]})
    o.fixes = fixes(verdict, mat, lam_y, lam_c, target, section, where_h)
    return o


def carrying_supports(sm, frame) -> np.ndarray | None:
    """Nodes of the supports that carry load at collapse. A symmetry plane along the part carries next to nothing,
    and leaving it out lets the beam check cut the part between the supports that do and the load."""
    if "RF" not in frame:
        return None
    X = sm.mesh.X
    size, c0 = float(np.linalg.norm(X.max(0) - X.min(0))), X.mean(0)
    sets = [(c.nodes, list(c.dofs)) for c in sm.constraints]
    sets += [(nc.nodes, [int(np.argmax(np.abs(nc.normal))) + 1]) for nc in sm.normal_constraints]
    carry = []
    for nodes, dofs in sets:
        R = np.zeros((len(nodes), 3))
        for d in dofs:
            R[:, d - 1] = frame["RF"][nodes, d - 1]
        # the net force (times the part size) and the net moment: nodal reactions of second order elements swing in
        # sign, so their sum of sizes says little
        carry.append(float(np.linalg.norm(R.sum(0))) * size + float(np.linalg.norm(np.cross(X[nodes] - c0, R).sum(0))))
    top = max(carry, default=0.0)
    keep = [nodes for (nodes, _), c in zip(sets, carry) if c >= 0.15 * top]
    return np.unique(np.concatenate(keep)) if keep else None


def _sanity(prep, mat, lam_y, lam_c, service) -> Check:
    issues = []
    if lam_c < lam_y * 0.98:
        issues.append(f"collapse at {fmt(lam_c)}x comes before first yield at {fmt(lam_y)}x, which plasticity "
                      "theory does not allow; check the supports")
    if lam_y > 1e4 or lam_y < 1e-4:
        issues.append(f"first yield at {fmt(lam_y)}x the load; check the load magnitude and units")
    if prep.diag < 1.0 or prep.diag > 20000:
        issues.append(f"the part is {fmt(prep.diag)} mm across; check the file units")
    if service and "error" not in service and service["disp"] > 0.2 * prep.diag:
        issues.append(f"under the load it moves {fmt(service['disp'])} mm, {100 * service['disp'] / prep.diag:.0f}% "
                      "of its size, so it would change shape a lot before collapsing")
    base = (f"part {fmt(prep.diag)} mm across, first yield at {fmt(lam_y)}x and collapse at {fmt(lam_c)}x the load, "
            f"a reserve of {lam_c / lam_y:.2g} past first yield")
    return Check("units and sanity", "warn" if issues else "pass", "; ".join(issues) if issues else base + ": plausible")


def assumptions(mat, curve, study, service) -> list[str]:
    from ..structural.outcome import process_caveat

    a = [f"{mat.name}: E {fmt(mat.E)} MPa, yield {fmt(mat.yield_strength)} MPa, tensile strength "
         f"{fmt(mat.tensile_strength)} MPa ({mat.source}). Past yield it follows a {curve.kind} true stress strain "
         f"curve: {curve.source}.",
         "The collapse load comes from an elastic perfectly plastic copy of the part at the yield stress with small "
         "deflection theory, the limit load of plasticity theory. Hardening usually carries a little more, while "
         "large deflections can carry more or less, and buckling or necking can carry less.",
         "Hardening is isotropic, so reversed yielding as the load comes off (the Bauschinger effect) is not "
         "modeled. The load is applied once; repeated loads that ratchet or fatigue the part are not.",
         "First yield is read away from sharp inside corners and supports, where linear stress is singular. A tiny "
         "region there yields first, which the plastic runs include.",
         "Loads keep their direction as the part deforms, and supports are as rigid as in a strength study."]
    made = process_caveat(mat)
    if made:
        a.append(made)
    if study.curve == "perfect":
        a.append("The load and unload run uses a flat curve at yield, as asked, so it bends more than with hardening.")
    if service is None:
        a.append("The part does not yield under the load, so there is no permanent set to find.")
    return a


def fixes(verdict, mat, lam_y, lam_c, target, section, where_h) -> list[dict]:
    """Fixes aimed at the collapse mechanism, with how much each must change. The limit load scales with the yield
    stress exactly; a bending hinge with the square of the section depth and linearly with its width."""
    if verdict not in ("collapses", "marginal"):
        return []
    need = target / max(lam_c, 1e-12)
    out = []
    if section is not None and section.get("bending_share", 0) > 0.6:
        depth = section["depth_mm"]
        out.append({"kind": "section", "title": f"Deepen the section at the hinge by {np.sqrt(need):.2g}x",
                    "summary": f"The hinge forms {where_h}, where the part bends through a section about "
                               f"{fmt(depth)} mm deep. Its plastic moment grows with the depth squared, so about "
                               f"{fmt(depth * np.sqrt(need))} mm deep (or {need:.2g}x as wide) reaches the "
                               f"{target:g} target.", "auto": False})
    elif section is not None:
        out.append({"kind": "section", "title": f"Add {need:.2g}x the material through the hinge",
                    "summary": f"The hinge forms {where_h}, where the section is pulled more than bent, so its "
                               f"collapse load grows with its area: about {need:.2g}x the "
                               f"{fmt(section['area_mm2'])} mm2 there.", "auto": False})
    else:
        out.append({"kind": "section", "title": "Stiffen the region that gives way",
                    "summary": f"The plastic strain at collapse concentrates {where_h}. Thicker walls or a rib "
                               f"across that region raise the collapse load: about {need:.2g}x the wall where it "
                               f"is pulled, {np.sqrt(need):.2g}x where it bends.", "auto": False})
    want = mat.yield_strength * need
    # a stronger grade of the same family first (a steel for a steel), so the stiffness and the joining stay the same
    better = sorted([m for m in LIBRARY.values() if m.category == "metal" and m.ductile and m.yield_strength >= want],
                    key=lambda m: (abs(np.log(m.E / mat.E)) > 0.25, m.yield_strength))
    pick = f", such as {better[0].name} at {fmt(better[0].yield_strength)} MPa" if better else ""
    out.append({"kind": "material", "title": f"Use a metal that yields above {fmt(want)} MPa",
                "summary": f"The collapse load scales with the yield stress, so {need:.2g}x the "
                           f"{fmt(mat.yield_strength)} MPa of {mat.name} reaches the target{pick}.", "auto": False})
    out.append({"kind": "load", "title": f"Keep the load under {fmt(lam_c / target)}x this one",
                "summary": f"With {mat.name} as drawn, the {target:g} margin on the {fmt(lam_c)}x collapse load "
                           f"allows {fmt(lam_c / target)}x the load in service.", "auto": False})
    return out


def playback(path: Path, study, mesh, lim, service) -> str:
    """The run as a player sees it: the load going on and coming off with the plastic strain spreading, then the
    elastic perfectly plastic ramp to collapse, on one decimated surface."""
    from .common import write_playback

    frames = []
    if service and "error" not in service:
        for fr in service["frames"]:
            t = float(fr["time"])
            frames.append({"t": t, "label": "loading" if t <= 1 + 1e-9 else "unloading", "U": fr["U"],
                           "plastic_strain": 100 * fr["PEEQ"], "values": {"load": t if t <= 1 else 2 - t}})
    start = frames[-1]["t"] if frames else 0.0
    ramp = [fr for fr in lim.get("frames", []) if "PEEQ" in fr]
    for i in np.unique(np.linspace(0, len(ramp) - 1, min(len(ramp), 10)).astype(int)) if ramp else []:
        fr = ramp[i]
        frames.append({"t": start + float(fr["time"]), "label": "collapse ramp", "U": fr["U"],
                       "plastic_strain": 100 * fr["PEEQ"], "values": {"load": float(fr["time"]) * lim["ramp"]}})
    top = max([float(np.max(f["plastic_strain"])) for f in frames] + [1e-9])
    return write_playback(
        path, mesh, frames,
        {"kind": "plastic", "title": Path(study.part).name, "question": study.question,
         "units": {"length": "mm", "load": "multiple of the load asked about", "plastic_strain": "%"},
         "phases": ["loading", "unloading", "collapse ramp"],
         "about": "Frames play in order of t. The load goes on (t 0 to 1) and comes off (t 1 to 2) with the true "
                 "stress strain curve and large deflection theory, then an elastic perfectly plastic copy on the "
                 "same mesh is ramped to collapse (t from 2 on). values.load is the load multiple at each frame; "
                 "the last frame is the mechanism, which moves much further than the frames before it."},
        {"plastic_strain": {"unit": "%", "min": 0.0, "max": top}},
        {"collapse_path": {"load": lim["lam"], "deflection_mm": lim["u"]}})


def render(out: Path, mesh, feats, y, lim, service) -> dict:
    from ..report.render import line_plot, render_scalar

    img = {}
    X, T, tris = mesh.X, mesh.tets, mesh.tris
    try:
        if service and "error" not in service:
            pe = 100 * service["on"]["PEEQ"]
            img["plastic"] = light_png(render_scalar(out / "plastic.png", X, T, tris, pe, "plastic strain", "%",
                                                     "magma", np.asarray(service["peeq_xyz"]), f"{pe.max():.2g}%"))
            um = np.linalg.norm(service["off"]["U"], axis=1)
            img["set"] = light_png(render_scalar(out / "set.png", X, T, tris, um, "permanent set", "mm", "viridis",
                                                 np.asarray(service["set_xyz"]), f"{fmt(float(um.max()))} mm"))
        if "PEEQ" in lim["frame"]:
            pe = 100 * lim["frame"]["PEEQ"]
            i = int(np.argmax(pe))
            img["collapse"] = light_png(render_scalar(out / "collapse.png", X, T, tris, pe,
                                                      "plastic strain at collapse", "%", "magma", X[i],
                                                      f"collapse {lim['factor']:.3g}x"))
        if len(lim["lam"]) > 1:
            img["path"] = light_png(line_plot(out / "path.png", np.asarray(lim["u"]),
                                              {"elastic perfectly plastic ramp": np.asarray(lim["lam"])},
                                              "deflection at the point that moves most, mm", "load multiple",
                                              1.0, "the load"))
    except Exception as e:  # noqa: BLE001
        img["error"] = str(e)[:200]
    return img
