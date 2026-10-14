"""The strength study: mesh, solve, refine where it matters, repeat until the answer stops moving."""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Callable

import numpy as np

from ..fe import NormalConstraint, StaticResult
from ..geometry.cad import PreparedGeometry, prepare, scale_geometry
from ..geometry.faces import PartFeatures, analyze, find_sharp_corners
from ..geometry.select import Selection, SelectionError, Selector
from ..materials import Material, resolve_material
from ..mesh.isolate import IsolatedCrash, IsolatedTimeout
from ..mesh.mesher import (Mesh, MeshError, Refinement, Sizing, auto_sizing, build_mesh, build_surface_mesh,
                           requested_size)
from ..solvers.base import SolverError
from ..solvers import base as solvers
from .budget import Budget, mesh_cost, solve_cost
from .build import BuiltModel, build_static
from .hotspots import Hotspot, ZoneMap, classify, find_hotspots, track
from .spec import SpecError, StructuralStudy

QUALITY = {
    "fast": dict(fineness=0.8, nodes=20000, levels=3, tol=0.05, max_nodes=80000),
    "normal": dict(fineness=1.0, nodes=35000, levels=5, tol=0.03, max_nodes=250000),
    "fine": dict(fineness=1.3, nodes=60000, levels=5, tol=0.02, max_nodes=400000),
}

Progress = Callable[[str, str], None]


@dataclass
class Level:
    level: int
    n_nodes: int
    n_elems: int
    h_global: float
    h_min: float
    t_mesh: float
    t_solve: float
    governing: float
    governing_xyz: list[float]
    raw_max: float
    raw_max_xyz: list[float]
    max_disp: float
    max_disp_xyz: list[float]
    reaction: list[float]
    applied: list[float]
    reaction_moment: list[float]
    applied_moment: list[float]
    wall_iterations: int = 0


@dataclass
class StructuralRun:
    """Everything the verification and report layers need."""
    study: StructuralStudy
    material: Material
    prep: PreparedGeometry
    features: PartFeatures
    selections: dict[str, dict]
    levels: list[Level]
    hotspots: list[Hotspot]
    mesh: Mesh
    result: StaticResult
    built: BuiltModel
    measure_name: str
    measure: np.ndarray
    excluded: np.ndarray
    exclusion_radius: float
    workdir: Path
    tol: float
    stopped_by: str = ""
    t_total: float = 0.0
    notes: list[str] = field(default_factory=list)
    nonlinear: dict | None = None
    load_path: dict | None = None
    budget: dict = field(default_factory=dict)   # the budget, the modeled seconds spent and the clock


def _say(progress: Progress | None, stage: str, detail: str = ""):
    if progress:
        progress(stage, detail)


def failure_measure(res: StaticResult, material: Material, print_axis: np.ndarray | None = None):
    """The stress Wobbl compares with the material limit, plus its name."""
    if material.ductile:
        m, name = res.von_mises(), "von Mises stress"
    else:
        m, name = np.maximum(res.principal()[:, 2], 0.0), "max principal stress"
    if print_axis is not None and material.layer_factor < 1.0:
        a = print_axis / np.linalg.norm(print_axis)
        sxx, syy, szz, sxy, syz, szx = res.S.T
        sn = (a[0] ** 2 * sxx + a[1] ** 2 * syy + a[2] ** 2 * szz
              + 2 * (a[0] * a[1] * sxy + a[1] * a[2] * syz + a[0] * a[2] * szx))
        across = np.maximum(sn, 0.0) / material.layer_factor   # scaled so one allowable applies everywhere
        m = np.maximum(m, across)
        name += " (layer pull scaled by layer strength)"
    return m, name


def resolve_selections(study: StructuralStudy, features: PartFeatures):
    sel = Selector(features.faces, features.bbox)
    loads, fixtures, info = [], [], {}
    for k, ld in enumerate(study.loads):
        s = None
        if ld.kind != "gravity":
            try:
                s = sel.select(ld.faces)
            except SelectionError as e:
                raise SpecError(f"load {k + 1} ({ld.describe()}): {e}") from None
            info[f"load{k + 1}"] = s.to_dict()
        loads.append((ld, s))
    for k, fx in enumerate(study.fixtures):
        try:
            s = sel.select(fx.faces)
        except SelectionError as e:
            raise SpecError(f"fixture {k + 1} ({fx.describe()}): {e}") from None
        fixtures.append((fx, s))
        info[f"fix{k + 1}"] = s.to_dict()
    if not fixtures:
        raise SpecError("no fixtures: say what holds the part, for example --fix 'holes d=4.5' or --fix 'x=min'")
    if not loads:
        raise SpecError("no loads: add one, for example --load '200 N down @ top'")
    return loads, fixtures, info


def initial_sizing(prep: PreparedGeometry, features: PartFeatures, study: StructuralStudy, q: dict,
                   loads) -> Sizing:
    t = features.thickness_typical
    s = auto_sizing(prep, thickness=t, fineness=q["fineness"], max_nodes=q["nodes"])
    if study.mesh_size:
        s.h = float(study.mesh_size)
    h = s.h
    s.curvature = 0
    s.refinements += feature_refinements(features, h)
    curves = sorted({c for cn in features.corners for c in cn.curves})
    if curves:
        s.refinements.append(Refinement(h / 3.0, curves=curves, grow=2 * h, label="sharp corners"))
    for ld, sel in loads:
        if sel is None:
            continue
        for reg in sel.regions:
            if reg[0] == "sphere":
                size = min(h, reg[2] / 4.0)
                s.refinements.append(Refinement(size, points=[tuple(reg[1])], radius=reg[2], grow=2 * h,
                                                label="load patch"))
    s.h_min = 0.0
    return s


def feature_refinements(features: PartFeatures, h: float) -> list[Refinement]:
    """Resolve curved features on the first mesh, but never finer than a floor tied to the wall thickness.

    Curvature based sizing would mesh every tiny fillet, slot and logo edge finely and blow a thin walled
    enclosure up to half a million nodes. Here each curved face gets about half its radius, clamped to the
    floor, and faces are grouped by size so gmsh evaluates only a handful of fields. The hotspot loop
    later goes below the floor exactly where the stress is.
    """
    floor = max(h / 6.0, 0.4 * features.thickness_min)
    buckets: dict[float, list[int]] = {}
    for f in features.faces.values():
        if f.kind == "plane":
            continue
        size = 0.5 * f.radius if f.radius else h / 2.0
        size = float(np.clip(size, floor, h))
        if size >= 0.95 * h:
            continue
        key = float(floor * 1.3 ** np.floor(np.log(size / floor) / np.log(1.3)))
        buckets.setdefault(key, []).append(f.id)
    return [Refinement(size, surfaces=ids, grow=3 * size, label=f"curved features {size:.3g} mm", corners_only=True)
            for size, ids in sorted(buckets.items())]


def exclusion_radius(features: PartFeatures, sizing: Sizing, mesh: Mesh, hot: np.ndarray) -> float:
    """How far from a sharp corner or an idealized support the reported stress is taken.

    A fifth of the typical wall, or one element of the first mesh where the part is most stressed, whichever is
    larger. The element size is the one the sizing rules ask for there, a property of the geometry, rather than
    the size of whichever elements gmsh made: a peak that sits at this radius (a rigid attachment beside the
    hole it grabs) otherwise moves with every small change in the mesh.
    """
    return float(max(0.2 * features.thickness_typical, np.median(requested_size(sizing, mesh, hot))))


def describe_part(path: str, workdir: str | Path, scale: float = 1.0, body: int | None = None,
                  params: dict | None = None):
    """Prepare the CAD (building parametric .py parts first) and analyze its faces. Returns (prepared, features)."""
    from ..geometry.parametric import build_step, is_parametric

    work = Path(workdir)
    if is_parametric(path):
        step, used = build_step(path, params, work / "geometry")
        prep = prepare(step, work / "geometry", body=body)
        prep.notes.append("Built from " + Path(path).name + " with " +
                          ", ".join(f"{k}={v:g}" if isinstance(v, (int, float)) else f"{k}={v}"
                                    for k, v in used.items()) + ".")
    else:
        prep = prepare(path, work / "geometry", body=body)
    if scale and scale != 1.0:
        prep = scale_geometry(prep, scale, work / "geometry")
    return prep, analyze(build_surface_mesh(prep))


def run_structural(study: StructuralStudy, workdir: str | Path, progress: Progress | None = None,
                   pre: tuple | None = None) -> StructuralRun:
    t_start = time.time()
    budget = Budget(study.budget, t_start)
    work = Path(workdir)
    work.mkdir(parents=True, exist_ok=True)
    q = QUALITY.get(study.quality, QUALITY["normal"])
    levels_max = min(study.max_levels, q["levels"])
    tol = q["tol"]
    material = resolve_material(study.material)
    adapter = solvers.for_domain("structural")
    adapter.require()

    if pre is not None:
        prep, features = pre
    else:
        _say(progress, "geometry", f"reading {Path(study.part).name}, finding faces, holes and fillets")
        prep, features = describe_part(study.part, work, study.scale, study.body, study.params)
    loads, fixtures, sel_info = resolve_selections(study, features)
    print_axis = None
    if study.print_direction:
        from ..geometry.select import parse_vector
        print_axis = parse_vector(study.print_direction)

    sizing = initial_sizing(prep, features, study, q, loads)
    levels: list[Level] = []
    spots: list[Hotspot] = []
    r_ex = None
    stopped_by = "converged"
    last = None
    wall_warm = None
    notes: list[str] = list(prep.notes)
    if any(fx.kind == "wall" for fx, _ in fixtures):
        _say(progress, "contact", "finding where the part presses on its wall support (coarse pre-solve)")
        wall_warm = _contact_presolve(prep, sizing, loads, fixtures, material, features, adapter,
                                      work / "contact", budget)
    for lv in range(levels_max):
        try:
            _say(progress, "mesh", f"level {lv}: meshing (element size {sizing.h:.3g} mm)")
            mesh = build_mesh(prep, sizing)
            cap = 2.0 * features.thickness_min  # never coarser than this: thin walls and struts need real elements
            if lv == 0 and mesh.n_nodes > 1.6 * q["nodes"] and not study.mesh_size and sizing.h < cap:
                # small features made the first mesh too big to solve quickly: coarsen once, by the right amount
                factor = min((mesh.n_nodes / q["nodes"]) ** (1 / 3), cap / sizing.h)
                kept = (sizing.h, [r.size for r in sizing.refinements])
                sizing.h *= factor
                for r in sizing.refinements:
                    if r.corners_only:
                        r.size *= factor
                _say(progress, "mesh", f"level 0 had {mesh.n_nodes:,} nodes; coarsening to {sizing.h:.3g} mm")
                budget.charge(mesh_cost(mesh.n_nodes))
                try:
                    coarse = build_mesh(prep, sizing)
                except MeshError:
                    coarse = None
                if coarse is not None and coarse.n_nodes < mesh.n_nodes:
                    mesh = coarse
                else:
                    # the coarser size folded a thin curved wall and only meshed finer than before, if at all;
                    # the first mesh worked, so keep it
                    budget.charge(mesh_cost(coarse.n_nodes) if coarse is not None else 0.0)
                    sizing.h = kept[0]
                    for r, size in zip(sizing.refinements, kept[1]):
                        r.size = size
                    notes.append(f"A coarser first mesh folded on the thin walls, so the study starts from the finer "
                                 f"one ({mesh.n_nodes:,} nodes).")
            if lv == 0 and mesh.n_nodes > 2 * q["nodes"]:
                notes.append(f"Thin walls need a large first mesh ({mesh.n_nodes:,} nodes), so this study takes longer "
                             "than usual.")
            notes += [n for n in mesh.notes if n not in notes]
            built = build_static(mesh, loads, fixtures, material, features.faces)
            _say(progress, "solve", f"level {lv}: solving {mesh.n_nodes:,} nodes with {adapter.name}")
            res, wall_iters, wall_warm = _solve(adapter, built, work / f"level{lv}", wall_warm,
                                                timeout=None if lv == 0 else budget.timeout())
        except (MeshError, SolverError, IsolatedTimeout, IsolatedCrash) as e:
            if lv == 0:
                raise
            # never come back empty handed: the last finished level is still a checked answer
            notes.append(f"Refinement level {lv} could not finish ({str(e)[:160]}); the answer comes from "
                         f"level {lv - 1}.")
            stopped_by = f"level {lv} failed"
            break
        measure, mname = failure_measure(res, material, print_axis)
        corners = find_sharp_corners(mesh, faces=features.faces, tangent=features.tangent_pairs)
        zmap = ZoneMap(mesh, built.zones, built.zone_kind, corners)
        if lv == 0:
            r_ex = exclusion_radius(features, sizing, mesh, mesh.X[measure > 0.5 * measure.max()])
        spots = _update_spots(spots, lv, mesh, measure, zmap, features, prep)
        excluded = zmap.excluded_mask(mesh.X, r_ex)
        levels.append(_record_level(lv, mesh, sizing, res, built, measure, excluded, wall_iters))
        budget.charge_level(mesh.n_nodes, wall_iters)
        last = (mesh, res, built, measure, mname, excluded)
        if lv == 0:
            first = (built, levels[0].governing, levels[0].max_disp, excluded, sizing)
        classify(spots, tol)
        _say(progress, "check", f"level {lv}: governing {levels[-1].governing:.4g} MPa"
                                + (f" ({_pct(levels)} change)" if lv else ""))
        stopped_by = _stop_reason(levels, spots, tol, 2 if study.quality == "fast" else 3, levels_max, q, budget)
        if stopped_by:
            break
        sizing = _refine(sizing, spots, levels, tol, lv, features)
    mesh, res, built, measure, mname, excluded = last
    if stopped_by == "clock limit":
        notes.append("Refinement stopped at the wall clock backstop (three times the budget): this machine ran "
                     "slower than the budget's cost model, so here the level reached depends on the machine. A "
                     "longer --budget reaches a finer mesh.")
    nonlinear = _large_deflection_check(adapter, first, levels, features, prep, material, print_axis, work,
                                        progress, budget, loads, fixtures, r_ex)
    load_path = _load_path_check(adapter, study, q, prep, sizing, mesh, loads, fixtures, material, features,
                                 print_axis, r_ex, tol, work, progress, levels[-1],
                                 _governing_spot(spots, levels[-1].governing_xyz), budget)
    run = StructuralRun(study, material, prep, features, sel_info, levels, spots, mesh, res, built, mname,
                        measure, excluded, r_ex, work, tol, stopped_by, time.time() - t_start, notes, nonlinear,
                        load_path, {"seconds": budget.seconds, "spent": round(budget.spent, 1),
                                    "clock": round(budget.clock(), 1)})
    _save_fields(run)
    return run


def support_reactions(built: BuiltModel, res: StaticResult, mesh: Mesh) -> tuple[np.ndarray, np.ndarray]:
    """Total force and moment the supports exert, counting only the directions each support holds.

    CalculiX reports nodal forces for every direction of a constrained node; at a node that is held in one
    direction and loaded in another, the free direction carries the applied load, not a reaction.
    """
    held = np.zeros((mesh.n_nodes, 3), bool)
    for c in built.model.constraints:
        for d in c.dofs:
            held[c.nodes, d - 1] = True
    for nc in built.model.normal_constraints:
        nz = np.nonzero(np.abs(np.asarray(nc.normal)) > 1e-9)[0]
        if len(nz) == 1:
            held[nc.nodes, nz[0]] = True
        else:  # an inclined roller carries its reaction along the normal through constraint equations
            held[nc.nodes, :] = True
    R = np.where(held, res.RF, 0.0)
    return R.sum(0), np.cross(mesh.X, R).sum(0)


def _update_spots(spots: list[Hotspot], lv: int, mesh: Mesh, measure: np.ndarray, zmap: ZoneMap,
                  features: PartFeatures, prep: PreparedGeometry) -> list[Hotspot]:
    """Find the peaks on the first mesh, add any the second mesh resolves for the first time, then record
    every peak's value on this mesh."""
    if lv == 0:
        spots = find_hotspots(mesh, measure, zmap, features.faces, prep.diag)
    elif lv == 1:
        fresh = find_hotspots(mesh, measure, zmap, features.faces, prep.diag)
        for f in fresh:
            if all(np.linalg.norm(np.asarray(f.xyz) - np.asarray(s.xyz)) > 2 * s.radius for s in spots):
                f.id = len(spots) + 1
                f.values = [float("nan")]
                f.sizes = [float("nan")]
                spots.append(f)
    track(spots, mesh, measure)
    return spots


def _record_level(lv: int, mesh: Mesh, sizing: Sizing, res: StaticResult, built: BuiltModel, measure: np.ndarray,
                  excluded: np.ndarray, wall_iters: int = 0) -> Level:
    """The governing stress (away from excluded zones), raw peak, deflection and balance on one mesh."""
    valid = ~excluded
    if not valid.any():
        valid = np.ones(len(measure), bool)
    gi = np.nonzero(valid)[0][np.argmax(measure[valid])]
    ri = int(np.argmax(measure))
    dmag = res.displacement_magnitude()
    di = int(np.argmax(dmag))
    fapp, mapp = built.model.applied_total()
    rf, rm = support_reactions(built, res, mesh)
    return Level(lv, mesh.n_nodes, mesh.n_elems, float(sizing.h), float(np.nanmin(mesh.node_sizes())), mesh.t_mesh,
                 res.t_solve, float(measure[gi]), mesh.X[gi].tolist(), float(measure[ri]), mesh.X[ri].tolist(),
                 float(dmag[di]), mesh.X[di].tolist(), rf.tolist(), fapp.tolist(), rm.tolist(), mapp.tolist(),
                 wall_iters)


def _stop_reason(levels: list[Level], spots: list[Hotspot], tol: float, min_levels: int, levels_max: int, q: dict,
                 budget: Budget) -> str:
    """Why refinement stops after the latest level, or an empty string to refine again.

    Every rule but the last depends only on the meshes and the answers, so a study stops at the same level on
    every run. The wall clock backstop catches a machine far slower than the budget's cost model.
    """
    L = levels[-1]
    if len(levels) >= 2 and _converged(levels, spots, tol, min_levels):
        return "converged"
    if len(levels) >= levels_max:
        return "level limit"
    if not budget.next_level_fits(L.n_nodes):
        return "time budget"
    growth = L.n_nodes / levels[-2].n_nodes if len(levels) > 1 else 1.25
    if L.n_nodes * max(growth, 1.15) > q["max_nodes"]:
        return "node limit"
    if budget.clock_ran_out():
        return "clock limit"
    return ""


def _pct(levels: list[Level]) -> str:
    a, b = levels[-2].governing, levels[-1].governing
    return f"{100 * (b - a) / a:+.1f}%" if a else "n/a"


def _governing_spot(spots: list[Hotspot], xyz) -> Hotspot | None:
    feats = [s for s in spots if s.kind == "feature"]
    if not feats:
        return None
    return min(feats, key=lambda s: float(np.linalg.norm(np.asarray(s.xyz) - np.asarray(xyz))))


def _converged(levels: list[Level], spots: list[Hotspot], tol: float, min_levels: int = 3) -> bool:
    if len(levels) < min_levels:  # two agreeing meshes can both be too coarse; three show the trend
        return False
    a, b = levels[-2].governing, levels[-1].governing
    if a <= 0 or abs(b - a) / a > tol:
        return False
    gs = _governing_spot(spots, levels[-1].governing_xyz)
    if gs is not None and gs.change is not None and abs(gs.change) > 2 * tol:
        return False
    return True


def refinement_targets(spots: list[Hotspot], governing: float, lv: int) -> list[Hotspot]:
    """The peaks worth refining: high, not yet settled, at most one per CAD face and three in all."""
    out, faces = [], set()
    for s in sorted([s for s in spots if s.kind == "feature"], key=lambda s: -s.value):
        if s.value < 0.85 * governing or s.converged:
            continue
        if s.face in faces:
            continue
        faces.add(s.face)
        out.append(s)
        if len(out) == 3:
            break
    if not out:
        feats = sorted([s for s in spots if s.kind == "feature"], key=lambda s: -s.value)
        out = feats[:1]
    if lv == 0:  # refine the worst idealized peak once, to show whether it is singular
        zone = sorted([s for s in spots if s.kind != "feature"], key=lambda s: -s.value)
        if zone and zone[0].value > governing:
            out.append(zone[0])
    return out


def _refine(sizing: Sizing, spots: list[Hotspot], levels: list[Level], tol: float, lv: int,
            features: PartFeatures | None = None) -> Sizing:
    new = Sizing(h=sizing.h, refinements=list(sizing.refinements), order=sizing.order, curvature=sizing.curvature)
    done_faces = set()
    for s in refinement_targets(spots, levels[-1].governing, lv):
        h = s.sizes[-1] if s.sizes and np.isfinite(s.sizes[-1]) else sizing.h / 4
        size = 0.5 * h
        face = features.faces.get(s.face) if (features and s.face is not None) else None
        if face is not None and face.radius and s.kind == "feature":
            # jump straight toward a size that resolves the curve, at most about seven times finer per level
            size = max(min(size, face.radius / 4.0), 0.15 * h)
        tight = face is not None and features is not None and face.role == "fillet" and \
            face.radius < 0.25 * features.thickness_typical
        if tight and s.kind == "feature":
            # a peak on a fillet runs along it: refine the whole fillet so neighbours do not trade places
            if face.id not in done_faces:
                new.refinements.append(Refinement(size, surfaces=[face.id], grow=6 * size,
                                                  label=f"fillet {face.id} at hotspot {s.id}"))
                done_faces.add(face.id)
            continue
        new.refinements.append(Refinement(size, points=[tuple(s.xyz)], radius=2.5 * h, grow=6 * size,
                                          label=f"hotspot {s.id}"))
    # the far field gets a little finer too, so global load paths converge
    new.h = sizing.h * 0.85
    return new


def _peak_refinement(mesh: Mesh, spots: list[Hotspot], xyz) -> list[Refinement]:
    """Refine around a governing point that no tracked feature peak covers, such as one just outside the zone
    a rigid attachment grabs; the hotspot targets alone would leave it on the coarse far field mesh."""
    xyz = np.asarray(xyz, float)
    gs = _governing_spot(spots, xyz)
    if gs is not None and np.linalg.norm(np.asarray(gs.xyz) - xyz) <= gs.radius:
        return []
    h = float(mesh.node_sizes()[int(np.argmin(np.linalg.norm(mesh.X - xyz, axis=1)))])
    if not np.isfinite(h) or h <= 0:
        return []
    return [Refinement(0.5 * h, points=[tuple(xyz)], radius=2.5 * h, grow=3 * h, label="governing point")]


NLGEOM_SOLVES = 16     # modeled linear solves in a large deflection re-solve (two load steps of a few iterations)
NLGEOM_NODES = 20000   # the re-solve runs on a mesh about this size: it measures a ratio, not the answer


def _large_deflection_check(adapter, first, levels, features, prep, material, print_axis, work, progress,
                            budget: Budget, loads, fixtures, r_ex: float):
    """When the part deflects a good fraction of its wall thickness, linear theory misses membrane
    stiffening and changing geometry. Solve one mesh both ways and report how much the peak and the
    deflection move; the converged answer is scaled by that ratio.

    The first mesh is used when it is small. A large one is coarsened to about NLGEOM_NODES first: every
    load step of a nonlinear solve costs a few linear solves, and on a 77,000 node drone arm the first mesh
    was still running after 17 minutes. Whether the check runs is decided from the modeled cost, never by a timeout,
    which would make it pass on one run and go missing on the next.
    """
    built, lin_peak, lin_disp, excluded, sizing0 = first
    d = levels[-1].max_disp
    if built.walls or (d < 0.3 * features.thickness_min and d < 0.03 * prep.diag):
        return None
    n0 = levels[0].n_nodes
    coarse = n0 > 1.3 * NLGEOM_NODES
    n = NLGEOM_NODES if coarse else n0
    cost = NLGEOM_SOLVES * solve_cost(n) + (mesh_cost(n) + solve_cost(n) if coarse else 0.0)
    if budget.spent + cost > 1.5 * budget.seconds:
        return {"error": "time budget used up"}
    _say(progress, "check", f"deflection {d:.3g} mm is large next to the {features.thickness_min:.3g} mm wall; "
                            "re-solving with large deflection theory")
    budget.charge(cost)
    timeout = max(budget.timeout(), 4 * cost)
    try:
        if coarse:
            # refinement sizes are floored by the walls, so the node count falls slower than the size cubed
            k = (n0 / NLGEOM_NODES) ** 0.5
            mesh = build_mesh(prep, Sizing(h=sizing0.h * k, order=sizing0.order, curvature=sizing0.curvature,
                                           refinements=[replace(r, size=r.size * k, grow=(r.grow or 6 * r.size) * k)
                                                        for r in sizing0.refinements]))
            built = build_static(mesh, loads, fixtures, material, features.faces)
            lin = adapter.solve_static(built.model, work / "nlgeom-linear", timeout=timeout)
            m, _ = failure_measure(lin, material, print_axis)
            zmap = ZoneMap(mesh, built.zones, built.zone_kind,
                           find_sharp_corners(mesh, faces=features.faces, tangent=features.tangent_pairs))
            excluded = zmap.excluded_mask(mesh.X, r_ex)
            valid = ~excluded if (~excluded).any() else np.ones(len(m), bool)
            lin_peak, lin_disp = float(m[valid].max()), float(lin.displacement_magnitude().max())
        res = adapter.solve_static(built.model, work / "nlgeom", nlgeom=True, timeout=timeout)
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)[:200]}
    m, _ = failure_measure(res, material, print_axis)
    valid = ~excluded if (~excluded).any() else np.ones(len(m), bool)
    return {"linear_peak": lin_peak, "nonlinear_peak": float(m[valid].max()), "linear_disp": lin_disp,
            "nonlinear_disp": float(res.displacement_magnitude().max()), "nodes": built.model.mesh.n_nodes,
            "own_mesh": coarse}


def variant_needs_refinement(primary: Level, variant: Level, primary_spot: Hotspot | None, allowable: float,
                             target: float) -> str:
    """Why the other load attachment needs refinement of its own, or an empty string when the mesh refined for
    the main answer already resolves its peak."""
    def holds(s: float) -> bool:
        return (allowable / s if s > 0 else float("inf")) >= target

    if holds(variant.governing) != holds(primary.governing):
        return "the verdict flips"
    reach = 2.5 * primary_spot.radius if primary_spot is not None else 0.0
    if np.linalg.norm(np.asarray(variant.governing_xyz) - np.asarray(primary.governing_xyz)) > reach:
        return "its peak sits away from the converged one"
    return ""


def _load_path_check(adapter, study, q, prep, sizing, mesh, loads, fixtures, material, features, print_axis, r_ex,
                     tol, work, progress, primary: Level, primary_spot: Hotspot | None, budget: Budget):
    """Loads that act off the part (a belt pull at a pulley, thrust on a motor) can enter it through a rigid
    attachment or spread over the faces. The two idealizations can give very different peaks, so solve the
    other one too and report the spread.

    The other idealization is first solved on the final mesh. When the verdict flips or its peak sits away from
    the converged one, that mesh says little about it, so it gets its own refinement loop around its own peaks,
    with the same stopping rules and time budget as the main study.
    """
    import copy

    remote = [i for i, (ld, _) in enumerate(loads) if ld.kind in ("remote", "moment")]
    if not remote or any(fx.kind == "wall" for fx, _ in fixtures):
        return None
    if budget.spent > 1.3 * budget.seconds:
        return {"skipped": "time budget"}
    alt = []
    for i, (ld, sel) in enumerate(loads):
        ld2 = copy.copy(ld)
        if i in remote:
            ld2.coupling = "distributed" if ld.coupling == "rigid" else "rigid"
        alt.append((ld2, sel))
    other = alt[remote[0]][0].coupling
    _say(progress, "check", f"re-solving with a {other} load attachment to see how much the answer depends on it")
    try:
        out = _converge_variant(adapter, study, q, prep, sizing, mesh, alt, fixtures, material, features,
                                print_axis, r_ex, tol, work / "load_path", progress, primary, primary_spot, budget,
                                other)
    except Exception as e:  # noqa: BLE001  the comparison is a bonus, never a reason to fail
        return {"error": str(e)[:200]}
    return {"primary": loads[remote[0]][0].coupling, "alternative": other, "primary_peak": primary.governing, **out}


def _converge_variant(adapter, study, q, prep, sizing, mesh, loads, fixtures, material, features, print_axis, r_ex,
                      tol, work, progress, primary: Level, primary_spot: Hotspot | None, budget: Budget,
                      label: str) -> dict:
    """Solve a variant of the study on the final mesh, then refine around its own peaks until they settle."""
    levels_max = min(study.max_levels, q["levels"])
    target = study.target_sf or material.recommended_sf
    levels: list[Level] = []
    spots: list[Hotspot] = []
    why, stopped_by = "", ""
    for lv in range(levels_max):
        try:
            if lv:
                _say(progress, "check", f"{label} attachment, level {lv}: meshing (element size {sizing.h:.3g} mm)")
                mesh = build_mesh(prep, sizing)
            built = build_static(mesh, loads, fixtures, material, features.faces)
            res, _, _ = _solve(adapter, built, work / f"level{lv}", None, timeout=None if lv == 0 else budget.timeout())
        except Exception as e:  # noqa: BLE001  a failed refinement keeps the last finished level
            if lv == 0:
                raise
            stopped_by = f"level {lv} failed ({str(e)[:80]})"
            break
        measure, _ = failure_measure(res, material, print_axis)
        zmap = ZoneMap(mesh, built.zones, built.zone_kind,
                       find_sharp_corners(mesh, faces=features.faces, tangent=features.tangent_pairs))
        spots = _update_spots(spots, lv, mesh, measure, zmap, features, prep)
        levels.append(_record_level(lv, mesh, sizing, res, built, measure, zmap.excluded_mask(mesh.X, r_ex)))
        budget.charge_level(mesh.n_nodes, meshed=lv > 0)
        # the variant's field, so readings from a test can be compared with both idealizations
        np.savez_compressed(work / "fields.npz", X=mesh.X, U=res.U, S=res.S, measure=measure, tris=mesh.tris,
                            tri_surf=mesh.tri_surf)
        classify(spots, tol)
        if lv == 0:
            why = variant_needs_refinement(primary, levels[0], primary_spot, material.allowable, target)
            if not why:
                stopped_by = "not needed"
                break
        _say(progress, "check", f"{label} attachment, level {lv}: governing {levels[-1].governing:.4g} MPa"
                                + (f" ({_pct(levels)} change)" if lv else f" ({why}, refining around it)"))
        stopped_by = _stop_reason(levels, spots, tol, 2 if study.quality == "fast" else 3, levels_max, q, budget)
        if stopped_by:
            break
        # lv + 1: only the variant's governing stress is reported, so its idealized zone peaks are not chased
        sizing = _refine(sizing, spots, levels, tol, lv + 1, features)
        sizing.refinements += _peak_refinement(mesh, spots, levels[-1].governing_xyz)
    L = levels[-1]
    spot = _governing_spot(spots, L.governing_xyz)
    return {"alternative_peak": L.governing, "alternative_xyz": L.governing_xyz, "alternative_raw_max": L.raw_max,
            "alternative_spot": {"xyz": spot.xyz, "radius": spot.radius, "detail": spot.detail} if spot else None,
            "alternative_levels": [{"level": x.level, "n_nodes": x.n_nodes, "h_min": x.h_min,
                                    "governing": x.governing, "governing_xyz": x.governing_xyz} for x in levels],
            "refined": bool(why), "refine_reason": why, "converged": stopped_by == "converged",
            "stopped_by": stopped_by, "tolerance": tol}


def _contact_presolve(prep, sizing, loads, fixtures, material, features, adapter, workdir, budget: Budget):
    """Find the contact patch on a coarse linear mesh, where each iteration costs a fraction of a second.

    The patch shape depends on the overall stiffness of the part, which a coarse mesh captures well;
    the real levels then start from it and usually settle in one or two solves.
    """
    coarse = Sizing(h=sizing.h * 0.8, order=1, curvature=12)
    try:
        mesh = build_mesh(prep, coarse)
        built = build_static(mesh, loads, fixtures, material, features.faces)
        _, iters, warm = _solve(adapter, built, workdir, None)
        budget.charge_level(mesh.n_nodes, iters)
        return warm
    except Exception:  # noqa: BLE001  the pre-solve is only a head start
        return None


def _solve(adapter, built: BuiltModel, workdir: Path, warm, timeout: float | None = None):
    """Solve, running an active set loop when some faces rest on a wall (compression only)."""
    model = built.model
    kw = {"timeout": timeout} if timeout else {}
    if not built.walls:
        return adapter.solve_static(model, workdir, **kw), 0, None
    base_normals = list(model.normal_constraints)
    active = {}
    for w in built.walls:
        if warm is not None and w.name in warm:
            from scipy.spatial import cKDTree
            pts, flags = warm[w.name]
            _, j = cKDTree(pts).query(model.mesh.X[w.nodes])
            act = w.nodes[flags[j]]
            active[w.name] = act if len(act) else w.nodes
        else:
            active[w.name] = w.nodes
    res = None
    it = 0
    solved = dict(active)
    for it in range(1, 9):
        model.normal_constraints = base_normals + [NormalConstraint(w.name, active[w.name], w.normal)
                                                   for w in built.walls if len(active[w.name])]
        res = adapter.solve_static(model, workdir, **kw)
        solved = {k: v.copy() for k, v in active.items()}
        changed = 0
        total = sum(len(w.nodes) for w in built.walls)
        rf_tol = 1e-4 * (1.0 + float(np.abs(res.RF).max()))
        u_tol = 1e-4 * (1e-12 + float(np.abs(res.U).max()))
        for w in built.walls:
            act = active[w.name]
            pull = (res.RF[act] @ w.normal) > rf_tol
            inactive = np.setdiff1d(w.nodes, act)
            pen = inactive[(res.U[inactive] @ w.normal) > u_tol]
            if pull.any() or len(pen):
                active[w.name] = np.union1d(act[~pull], pen)
                changed += int(pull.sum()) + len(pen)
        # a handful of nodes flipping at the rim of the contact patch does not move the answer
        if changed <= max(2, 0.02 * total):
            break
    model.normal_constraints = base_normals + [NormalConstraint(w.name, solved[w.name], w.normal)
                                               for w in built.walls if len(solved[w.name])]
    active = solved
    for w in built.walls:  # the zone where an idealized support acts is the contact patch
        built.zones[w.name] = active[w.name]
    warm = {w.name: (model.mesh.X[w.nodes], np.isin(w.nodes, active[w.name])) for w in built.walls}
    return res, it, warm


def _save_fields(run: StructuralRun):
    out = run.workdir
    np.savez_compressed(out / "fields.npz", X=run.mesh.X, tets=run.mesh.tets, U=run.result.U, S=run.result.S,
                        measure=run.measure, excluded=run.excluded, tris=run.mesh.tris, tri_surf=run.mesh.tri_surf)
    lv = [asdict(x) for x in run.levels]
    (out / "levels.json").write_text(json.dumps(lv, indent=1))
    (out / "budget.json").write_text(json.dumps(run.budget, indent=1))
