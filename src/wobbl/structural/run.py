"""The strength study: mesh, solve, refine where it matters, repeat until the answer stops moving."""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from ..fe import NormalConstraint, StaticResult
from ..geometry.cad import PreparedGeometry, prepare, scale_geometry
from ..geometry.faces import PartFeatures, analyze, find_sharp_corners
from ..geometry.select import Selection, SelectionError, Selector
from ..materials import Material, resolve_material
from ..mesh.mesher import Mesh, Refinement, Sizing, auto_sizing, build_mesh, build_surface_mesh
from ..solvers import base as solvers
from .build import BuiltModel, build_static
from .hotspots import Hotspot, ZoneMap, classify, find_hotspots, track
from .spec import SpecError, StructuralStudy

QUALITY = {
    "fast": dict(fineness=0.8, nodes=20000, levels=3, tol=0.05, max_nodes=60000),
    "normal": dict(fineness=1.0, nodes=35000, levels=4, tol=0.03, max_nodes=110000),
    "fine": dict(fineness=1.3, nodes=60000, levels=5, tol=0.02, max_nodes=250000),
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
    fillets = [f for f in features.fillets() if f.radius < 1.5 * h]
    for f in fillets:
        size = float(np.clip(f.radius / 2.0, h / 10.0, h / 2.0))
        s.refinements.append(Refinement(size, surfaces=[f.id], grow=4 * size, label=f"fillet {f.id}"))
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


def run_structural(study: StructuralStudy, workdir: str | Path, progress: Progress | None = None) -> StructuralRun:
    t_start = time.time()
    work = Path(workdir)
    work.mkdir(parents=True, exist_ok=True)
    q = QUALITY.get(study.quality, QUALITY["normal"])
    levels_max = min(study.max_levels, q["levels"])
    tol = q["tol"]
    material = resolve_material(study.material)
    adapter = solvers.for_domain("structural")
    adapter.require()

    _say(progress, "geometry", f"reading {Path(study.part).name}")
    prep = prepare(study.part, work / "geometry", body=study.body)
    if study.scale and study.scale != 1.0:
        prep = scale_geometry(prep, study.scale, work / "geometry")
    _say(progress, "geometry", "finding faces, holes, fillets and wall thickness")
    surf = build_surface_mesh(prep)
    features = analyze(surf)
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
    for lv in range(levels_max):
        _say(progress, "mesh", f"level {lv}: meshing (element size {sizing.h:.3g} mm)")
        mesh = build_mesh(prep, sizing)
        notes += [n for n in mesh.notes if n not in notes]
        built = build_static(mesh, loads, fixtures, material, features.faces)
        _say(progress, "solve", f"level {lv}: solving {mesh.n_nodes:,} nodes with {adapter.name}")
        res, wall_iters, wall_warm = _solve(adapter, built, work / f"level{lv}", wall_warm)
        measure, mname = failure_measure(res, material, print_axis)
        corners = find_sharp_corners(mesh)
        zmap = ZoneMap(mesh, built.zones, built.zone_kind, corners)
        if lv == 0:
            sizes = mesh.node_sizes()
            r_ex = float(max(0.2 * features.thickness_typical, 1.0 * np.nanmedian(sizes[measure > 0.5 * measure.max()])))
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
        excluded = zmap.excluded_mask(mesh.X, r_ex)
        valid = ~excluded
        if not valid.any():
            valid = np.ones(len(measure), bool)
        gi = np.nonzero(valid)[0][np.argmax(measure[valid])]
        ri = int(np.argmax(measure))
        dmag = res.displacement_magnitude()
        di = int(np.argmax(dmag))
        fapp, mapp = built.model.applied_total()
        cons = np.unique(np.concatenate([c.nodes for c in built.model.constraints] +
                                        [c.nodes for c in built.model.normal_constraints] or [np.zeros(0, int)]))
        rf = res.RF[cons].sum(0) if len(cons) else np.zeros(3)
        rm = np.cross(mesh.X[cons], res.RF[cons]).sum(0) if len(cons) else np.zeros(3)
        levels.append(Level(lv, mesh.n_nodes, mesh.n_elems, float(sizing.h),
                            float(np.nanmin(mesh.node_sizes())), mesh.t_mesh, res.t_solve,
                            float(measure[gi]), mesh.X[gi].tolist(), float(measure[ri]), mesh.X[ri].tolist(),
                            float(dmag[di]), mesh.X[di].tolist(), rf.tolist(), fapp.tolist(), rm.tolist(),
                            mapp.tolist(), wall_iters))
        last = (mesh, res, built, measure, mname, excluded)
        classify(spots, tol)
        _say(progress, "check", f"level {lv}: governing {measure[gi]:.4g} MPa"
                                + (f" ({_pct(levels)} change)" if lv else ""))
        if lv >= 1 and _converged(levels, spots, tol):
            stopped_by = "converged"
            break
        if lv == levels_max - 1:
            stopped_by = "level limit"
            break
        elapsed = time.time() - t_start
        est_next = res.t_solve * 2.6 + mesh.t_mesh * 2
        if elapsed + est_next > study.budget:
            stopped_by = "time budget"
            break
        if mesh.n_nodes * 1.8 > q["max_nodes"]:
            stopped_by = "node limit"
            break
        sizing = _refine(sizing, spots, levels, tol, lv)
    mesh, res, built, measure, mname, excluded = last
    run = StructuralRun(study, material, prep, features, sel_info, levels, spots, mesh, res, built, mname,
                        measure, excluded, r_ex, work, tol, stopped_by, time.time() - t_start, notes)
    _save_fields(run)
    return run


def _pct(levels: list[Level]) -> str:
    a, b = levels[-2].governing, levels[-1].governing
    return f"{100 * (b - a) / a:+.1f}%" if a else "n/a"


def _governing_spot(spots: list[Hotspot], xyz) -> Hotspot | None:
    feats = [s for s in spots if s.kind == "feature"]
    if not feats:
        return None
    return min(feats, key=lambda s: float(np.linalg.norm(np.asarray(s.xyz) - np.asarray(xyz))))


def _converged(levels: list[Level], spots: list[Hotspot], tol: float) -> bool:
    a, b = levels[-2].governing, levels[-1].governing
    if a <= 0 or abs(b - a) / a > tol:
        return False
    gs = _governing_spot(spots, levels[-1].governing_xyz)
    if gs is not None and gs.change is not None and abs(gs.change) > tol:
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


def _refine(sizing: Sizing, spots: list[Hotspot], levels: list[Level], tol: float, lv: int) -> Sizing:
    new = Sizing(h=sizing.h, refinements=list(sizing.refinements), order=sizing.order, curvature=sizing.curvature)
    for s in refinement_targets(spots, levels[-1].governing, lv):
        h = s.sizes[-1] if s.sizes and np.isfinite(s.sizes[-1]) else sizing.h / 4
        size = 0.5 * h
        new.refinements.append(Refinement(size, points=[tuple(s.xyz)], radius=2.5 * h, grow=6 * size,
                                          label=f"hotspot {s.id}"))
    # the far field gets a little finer too, so global load paths converge
    new.h = sizing.h * 0.85
    return new


def _solve(adapter, built: BuiltModel, workdir: Path, warm):
    """Solve, running an active set loop when some faces rest on a wall (compression only)."""
    model = built.model
    if not built.walls:
        return adapter.solve_static(model, workdir), 0, None
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
    for it in range(1, 16):
        model.normal_constraints = base_normals + [NormalConstraint(w.name, active[w.name], w.normal)
                                                   for w in built.walls if len(active[w.name])]
        res = adapter.solve_static(model, workdir)
        solved = {k: v.copy() for k, v in active.items()}
        changed = False
        rf_tol = 1e-5 * (1.0 + float(np.abs(res.RF).max()))
        u_tol = 1e-5 * (1e-12 + float(np.abs(res.U).max()))
        for w in built.walls:
            act = active[w.name]
            pull = (res.RF[act] @ w.normal) > rf_tol
            inactive = np.setdiff1d(w.nodes, act)
            pen = inactive[(res.U[inactive] @ w.normal) > u_tol]
            if pull.any() or len(pen):
                active[w.name] = np.union1d(act[~pull], pen)
                changed = True
        if not changed:
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
