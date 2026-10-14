"""Submodeling: a fine local model of the critical region, driven by the displacements of a coarse global one.

A big part needs a fine mesh only where its stress peaks. A submodel cuts a ball around the peak out of the
part, meshes it finely and holds its cut faces where the global solution says they moved, through CalculiX's
*SUBMODEL, which interpolates the global displacements at every cut node from the global elements. Loads and
supports inside the ball act on it as they act on the whole part, and a force spread over a face keeps its
pressure, so the ball carries its own share of it. Everything outside reaches the ball only through the cut.

The answer is only as good as the cut. By Saint-Venant's principle what the cut gets wrong dies away over about
the size of the feature it crosses, so the check asks that the peak sit at least one and a half feature sizes
from the cut, and it compares the stress on the cut with the global stress there. A global mesh too coarse to
be right at the cut shows up as a mismatch.
"""
from __future__ import annotations

import copy
import json
import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from .spec import SpecError

FEATURES = 2.5          # default ball radius, in feature sizes (the radius of the curve that peaks, or the wall)
CUT_CLEARANCE = 1.5     # the peak must sit this many feature sizes inside the cut
CUT_MISMATCH = 0.10     # stress on the cut may differ from the global one by this share of the local peak


class SubmodelError(SpecError):
    def __init__(self, message: str, hint: str = ""):
        super().__init__(message)
        self.hint = hint


@dataclass
class Global:
    """What a finished strength run leaves behind that a submodel needs."""
    folder: Path
    outcome: dict
    study: object
    prep: object
    fields: dict
    frd: Path
    h_peak: float
    features: object = None
    selections: list = field(default_factory=list)


class _Surface:
    """Just enough of a mesh for a selection's triangle mask: nodes, boundary triangles and their faces."""

    def __init__(self, X, tris, tri_surf):
        self.X, self.tris, self.tri_surf = X, tris, tri_surf

    def surface_mask(self, surfaces):
        return np.isin(self.tri_surf, np.asarray(list(surfaces), dtype=int))

    def tri_geometry(self, mask=None):
        from ..mesh.mesher import Mesh
        return Mesh.tri_geometry(self, mask)


def frd_nodes(path: Path) -> int:
    """The node count in a .frd header, read from its first lines only."""
    try:
        with open(path, "r", errors="replace") as fh:
            for _ in range(50):
                line = fh.readline()
                if not line:
                    break
                if line.startswith("    2C"):
                    return int(line[6:].split()[0])
    except (OSError, ValueError, IndexError):
        pass
    return -1


def load_global(result_dir: str | Path) -> Global:
    from ..geometry.cad import PreparedGeometry
    from .spec import StructuralStudy

    d = Path(result_dir)
    if not (d / "outcome.json").exists():
        raise SubmodelError(f"no Wobbl strength result in {d}", "Pass the folder a wobbl run printed")
    outcome = json.loads((d / "outcome.json").read_text())
    if outcome.get("domain") != "strength":
        raise SubmodelError("a submodel needs a strength result")
    sd = json.loads((d / "study.json").read_text())
    if sd.get("shell"):
        raise SubmodelError("this result was solved as a shell, and a submodel needs a solid result",
                            "Run the study as a solid, for example with a finer --mesh-size")
    sd["part"] = str((d / sd["part"]).resolve())
    gd = json.loads((d / "geometry.json").read_text())
    gd["source"], gd["path"] = (str((d / gd[k]).resolve()) for k in ("source", "path"))
    f = np.load(d / "work" / "fields.npz")
    fields = {k: f[k] for k in f.files}
    levels = [p for p in sorted((d / "work").glob("level*/static.frd"), key=lambda p: int(p.parent.name[5:]))
              if frd_nodes(p) == len(fields["X"])]
    if not levels:
        raise SubmodelError("the global result no longer has the CalculiX results of its final mesh "
                            "(work/level*/static.frd)", "Run the global study again and keep its work folder")
    X, tets = fields["X"], fields["tets"]
    gov = np.asarray(outcome["governing_xyz"], float)
    i = int(np.argmin(np.linalg.norm(X - gov, axis=1)))
    near = np.any(tets == i, axis=1)
    c = X[tets[near][:, :4]]
    h = float(np.mean([np.linalg.norm(c[:, a] - c[:, b], axis=1).mean() for a, b in ((0, 1), (1, 2), (0, 3))]))
    return Global(d, outcome, StructuralStudy.from_dict(sd), PreparedGeometry.from_dict(gd), fields, levels[-1], h)


def curvature_radius(X, tris, tri_surf, xyz: np.ndarray, reach: float) -> tuple[float | None, int]:
    """The tighter radius of curvature of the surface at ``xyz``, from how fast the normals of the nearby
    triangles on the same face turn, and that face. None on a flat face."""
    c = X[tris[:, :3]].mean(1)
    near = int(np.argmin(np.linalg.norm(c - xyz, axis=1)))
    face = int(tri_surf[near])
    pick = np.flatnonzero((tri_surf == face) & (np.linalg.norm(c - xyz, axis=1) <= reach))
    if len(pick) < 4:
        return None, face
    p0, p1, p2 = (X[tris[pick, k]] for k in range(3))
    n = np.cross(p1 - p0, p2 - p0)
    n /= np.maximum(np.linalg.norm(n, axis=1), 1e-30)[:, None]
    cc = c[pick]
    i, j = np.triu_indices(len(pick), 1)
    dn = np.linalg.norm(n[i] - n[j], axis=1)
    turn = dn > 0.05
    if turn.sum() < 3:
        return None, face
    return float(np.percentile(np.linalg.norm(cc[i] - cc[j], axis=1)[turn] / dn[turn], 25)), face


def feature_size(g: Global, xyz: np.ndarray) -> tuple[float, str]:
    """The size the stress concentration at ``xyz`` dies away over: the radius of the curved face it sits on, or
    else the typical wall."""
    r, face = curvature_radius(g.fields["X"], g.fields["tris"], g.fields["tri_surf"], xyz, 3.0 * g.h_peak)
    wall = float(g.features.thickness_typical)
    if r is None or r < 0.25 * wall or r > 20 * wall:
        return wall, f"the {wall:.3g} mm wall"
    return r, f"the {r:.3g} mm radius of face {face}"


def local_geometry(g: Global, center: np.ndarray, radius: float, work: Path):
    """The part cut down to a ball, prepared and described like any part."""
    from ..geometry.cad import gmsh_session, prepare
    from ..geometry.faces import analyze
    from ..mesh.mesher import build_surface_mesh

    work.mkdir(parents=True, exist_ok=True)
    with gmsh_session() as gmsh:
        gmsh.model.occ.importShapes(str(g.prep.path), highestDimOnly=True)
        gmsh.model.occ.synchronize()
        ball = gmsh.model.occ.addSphere(*center, radius)
        gmsh.model.occ.intersect(gmsh.model.getEntities(3), [(3, ball)])
        gmsh.model.occ.synchronize()
        if not gmsh.model.getEntities(3):
            raise SubmodelError("the ball around the peak holds none of the part")
        gmsh.write(str(work / "local.brep"))
    prep = prepare(work / "local.brep", work / "geometry", defeature=False)
    return prep, analyze(build_surface_mesh(prep))


def parent_faces(mesh, g: Global, center, radius) -> tuple[dict, list[int]]:
    """For each local face, the global face it lies on, by its triangles' nearest global triangles. Faces on the
    sphere are the cut and have no parent."""
    from scipy.spatial import cKDTree

    GX, gtris, gsurf = g.fields["X"], g.fields["tris"], g.fields["tri_surf"]
    gc = GX[gtris[:, :3]].mean(1)
    tree = cKDTree(gc)
    d_node = np.linalg.norm(mesh.X - center, axis=1)
    parents, cut = {}, []
    for sid in mesh.surfaces():
        nodes = mesh.surface_nodes([sid])
        if np.all(np.abs(d_node[nodes] - radius) < 1e-3 * radius + 1e-6):
            cut.append(int(sid))
            continue
        tc = mesh.X[mesh.tris[mesh.tri_surf == sid][:, :3]].mean(1)
        _, j = tree.query(tc)
        vals, counts = np.unique(gsurf[j], return_counts=True)
        parents[int(sid)] = int(vals[np.argmax(counts)])
    return parents, cut


def _local_selection(sel, parents: dict):
    from ..geometry.select import Selection

    faces = sorted(k for k, v in parents.items() if v in set(sel.faces))
    return Selection(sel.text, faces, list(sel.regions))


def _area(mesh, sel) -> float:
    m = sel.tri_mask(mesh)
    if not m.any():
        return 0.0
    _, a, _ = mesh.tri_geometry(m)
    return float(a.sum())


def local_loads(g: Global, mesh, parents: dict):
    """The study's loads and supports, cut down to the ball. A force spread over a face keeps its pressure, so the
    ball gets its share. A load that is not spread evenly (a bearing, a remote force) must lie wholly inside."""
    from .run import resolve_selections

    gsurf = _Surface(g.fields["X"], g.fields["tris"], g.fields["tri_surf"])
    loads, fixtures, _ = resolve_selections(g.study, g.features)
    out_loads, out_fix, notes = [], [], []
    for k, (ld, sel) in enumerate(loads):
        if ld.kind == "gravity":
            out_loads.append((ld, None))
            continue
        ls = _local_selection(sel, parents)
        a_loc = _area(mesh, ls) if ls.faces else 0.0
        if a_loc <= 0:
            continue
        a_glob = _area(gsurf, sel)
        share = a_loc / a_glob if a_glob > 0 else 1.0
        if ld.kind in ("bolt", "remote", "moment") and share < 0.98:
            raise SubmodelError(f"load {k + 1} ({ld.describe()}) acts partly inside the ball, and only a load "
                                "spread evenly over its faces can be cut",
                                "Grow --radius to hold all of it, or move the ball with --at")
        if ld.kind == "force" and share < 0.999:
            ld = copy.copy(ld)
            ld.magnitude *= share
            notes.append(f"{100 * share:.3g}% of load {k + 1} ({ld.describe()}) acts inside the ball.")
        out_loads.append((ld, ls))
    for k, (fx, sel) in enumerate(fixtures):
        ls = _local_selection(sel, parents)
        if not ls.faces or not _area(mesh, ls):
            continue
        if fx.kind == "wall":
            notes.append(f"Support {k + 1} ({fx.describe()}) holds the ball only where the global run found the "
                         "part pressing on it.")
        out_fix.append((fx, ls))
    return out_loads, out_fix, notes


def _walls_from_global(built, g: Global):
    """A wall inside the ball holds only where the global solution keeps the part on it."""
    from scipy.spatial import cKDTree

    from ..fe import NormalConstraint

    if not built.walls:
        return
    GX, GU = g.fields["X"], g.fields["U"]
    _, j = cKDTree(GX).query(built.model.mesh.X)
    tol = 1e-3 * float(np.abs(GU).max() + 1e-30)
    for w in built.walls:
        un = GU[j[w.nodes]] @ w.normal
        on = w.nodes[un <= tol]
        if len(on):
            built.model.normal_constraints.append(NormalConstraint(w.name, on, w.normal))
    built.walls = []


def submodel_cards(text: str, nset: str, global_frd: str) -> str:
    """A static deck rewritten so the node set ``nset`` follows the displacements of the global .frd: the set
    goes on *SUBMODEL before the step, and inside it a *BOUNDARY, SUBMODEL card drives its three directions from
    the global step 1."""
    k = text.index("*STEP")
    text = text[:k] + f"*SUBMODEL, TYPE=NODE, INPUT={global_frd}\n{nset}\n" + text[k:]
    k = text.index("*NODE FILE", k)
    return text[:k] + f"*BOUNDARY, SUBMODEL, STEP=1\n{nset},1,3\n" + text[k:]


def write_submodel_inp(path: Path, model, cut: np.ndarray, global_frd: str):
    """A static deck whose cut nodes follow the global displacements through *SUBMODEL."""
    import io

    from ..solvers.calculix import _nset, write_static_inp

    write_static_inp(path, model)
    text = path.read_text()
    buf = io.StringIO()
    _nset(buf, "NCUT", cut)
    k = text.index("*STEP")
    text = text[:k] + buf.getvalue() + text[k:]
    path.write_text(submodel_cards(text, "NCUT", global_frd))


def run_submodel(result_dir: str | Path, center=None, radius: float | None = None, size: float | None = None,
                 out: str | Path | None = None, progress: Callable[[str, str], None] | None = None) -> dict:
    """Solve a fine local model around the global peak (or ``center``) and check that its cut is far enough."""
    from scipy.spatial import cKDTree

    from ..fe import StaticResult
    from ..geometry.select import Selection
    from ..materials import resolve_material
    from ..mesh.mesher import Sizing, build_mesh
    from ..solvers import base as solvers
    from ..solvers.calculix import _run, read_frd
    from .build import build_static
    from ..geometry.faces import analyze
    from ..mesh.mesher import build_surface_mesh
    from .run import failure_measure
    from .spec import Fixture

    say = progress or (lambda a, b="": None)
    t0 = time.time()
    g = load_global(result_dir)
    say("geometry", "reading the global result")
    g.features = analyze(build_surface_mesh(g.prep))      # the very faces the global study selected on
    gov = np.asarray(g.outcome["governing_xyz"], float)
    c = np.asarray(center, float) if center is not None else gov
    feat, feat_label = feature_size(g, c)
    R = float(radius) if radius else FEATURES * feat
    h = float(size) if size else min(feat / 6.0, g.h_peak / 3.0)
    work = Path(out) if out else g.folder / "submodel"
    work.mkdir(parents=True, exist_ok=True)
    say("geometry", f"cutting a ball of radius {R:.3g} mm around ({', '.join(f'{v:.3g}' for v in c)})")
    lprep, lfeat = local_geometry(g, c, R, work)
    say("mesh", f"meshing the ball with {h:.3g} mm elements")
    mesh = build_mesh(lprep, Sizing(h=h))
    parents, cut_surfs = parent_faces(mesh, g, c, R)
    if not cut_surfs:
        raise SubmodelError("the ball holds the whole part, so there is nothing to cut",
                            "Run the study itself with a finer --mesh-size instead")
    loads, fixtures, notes = local_loads(g, mesh, parents)
    cut_nodes = mesh.surface_nodes(cut_surfs)
    held = Fixture("fixed", "cut", "the cut")
    material = resolve_material(g.study.material)
    built = build_static(mesh, loads, fixtures + [(held, Selection("cut", cut_surfs))], material, lfeat.faces)
    _walls_from_global(built, g)
    cut_c = built.model.constraints.pop()           # the cut follows the global solution, not zero
    others = [cc.nodes for cc in built.model.constraints] + [nc.nodes for nc in built.model.normal_constraints]
    if others:
        cut_nodes = np.setdiff1d(cut_c.nodes, np.concatenate(others))      # a support wins over the cut
    ccx = solvers.get("calculix").require()
    run = work / "solve"
    run.mkdir(exist_ok=True)
    link = run / "global.frd"
    link.unlink(missing_ok=True)
    try:
        os.symlink(g.frd.resolve(), link)
    except (OSError, NotImplementedError):
        shutil.copy2(g.frd, link)
    write_submodel_inp(run / "sub.inp", built.model, cut_nodes, "global.frd")
    say("solve", f"solving {mesh.n_nodes:,} nodes driven by the global displacements")
    log, t_solve = _run(ccx, run, "sub", 3600)
    blocks = read_frd(run / "sub.frd", mesh.n_nodes)
    U, S = blocks["DISP"][-1][1][:, :3], blocks["STRESS"][-1][1][:, :6]
    res = StaticResult(U, S, np.zeros_like(U), t_solve=t_solve)
    measure, mname = failure_measure(res, material)
    # the peak is read away from the cut, and away from anything the global study excluded
    d_cut = cKDTree(mesh.X[cut_nodes]).query(mesh.X)[0]
    _, gj = cKDTree(g.fields["X"]).query(mesh.X)
    excluded = g.fields.get("excluded", np.zeros(len(g.fields["X"]), bool))[gj].astype(bool)
    valid = (d_cut > 0.2 * R) & ~excluded
    if not valid.any():
        valid = d_cut > 0.2 * R
    i = int(np.flatnonzero(valid)[np.argmax(measure[valid])])
    peak, where = float(measure[i]), mesh.X[i]
    # the global stress at the cut, from the global nodes nearest each cut node
    gmeasure = g.fields["measure"][gj]
    mismatch = float(np.max(np.abs(measure[cut_nodes] - gmeasure[cut_nodes]))) / max(peak, 1e-30)
    clearance = float(d_cut[i])
    checks = [
        {"name": "cut distance", "status": "pass" if clearance >= CUT_CLEARANCE * feat else "fail",
         "summary": f"the peak sits {clearance:.3g} mm inside the cut, {clearance / feat:.2g} times {feat_label}"
                    f" (at least {CUT_CLEARANCE:g} wanted)"},
        {"name": "cut agreement", "status": "pass" if mismatch <= CUT_MISMATCH else "warn",
         "summary": f"stress on the cut differs from the global solution by at most {100 * mismatch:.3g}% of the "
                    f"local peak (up to {100 * CUT_MISMATCH:g}% expected from the coarse global mesh)"}]
    gpeak = float(g.outcome["governing_MPa"])
    out_d = {"global_peak_MPa": round(gpeak, 4), "submodel_peak_MPa": round(peak, 4), "measure": mname,
             "change_pct": round(100 * (peak - gpeak) / gpeak, 2), "where_mm": [round(v, 3) for v in where],
             "center_mm": [round(v, 3) for v in c], "radius_mm": round(R, 4), "element_size_mm": round(h, 4),
             "global_element_size_mm": round(g.h_peak, 4), "nodes": mesh.n_nodes, "cut_nodes": int(len(cut_nodes)),
             "feature": feat_label, "checks": checks, "notes": notes,
             "trusted": all(ch["status"] == "pass" for ch in checks), "time_s": round(time.time() - t0, 1),
             "folder": str(work)}
    np.savez_compressed(work / "fields.npz", X=mesh.X, tets=mesh.tets, tris=mesh.tris, U=U, S=S, measure=measure,
                        cut=cut_nodes)
    (work / "submodel.json").write_text(json.dumps(out_d, indent=1))
    for ext in (".frd", ".dat", ".sta", ".cvg", ".12d"):
        if ext != ".frd":
            (run / f"sub{ext}").unlink(missing_ok=True)
    return out_d
