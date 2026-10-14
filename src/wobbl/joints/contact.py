"""Contact: parts that press on each other, slide and open apart.

"Does the clamp hold the rod, and how hard does the jaw press?" needs the two parts to stay two parts. Wobbl keeps the
bodies of an assembly apart where they touch, pairs the faces that lie on each other, and solves surface to surface
penalty contact with Coulomb friction in CalculiX. The usual traps are dealt with before the solve: gaps and overlaps
that are only CAD tolerance start in exact contact, bodies held by nothing but contact get springs far too soft to
matter so they cannot drift before it closes, and the penalty stiffness follows the element size. The checks compare
the answer over two meshes, balance the reactions against the loads, close each free body's forces through its
contact faces, measure how far the penalty lets faces overlap, hold shear to friction where faces slide, and compare
the peak pressure with Hertz where a round face presses on a flat one.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from ..mesh.mesher import Refinement, Sizing, auto_sizing
from ..solvers import calculix_nl as nl
from ..solvers.base import SolverError
from ..structural.build import build_static, tri_area_weights
from ..structural.run import support_reactions
from ..structural.spec import SpecError, StructuralStudy, parse_fixture, parse_load
from ..units import fmt
from ..verify.checks import Check, balance_check, confidence, convergence_check
from ..plastic.common import NonlinearOutcome, clean_work, light_png, where
from . import setup as S
from .assembly import Assembly, prepare_assembly

QUALITY = {"fast": 2000, "normal": 5000, "fine": 12000}   # nodes on the first of two meshes, before the contact
CONTACT_SIZE = 0.6                                        # element size on contact faces against the global size
REFINE = 0.75
TOL = 0.05                                                # the two meshes agree within this


@dataclass
class ContactStudy:
    part: str
    material: str
    loads: list[str] = field(default_factory=list)
    fixtures: list[str] = field(default_factory=list)
    moves: list[str] = field(default_factory=list)       # prescribed displacements, after the loads
    friction: float = 0.2
    body_materials: dict = field(default_factory=dict)   # body number -> material
    bonded: list = field(default_factory=list)           # pairs of body numbers glued together
    contact_size: float | None = None                    # element size on the contact faces, mm
    target: float | None = None
    quality: str = "normal"
    question: str = ""
    scale: float = 1.0
    params: dict = field(default_factory=dict)
    keep: bool = False
    playback: bool = False               # also write playback.json with every increment, for a player

    def to_dict(self) -> dict:
        return asdict(self)


def run_contact(study: ContactStudy, out_dir: str | Path, progress=None, asm: Assembly | None = None,
                render_images: bool = True) -> NonlinearOutcome:
    t0 = time.time()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    work = out / "work"
    say = progress or (lambda a, b: None)
    if asm is None:
        say("geometry", f"reading {Path(study.part).name} and finding the faces that touch")
        asm = prepare_assembly(study.part, work, [tuple(b) for b in study.bonded], study.params, study.scale)
    if not asm.pairs:
        raise SpecError("no faces of different bodies touch or face each other closely, so there is no contact "
                          "to solve. Check that the file holds separate bodies, or run 'wobbl run' for one part")
    base = StructuralStudy(str(study.part), study.material, [parse_load(x) for x in study.loads],
                           [parse_fixture(x) for x in study.fixtures])
    loads, fixtures, selections = S.resolve(asm, base.loads, base.fixtures)
    moves = [S.parse_move(m) for m in study.moves]
    move_sel = [S.select(asm, m.faces) for m in moves]
    levels, last = [], None
    for k, sizing in enumerate(level_sizes(asm, study)):
        say("mesh", f"level {k}: element size {sizing.h:.3g} mm")
        from .assembly import build_assembly_mesh
        mesh = build_assembly_mesh(asm, sizing)
        r = solve_level(asm, study, mesh, loads, fixtures, moves, move_sel, work / f"level{k}", say, k)
        levels.append({"level": k, "n_nodes": mesh.n_nodes, "h": sizing.h, "governing": r["governing"],
                       "contact_force": r["contact_force"], "peak_pressure": r["peak_pressure"],
                       "increments": r["res"].increments, "iterations": r["res"].iterations,
                       "t_solve": r["res"].t_solve})
        say("check", f"level {k}: peak stress {r['governing']:.4g} MPa, peak contact pressure "
                     f"{r['peak_pressure']:.4g} MPa")
        last = r
    o = evaluate(study, asm, last, levels, moves)
    o.t_total = time.time() - t0
    if render_images:
        say("render", "drawing the stress and the contact pressure")
        o.images = render(out / "images", last)
    if study.playback:
        o.images["playback"] = playback(out / "playback.json", study, last, bool(moves))
    o.notes = list(asm.prep.notes)
    try:
        from .scene import write_scene
        write_scene(out, "contact", study, o, last, base, selections)
    except Exception as e:  # noqa: BLE001  a view that cannot be built never costs the result
        o.notes.append(f"The 3D view could not be built ({str(e)[:120]}).")
    o.write(out)
    clean_work(work, study.keep)
    return o


def level_sizes(asm: Assembly, study: ContactStudy) -> list[Sizing]:
    """Two meshes, each finer at the contact faces (0.6 of the global size there, or the size asked for)."""
    s0 = auto_sizing(asm.prep, thickness=asm.features.thickness_typical,
                     max_nodes=QUALITY.get(study.quality, QUALITY["normal"]))
    faces = sorted({f for p in asm.pairs for f in (p.slave, p.master)})
    out = []
    for k in range(2):
        h = s0.h * REFINE ** k
        c = (study.contact_size * REFINE ** k) if study.contact_size else CONTACT_SIZE * h
        out.append(Sizing(h=h, refinements=[Refinement(min(c, h), surfaces=faces, grow=3 * h,
                                                       label="contact faces")]))
    return out


def solve_level(asm, study, mesh, loads, fixtures, moves, move_sel, work: Path, say, k: int) -> dict:
    notes: list[str] = []
    mats, mat_of = S.materials(mesh, study.material, {int(b): m for b, m in study.body_materials.items()})
    built = build_static(mesh, loads, fixtures, mat_of[1], asm.features.faces)
    sm = built.model
    floating = S.floating_bodies(mesh, sm)
    move_sets = {}
    for i, sel in enumerate(move_sel):
        nodes = np.unique(mesh.tris[sel.tri_mask(mesh)].ravel())
        move_sets[f"MOVE{i + 1}"] = nodes
    if move_sets:
        # a moved face holds its body in the moved directions, so only the rest of it may need springs
        held = np.unique(np.concatenate(list(move_sets.values())))
        floating = [b for b in floating if not np.isin(np.unique(mesh.tets[mesh.body == b].ravel()), held).any()]
    springs = S.springs_for(mesh, floating, mat_of)
    pairs = S.contact_pairs(asm, mesh, asm.pairs, mat_of, study.friction, notes)
    hold = [(f"MOVE{i + 1}", d + 1, 0.0) for i, m in enumerate(moves) for d in range(3) if abs(m.vector[d]) > 1e-12]
    every = study.playback
    steps = [nl.Step(1.0, first=0.25, largest=0.5 if not every else 0.25, smallest=1e-5, prescribed=hold,
                     every_increment=every)]
    if moves:
        go = [(f"MOVE{i + 1}", d + 1, float(m.vector[d])) for i, m in enumerate(moves) for d in range(3)
              if abs(m.vector[d]) > 1e-12]
        steps.append(nl.Step(1.0, first=0.1, largest=0.25, smallest=1e-5, prescribed=go, every_increment=every))
    model = nl.NonlinearModel(sm, mats, steps, contact=pairs, springs=springs, node_sets=move_sets,
                              fields=("U", "RF", "S"))
    say("solve", f"level {k}: contact on {mesh.n_nodes:,} nodes, {len(pairs)} contact pair"
                 + ("s" if len(pairs) != 1 else "") + (f", {len(floating)} body held only by contact"
                                                         if floating else ""))
    try:
        res = nl.solve_nonlinear(model, work, "contact")
    except SolverError as e:
        if floating and not moves:
            raise SolverError(f"{e}", f"Body {floating[0]} is held only by contact and friction, so a load "
                                 "larger than friction can carry slides it away. Check the friction coefficient, "
                                 "or hold the body with a support") from None
        raise
    fr = res.frames[-1]
    return results(asm, study, mesh, built, pairs, springs, move_sets, floating, mat_of, fr, res, notes, moves)


def pressed_edges(mesh, pairs, fr) -> tuple[np.ndarray, float]:
    """Points on the rim of each contact face that is pressed, with the size of the elements there. Where the edge of
    a flat face presses on another, the elastic stress and pressure under it grow without bound as the mesh
    refines, the way they do at a sharp inside corner, so they are kept out of the governing numbers."""
    pres = np.clip(fr.get("pressure", np.zeros(mesh.n_nodes)), 0, None)
    pts, sizes = [], []
    for cp in pairs:
        own = np.unique(mesh.tris[cp.slave].ravel())
        rest = np.unique(mesh.tris[~cp.slave & ~cp.master].ravel())
        rim = np.intersect1d(own, rest)
        top = float(pres[own].max()) if len(own) else 0.0
        rim = rim[pres[rim] > 1e-3 * top] if top > 0 else rim[:0]
        if len(rim):
            pts.append(mesh.X[rim])
            sizes.append(float(np.median(np.sqrt(2 * tri_area_weights(mesh, cp.slave)[1]))))
    if not pts:
        return np.zeros((0, 3)), 0.0
    return np.vstack(pts), max(sizes)


def results(asm, study, mesh, built, pairs, springs, move_sets, floating, mat_of, fr, res, notes, moves) -> dict:
    vm = nl.von_mises(fr["S"])
    from ..geometry.faces import find_sharp_corners
    from ..structural.hotspots import ZoneMap

    corners = find_sharp_corners(mesh, faces=asm.features.faces, tangent=asm.features.tangent_pairs)
    zones = dict(built.zones, **{k: v for k, v in move_sets.items()})
    zmap = ZoneMap(mesh, zones, {**built.zone_kind, **{k: "fixture" for k in move_sets}}, corners)
    r_ex = 0.2 * asm.features.thickness_typical
    excluded = zmap.excluded_mask(mesh.X, r_ex)
    edges, h_edge = pressed_edges(mesh, pairs, fr)
    if len(edges):
        from scipy.spatial import cKDTree

        r_edge = max(r_ex, 2.0 * h_edge)
        excluded |= cKDTree(edges).query(mesh.X, distance_upper_bound=r_edge)[0] <= r_edge
    else:
        r_edge = 0.0
    valid = ~excluded if (~excluded).any() else np.ones(len(vm), bool)
    gi = int(np.nonzero(valid)[0][np.argmax(vm[valid])])
    per = []
    for cp, fp in zip(pairs, asm.pairs):
        tot = (res.contact_totals.get(cp.name) or [{}])[-1]
        w, _ = tri_area_weights(mesh, cp.slave)
        nodes = np.nonzero(w > 0)[0]
        p = np.clip(fr.get("pressure", np.zeros(mesh.n_nodes))[nodes], 0, None)
        tau = fr.get("shear", np.zeros(mesh.n_nodes))[nodes]
        opening = fr.get("open", np.zeros(mesh.n_nodes))[nodes]
        slip = fr.get("slip", np.zeros(mesh.n_nodes))[nodes]
        ww = w[nodes]
        touching = p > 1e-6 * max(p.max(), 1e-30)
        sliding = touching & (tau >= 0.95 * cp.mu * p) if cp.mu > 0 else touching
        # the peak away from a pressed rim, where it is mesh dependent; the whole face when nothing is left
        inner = ~excluded[nodes] if (~excluded[nodes] & touching).any() else np.ones(len(nodes), bool)
        i = int(np.argmax(np.where(inner, p, -1.0))) if len(p) else 0
        # forces and area from CalculiX's own totals; the nodal values say where and how the faces press and slide
        per.append({"pair": fp.describe(), "slave": fp.slave, "master": fp.master,
                    "normal_force_N": float(tot.get("normal_force", ww @ p)),
                    "shear_force_N": float(tot.get("shear_force", ww @ tau)),
                    "peak_pressure_MPa": float(p[i]) if len(p) else 0.0,
                    "edge_pressure_MPa": float(p.max()) if len(p) else 0.0,
                    "mean_pressure_MPa": float(tot.get("normal_force", ww @ p)) / max(float(
                        tot.get("area", ww[touching].sum())), 1e-30),
                    "peak_xyz": mesh.X[nodes[i]].tolist() if len(p) else None,
                    "contact_area_mm2": float(tot.get("area", ww[touching].sum())), "area_mm2": float(ww.sum()),
                    "sliding_share": float(ww[sliding].sum() / max(ww[touching].sum(), 1e-30)),
                    # gross slip: the shear the pair carries against friction on its normal force
                    "friction_use": float(tot["shear_force"] / (cp.mu * tot["normal_force"]))
                    if cp.mu > 0 and tot.get("normal_force", 0) > 0 else 0.0,
                    "max_opening_mm": float(opening[~touching].max()) if (~touching).any() else 0.0,
                    "max_slip_mm": float(slip.max()) if len(slip) else 0.0,
                    "penetration_mm": float(p[i] / cp.stiffness) if len(p) else 0.0,
                    "element_mm": float(np.median(np.sqrt(2 * tri_area_weights(mesh, cp.slave)[1]))),
                    "mu": cp.mu, "stiffness": cp.stiffness, "clearance": cp.clearance, "gap": fp.gap,
                    "body_slave": fp.body_slave, "body_master": fp.body_master})
    spring_force, spring_moment = np.zeros(3), np.zeros(3)
    for nodes, kk in springs:
        f = -kk * fr["U"][nodes]
        spring_force += f.sum(0)
        spring_moment += np.cross(mesh.X[nodes], f).sum(0)
    moved, moved_m = np.zeros(3), np.zeros(3)
    for m, nodes in zip(moves, move_sets.values()):
        # only the directions the move prescribes: a node on the moved face that is also pressed by contact carries
        # contact force in its free directions
        rf = fr["RF"][nodes] * (np.abs(m.vector) > 1e-12)
        moved += rf.sum(0)
        moved_m += np.cross(mesh.X[nodes], rf).sum(0)
    return {"mesh": mesh, "built": built, "frame": fr, "res": res, "vm": vm, "governing": float(vm[gi]), "gi": gi,
            "governing_xyz": mesh.X[gi].tolist(), "pairs": per, "spring_force": spring_force, "floating": floating,
            "move_force": moved, "move_moment": moved_m, "spring_moment": spring_moment, "mat_of": mat_of,
            "notes": notes, "excluded": excluded, "edge_radius": r_edge,
            "contact_force": float(sum(x["normal_force_N"] for x in per)),
            "peak_pressure": float(max((x["peak_pressure_MPa"] for x in per), default=0.0))}


def evaluate(study, asm, r, levels, moves) -> NonlinearOutcome:
    mesh, built, fr, sm = r["mesh"], r["built"], r["frame"], r["built"].model
    mat_of = r["mat_of"]
    mat = mat_of[int(mesh.body[np.nonzero((mesh.tets == r["gi"]).any(1))[0][0]])]
    target = study.target or mat.recommended_sf
    checks: list[Check] = []
    extra: list[tuple[str, str]] = []
    checks.append(convergence_check([L["governing"] for L in levels], TOL, "peak von Mises stress", "MPa",
                                    min_levels=2))
    forces = [L["contact_force"] for L in levels]
    if len(forces) > 1 and max(forces) > 0:
        ch = abs(forces[-1] - forces[-2]) / max(abs(forces[-1]), 1e-12)
        checks.append(Check("contact convergence", "pass" if ch <= TOL else "warn",
                            f"force through the contact faces {' -> '.join(fmt(f) for f in forces)} N over two meshes, "
                            f"last change {100 * ch:.1f}%; peak pressure "
                            f"{' -> '.join(fmt(L['peak_pressure']) for L in levels)} MPa",
                            {"forces": forces}))
    F, M = sm.applied_total()
    held = SimpleNamespace(RF=fr["RF"])
    R, RM = support_reactions(built, held, mesh)
    R, RM = R + r["move_force"], RM + r["move_moment"]
    springs = r["spring_force"]
    scale = max(float(np.linalg.norm(F)), float(np.linalg.norm(r["move_force"])), 1e-12)
    c = balance_check(F + springs, R, M + r["spring_moment"], RM, asm.prep.diag)
    c.summary = c.summary.replace("reactions balance the loads", "supports and moved faces balance the loads")
    checks.append(Check(c.name, c.status if np.linalg.norm(F) > 0 or moves else "info", c.summary, c.detail))
    if r["floating"]:
        share = float(np.linalg.norm(springs)) / scale
        st = "pass" if share < 0.005 else ("warn" if share < 0.02 else "fail")
        checks.append(Check("stabilizing springs", st, f"bod{'ies' if len(r['floating']) > 1 else 'y'} "
                            f"{', '.join(map(str, r['floating']))} rest{'' if len(r['floating']) > 1 else 's'} on "
                            f"contact alone, so soft springs hold them before contact closes; they carry "
                            f"{100 * share:.2g}% of the load", {"share": share}))
    checks += _contact_checks(asm, r, F, moves)
    hertz = hertz_check(asm, r)
    if hertz is not None:
        checks.append(hertz)
    sf = mat.allowable / r["governing"] if r["governing"] > 0 else float("inf")
    gwhere = where(mesh, asm.features.faces, r["governing_xyz"])
    checks.append(Check("units and sanity", "pass" if sf < 1e4 and asm.prep.diag < 2e4 else "warn",
                        f"assembly {fmt(asm.prep.diag)} mm across, peak strain {r['governing'] / mat.E:.2g}, largest "
                        f"movement {fmt(float(np.linalg.norm(fr['U'], axis=1).max()))} mm"))
    level, reasons = confidence(checks, extra)
    verdict = "holds" if sf >= target else ("marginal" if sf >= 1 else "fails")
    slid = [p for p in r["pairs"] if p["mu"] > 0 and p["friction_use"] >= 0.95]
    opened = [p for p in r["pairs"] if p["contact_area_mm2"] <= 0.02 * p["area_mm2"] and p["gap"] <= 1e-3 * asm.prep.diag]
    if slid and not moves:
        verdict = "slips"
    elif opened and verdict == "holds":
        verdict = "separates"
    top = max(r["pairs"], key=lambda p: p["normal_force_N"])
    head = (f"Peak {fmt(r['governing'])} MPa {gwhere}, safety factor {sf:.2f} against "
            f"{'yield' if mat.ductile else 'tensile strength'} (target {target:g}). ")
    if top["normal_force_N"] > 0:
        head += (f"{top['pair'].capitalize()} press with {fmt(top['normal_force_N'])} N, up to "
                 f"{fmt(top['peak_pressure_MPa'])} MPa over {fmt(top['contact_area_mm2'])} mm2.")
    for p in slid:
        head += f" {p['pair'].capitalize()} slide, by up to {fmt(p['max_slip_mm'])} mm."
    for p in opened:
        head += f" {p['pair'].capitalize()} open apart."
    if moves:
        head += f" Moving them takes {fmt(float(np.linalg.norm(r['move_force'])))} N."
    if level != "high":
        head += f" Confidence is {level}."
    metrics = [("peak stress", f"{fmt(r['governing'])} MPa"), ("safety factor", f"{sf:.2f}"),
               ("peak contact pressure", f"{fmt(r['peak_pressure'])} MPa"),
               ("contact force", f"{fmt(r['contact_force'])} N")]
    if moves:
        metrics.append(("force to move", f"{fmt(float(np.linalg.norm(r['move_force'])))} N"))
    o = NonlinearOutcome("contact", str(study.part), study.question, verdict, head, metrics, level, reasons, checks,
                         assumptions(study, mat_of, r), solver=r["res"].solver)
    o.values = {"governing_MPa": r["governing"], "governing_xyz": r["governing_xyz"], "safety_factor": sf,
                "target": target, "pairs": r["pairs"], "levels": levels,
                "move_force_N": r["move_force"].tolist(), "spring_force_N": springs.tolist(),
                "_figures": [["stress", "von Mises stress across the assembly"],
                             ["pressure", "Contact pressure on the pressed faces"]],
                "_tables": [["Contact pairs", ["pair", "force N", "peak MPa", "area mm2", "sliding", "opening mm"],
                             [[p["pair"], fmt(p["normal_force_N"]), fmt(p["peak_pressure_MPa"]),
                               fmt(p["contact_area_mm2"]), f"{100 * p['sliding_share']:.0f}%",
                               fmt(p["max_opening_mm"])] for p in r["pairs"]]]]}
    o.fixes = fixes(verdict, mat, sf, target, slid, study)
    return o


def _contact_checks(asm, r, F, moves) -> list[Check]:
    out = []
    worst = max(r["pairs"], key=lambda p: p["penetration_mm"] / max(p["element_mm"], 1e-12))
    rel = worst["penetration_mm"] / max(worst["element_mm"], 1e-12)
    st = "pass" if rel <= 0.01 else ("warn" if rel <= 0.05 else "fail")
    out.append(Check("penalty overlap", st, f"the penalty stiffness of {fmt(worst['stiffness'])} MPa/mm, twenty "
                     f"times the modulus over the {fmt(worst['element_mm'])} mm elements, lets {worst['pair']} "
                     f"overlap by {fmt(worst['penetration_mm'])} mm at the peak pressure, {100 * rel:.2g}% of an "
                     "element", {"overlap_mm": worst["penetration_mm"], "ratio": rel}))
    # a body pressed only through one flat pair: the pressure has to add up to the load pressing it
    for b in r["floating"]:
        mine = [p for p in r["pairs"] if b in (p["body_slave"], p["body_master"])]
        faces = [asm.features.faces[p["master"]] for p in mine]
        if len(mine) != 1 or faces[0].kind != "plane":
            continue
        n = np.asarray(faces[0].normal)
        mesh, sm = r["mesh"], r["built"].model
        nodes = np.unique(mesh.tets[mesh.body == b].ravel())
        load = float(abs((sm.nodal_forces() + nl.rigid_as_nodal(sm))[nodes].sum(0) @ n))
        got = mine[0]["normal_force_N"]
        if load <= 0:
            continue
        err = abs(got - load) / load
        out.append(Check("contact balance", "pass" if err < 0.01 else ("warn" if err < 0.03 else "fail"),
                         f"body {b} is pressed onto its contact with {fmt(load)} N and the contact pressure adds up "
                         f"to {fmt(got)} N ({100 * err:.2g}% apart)", {"load_N": load, "contact_N": got}))
    slid = [p for p in r["pairs"] if p["mu"] > 0 and p["friction_use"] >= 0.95 and p["normal_force_N"] > 0]
    for p in slid:
        ratio = p["shear_force_N"] / p["normal_force_N"]
        err = abs(ratio - p["mu"]) / p["mu"]
        out.append(Check("friction", "pass" if err < 0.03 else "warn",
                         f"where {p['pair']} slide, shear over pressure is {ratio:.3g} against the friction "
                         f"coefficient {p['mu']:g} ({100 * err:.2g}% apart), as Coulomb's law has it",
                         {"ratio": ratio, "mu": p["mu"]}))
    return out


def hertz_check(asm, r) -> Check | None:
    """Hertz's peak pressure where a round face presses on a flat one: a cylinder along a line (pressure
    sqrt(P' E* / (pi R)) per unit length P'), or a sphere at a point ((6 P E*^2 / (pi^3 R^2))^(1/3))."""
    for p in r["pairs"]:
        a, b = asm.features.faces[p["slave"]], asm.features.faces[p["master"]]
        flat, round_ = (a, b) if a.kind == "plane" else (b, a)
        if flat.kind != "plane" or round_.radius is None or round_.concave or p["normal_force_N"] <= 0:
            continue
        m1, m2 = r["mat_of"][p["body_slave"]], r["mat_of"][p["body_master"]]
        e_star = 1.0 / ((1 - m1.nu ** 2) / m1.E + (1 - m2.nu ** 2) / m2.E)
        R, P = round_.radius, p["normal_force_N"]
        if round_.kind == "sphere":
            p0 = (6 * P * e_star ** 2 / (np.pi ** 3 * R ** 2)) ** (1 / 3)
            what = f"a sphere of radius {fmt(R)} mm pressed with {fmt(P)} N"
        elif round_.length:
            p0 = np.sqrt(P / round_.length * e_star / (np.pi * R))
            what = f"a cylinder of radius {fmt(R)} mm pressed with {fmt(P / round_.length)} N per mm"
        else:
            continue
        got = p["peak_pressure_MPa"]
        ratio = got / p0
        st = "pass" if abs(ratio - 1) <= 0.1 else "warn"
        return Check("hand calculation", st, f"Hertz gives a peak pressure of {fmt(p0)} MPa for {what}; the FEA peak "
                     f"is {fmt(got)} MPa ({ratio:.2f}x)", {"hertz_MPa": p0, "fea_MPa": got})
    return None


def assumptions(study, mat_of, r) -> list[str]:
    mats = sorted({m.name for m in mat_of.values()})
    a = [f"Materials are linear elastic: {', '.join(mats)}.",
         f"Contact is surface to surface with a penalty stiffness and Coulomb friction, coefficient {study.friction:g}"
         + (" (frictionless)" if study.friction == 0 else "") + ". Real friction varies with finish, lubrication "
         "and wear, often by a factor of two.",
         "Deflections are small, so faces slide by small amounts compared with their size.",
         "Loads are applied once and stay in their direction."]
    a += r["notes"]
    if r["floating"]:
        a.append(f"Bodies {', '.join(map(str, r['floating']))} rest on contact alone and carry soft springs to "
                 "ground so they cannot drift before contact closes.")
    if r.get("edge_radius"):
        rim = max(r["pairs"], key=lambda p: p["edge_pressure_MPa"])
        a.append(f"Where the rim of a flat face presses on another, elastic stress and pressure have no finite peak, "
                 f"so the {fmt(r['edge_radius'])} mm next to it is left out of the governing numbers, as at sharp "
                 f"corners. The rim of {rim['pair']} reads {fmt(rim['edge_pressure_MPa'])} MPa on this mesh; a real "
                 "edge rounds off or yields a little and spreads it.")
    return a


def fixes(verdict, mat, sf, target, slid, study) -> list[dict]:
    out = []
    if verdict in ("fails", "marginal"):
        need = target / max(sf, 1e-12)
        out.append({"kind": "stress", "title": f"Lower the peak stress by {need:.2g}x",
                    "summary": "Spread the contact over a larger area (a bigger radius or a wider face), or thicken "
                               "the part where the peak sits, until it is under "
                               f"{fmt(mat.allowable / target)} MPa.", "auto": False})
    if slid and not study.moves:
        p = slid[0]
        out.append({"kind": "friction", "title": "Clamp harder or add a positive stop",
                    "summary": f"{p['pair'].capitalize()} slide because friction at {p['mu']:g} carries at most "
                               f"{fmt(p['mu'] * p['normal_force_N'])} N there. More clamping force, a rougher "
                               "interface or a shoulder that takes the load in bearing stops it.", "auto": False})
    return out


def playback(path: Path, study, r, moved: bool) -> str:
    """Every increment of the run on the assembly's surface: the loads coming on, then any prescribed movement,
    with the stress and the contact pressure at each."""
    from ..plastic.common import write_playback

    mesh = r["mesh"]
    frames = []
    top_s = top_p = 1e-9
    for fr in r["res"].frames:
        t = float(fr["time"])
        vm = nl.von_mises(fr["S"])
        pres = np.clip(fr.get("pressure", np.zeros(mesh.n_nodes)), 0, None)
        top_s, top_p = max(top_s, float(np.percentile(vm, 99.5))), max(top_p, float(pres.max()))
        frames.append({"t": t, "label": "loading" if t <= 1 + 1e-9 else "moving", "U": fr["U"], "stress": vm,
                       "pressure": pres, "values": {"load": min(t, 1.0), "move": max(t - 1.0, 0.0)}})
    body = np.zeros(mesh.n_nodes)
    for b in np.unique(mesh.body):
        body[np.unique(mesh.tets[mesh.body == b].ravel())] = b
    return write_playback(
        path, mesh, frames,
        {"kind": "contact", "title": Path(study.part).name, "question": study.question,
         "units": {"length": "mm", "stress": "MPa", "pressure": "MPa"},
         "phases": ["loading"] + (["moving"] if moved else []),
         "about": "Frames play in order of t. From t 0 to 1 the loads come on in proportion; past 1 the faces given "
                  "a movement move to it. stress is von Mises (colours clip at the 99.5th percentile, so a single "
                  "corner does not wash the rest out) and pressure is the contact pressure, zero where faces are "
                  "apart. mesh.body numbers the parts."},
        {"stress": {"unit": "MPa", "min": 0.0, "max": top_s}, "pressure": {"unit": "MPa", "min": 0.0, "max": top_p}},
        body=body)


def render(out: Path, r) -> dict:
    from ..report.render import render_scalar

    img = {}
    mesh = r["mesh"]
    try:
        img["stress"] = light_png(render_scalar(out / "stress.png", mesh.X, mesh.tets, mesh.tris, r["vm"],
                                                "von Mises", "MPa", "turbo", np.asarray(r["governing_xyz"]),
                                                f"{fmt(r['governing'])} MPa"))
        top = max(r["pairs"], key=lambda p: p["normal_force_N"])
        b = top["body_slave"]
        keep = mesh.body == b
        used = np.unique(mesh.tets[keep].ravel())
        lut = np.full(mesh.n_nodes, -1)
        lut[used] = np.arange(len(used))
        tris = mesh.tris[np.isin(mesh.tri_elem, np.nonzero(keep)[0])]
        pres = np.clip(r["frame"].get("pressure", np.zeros(mesh.n_nodes)), 0, None)[used]
        if top["peak_xyz"] is not None:
            img["pressure"] = light_png(render_scalar(out / "pressure.png", mesh.X[used], lut[mesh.tets[keep]],
                                                      lut[tris], pres, f"contact pressure on body {b}", "MPa",
                                                      "inferno", np.asarray(top["peak_xyz"]),
                                                      f"{fmt(top['peak_pressure_MPa'])} MPa"))
    except Exception as e:  # noqa: BLE001
        img["error"] = str(e)[:200]
    return img
