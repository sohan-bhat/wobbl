"""The fast path for thin walled parts: the strength study on a shell model of the walls, checked against a
coarse solid.

A 3 mm wall on a 100 mm enclosure needs over 100,000 nodes of solid elements, because each element has to be
about as small as the wall is thick. A shell carries the wall's thickness as a number instead, so its elements
can be several walls wide. Wobbl meshes the midsurface (shells/mesh.py), turns screw bosses into beams
(shells/posts.py), refines the shell until the governing stress settles, and solves the solid once on a coarse
mesh to check that the shell idealization gives the same deflection and stress. The answer comes back in the
strength study's own outcome, so the report, the summary and the agent tools read it as they read any other.
"""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import numpy as np

from ..materials import resolve_material
from ..shells.build import build_shell, excluded_nodes
from ..shells.ccx import ShellResult, solve_shell
from ..shells.checks import beam_hand_check, force_balance
from ..shells.mesh import middle_dimension, midsurface, thin_walled
from ..shells.posts import find_posts
from ..solvers import base as solvers
from ..units import fmt
from ..verify.checks import Check, confidence, convergence_check, rel_up
from .outcome import Outcome, _headline, _verdict, process_caveat
from .spec import SpecError, StructuralStudy

LEVELS = {"fast": (1.0, 0.7), "normal": (1.0, 0.7, 0.5), "fine": (1.0, 0.7, 0.5, 0.35)}
TOL = {"fast": 0.05, "normal": 0.03, "fine": 0.02}
CROSS_SIZE = 1.5         # the solid cross check's element size, in typical walls


def shell_measure(res: ShellResult, material) -> tuple[np.ndarray, str]:
    """The stress compared with the limit at each shell node, the larger of the two faces: von Mises for ductile
    materials, the largest principal stress for brittle ones."""
    from ..fe import StaticResult

    def one(S):
        r = StaticResult(np.zeros((len(S), 3)), S, np.zeros((len(S), 3)))
        return r.von_mises() if material.ductile else np.maximum(r.principal()[:, 2], 0.0)

    m = np.maximum(one(res.S_bot), one(res.S_top))
    return m, "von Mises stress" if material.ductile else "max principal stress"


def analyze_shell(study: StructuralStudy, out_dir: str | Path, progress=None, render: bool = True,
                  pre: tuple | None = None, cross_check: bool = True, keep: bool = False) -> Outcome:
    from ..report.render import convergence_plot
    from .report import write_outputs
    from .run import describe_part, resolve_selections

    t0 = time.time()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    work = out / "work"
    say = progress or (lambda a, b: None)
    material = resolve_material(study.material)
    solvers.for_domain("structural").require()
    if pre is None:
        say("geometry", f"reading {Path(study.part).name}")
        pre = describe_part(study.part, work, study.scale, study.body, study.params)
    prep, features = pre
    thin, why = thin_walled(features)
    if not thin:
        raise SpecError(f"--shell needs a thin walled part, and this one is not: {why}. Run it without --shell")
    loads, fixtures, sel_info = resolve_selections(study, features)
    if any(fx.kind == "wall" for fx, _ in fixtures):
        raise SpecError("wall supports that can lift off are solved on the solid; run this part without --shell")
    p = material.internal()
    q = study.quality if study.quality in LEVELS else "normal"
    base = study.mesh_size or min(prep.diag / 30.0, middle_dimension(features) / 8.0)
    levels, last, notes = [], None, list(prep.notes)
    r_ex = None
    for k, f in enumerate(LEVELS[q]):
        h = base * f
        say("mesh", f"level {k}: midsurface with {h:.3g} mm shell elements")
        shell = midsurface(prep, features, h)
        posts = find_posts(shell, features.faces, h)
        model, zones = build_shell(shell, loads, fixtures, features.faces, p["E"], p["nu"], p["density"], posts=posts)
        say("solve", f"level {k}: {shell.n_nodes:,} shell nodes" + (f" and {len(posts)} bosses as beams" if posts
                                                                   else ""))
        res = solve_shell(model, work / f"level{k}")
        measure, mname = shell_measure(res, material)
        if r_ex is None:
            r_ex = float(max(features.thickness_typical, 0.5 * h))
        excl = excluded_nodes(shell, zones, r_ex)
        levels.append(_level(k, h, shell, res, measure, excl, model, material))
        last = (shell, posts, model, zones, res, measure, excl)
        say("check", f"level {k}: governing {levels[-1]['governing']:.4g} MPa")
        if len(levels) >= len(LEVELS[q]) or (len(levels) >= 3 and _change(levels) <= TOL[q]):
            break
    shell, posts, model, zones, res, measure, excl = last
    notes += [n for n in shell.notes if n not in notes]
    L = levels[-1]
    gov, gxyz = L["governing"], np.asarray(L["governing_xyz"])
    allow = material.allowable
    sf = allow / gov if gov > 0 else float("inf")
    target = study.target_sf or material.recommended_sf
    verdict = _verdict(sf, target)
    where = _where(shell, features, gxyz, L.get("on_post"))
    checks = [convergence_check([x["governing"] for x in levels], TOL[q], mname, "MPa",
                                "converged" if _change(levels) <= TOL[q] else "level limit",
                                [x["max_disp"] for x in levels])]
    checks.append(force_balance(model, res))
    checks.append(beam_hand_check(model, _axial_strain(shell, res), zones, p["E"], f"E {fmt(material.E)} MPa"))
    checks.append(_supports(measure, excl, gov, r_ex, shell))
    checks.append(_coverage(shell, posts))
    cross = None
    if cross_check:
        say("check", "solving the solid on a coarse mesh to check the shell idealization")
        cross = solid_cross_check(prep, features, loads, fixtures, material, levels[0], shell, work / "solid")
        checks.append(cross[0])
    checks.append(_sanity(prep.diag, L, gov, allow, material))
    level, reasons = confidence(checks)
    head = _headline(verdict, gov, sf, target, material, where, study.imperial, level)
    mass = model.mass() * 1000.0 + sum(pp.area * pp.length for pp in posts) * p["density"] * 1000.0
    o = Outcome("strength", str(study.part), material.name, study.question, mname + " (shell model)", gov,
                gxyz.tolist(), where, allow, sf, target, verdict, head, L["max_disp"], L["max_disp_xyz"], level,
                reasons, checks, _assumptions(material, study, shell, posts, r_ex), _spots(measure, excl, shell, gov),
                levels, [ld.describe() for ld in study.loads], [fx.describe() for fx in study.fixtures], sel_info,
                features.to_dict(max_faces=60), notes, study.imperial, mass, None, workdir=str(out),
                solver=res.solver)
    o.fixes = _fixes(o, material, features.thickness_typical)
    if render:
        say("render", "drawing heatmaps")
        o.images = _images(out / "images", shell, res, measure, gxyz, gov, allow, target, mname)
        o.images["convergence"] = convergence_plot(out / "images" / "convergence.png", levels, o.hotspots)
    sd = study.to_dict()
    sd["part"] = rel_up(sd["part"], out)
    sd["shell"] = True
    (out / "study.json").write_text(json.dumps(sd, indent=1))
    np.savez_compressed(out / "fields.npz", X=shell.X.astype(np.float32), tris=shell.tris.astype(np.int32),
                        U=res.U.astype(np.float32), measure=measure.astype(np.float32), excluded=excl)
    if not keep:
        shutil.rmtree(work, ignore_errors=True)
    o.t_total = time.time() - t0
    return write_outputs(o, out)


def _level(k, h, shell, res, measure, excl, model, material) -> dict:
    keep = ~excl if (~excl).any() else np.ones(shell.n_nodes, bool)
    i = np.flatnonzero(keep)[int(np.argmax(measure[keep]))]
    gov, gxyz, on_post = float(measure[i]), shell.X[i].tolist(), False
    if res.post_vm is not None and len(res.post_vm) and material.ductile and res.post_vm.max() > gov:
        j = int(np.argmax(res.post_vm))
        gov, gxyz, on_post = float(res.post_vm[j]), res.post_xyz[j].tolist(), True
    um = np.linalg.norm(res.U, axis=1)
    d = int(np.argmax(um))
    applied, _ = model.applied_total()
    return {"level": k, "n_nodes": shell.n_nodes, "n_elems": shell.n_elems, "h_global": h, "h_min": h,
            "t_mesh": 0.0, "t_solve": res.t_solve, "governing": gov, "governing_xyz": gxyz, "on_post": on_post,
            "raw_max": float(measure.max()), "raw_max_xyz": shell.X[int(np.argmax(measure))].tolist(),
            "max_disp": float(um[d]), "max_disp_xyz": shell.X[d].tolist(), "reaction": res.reaction.tolist(),
            "applied": applied.tolist(), "expanded_nodes": res.n_expanded}


def _change(levels) -> float:
    if len(levels) < 2:
        return 1.0
    a, b = levels[-2]["governing"], levels[-1]["governing"]
    return abs(b - a) / max(abs(b), 1e-12)


def _where(shell, features, xyz, on_post) -> str:
    at = f"({', '.join(f'{v:.3g}' for v in xyz)})"
    if on_post:
        return f"on a boss at {at}"
    i = int(np.argmin(np.linalg.norm(shell.X - xyz, axis=1)))
    near = np.flatnonzero((shell.tris == i).any(axis=1))
    fid = int(shell.tri_surf[near[0]]) if len(near) else None
    f = features.faces.get(fid) if fid is not None else None
    if f is None:
        return f"in the wall at {at}"
    if f.role == "fillet" or (f.radius and not f.concave):
        return f"in the wall under the {f.radius:.3g} mm round (face {f.id}) at {at}"
    return f"in the wall under face {f.id} at {at}"


def _axial_strain(shell, res):
    """Strain along the part's long axis at points, offset through the wall along its normal, from the strain
    profile of the nearest node."""
    from scipy.spatial import cKDTree

    tree = cKDTree(shell.X)
    a = np.eye(3)[int(np.argmax(np.ptp(shell.X, axis=0)))]

    def at(pts, off):
        _, idx = tree.query(pts)
        out = []
        for i, z in zip(idx, off):
            s, E = res.profiles[i]
            if not len(s):
                out.append(0.0)
                continue
            exx, eyy, ezz, exy, eyz, ezx = E.T
            T = np.stack([np.stack([exx, exy, ezx], -1), np.stack([exy, eyy, eyz], -1),
                          np.stack([ezx, eyz, ezz], -1)], -2)
            ea = np.einsum("i,kij,j->k", a, T, a)
            o = np.argsort(s, kind="stable")
            out.append(float(np.interp(z, s[o], ea[o])))
        return np.asarray(out)
    return at


def _supports(measure, excl, gov, r_ex, shell) -> Check:
    if not excl.any() or measure[excl].max() <= gov:
        return Check("singularities", "pass", "no support, boss joint or rigid attachment is above the reported "
                                              "stress")
    i = np.flatnonzero(excl)[int(np.argmax(measure[excl]))]
    v = float(measure[i])
    return Check("singularities", "info" if v <= 1.5 * gov else "warn",
                 f"excluded from the answer: {fmt(v)} MPa at ({', '.join(fmt(x) for x in shell.X[i])}), within "
                 f"{fmt(r_ex)} mm of a support, boss joint or rigid attachment, where the idealization sets the "
                 "stress. Wobbl reports the stress away from these spots", {"value": v, "radius_mm": r_ex})


def _coverage(shell, posts) -> Check:
    share = shell.volume() / max(shell.solid_volume, 1e-9)
    post_v = sum(p.area * p.length for p in posts)
    share_all = (shell.volume() + post_v) / max(shell.solid_volume, 1e-9)
    what = f"the shell holds {100 * share:.0f}% of the part's material" + (
        f" and {len(posts)} bosses as beams bring it to {100 * share_all:.0f}%" if posts else "")
    left = len(shell.left_out)
    if share_all >= 0.85:
        return Check("shell model", "pass", what + (f"; {left} rim or edge faces are not walls" if left else ""),
                     {"share": share_all})
    return Check("shell model", "warn", what + ". Features the shell leaves out (ribs, chunky blocks) may carry "
                                               "load, so trust the solid cross check over the shell where they differ",
                 {"share": share_all})


def solid_cross_check(prep, features, loads, fixtures, material, first: dict, shell, work: Path):
    """Solve the solid once on a coarse mesh, elements CROSS_SIZE walls across, and compare its deflection and stress
    near the shell's governing point with the shell's first level. Coarse on purpose: the check is about the
    idealization (walls as shells, bosses as beams), which shows at any mesh, and a fine solid is the slow path
    the shell replaces. Returns the check and the numbers."""
    from ..mesh.mesher import Sizing, build_mesh
    from .build import build_static
    from .run import failure_measure

    try:
        # elements a wall and a half across: one second order element through each wall, which carries bending
        # well enough to show what the shell idealization changes
        mesh = build_mesh(prep, Sizing(h=CROSS_SIZE * features.thickness_typical, curvature=8))
        built = build_static(mesh, loads, fixtures, material, features.faces)
        res = solvers.for_domain("structural").solve_static(built.model, work)
    except Exception as e:  # noqa: BLE001  the cross check is a check, never a reason to fail
        return Check("solid cross check", "info", f"the coarse solid could not be solved ({str(e)[:120]})"), None
    m, _ = failure_measure(res, material)
    g = np.asarray(first["governing_xyz"])
    reach = max(2.0 * features.thickness_typical, first["h_global"])
    near = np.linalg.norm(mesh.X - g, axis=1) <= reach
    s_solid = float(m[near].max()) if near.any() else float("nan")
    d_solid = float(res.displacement_magnitude().max())
    s_shell, d_shell = first["governing"], first["max_disp"]
    ds = d_shell / d_solid - 1.0 if d_solid > 0 else float("nan")
    ss = s_shell / s_solid - 1.0 if s_solid > 0 else float("nan")
    s = (f"a coarse solid ({mesh.n_nodes:,} nodes) gives {fmt(d_solid)} mm of deflection against the shell's "
         f"{fmt(d_shell)} mm ({100 * ds:+.0f}%), and {fmt(s_solid)} MPa within {fmt(reach)} mm of the shell's "
         f"governing point against {fmt(s_shell)} MPa ({100 * ss:+.0f}%), both on first meshes")
    worst = max(abs(ds), abs(ss))
    if worst <= 0.15:
        st = "pass"
    elif worst <= 0.3:
        st = "warn"
        s += ". The shell idealization moves the answer that much here, so read the shell numbers with that margin"
    else:
        st = "fail"
        s += ". The shell model does not stand in for this part; run it without --shell"
    return Check("solid cross check", st, s, {"solid_disp_mm": d_solid, "shell_disp_mm": d_shell,
                                             "solid_stress_MPa": s_solid, "shell_stress_MPa": s_shell,
                                             "solid_nodes": mesh.n_nodes}), (d_solid, s_solid)


def _sanity(diag, L, gov, allow, material) -> Check:
    d_rel = L["max_disp"] / diag if diag > 0 else 0.0
    issues, st = [], "pass"
    if d_rel > 0.2:
        issues.append(f"max deflection {fmt(L['max_disp'])} mm is {100 * d_rel:.0f}% of the part size, far outside "
                      "small deflection theory")
        st = "fail"
    elif d_rel > 0.06:
        issues.append(f"max deflection {fmt(L['max_disp'])} mm is {100 * d_rel:.1f}% of the part size; linear "
                      "results get less accurate past a few percent")
        st = "warn"
    if gov > 30 * allow or 0 < gov < 1e-4 * allow:
        issues.append("the stress is far from the material limit; check the load magnitude and units")
        st = "warn" if st == "pass" else st
    base = (f"part {fmt(diag)} mm across, peak strain {gov / material.E:.2g}, max deflection {fmt(L['max_disp'])} mm "
            f"({100 * d_rel:.2g}% of size)")
    return Check("units and sanity", st, "; ".join(issues) if issues else base + ": all plausible",
                 {"diag_mm": diag, "deflection_ratio": d_rel})


def _spots(measure, excl, shell, gov) -> list[dict]:
    if not excl.any() or measure[excl].max() <= gov:
        return []
    i = np.flatnonzero(excl)[int(np.argmax(measure[excl]))]
    return [{"id": 1, "kind": "support", "detail": "at a support, boss joint or rigid attachment",
             "xyz": shell.X[i].tolist(), "values": [float(measure[i])], "excluded": True}]


def _assumptions(material, study, shell, posts, r_ex) -> list[str]:
    a = ["Linear elastic material and small deflections (no yielding redistribution, no buckling).",
         f"{material.name}: E {fmt(material.E)} MPa, Poisson {material.nu:g}, {material.criterion} "
         f"{fmt(material.allowable)} MPa ({material.source}).",
         "Loads are static and applied all at once; no impact, vibration or fatigue.",
         "Walls are shells on their midsurface with the thickness the CAD gives at each point; stress is taken "
         "on both faces. Through the thickness stress under a load and at sharp inside corners is not modeled.",
         "Fixed faces hold the walls they touch against moving and turning."]
    if posts:
        a.append(f"{len(posts)} bosses are beams with their own round or pipe section, joined to the wall they meet "
                 "by a stiff spider over their outline; their own weight is left out.")
    if shell.left_out:
        a.append("Rims, edges and chunky features that are not walls are left out of the shell, so their stiffness "
                 "and stress are not in the model; the solid cross check shows what that costs.")
    made = process_caveat(material, study.print_direction)
    if made:
        a.append(made)
    a.append(f"Points within {fmt(r_ex)} mm of supports, boss joints and rigid attachments are left out of the "
             "answer.")
    return a + list(study.assumptions)


def _fixes(o: Outcome, material, t: float) -> list[dict]:
    if o.verdict == "holds":
        return []
    need = o.target_sf / max(o.safety_factor, 1e-9)
    # bending stress in a wall falls with the square of its thickness
    t_new = t * need ** 0.5
    return [{"kind": "wall", "auto": False, "title": f"Thicken the walls to about {fmt(t_new)} mm",
             "summary": f"Stress piles up in the wall, which bends, and bending stress falls with the square of the "
                        f"thickness. Taking the {fmt(t)} mm walls to about {fmt(t_new)} mm reaches the {o.target_sf:g} "
                        f"target, and the shell path solves the change in about the same time."}]


def _images(out, shell, res, measure, gxyz, gov, allow, target, mname) -> dict:
    from ..shells.render import render_shell

    img = {}
    try:
        img["stress"] = render_shell(out / "stress.png", shell.X, shell.tris, measure, mname, "MPa",
                                     mark_xyz=gxyz, mark_label=f"{fmt(gov)} MPa", clim=[0.0, max(gov, 1e-9)])
        sf = np.where(measure > 0, allow / np.maximum(measure, 1e-30), 99.0)
        img["safety"] = render_shell(out / "safety.png", shell.X, shell.tris, sf, "safety factor", "", cmap="RdYlGn",
                                     mark_xyz=gxyz, mark_label=f"SF {allow / gov:.2f}" if gov > 0 else "",
                                     clim=[0.0, max(2 * target, 3.0)])
        um = np.linalg.norm(res.U, axis=1)
        img["deformed"] = render_shell(out / "deformed.png", shell.X, shell.tris, um, "deflection", "mm",
                                       cmap="viridis", U=res.U, note="shape exaggerated")
    except Exception as e:  # noqa: BLE001  pictures are a bonus
        img["error"] = str(e)[:200]
    return img


__all__ = ["analyze_shell", "solid_cross_check", "shell_measure"]
