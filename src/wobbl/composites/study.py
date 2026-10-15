"""Laminate strength: a ply stack on the midsurface of a thin walled part, solved with CalculiX composite shells.

"Is this carbon drone arm strong enough?" becomes: the arm's walls are a [0/±45/90]s stack of T300/5208, its
root is clamped and the motor pushes 40 N up at the tip. Wobbl meshes the midsurface, gives every element a
composite section with each ply turned to its angle, solves with CalculiX and reads the strain through the
thickness at every node. Each ply's stresses along and across its fibers come from that strain, and Tsai-Wu
and Hashin turn them into the factor on the load at which the first ply fails. The mesh is refined until that
factor stops moving, and classical lamination theory checks the laminate's stiffness through beam theory.
"""
from __future__ import annotations

import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from ..geometry.select import parse_vector
from ..shells.build import build_shell, excluded_nodes
from ..shells.ccx import Layer, solve_shell
from ..shells.checks import beam_hand_check, force_balance
from ..shells.mesh import middle_dimension, midsurface, thin_walled
from ..structural.spec import SpecError, parse_fixture, parse_load
from ..units import fmt
from ..verify.checks import Check, confidence, convergence_check
from .laminate import HASHIN_MODES, Laminate, _wrap, clt_first_ply_failure, parse_layup, ply_failure

TARGET = 1.5          # default required factor on the load at first ply failure
LEVELS = {"fast": (1.0, 0.7), "normal": (1.0, 0.7, 0.5)}


@dataclass
class LaminateStudy:
    part: str
    layup: str
    ply: str = "t300-5208"
    ply_thickness: float | None = None
    loads: list[str] = field(default_factory=list)
    fixtures: list[str] = field(default_factory=list)
    target_sf: float | None = None
    reference: str = ""             # the 0 degree direction; default the part's longest axis
    quality: str = "normal"
    question: str = ""
    scale: float = 1.0
    params: dict = field(default_factory=dict)
    keep: bool = False              # keep the solver work folder

    def to_dict(self) -> dict:
        return asdict(self)


def node_frames(mesh, ref: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """At each node: the reference direction laid onto the wall, the second in plane axis and the normal."""
    n = mesh.normals
    e1 = ref[None] - (n @ ref)[:, None] * n
    weak = np.linalg.norm(e1, axis=1) < 0.2
    if weak.any():
        alt = np.eye(3)[int(np.argmin(np.abs(ref)))]
        e1[weak] = alt[None] - (n[weak] @ alt)[:, None] * n[weak]
    e1 /= np.linalg.norm(e1, axis=1)[:, None]
    return e1, np.cross(n, e1), n


def _tensor(e6: np.ndarray) -> np.ndarray:
    xx, yy, zz, xy, yz, zx = np.asarray(e6, float).T
    return np.stack([np.stack([xx, xy, zx], -1), np.stack([xy, yy, yz], -1), np.stack([zx, yz, zz], -1)], -2)


def inplane_strain(profile, e1, e2, z: np.ndarray) -> np.ndarray:
    """(len(z),3) engineering strains (eps_x, eps_y, gamma_xy) in the node's frame at heights z through the
    wall, interpolated in the strain profile CalculiX gave through the thickness."""
    s, E = profile
    if not len(s):
        return np.full((len(z), 3), np.nan)
    T = _tensor(E)
    ex = np.einsum("i,kij,j->k", e1, T, e1)
    ey = np.einsum("i,kij,j->k", e2, T, e2)
    gxy = 2.0 * np.einsum("i,kij,j->k", e1, T, e2)
    order = np.argsort(s, kind="stable")
    s = s[order]
    return np.column_stack([np.interp(z, s, v[order]) for v in (ex, ey, gxy)])


def laminate_layers(lam: Laminate) -> list[Layer]:
    return [Layer(p.material.key, p.material.engineering_constants(), p.material.density * 1e-12, p.thickness,
                  p.angle) for p in lam.plies]


@dataclass
class PlyField:
    tsai_wu: np.ndarray        # (n, plies) lowest strength ratio over each ply's two faces
    hashin: np.ndarray
    mode: np.ndarray           # (n, plies) governing Hashin mode
    strain: np.ndarray         # (n, plies + 1, 3) in plane strain at every ply boundary


def ply_field(lam: Laminate, mesh, res, ref: np.ndarray) -> PlyField:
    e1, e2, _ = node_frames(mesh, ref)
    z = lam.z()
    n, k = mesh.n_nodes, len(lam.plies)
    strain = np.stack([inplane_strain(res.profiles[i], e1[i], e2[i], z) for i in range(n)])
    tw, hs = np.full((n, k), np.inf), np.full((n, k), np.inf)
    mode = np.empty((n, k), dtype=object)
    ok = np.isfinite(strain).all(axis=(1, 2))
    for j in range(k):
        both = np.vstack([strain[ok, j], strain[ok, j + 1]])
        f = ply_failure(lam, j, both)
        m = ok.sum()
        tw[ok, j] = np.minimum(f["tsai_wu"][:m], f["tsai_wu"][m:])
        lower = f["hashin"][:m] <= f["hashin"][m:]
        hs[ok, j] = np.where(lower, f["hashin"][:m], f["hashin"][m:])
        mode[ok, j] = np.where(lower, f["hashin_mode"][:m], f["hashin_mode"][m:])
    return PlyField(tw, hs, mode, strain)


def resultants(lam: Laminate, strain: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Force and moment resultants (N/mm, N mm/mm) at a node from its strains at the ply boundaries: a straight
    line fit of strain through the thickness, then the laminate's ABD matrix."""
    z = lam.z()
    Z = np.column_stack([np.ones_like(z), z])
    coef, *_ = np.linalg.lstsq(Z, strain, rcond=None)
    A, B, D = lam.abd()
    eps0, kap = coef[0], coef[1]
    return A @ eps0 + B @ kap, B @ eps0 + D @ kap


def run_laminate(study: LaminateStudy, out_dir: str | Path, progress=None, pre=None, render: bool = True):
    from ..outcome_generic import GenericOutcome
    from ..structural.run import describe_part, resolve_selections
    from ..structural.spec import StructuralStudy

    t0 = time.time()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    say = progress or (lambda a, b: None)
    lam = parse_layup(study.layup, study.ply, study.ply_thickness)
    if pre is None:
        say("geometry", f"reading {Path(study.part).name}")
        pre = describe_part(study.part, out / "work", study.scale, None, study.params)
    prep, feats = pre
    thin, why = thin_walled(feats)
    if not thin:
        raise SpecError(f"a laminate needs a thin walled part, and this one is not: {why}")
    ss = StructuralStudy(study.part, "6061-t6", [parse_load(x) for x in study.loads],
                         [parse_fixture(x) for x in study.fixtures])
    loads, fixtures, _ = resolve_selections(ss, feats)
    ref = parse_vector(study.reference) if study.reference else np.eye(3)[int(np.argmax(prep.size))]
    layers = laminate_layers(lam)
    base = min(prep.diag / 30.0, middle_dimension(feats) / 4.0)
    levels, last, notes = [], None, list(prep.notes)
    for k, f in enumerate(LEVELS.get(study.quality, LEVELS["normal"])):
        h = base * f
        say("mesh", f"level {k}: midsurface with {h:.3g} mm elements")
        shell = midsurface(prep, feats, h)
        model, zones = build_shell(shell, loads, fixtures, feats.faces, layers=layers, ref_dir=ref)
        say("solve", f"level {k}: {shell.n_nodes:,} shell nodes, {len(lam.plies)} plies")
        res = solve_shell(model, out / "work" / f"level{k}")
        pf = ply_field(lam, shell, res, ref)
        r_ex = max(2.0 * lam.thickness, 0.5 * base)
        excl = excluded_nodes(shell, zones, r_ex)
        lv = _record(k, h, shell, res, pf, excl, lam)
        levels.append(lv)
        last = (shell, model, zones, res, pf, excl, r_ex)
        say("check", f"level {k}: first ply fails at {lv['strength_ratio']:.3g}x the load")
        if len(levels) >= 3 and _change(levels) <= 0.03:
            break
    shell, model, zones, res, pf, excl, r_ex = last
    notes += [n for n in shell.notes if n not in notes]
    gov = levels[-1]
    target = study.target_sf or TARGET
    R = gov["strength_ratio"]
    verdict = "holds" if R >= target else ("marginal" if R >= 1.0 else "fails")
    gi, gj = gov["node"], gov["ply"]
    gxyz = shell.X[gi]
    ply = lam.plies[gj]
    mode = gov["hashin_mode"]

    checks = [convergence_check([L["strength_ratio"] for L in levels], 0.03, "load factor at first ply failure", "x",
                                "converged" if _change(levels) <= 0.03 else "level limit")]
    checks.append(force_balance(model, res))
    eff = lam.effective()
    checks.append(_clt_check(model, res, lam, ref, zones, eff))
    checks.append(_layup_check(lam))
    checks.append(_support_check(pf, excl, R, r_ex))
    checks.append(_sanity(shell, res, prep.diag, pf))
    level, reasons = confidence(checks)
    umax = float(np.linalg.norm(res.U, axis=1).max())
    crit = gov["criterion"]
    other = (f"Tsai-Wu gives {gov['tsai_wu']:.3g}x" if crit == "hashin" else
             f"Hashin gives {gov['hashin']:.3g}x by {mode}")
    by = f"by {mode} (Hashin)" if crit == "hashin" else "by Tsai-Wu"
    head = (f"First ply failure at {R:.3g}x the load, target {target:g}. Ply {gj + 1} at {fmt(_wrap(ply.angle))} "
            f"degrees goes first {by} at ({', '.join(fmt(v) for v in gxyz)}) mm, and {other}.")
    if level != "high":
        head += f" Confidence is {level}."
    N, M = resultants(lam, pf.strain[gi])
    fixes = layup_fixes(lam, N, M, R, target, gj, mode, study.layup, study.ply) if R < target * 1.0001 else []
    assumptions = _assumptions(lam, study, shell, r_ex)
    table = _ply_table(lam, pf, excl)
    img = {}
    if render:
        say("render", "drawing the failure index and ply charts")
        img = _images(out / "images", shell, res, pf, excl, gxyz, R, lam, table, target)
    metrics = [("first ply failure", f"{R:.3g}x load"), ("target", f"{target:g}"),
               ("governing ply", f"{gj + 1} ({fmt(_wrap(ply.angle))} deg)"), ("max deflection", f"{fmt(umax)} mm"),
               ("laminate", f"{len(lam.plies)} plies, {fmt(lam.thickness)} mm"), ("Ex (CLT)", f"{fmt(eff['Ex'])} MPa")]
    values = {"strength_ratio": R, "criterion": crit, "tsai_wu": gov["tsai_wu"], "hashin": gov["hashin"],
              "hashin_mode": mode, "governing_ply": gj + 1, "governing_angle": _wrap(ply.angle),
              "governing_xyz": gxyz.tolist(), "max_deflection_mm": umax, "layup": lam.angles(),
              "laminate_thickness_mm": lam.thickness, "clt": eff, "plies": table, "levels": levels,
              "resultants": {"N_per_mm": N.tolist(), "M_per_mm": M.tolist()}, "mass_kg": model.mass() * 1000,
              "_figures": [["failure", "Failure index (1 over the strength ratio) of the worst ply at each point"],
                           ["plies", "Lowest strength ratio of each ply"], ["deformed", "Deflection"]]}
    o = GenericOutcome("laminate", str(study.part), study.question, verdict, head, metrics, level, reasons, checks,
                       assumptions, values, img, fixes, notes, solver=res.solver, t_total=time.time() - t0)
    np.savez_compressed(out / "fields.npz", X=shell.X.astype(np.float32), tris=shell.tris.astype(np.int32),
                        U=res.U.astype(np.float32), tsai_wu=pf.tsai_wu.min(1).astype(np.float32),
                        hashin=pf.hashin.min(1).astype(np.float32))
    try:
        playback(out / "playback.json", shell, res, pf, excl, lam, R, gov, target)
    except Exception as e:  # noqa: BLE001  the playback file is a bonus, never a reason to fail
        notes.append(f"playback.json was not written ({str(e)[:120]})")
    try:
        from ..shells.scene import ramp_scene

        fi = 1.0 / np.maximum(np.minimum(pf.tsai_wu, pf.hashin).min(1), 1e-12)
        fi = np.where(np.isfinite(fi), fi, 0.0)
        tone = {"holds": "pass", "marginal": "warn", "fails": "fail"}[verdict]
        ramp_scene(out, title=f"{Path(study.part).stem.replace('_', ' ').capitalize()}, {lam.angles()} laminate",
                   domain="laminate", question=study.question, model=model, values=fi, field="failure index",
                   unit="", U=res.U, reserve=R, limit=1.0, limit_label="first ply failure", keep=int(gi),
                   peak_label=f"ply {gj + 1} at {fmt(_wrap(ply.angle))} deg",
                   readouts=[("worst ply index", 1.0 / R, "", 1.0), ("deflection", umax, "mm", None)],
                   events=[(R, "first ply failure", f"ply {gj + 1} at {fmt(_wrap(ply.angle))} degrees, {mode}",
                            "fail")],
                   verdict=(verdict.upper(), f"The first ply cracks at {R:.3g} times the load, target {target:g}.",
                            tone),
                   source="The laminate strains at any load are the solved ones scaled by it, exact for this "
                          "linear model up to first ply failure.")
    except Exception as e:  # noqa: BLE001  the 3D view is a bonus, never a reason to fail
        import os
        if os.environ.get("WOBBL_DEBUG"):
            raise
        notes.append(f"the 3D view was not written ({str(e)[:120]})")
    if not study.keep:
        shutil.rmtree(out / "work", ignore_errors=True)
    return o.write(out, [_ply_section(table)])


def playback(path, shell, res, pf, excl, lam: Laminate, R: float, gov: dict, target: float, frames: int = 30) -> str:
    """playback.json: the laminate loading up from nothing to first ply failure, ply by ply.

    The study is linear, so at a load factor lam every displacement is lam times the solved one and every ply's
    share of its strength used is lam over its strength ratio. The file carries the midsurface, the solved
    displacement and, for every ply, the failure index at the applied load (1 over the strength ratio, the lower
    of Tsai-Wu and Hashin); a viewer plays index(lam) = lam * index and x(lam) = x + lam * u.
    """
    from ..life.playback import encode, mesh_block, quantize, surface, write

    pts, faces, nodes = surface(shell)
    fi = 1.0 / np.maximum(np.minimum(pf.tsai_wu, pf.hashin)[nodes], 1e-12)          # (v, plies)
    fi = np.where(np.isfinite(fi), fi, 0.0)
    top = float(fi.max()) if fi.size else 1.0
    q = np.round(np.clip(fi / max(top, 1e-30), 0, 1) * 255).astype("uint8")
    lam_ = np.linspace(0.0, R, frames)
    worst = np.argmin(np.minimum(pf.tsai_wu, pf.hashin)[nodes], axis=1)
    doc = {"format": "wobbl-playback", "version": 1, "kind": "laminate",
           "about": "A composite laminate loading up from nothing to first ply failure. At load factor lam (times "
                    "the applied load) a ply's failure index is lam * index, and it fails where that reaches one; "
                    "the part moves to x + lam * displacement. Linear elastic, so this is exact up to first ply "
                    "failure; past it Wobbl does not follow the damage.",
           "units": {"length": "mm", "load_factor": "times the applied load", "failure_index": "fraction of "
                     "strength used"},
           "mesh": mesh_block(pts, faces),
           "displacement": quantize(res.U[nodes].astype(float), "int16"),
           "failure_index": encode(q, "uint8", scale=top / 255.0, offset=0.0),
           "worst_ply": encode(worst.astype("uint8"), "uint8"),
           "excluded": encode(excl[nodes].astype("uint8"), "uint8"),
           "plies": [{"ply": j + 1, "angle_deg": round(_wrap(p.angle), 3), "material": p.material.key,
                      "thickness_mm": p.thickness, "z_mm": [round(float(a), 5) for a in lam.z()[j:j + 2]]}
                     for j, p in enumerate(lam.plies)],
           "frames": {"load_factor": [round(float(x), 5) for x in lam_]},
           "first_ply_failure": {"load_factor": round(R, 5), "ply": gov["ply"] + 1,
                                 "angle_deg": round(_wrap(lam.plies[gov["ply"]].angle), 3),
                                 "criterion": gov["criterion"], "hashin_mode": gov["hashin_mode"],
                                 "at_mm": [round(float(v), 4) for v in gov["xyz"]]},
           "target": target}
    return write(path, doc)


def _record(k, h, shell, res, pf, excl, lam) -> dict:
    keep = ~excl if (~excl).any() else np.ones(shell.n_nodes, bool)
    both = np.minimum(pf.tsai_wu, pf.hashin)
    masked = np.where(keep[:, None], both, np.inf)
    i, j = np.unravel_index(int(np.argmin(masked)), masked.shape)
    return {"level": k, "h": h, "n_nodes": shell.n_nodes, "t_solve": res.t_solve, "node": int(i), "ply": int(j),
            "strength_ratio": float(both[i, j]), "tsai_wu": float(pf.tsai_wu[i, j]),
            "hashin": float(pf.hashin[i, j]), "hashin_mode": str(pf.mode[i, j]),
            "criterion": "tsai-wu" if pf.tsai_wu[i, j] <= pf.hashin[i, j] else "hashin",
            "max_disp": float(np.linalg.norm(res.U, axis=1).max()), "xyz": shell.X[i].tolist()}


def _change(levels) -> float:
    if len(levels) < 2:
        return 1.0
    a, b = levels[-2]["strength_ratio"], levels[-1]["strength_ratio"]
    return abs(b - a) / max(abs(b), 1e-12)


def _clt_check(model, res, lam, ref, zones, eff) -> Check:
    """Beam theory with the laminate's CLT membrane modulus along the part's axis, against the shell's strain.
    Only meaningful when the plies are measured from that axis."""
    from ..shells.checks import slender_axis

    sl = slender_axis(model.mesh.X)
    if sl is None:
        return Check("hand calculation", "skip", f"no simple beam model applies (the part is not slender); CLT gives "
                                                 f"Ex {fmt(eff['Ex'])}, Ey {fmt(eff['Ey'])} and Gxy {fmt(eff['Gxy'])} MPa")
    if abs(abs(ref[sl[0]]) - 1.0) > 1e-6:
        return Check("hand calculation", "skip", "the plies are not measured from the part's long axis, so no single "
                                                 "laminate modulus runs along the beam")
    from scipy.spatial import cKDTree

    mesh = model.mesh
    a = np.eye(3)[sl[0]]
    tree = cKDTree(mesh.X)

    def axial(pts, off):
        _, idx = tree.query(pts)
        out = []
        for i, z in zip(idx, off):
            s, E = res.profiles[i]
            T = _tensor(E)
            ea = np.einsum("i,kij,j->k", a, T, a)
            out.append(np.interp(z, np.sort(s), ea[np.argsort(s, kind="stable")]) if len(s) else 0.0)
        return np.asarray(out)

    return beam_hand_check(model, axial, zones, eff["Ex"], f"the laminate's CLT moduli, {fmt(eff['Ex'])} MPa "
                           f"stretching and {fmt(eff['Efx'])} MPa bending", eff["Efx"])


def _layup_check(lam: Laminate) -> Check:
    """Symmetry, balance and the 10% rule of thumb for the four main angles."""
    issues = []
    if not lam.symmetric():
        issues.append("not symmetric, so stretching it also bends and twists it (B is not zero)")
    if not lam.balanced():
        issues.append("not balanced, so pulling it along its axes also shears it")
    shares = {a: lam.share(a) for a in (0, 45, 90)}
    thin = [f"{a}" for a, s in shares.items() if s < 0.1]
    if thin:
        issues.append("under 10% of the plies at " + " and ".join(thin) + " degrees, so loads off the main axis "
                      "fall on the matrix")
    s = (f"{lam.angles()}: " + "; ".join(issues)) if issues else \
        f"{lam.angles()} is symmetric and balanced, with at least 10% of the plies at 0, 45 and 90 degrees"
    return Check("layup", "info" if issues else "pass", s, {"symmetric": lam.symmetric(), "balanced": lam.balanced(),
                                                            "shares": shares})


def _support_check(pf: PlyField, excl: np.ndarray, R: float, r_ex: float) -> Check:
    both = np.minimum(pf.tsai_wu, pf.hashin).min(1)
    if not excl.any() or both[excl].min() >= R:
        return Check("singularities", "pass", "no support or rigid attachment sets a lower strength ratio than the "
                                              "reported one")
    v = float(both[excl].min())
    return Check("singularities", "info" if v >= 0.67 * R else "warn",
                 f"within {fmt(r_ex)} mm of the supports and rigid attachments the first ply reads {v:.3g}x, set by how "
                 f"the model holds the laminate there; Wobbl reports the value away from them. A real clamp spreads "
                 "the load over its length, so check the joint itself with a test or a detailed model",
                 {"support_ratio": v, "radius_mm": r_ex})


def _sanity(shell, res, diag, pf) -> Check:
    umax = float(np.linalg.norm(res.U, axis=1).max())
    rel = umax / diag
    emax = float(np.nanmax(np.abs(pf.strain)))
    issues, st = [], "pass"
    if rel > 0.1:
        issues.append(f"max deflection {fmt(umax)} mm is {100 * rel:.0f}% of the part size, far outside small "
                      "deflection theory")
        st = "fail" if rel > 0.2 else "warn"
    if emax > 0.02:
        issues.append(f"strain {fmt(100 * emax)}% is beyond any laminate's failure strain; check the load units")
        st = "warn" if st == "pass" else st
    base = f"part {fmt(diag)} mm across, max deflection {fmt(umax)} mm ({100 * rel:.2g}% of size), peak ply strain " \
           f"{fmt(1e6 * emax)} microstrain"
    return Check("units and sanity", st, "; ".join(issues) if issues else base + ": all plausible",
                 {"deflection_ratio": rel, "max_strain": emax})


def _ply_table(lam: Laminate, pf: PlyField, excl) -> list[dict]:
    keep = ~excl if (~excl).any() else np.ones(len(excl), bool)
    rows = []
    for j, p in enumerate(lam.plies):
        tw = pf.tsai_wu[keep, j]
        hs = pf.hashin[keep, j]
        i = int(np.argmin(hs))
        rows.append({"ply": j + 1, "angle": _wrap(p.angle), "material": p.material.key, "thickness": p.thickness,
                     "tsai_wu": float(tw.min()), "hashin": float(hs.min()),
                     "hashin_mode": str(pf.mode[keep, j][i])})
    return rows


def _ply_section(table: list[dict]) -> str:
    from ..report.html import table as html_table

    rows = [[r["ply"], f"{r['angle']:g}", r["material"], f"{r['thickness']:g}", f"{r['tsai_wu']:.3g}",
             f"{r['hashin']:.3g}", r["hashin_mode"]] for r in table]
    return "<h2>Ply by ply</h2>" + html_table(["ply", "angle", "material", "mm", "Tsai-Wu ratio", "Hashin ratio",
                                               "Hashin mode"], rows, {0, 1, 3, 4, 5})


def _images(out, shell, res, pf, excl, gxyz, R, lam, table, target) -> dict:
    from ..shells.render import bar_chart, render_shell

    img = {}
    try:
        fi = 1.0 / np.minimum(pf.tsai_wu, pf.hashin).min(1)
        img["failure"] = render_shell(out / "failure.png", shell.X, shell.tris, fi, "failure index", "",
                                      mark_xyz=gxyz, mark_label=f"FI {1 / R:.2f}", clim=[0.0, max(1.0 / R, 1e-9)])
        img["plies"] = bar_chart(out / "plies.png", [f"{r['ply']}: {r['angle']:g}" for r in table],
                                 [min(r["tsai_wu"], r["hashin"]) for r in table], "strength ratio", target,
                                 f"target {target:g}", int(np.argmin([min(r["tsai_wu"], r["hashin"]) for r in table])))
        um = np.linalg.norm(res.U, axis=1)
        img["deformed"] = render_shell(out / "deformed.png", shell.X, shell.tris, um, "deflection", "mm",
                                       cmap="viridis", U=res.U, note="shape exaggerated")
    except Exception as e:  # noqa: BLE001  pictures are a bonus, never a reason to fail
        img["error"] = str(e)[:200]
    return img


def layup_fixes(lam: Laminate, N, M, R: float, target: float, ply: int, mode: str, text: str, ply_name: str):
    """Layup changes that reach the target, checked with CLT at the governing point.

    The force and moment per unit width there stay the same when plies are added to a part whose loads do not
    depend on its stiffness, so CLT at that point gives each candidate's new strength ratio without a new solve.
    Candidates add plies on both faces at the angle that carries the failing stress (fibers along it for a fiber
    mode, across the crack for matrix tension, a pair at 45 degrees for shear), or move the strongest plies out
    where bending strains are largest.
    """
    from .laminate import Ply

    p = lam.plies[ply]
    a = _wrap(p.angle)
    if "fiber" in mode:
        options = [(a, f"{fmt(a)} degree plies, along the fibers that are failing")]
    elif mode == "matrix tension":
        options = [(_wrap(a + 90), f"{fmt(_wrap(a + 90))} degree plies, which carry the load the cracking ply takes "
                                   "across its fibers"), (45, "a 45 degree pair")]
    else:
        options = [(45, "a 45 degree pair, which carries in plane shear"), (_wrap(a + 90), f"{fmt(_wrap(a + 90))} "
                                                                                            "degree plies")]
    out = []
    for ang, why in options:
        for pairs in range(1, 7):
            if abs(ang) == 45:
                add = [Ply(45, p.material, p.thickness), Ply(-45, p.material, p.thickness)] * pairs
            else:
                add = [Ply(ang, p.material, p.thickness)] * pairs
            new = Laminate(add + lam.plies + add[::-1])
            r = clt_first_ply_failure(new, N, M)
            r_new = min(r["tsai_wu"], r["hashin"])
            if r_new >= target:
                added = 2 * len(add)
                out.append({"kind": "layup", "auto": False, "layup": new.angles(), "strength_ratio": r_new,
                            "title": f"Add {added} plies, {why.split(',')[0]}",
                            "summary": f"Adding {why} on both faces ({added} plies, the laminate goes from "
                                       f"{fmt(lam.thickness)} to {fmt(new.thickness)} mm) lifts first ply failure "
                                       f"from {R:.3g}x to about {r_new:.3g}x the load by CLT at the governing point. "
                                       f"Check it with --layup \"{new.angles()}\" --ply {ply_name}."})
                break
    if np.abs(M).max() * 6 / lam.thickness ** 2 > np.abs(N).max() / lam.thickness:
        # bending dominates: the plies at the failing angle go to the faces, the rest inside
        half = sorted(lam.plies[:len(lam.plies) // 2], key=lambda q: abs(_wrap(q.angle) - a))
        new = Laminate(half + half[::-1] + ([lam.plies[len(lam.plies) // 2]] if len(lam.plies) % 2 else []))
        if len(new.plies) == len(lam.plies):
            r = clt_first_ply_failure(new, N, M)
            r_new = min(r["tsai_wu"], r["hashin"])
            if r_new > R * 1.05:
                out.append({"kind": "stack", "auto": False, "layup": new.angles(), "strength_ratio": r_new,
                            "title": "Move the failing direction's plies to the faces",
                            "summary": f"The part bends, so the outer plies see the most strain. Stacking it as "
                                       f"{new.angles()} with the same plies lifts first ply failure from {R:.3g}x to "
                                       f"about {r_new:.3g}x by CLT at the governing point, at no extra weight."})
    return out


def _assumptions(lam: Laminate, study: LaminateStudy, shell, r_ex: float) -> list[str]:
    mats = lam.materials()
    a = [f"Laminate {lam.describe()}, plies listed from the inner face out, angles measured from the "
         f"{study.reference or 'longest axis of the part'}."]
    for m in mats:
        a.append(f"{m.summary()} ({m.source}). {m.notes}".strip())
    a += ["Plies are linear elastic up to first ply failure; Wobbl does not follow the damage past it, where the "
          "laminate often carries more load.",
          "Tsai-Wu uses the interaction term F12 = -sqrt(F11 F22)/2; Hashin's matrix compression takes the "
          "transverse shear strength equal to S12.",
          "Shell theory: in plane ply stresses only. Free edge, interlaminar and through thickness stresses, and "
          "delamination, are not modeled.",
          f"The laminate sits on the midsurface of the CAD walls; {shell.notes[0][0].lower()}{shell.notes[0][1:]}"
          if shell.notes else "The laminate sits on the midsurface of the CAD walls.",
          f"Points within {fmt(r_ex)} mm of supports and rigid attachments are left out of the answer.",
          "Loads are static. Fatigue, impact, moisture and temperature knockdowns are not applied."]
    return a


__all__ = ["LaminateStudy", "run_laminate", "layup_fixes", "ply_field", "resultants", "HASHIN_MODES"]
