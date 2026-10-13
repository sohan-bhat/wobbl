"""The answer to a strength question: number, place, safety factor, verdict, checks, confidence, assumptions."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from ..units import fmt, fmt_force, fmt_length, fmt_stress
from ..verify.beam import beam_estimate
from ..verify.checks import Check, _jsonable, balance_check, confidence, convergence_check
from .run import StructuralRun


@dataclass
class Outcome:
    domain: str
    part: str
    material: str
    question: str
    measure: str
    governing_MPa: float
    governing_xyz: list[float]
    governing_where: str
    allowable_MPa: float
    safety_factor: float
    target_sf: float
    verdict: str                 # holds, marginal, fails
    headline: str
    max_disp_mm: float
    max_disp_xyz: list[float]
    confidence: str
    confidence_reasons: list[str]
    checks: list[Check]
    assumptions: list[str]
    hotspots: list[dict]
    levels: list[dict]
    loads: list[str]
    fixtures: list[str]
    selections: dict
    features: dict
    notes: list[str]
    imperial: bool = False
    mass_kg: float = 0.0
    beam: dict | None = None
    fixes: list[dict] = field(default_factory=list)
    images: dict[str, str] = field(default_factory=dict)
    report: str = ""
    workdir: str = ""
    t_total: float = 0.0
    solver: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["checks"] = [c.to_dict() for c in self.checks]
        return _jsonable(d)

    def save(self, path: str | Path):
        from ..verify.checks import relative_paths
        Path(path).write_text(json.dumps(relative_paths(self.to_dict(), Path(path).parent), indent=1))

    @classmethod
    def load(cls, path: str | Path) -> "Outcome":
        from ..verify.checks import absolute_paths
        d = absolute_paths(json.loads(Path(path).read_text()), Path(path).parent)
        d["checks"] = [Check(**c) for c in d["checks"]]
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def _where(run, gspot, xyz: np.ndarray) -> str:
    """Name the governing location: the feature it sits on, else the nearest face and the coordinates."""
    from .hotspots import nearest_face

    at = f"({', '.join(f'{v:.3g}' for v in xyz)})"
    if gspot is not None and np.linalg.norm(np.asarray(gspot.xyz) - xyz) <= 2.5 * gspot.radius and gspot.detail \
            and "body of the part" not in gspot.detail:
        return f"{gspot.detail} at {at}"
    fid = nearest_face(run.mesh, xyz)
    f = run.features.faces.get(fid) if fid is not None else None
    if f is None:
        return f"inside the part at {at}"
    if f.role == "fillet":
        return f"on the {f.radius:.3g} mm inside fillet (face {f.id}) at {at}"
    if f.role == "hole":
        return f"at the edge of the {2 * f.radius:.3g} mm hole (face {f.id}) at {at}"
    return f"on face {f.id} at {at}"


def _verdict(sf: float, target: float) -> str:
    if sf >= target:
        return "holds"
    if sf >= 1.0:
        return "marginal"
    return "fails"


def evaluate(run: StructuralRun) -> Outcome:
    st, mat, lv = run.study, run.material, run.levels[-1]
    imperial = st.imperial or any(l.imperial for l in st.loads)
    mesh, res = run.mesh, run.result
    gov = lv.governing
    allow = mat.allowable
    sf = allow / gov if gov > 0 else float("inf")
    target = st.target_sf or mat.recommended_sf
    verdict = _verdict(sf, target)
    gspot = None
    feats = [h for h in run.hotspots if h.kind == "feature"]
    if feats:
        gspot = min(feats, key=lambda h: float(np.linalg.norm(np.asarray(h.xyz) - np.asarray(lv.governing_xyz))))
    where = _where(run, gspot, np.asarray(lv.governing_xyz))
    checks: list[Check] = []
    extra: list[tuple[str, str]] = []

    # 1. mesh convergence of the governing stress (and the deflection alongside)
    gov_vals = [L.governing for L in run.levels]
    conv = convergence_check(gov_vals, run.tol, run.measure_name, "MPa", run.stopped_by,
                             [L.max_disp for L in run.levels])
    bound = max(max(L.raw_max for L in run.levels), max(gov_vals))
    if conv.status != "pass" and bound > 0 and allow / bound >= target * 1.2:
        # the number is still moving, but even the highest peak seen on any mesh, artifacts included, leaves
        # plenty of margin: the verdict does not depend on the unsettled digits
        conv = Check("mesh convergence", "warn", conv.summary + f". Even the highest peak on any mesh "
                     f"({fmt(bound)} MPa, artifacts included) leaves safety factor {allow / bound:.2f}, so the "
                     "verdict holds either way", conv.detail)
    checks.append(conv)
    dvals = [L.max_disp for L in run.levels]
    if len(dvals) >= 2 and dvals[-1] > 0 and abs(dvals[-1] - dvals[-2]) / dvals[-1] > 0.05:
        extra.append(("medium", f"max deflection still changing ({fmt(dvals[-2])} -> {fmt(dvals[-1])} mm)"))

    # 2. equilibrium
    partial = bool(run.built.model.normal_constraints) and any(
        np.count_nonzero(np.abs(nc.normal) > 1e-6) > 1 for nc in run.built.model.normal_constraints)
    checks.append(balance_check(np.asarray(lv.applied), np.asarray(lv.reaction), np.asarray(lv.applied_moment),
                                np.asarray(lv.reaction_moment), run.prep.diag, partial))

    # 3. hand calculation: beam theory through the governing point
    beam = None
    model = run.built.model
    fix_nodes = np.unique(np.concatenate([c.nodes for c in model.constraints] +
                                         [c.nodes for c in model.normal_constraints] or [np.zeros(0, int)]))
    load_nodes = np.nonzero(np.abs(model.nodal_forces()).sum(1) > 0)[0]
    extra_pts = [np.asarray(r.ref_point) for r in model.rigid]
    if model.gravity is not None and np.linalg.norm(model.gravity) > 0:
        checks.append(Check("hand calculation", "skip", "beam check skipped: self weight acts everywhere"))
    else:
        F, M = model.applied_total()
        est, why = beam_estimate(mesh, np.asarray(lv.governing_xyz), F, M, load_nodes, fix_nodes, extra_pts)
        if est is None:
            checks.append(Check("hand calculation", "skip", f"no simple beam model applies ({why})"))
        else:
            beam = est
            nom = est["nominal_peak_MPa"]
            ratio = gov / nom if nom > 0 else float("inf")
            est["ratio"] = ratio
            s = (f"beam theory through the critical section gives {fmt(nom)} MPa nominal; FEA peak is "
                 f"{ratio:.2f}x that")
            if 0.85 <= ratio <= 1.2:
                checks.append(Check("hand calculation", "pass", s + ", a plain beam region", est))
            elif 1.2 < ratio <= 4.0:
                checks.append(Check("hand calculation", "pass",
                                    s + f", a stress concentration of about {ratio:.1f} from the local shape", est))
            elif 0.6 <= ratio < 0.85:
                checks.append(Check("hand calculation", "info",
                                    s + "; the part is stiffer than a plain beam there (load spreads into more "
                                        "material)", est))
            else:
                checks.append(Check("hand calculation", "warn",
                                    s + "; that is far from beam theory, so check the loads, units and fixtures",
                                    est))

    # 4. singularities and idealized supports
    raw = lv.raw_max
    zone_spots = [h for h in run.hotspots if h.kind != "feature"]
    high_zone = [h for h in zone_spots if h.value > gov]
    if high_zone:
        parts = []
        for h in high_zone[:3]:
            vals = [v for v in h.values if v == v]
            rising = len(vals) >= 2 and all(b > a * 1.03 for a, b in zip(vals, vals[1:]))
            if h.kind == "corner":
                kind = ("singular, keeps rising with refinement" if rising else
                        "a sharp corner: singular in theory, so the value depends on the mesh")
            else:
                kind = "keeps rising with refinement" if rising else "set by the idealized support"
            parts.append(f"{fmt(h.value)} MPa {h.detail} ({kind}: {h.trend()} MPa)")
        msg = ("excluded from the answer: " + "; ".join(parts) +
               f". Wobbl reports the stress {fmt(run.exclusion_radius)} mm or more away from these spots")
        corner = any(h.kind == "corner" for h in high_zone)
        status = "warn" if (corner and not mat.ductile) or max(h.value for h in high_zone) > 1.5 * gov else "info"
        checks.append(Check("singularities", status, msg, {"spots": [h.to_dict() for h in high_zone]}))
        if corner and not mat.ductile:
            extra.append(("medium", "a sharp inside corner in a brittle material: the real peak depends on the "
                                    "actual corner radius, add a fillet to get a trustworthy number"))
    else:
        checks.append(Check("singularities", "pass", "no sharp corner or support artifact is above the reported "
                                                     "stress"))

    # 5. bolt bearing at bolted or fixed holes, the honest number where the support was idealized
    bc = _bearing_checks(run)
    checks += bc

    # 6. sanity: units, size, deflection
    checks.append(_sanity(run, gov, allow))

    # 7. large deflection: compare a geometrically nonlinear solve on the first mesh
    nl = run.nonlinear
    if nl and "error" not in nl and nl["linear_peak"] > 0:
        ps = nl["nonlinear_peak"] / nl["linear_peak"] - 1
        ds = nl["nonlinear_disp"] / max(nl["linear_disp"], 1e-12) - 1
        s = (f"deflection is large next to the wall thickness, so Wobbl re-solved the first mesh with large "
             f"deflection theory: the peak moves {100 * ps:+.0f}% and the deflection {100 * ds:+.0f}%")
        if abs(ps) <= 0.10:
            checks.append(Check("large deflection", "pass", s + "; linear theory holds", nl))
        else:
            est = gov * (1 + ps)
            checks.append(Check("large deflection", "warn", s + f". Scaling the converged answer gives about "
                                f"{fmt(est)} MPa; " + ("linear theory is conservative here" if ps < 0 else
                                                       "linear theory underestimates here"), nl))
    elif nl and "error" in nl:
        checks.append(Check("large deflection", "info", "deflection is large next to the wall thickness, but the "
                                                        f"large deflection solve did not finish ({nl['error']})"))

    level, reasons = confidence(checks, extra)
    if run.stopped_by in ("time budget", "node limit") and checks[0].status != "pass":
        reasons.append(f"refinement stopped by the {run.stopped_by}")
    mass = float(model.density * mesh.volume() * 1000.0)  # t -> kg
    head = _headline(verdict, gov, sf, target, mat, where, imperial, level)
    out = Outcome(
        "strength", str(st.part), mat.name, st.question, run.measure_name, gov, lv.governing_xyz, where, allow, sf,
        target, verdict, head, lv.max_disp, lv.max_disp_xyz, level, reasons, checks, assumptions(run),
        [h.to_dict() for h in run.hotspots], [asdict(L) for L in run.levels],
        [l.describe() for l in st.loads], [f.describe() for f in st.fixtures], run.selections,
        run.features.to_dict(max_faces=60), run.notes, imperial, mass, beam, workdir=str(run.workdir),
        t_total=run.t_total, solver=res.solver)
    return out


def _headline(verdict, gov, sf, target, mat, where, imperial, level) -> str:
    s = fmt_stress(gov, imperial)
    limit = mat.criterion.split(" vs ")[1]
    if verdict == "holds":
        v = f"Peak {s} {where}, safety factor {sf:.2f} against {limit} (target {target:g})."
    elif verdict == "marginal":
        v = (f"Peak {s} {where}. It survives the load, but the safety factor {sf:.2f} is below the "
             f"{target:g} target.")
    else:
        what = "yields" if mat.ductile else "cracks"
        v = f"Peak {s} {where}. The part {what} at about {sf:.2f}x this load."
    if level != "high":
        v += f" Confidence is {level}."
    return v


def _bearing_checks(run: StructuralRun) -> list[Check]:
    out = []
    model, mesh, res = run.built.model, run.mesh, run.result
    faces = run.features.faces
    allow = run.material.allowable * (1.5 if run.material.ductile and run.material.category == "metal" else 1.0)
    for k, fx in enumerate(run.study.fixtures):
        name = f"fix{k + 1}"
        sel = run.selections.get(name, {})
        holes = [faces[i] for i in sel.get("faces", []) if i in faces and faces[i].role == "hole"]
        if not holes or fx.kind not in ("fixed", "bolt"):
            continue
        groups = {}
        for h in holes:
            groups.setdefault(h.group if h.group is not None else h.id, []).append(h)
        worst = None
        for g in groups.values():
            h = g[0]
            nodes = mesh.surface_nodes([x.id for x in g])
            if fx.kind == "bolt":
                c = next((c for c in model.constraints if c.name == name), None)
                if c is not None:
                    a = np.asarray(h.axis)
                    rel = mesh.X[c.nodes] - np.asarray(h.axis_point)
                    radial = np.linalg.norm(rel - np.outer(rel @ a, a), axis=1)
                    nodes = c.nodes[radial <= 2.05 * h.radius]
            R = res.RF[nodes].sum(0)
            a = np.asarray(h.axis)
            Rp = R - (R @ a) * a
            L = max(h.length or 0.0, 1e-6)
            sb = float(np.linalg.norm(Rp)) / (2 * h.radius * L)
            if worst is None or sb > worst[0]:
                worst = (sb, h, float(np.linalg.norm(Rp)), float(abs(R @ a)))
        if worst is None:
            continue
        sb, h, Rp, Ra = worst
        sf = allow / sb if sb > 0 else float("inf")
        msg = (f"worst hole of {name} (d {fmt(2 * h.radius)} mm) carries {fmt(Rp)} N sideways and {fmt(Ra)} N along "
               f"its axis; bearing stress {fmt(sb)} MPa, safety factor {sf:.2f} on bearing")
        status = "pass" if sf >= (run.study.target_sf or run.material.recommended_sf) else ("warn" if sf >= 1 else "fail")
        out.append(Check("bolt bearing", status, msg, {"fixture": name, "bearing_MPa": sb, "shear_N": Rp,
                                                         "axial_N": Ra, "sf": sf}))
    return out


def _sanity(run: StructuralRun, gov: float, allow: float) -> Check:
    diag = run.prep.diag
    lv = run.levels[-1]
    issues, status = [], "pass"
    d_rel = lv.max_disp / diag if diag > 0 else 0
    if d_rel > 0.20:
        issues.append(f"max deflection {fmt(lv.max_disp)} mm is {100 * d_rel:.0f}% of the part size, far outside "
                      "small deflection theory; the real part bends differently or collapses")
        status = "fail"
    elif d_rel > 0.06:
        issues.append(f"max deflection {fmt(lv.max_disp)} mm is {100 * d_rel:.1f}% of the part size; linear "
                      "results get less accurate past a few percent")
        status = "warn"
    if gov > 30 * allow:
        issues.append(f"stress is {gov / allow:.0f}x the material limit; check the load magnitude and units")
        status = "warn" if status == "pass" else status
    if 0 < gov < 1e-4 * allow:
        issues.append("stress is a ten thousandth of the material limit; check the load magnitude and units")
        status = "warn" if status == "pass" else status
    if diag < 1.0 or diag > 20000:
        issues.append(f"the part is {fmt(diag)} mm across; check the file units")
        status = "warn" if status == "pass" else status
    E = run.material.E
    strain = gov / E
    base = (f"part {fmt(diag)} mm across, peak strain {strain:.2g}, max deflection {fmt(lv.max_disp)} mm "
            f"({100 * d_rel:.2g}% of size)")
    return Check("units and sanity", status, "; ".join(issues) if issues else base + ": all plausible",
                 {"diag_mm": diag, "strain": strain, "deflection_ratio": d_rel})


def assumptions(run: StructuralRun) -> list[str]:
    st, mat = run.study, run.material
    a = [
        "Linear elastic material and small deflections (no yielding redistribution, no buckling).",
        f"{mat.name}: E {fmt(mat.E)} MPa, Poisson {mat.nu:g}, {mat.criterion} {fmt(mat.allowable)} MPa "
        f"({mat.source}).",
        "Loads are static and applied all at once; no impact, vibration or fatigue.",
    ]
    if mat.category == "printed plastic":
        if st.print_direction:
            a.append(f"Printed layers stack along {st.print_direction}; pull across layers is judged against "
                     f"{mat.layer_factor:.0%} of the in-plane strength.")
        else:
            a.append("Printed part treated as isotropic at 100% infill; across the layers it is roughly "
                     f"{mat.layer_factor:.0%} as strong. Pass --print-direction to check that.")
    if not mat.ductile:
        a.append("Brittle failure: judged by the largest tensile principal stress.")
    kinds = {f.kind for f in st.fixtures}
    if "fixed" in kinds:
        a.append("Fixed faces are perfectly rigid; real mounts flex a little, which lowers peak stress next to them.")
    if "bolt" in kinds:
        a.append("Bolted holes are rigidly held over the hole and a washer sized ring (twice the hole radius).")
    if "wall" in kinds:
        a.append("Wall supports push but cannot pull (frictionless contact, solved iteratively).")
    if "roller" in kinds:
        a.append("Roller faces slide freely in their plane.")
    if any(l.kind == "remote" and l.coupling == "rigid" for l in st.loads):
        a.append("Remote loads reach the part through a rigid attachment (the attached part does not flex).")
    if any(l.kind in ("remote", "moment") and l.coupling != "rigid" for l in st.loads):
        a.append("Remote loads and moments are spread over their faces without stiffening them.")
    if any(l.kind == "bolt" for l in st.loads):
        a.append("Bolt loads press on the loaded half of each hole with a cosine distribution.")
    a.append("Contact between separate parts, friction and preload are not modeled.")
    if run.notes:
        a += [n for n in run.notes if "assume" in n.lower() or "bonded" in n.lower()]
    return a + list(st.assumptions)


def text_summary(o: Outcome) -> str:
    """Plain language summary an agent can read in one pass."""
    imp = o.imperial
    lines = [f"{o.verdict.upper()}. {o.headline}", ""]
    lines.append(f"Confidence: {o.confidence}. " + "; ".join(o.confidence_reasons[:4]))
    lines.append(f"Max deflection {fmt_length(o.max_disp_mm, imp)} at ({', '.join(fmt(v) for v in o.max_disp_xyz)}).")
    lines.append(f"Material {o.material}, limit {fmt_stress(o.allowable_MPa, imp)}; mass {o.mass_kg * 1000:.4g} g.")
    lines.append("Loads: " + "; ".join(o.loads) + ". Fixtures: " + "; ".join(o.fixtures) + ".")
    lines.append("")
    lines.append("Checks:")
    for c in o.checks:
        lines.append(f"  [{c.status}] {c.name}: {c.summary}")
    lines.append("")
    lines.append("Assumptions:")
    for a in o.assumptions:
        lines.append(f"  - {a}")
    if o.fixes:
        lines.append("")
        lines.append("Suggested fixes:")
        for f in o.fixes:
            lines.append(f"  - {f.get('summary', '')}")
    from ..verify.checks import display_path
    if o.images:
        lines.append("")
        lines.append("Images: " + ", ".join(f"{k}: {display_path(v, o.workdir)}" for k, v in o.images.items()))
    if o.report:
        lines.append(f"Report: {display_path(o.report, o.workdir)}")
    _ = fmt_force
    return "\n".join(lines)
