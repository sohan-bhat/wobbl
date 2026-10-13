"""Apply suggested fixes and re-run the study, so the user sees before and after numbers."""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import numpy as np

from ..geometry.cad import PreparedGeometry
from ..geometry.faces import analyze
from ..geometry.remap import remap_selector
from ..materials import resolve_material
from ..mesh.mesher import build_surface_mesh
from ..structural.outcome import Outcome
from ..structural.spec import StructuralStudy
from . import cad_edit


def _load(result_dir: Path):
    o = Outcome.load(result_dir / "outcome.json")
    study = StructuralStudy.from_dict(json.loads((result_dir / "study.json").read_text()))
    prep = PreparedGeometry.from_dict(json.loads((result_dir / "geometry.json").read_text()))
    return o, study, prep


def _corner_frame(feats, sm, gov_xyz: np.ndarray, face_id: int | None, corner=None):
    """The two flat walls meeting at the inside corner: normals, corner line point and direction."""
    if corner is not None:
        a, b = corner.faces
    else:
        curves = set(sm.surface_curves.get(face_id, []))
        adj = [s for s, cs in sm.surface_curves.items() if s != face_id and curves & set(cs)]
        flats = [feats.faces[s] for s in adj if s in feats.faces and feats.faces[s].kind == "plane"]
        fil = feats.faces.get(face_id)
        axis = np.asarray(fil.axis) if fil is not None and fil.axis is not None else None
        best, score = None, -1
        for i in range(len(flats)):
            for j in range(i + 1, len(flats)):
                n1, n2 = np.asarray(flats[i].normal), np.asarray(flats[j].normal)
                sc = 1 - abs(n1 @ n2)
                if axis is not None:  # the two walls must meet along the fillet, not across its ends
                    line = np.cross(n1, n2)
                    sc *= abs(line @ axis) / max(np.linalg.norm(line), 1e-12)
                if sc > score:
                    best, score = (flats[i].id, flats[j].id), sc
        if best is None or score < 0.3:
            return None
        a, b = best
    fa, fb = feats.faces[a], feats.faces[b]
    if fa.normal is None or fb.normal is None:
        return None
    na, nb = np.asarray(fa.normal), np.asarray(fb.normal)
    e = np.cross(na, nb)
    if np.linalg.norm(e) < 1e-6:
        return None
    e /= np.linalg.norm(e)
    A = np.array([na, nb, e])
    rhs = np.array([na @ np.asarray(fa.centroid), nb @ np.asarray(fb.centroid), e @ gov_xyz])
    p0 = np.linalg.solve(A, rhs)
    ext_a = max(((np.array(c) - p0) @ nb) for c in _corners(fa.bbox))
    ext_b = max(((np.array(c) - p0) @ na) for c in _corners(fb.bbox))
    span = [((np.array(c) - p0) @ e) for c in _corners(fa.bbox) + _corners(fb.bbox)]
    mid = p0 + e * 0.5 * (max(span) + min(span))
    return dict(na=na, nb=nb, e=e, p0=p0, ext=min(ext_a, ext_b), width=max(span) - min(span), mid=mid)


def _corners(bbox):
    lo, hi = bbox
    return [[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]


def _after(o2, extra: dict) -> dict:
    return {"governing_MPa": o2.governing_MPa, "safety_factor": o2.safety_factor, "verdict": o2.verdict,
            "confidence": o2.confidence, "report": o2.report, "stress_image": o2.images.get("stress", ""),
            "max_disp_mm": o2.max_disp_mm, **extra}


def _parametric_change(study, fix) -> dict | None:
    """For a parametric .py part, the parameter edit that implements a fix (or None)."""
    from ..geometry.parametric import is_parametric, load_params, radius_param, thickness_param

    if not is_parametric(study.part):
        return None
    params = {**load_params(study.part), **study.params}
    if fix["kind"] == "thicken":
        k = thickness_param(params)
        if k:
            return {k: math.ceil(float(fix["params"]["thickness"]) * 2) / 2}
    if fix["kind"] == "fillet":
        k = radius_param(params)
        if k:
            return {k: math.ceil(float(fix["params"]["radius"]) * 2) / 2}
    return None


def _rerun_parametric(study, change: dict, old_feats, out: Path, progress):
    from ..structural.pipeline import analyze_strength
    from ..structural.run import describe_part

    st2 = copy.deepcopy(study)
    st2.params = {**study.params, **{k: str(v) for k, v in change.items()}}
    pre = describe_part(st2.part, out / "work", st2.scale, st2.body, st2.params)
    for ld in st2.loads:
        if ld.faces:
            ld.faces = remap_selector(ld.faces, old_feats.faces, pre[1].faces, old_feats.bbox)
    for fx in st2.fixtures:
        fx.faces = remap_selector(fx.faces, old_feats.faces, pre[1].faces, old_feats.bbox)
    return analyze_strength(st2, out, progress, fixes=False, pre=pre)


def apply_fixes(result_dir: str | Path, which: list[str] | None = None, progress=None) -> dict:
    from ..structural.pipeline import analyze_strength
    from ..structural.report import write_outputs

    rd = Path(result_dir)
    o, study, prep = _load(rd)
    sm = build_surface_mesh(prep)
    feats = analyze(sm)
    results = []
    gov = np.asarray(o.governing_xyz)
    for fix in o.fixes:
        if not fix.get("auto") or (which and fix["kind"] not in which):
            continue
        try:
            if fix["kind"] == "material":
                m = resolve_material(fix["params"]["material"])
                old = resolve_material(study.material)
                fix["after"] = {"governing_MPa": o.governing_MPa, "safety_factor": m.allowable / o.governing_MPa,
                                "verdict": "holds" if m.allowable / o.governing_MPa >= o.target_sf else "marginal",
                                "max_disp_mm": o.max_disp_mm * old.E / m.E, "method": "same stress field, new limit"}
                results.append({"fix": fix["title"], **fix["after"]})
                continue
            new_prep = None
            out_geo = rd / f"fix-{fix['kind']}" / "geometry"
            param_change = _parametric_change(study, fix)
            if param_change:
                o2 = _rerun_parametric(study, param_change, feats, rd / f"fix-{fix['kind']}", progress)
                fix["after"] = _after(o2, extra={"params": param_change})
                results.append({"fix": fix["title"], **fix["after"]})
                continue
            if fix["kind"] == "fillet":
                p = fix["params"]
                face = p.get("face")
                reach = max(2.5 * (feats.faces[face].radius if face in feats.faces else 0.5), 1.0)
                new_prep = cad_edit.fillet(prep, out_geo, p["point"], p["radius"], face, reach)
            elif fix["kind"] == "gusset":
                p = fix["params"]
                corner = None
                if p.get("corner_faces"):
                    corner = next((c for c in feats.corners if set(c.faces) == set(p["corner_faces"])), None)
                fr = _corner_frame(feats, sm, gov, p.get("face"), corner)
                if fr is None:
                    raise cad_edit.EditError("could not find two flat walls meeting at the peak")
                t = min(float(p["thickness"]), 0.8 * fr["width"])
                leg = min(0.5 * fr["ext"], 5 * t)
                new_prep = cad_edit.gusset(prep, out_geo, fr["p0"], fr["na"], fr["nb"], fr["e"], fr["mid"], leg, t)
            if new_prep is None:
                continue
            new_feats = analyze(build_surface_mesh(new_prep))
            st2 = copy.deepcopy(study)
            st2.part = str(new_prep.path.with_suffix(".step"))
            for ld in st2.loads:
                if ld.faces:
                    ld.faces = remap_selector(ld.faces, feats.faces, new_feats.faces, feats.bbox)
            for fx in st2.fixtures:
                fx.faces = remap_selector(fx.faces, feats.faces, new_feats.faces, feats.bbox)
            o2 = analyze_strength(st2, rd / f"fix-{fix['kind']}", progress, fixes=False, pre=(new_prep, new_feats))
            fix["after"] = _after(o2, extra={"step": st2.part})
            results.append({"fix": fix["title"], **fix["after"]})
        except Exception as e:  # noqa: BLE001  one failed edit should not stop the others
            fix["after"] = {"error": str(e)}
            results.append({"fix": fix["title"], "error": str(e)})
    best = max((f for f in o.fixes if f.get("after", {}).get("stress_image")),
               key=lambda f: f["after"]["safety_factor"], default=None)
    if best is not None:
        o.images["fix_before"] = o.images.get("stress", "")
        o.images["fix_after"] = best["after"]["stress_image"]
    write_outputs(o, rd)
    return {"before": {"governing_MPa": o.governing_MPa, "safety_factor": o.safety_factor, "verdict": o.verdict},
            "after": results, "report": o.report}
