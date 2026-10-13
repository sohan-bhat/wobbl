"""Concrete fixes with numbers: thicken, fillet, add a gusset, swap material, shorten the lever.

Each suggestion carries the reasoning and a number. Fixes Wobbl can apply to
the geometry by itself (fillets and gussets on STEP files, any parameter of a
parametric part, material swaps) are marked ``auto`` and can be re-run with
``wobbl fix --apply`` to show before and after.
"""
from __future__ import annotations

import numpy as np

from ..geometry.faces import ray_hits
from ..materials import LIBRARY, PRINTABLE, SWAP_ORDER
from ..units import fmt


def local_thickness(mesh, point: np.ndarray) -> tuple[float | None, np.ndarray | None]:
    """Wall thickness through a surface point, measured along the inward normal."""
    n, a, c = mesh.tri_geometry()
    i = np.argsort(np.linalg.norm(c - point, axis=1))[:5]
    nrm = (n[i] * a[i][:, None]).sum(0)
    if np.linalg.norm(nrm) == 0:
        return None, None
    nrm /= np.linalg.norm(nrm)
    t = mesh.tris[:, :3]
    d = ray_hits(np.atleast_2d(point - nrm * 1e-6), np.atleast_2d(-nrm), mesh.X[t[:, 0]], mesh.X[t[:, 1]],
                 mesh.X[t[:, 2]])
    return (float(d[0]) if np.isfinite(d[0]) else None), nrm


def suggest_fixes(run, o) -> list[dict]:
    mat = run.material
    gov = o.governing_MPa
    target = o.target_sf
    need = gov * target / mat.allowable          # how many times too stressed for the target
    fixes: list[dict] = []
    corner_spots = [h for h in run.hotspots if h.kind == "corner" and h.value > 0.8 * gov]
    gspot = None
    feats = [h for h in run.hotspots if h.kind == "feature"]
    if feats:
        gspot = min(feats, key=lambda h: float(np.linalg.norm(np.asarray(h.xyz) - np.asarray(o.governing_xyz))))
    face = run.features.faces.get(gspot.face) if gspot and gspot.face is not None else None
    t_loc, _ = local_thickness(run.mesh, np.asarray(o.governing_xyz))
    if t_loc and face is not None and face.role == "fillet":
        t_loc = min(t_loc, run.features.thickness_typical)  # a ray through a corner runs diagonally
    beam = o.beam or {}

    if o.verdict == "holds" and not corner_spots:
        if o.safety_factor > 4 * target:
            fixes.append(_lighten(o, mat, t_loc, beam))
        return [f for f in fixes if f]

    # 1. a fillet where the peak sits on a sharp corner or a tight fillet
    small_fillet = face is not None and face.role == "fillet" and t_loc and face.radius < 0.4 * t_loc
    if corner_spots or small_fillet:
        r_now = face.radius if small_fillet else 0.0
        r_new = max(0.5 * (t_loc or run.features.thickness_typical), 2 * r_now, 1.0)
        nominal = beam.get("nominal_peak_MPa")
        est = ""
        if nominal:
            peak = 1.4 * nominal
            est = (f" With that radius the peak is usually about 1.4x the {fmt(nominal)} MPa beam stress, roughly "
                   f"{fmt(peak)} MPa (safety factor {mat.allowable / peak:.2f}).")
        where = "the sharp inside corner" if corner_spots else f"the {fmt(r_now)} mm fillet"
        fixes.append({
            "kind": "fillet", "title": f"Round {where} to a {fmt(r_new)} mm radius",
            "summary": f"Stress piles up in {where}. A fillet of {fmt(r_new)} mm (about half the "
                       f"{fmt(t_loc or run.features.thickness_typical)} mm wall) spreads it out." + est,
            "params": {"radius": r_new, "point": o.governing_xyz if not corner_spots else corner_spots[0].xyz},
            "auto": True})

    # 2. thicker wall at the peak
    if t_loc and need > 1:
        bending = True
        if beam:
            area = beam.get("area_mm2", 0)
            axial = abs(beam.get("axial_N", 0)) / area if area else 0
            bending = beam.get("nominal_peak_MPa", 0) > 2 * axial
        factor = np.sqrt(need) if bending else need
        t_new = t_loc * factor
        mode = "bending stress falls with thickness squared" if bending else "direct stress falls with thickness"
        fixes.append({
            "kind": "thicken", "title": f"Thicken the wall at the peak from {fmt(t_loc)} to {fmt(t_new)} mm",
            "summary": f"The wall through the peak is {fmt(t_loc)} mm. To reach safety factor {target:g} it needs "
                       f"{fmt(t_new)} mm (+{100 * (factor - 1):.0f}%), because {mode}.",
            "params": {"thickness": t_new, "from": t_loc, "point": o.governing_xyz}, "auto": False})

    # 3. a gusset or rib across an inside corner carrying bending
    corner_like = corner_spots or (face is not None and face.role == "fillet")
    if corner_like and need > 1:
        fixes.append({
            "kind": "gusset", "title": "Add a triangular gusset across the inside corner",
            "summary": "The peak sits where two walls meet and bend. A gusset as thick as the wall, with legs about "
                       "half as long as the shorter wall, ties them together and usually cuts the corner stress by "
                       "half or more.",
            "params": {"point": o.governing_xyz, "thickness": t_loc or run.features.thickness_typical},
            "auto": True})

    # 4. a stronger material
    swap = _material_swap(mat, gov, target)
    if swap:
        fixes.append(swap)

    # 5. shorter lever
    if beam and beam.get("bending_Nmm"):
        F = np.linalg.norm(run.levels[-1].applied)
        if F > 0:
            arm = beam["bending_Nmm"] / F
            if need > 1:
                new_arm = arm / need
                fixes.append({
                    "kind": "lever", "title": f"Shorten the lever arm from {fmt(arm)} to {fmt(new_arm)} mm",
                    "summary": f"The load acts {fmt(arm)} mm from the critical section. Moving a support or the load "
                               f"to within {fmt(new_arm)} mm brings the bending stress to the target.",
                    "params": {"arm": arm, "new_arm": new_arm}, "auto": False})
    return [f for f in fixes if f]


def _material_swap(mat, gov, target) -> dict | None:
    family = PRINTABLE if mat.key in PRINTABLE else None
    options = []
    for key in SWAP_ORDER:
        m = LIBRARY[key]
        if key == mat.key or (family is not None and key not in family) or \
                (family is None and m.category != mat.category):
            continue
        sf = m.allowable / gov
        if sf >= min(target, m.recommended_sf if family is None else target):
            options.append((m, sf))
    if not options:
        return None
    best = options[0]
    others = ", ".join(f"{m.name} ({sf:.2f})" for m, sf in options[1:3])
    stiff = best[0].E / mat.E
    return {
        "kind": "material", "title": f"Switch to {best[0].name}",
        "summary": f"Stress stays about the same for the same shape and load, so {best[0].name} "
                   f"({fmt(best[0].allowable)} MPa) gives safety factor {best[1]:.2f}. It is {stiff:.2g}x as stiff, "
                   f"so deflection scales by {1 / stiff:.2g}." + (f" Other options: {others}." if others else ""),
        "params": {"material": best[0].key, "safety_factor": best[1]}, "auto": True}


def _lighten(o, mat, t_loc, beam) -> dict | None:
    if not t_loc:
        return None
    ratio = o.safety_factor / o.target_sf
    t_new = t_loc / np.sqrt(ratio)
    return {"kind": "lighten", "title": f"Room to save material: the wall at the peak could be about {fmt(t_new)} mm",
            "summary": f"Safety factor {o.safety_factor:.1f} is {ratio:.1f}x the target. Bending stress grows with "
                       f"1/t^2, so the {fmt(t_loc)} mm wall could thin to about {fmt(t_new)} mm and still meet "
                       f"{o.target_sf:g}. Re-run after changing it.",
            "params": {"thickness": t_new, "from": t_loc}, "auto": False}
