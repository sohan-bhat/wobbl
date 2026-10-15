"""Forced convection cooling of any part: the flow sets how well each face sheds heat, CalculiX the temperature.

The CFD carries temperature as a passive scalar with every wetted wall held at one and the incoming fluid at zero,
so the heat each wall face passes to the cell beside it, divided by that unit difference, is the face's heat
transfer coefficient, the way a datasheet defines it against the incoming fluid. In laminar flow that heat is
plain conduction across the first cell. In turbulent flow the first cell sits in the wall layer, so the heat
follows Jayatilleke's thermal wall function: the friction velocity from the wall function's shear, the linear
T+ = Pr y+ inside the thermal sublayer and T+ = Prt (u+ + P) beyond it, whose P term carries the extra resistance
a fluid with a Prandtl number above one has in the sublayer (water and oils, where a plain eddy diffusivity would
double the answer). Each CAD face gets the area weighted mean of its wall faces, and faces the flow never touches
(the outside of a cold plate) get still air. The fluid warms as it takes the heat, so the films use
its mean temperature, the inlet plus half the rise. This is one way coupling, which holds when the part's
temperature changes the fluid's properties little; it is how heat sinks and cold plates are sized.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from ..units import fmt
from ..verify.checks import Check, confidence
from .fluids import get_fluid, qty

STILL_AIR = 5.0          # W/(m2 K) on faces the flow does not reach
PRANDTL_T = 0.85
KAPPA, E_WALL = 0.41, 9.8


def jayatilleke_tplus(y_plus: np.ndarray, pr: float, prt: float = PRANDTL_T) -> np.ndarray:
    """Dimensionless temperature at y+ for fluid of Prandtl number pr (Jayatilleke 1969, as OpenFOAM uses it)."""
    ratio = pr / prt
    pfun = 9.24 * (ratio ** 0.75 - 1) * (1 + 0.28 * np.exp(-0.007 * ratio))
    lo, hi = 0.1, 200.0                      # where the linear and log laws meet
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        lin, log = pr * mid, prt * (np.log(E_WALL * mid) / KAPPA + pfun)
        lo, hi = (mid, hi) if lin < log else (lo, mid)
    y_t = 0.5 * (lo + hi)
    y = np.maximum(np.asarray(y_plus, float), 1e-6)
    return np.where(y < y_t, pr * y, prt * (np.log(E_WALL * y) / KAPPA + pfun))


def face_coefficients(o) -> dict[int, tuple[float, float]]:
    """Each wetted CAD face's heat transfer coefficient, W/(m2 K), and wetted area, mm2, from the finest grid."""
    from scipy.spatial import cKDTree

    g, p, s, f = o.geometry, o.plan, o.setup, o.fine["fields"]
    fl = get_fluid(o.values["fluid"], s.temp)
    C, T = f["C"], f["T"]
    own, cen, area = f["wall_owner"], f["wall_centres"], f["wall_areas"]
    a = np.linalg.norm(area, axis=1)
    n = area / np.maximum(a[:, None], 1e-300)
    d = np.abs(np.einsum("ij,ij->i", cen - C[own], n))
    d = np.maximum(d, 1e-12)
    drive = 1.0 - T[own]
    if s.turb.model == "laminar":
        h = fl.k * drive / d
    else:
        nut = f.get("wall_nut")
        nut = np.zeros(len(own)) if nut is None or np.size(nut) != len(own) else np.asarray(nut).ravel()
        u = f["U"][own]
        ut = np.linalg.norm(u - np.einsum("ij,ij->i", u, n)[:, None] * n, axis=1)
        u_tau = np.sqrt((s.nu + nut) * ut / d)
        h = fl.rho * fl.cp * u_tau * drive / jayatilleke_tplus(u_tau * d / s.nu, fl.Pr)
    pts = cen * 1e3
    if s.mode in ("external", "ducted"):
        pts = pts @ p.R + (g.lo + g.hi) / 2
    tri_c = g.xyz.mean(1)
    _, tri_of = cKDTree(tri_c).query(pts)
    face_of = g.tri_face[tri_of]
    out = {}
    for fid in np.unique(face_of):
        m = face_of == fid
        out[int(fid)] = (float(np.sum(h[m] * a[m]) / np.sum(a[m])), float(a[m].sum() * 1e6))
    return out


def run_cooling(st, out_dir: str | Path, power: str, heat_faces: str, material: str, limit: str | None,
                progress=None, geometry=None):
    from ..outcome_generic import GenericOutcome
    from ..thermal.study import Film, ThermalStudy, run_thermal
    from .partstudy import run_partflow

    t0 = time.time()
    say = progress or (lambda a, b: None)
    out = Path(out_dir)
    flow_out = out / "flow"
    o = run_partflow(st, flow_out, say, geometry=geometry, heat=True)
    if o.plan.mode == "volume":
        raise ValueError("the part is the fluid itself here, so there is no solid to cool; give the solid part")
    s = o.setup
    fl = get_fluid(st.fluid, st.temp)
    watts = qty(power, "power")
    coeffs = face_coefficients(o)
    q = s.flow if o.plan.mode != "external" else 0.0
    rise = watts / (fl.rho * fl.cp * q) if q else 0.0
    t_ref = st.temp + rise / 2
    films, notes = [], []
    wet = set()
    for fid, (h, wet_area) in coeffs.items():
        face_area = o.geometry.faces[fid].area if fid in o.geometry.faces else wet_area
        if wet_area < 0.2 * face_area:      # a face the grid barely touched keeps still air
            continue
        wet.add(fid)
        films.append(Film(h * 1e-3, t_ref, f"face {fid}", f"{fmt(h)} W/m2K from CFD"))
    dry = [fid for fid in o.geometry.faces if fid not in wet]
    for fid in dry:
        films.append(Film(STILL_AIR * 1e-3, st.temp, f"face {fid}", "still air"))
    say("heat", f"{len(wet)} faces cooled by the flow, {len(dry)} by still air, CalculiX")
    th = ThermalStudy(str(Path(st.part).resolve()), material, watts * 1e3, heat_faces, films, ambient=st.temp,
                      limit=qty(limit, "temperature") if limit else None, question=st.question)
    to = run_thermal(th, out / "heat", say)
    hs = np.array([coeffs[f][0] for f in wet]) if wet else np.zeros(1)
    aw = np.array([coeffs[f][1] for f in wet]) if wet else np.ones(1)
    h_mean = float(np.sum(hs * aw) / np.sum(aw))
    checks = list(o.checks) + list(to.checks)
    ua = h_mean * float(aw.sum()) * 1e-6
    t_lump = t_ref + watts / max(ua, 1e-12)
    checks.append(Check("lumped estimate", "info",
                        f"the wetted area of {fmt(aw.sum() / 100)} cm2 at the mean {fmt(h_mean)} W/(m2 K) puts a "
                        f"part at one temperature at {fmt(t_lump)} C, and the FEA's hottest point is "
                        f"{fmt(to.values['t_max_C'])} C", {"uniform_C": t_lump}))
    if o.plan.mode in ("internal", "ducted"):
        from . import correlations as C
        re_ = s.speed * s.ref_length / s.nu
        nu_c, src = ((C.nu_gnielinski(re_, fl.Pr), "Gnielinski's correlation for a straight tube") if re_ > 3000 else
                     (3.66, "the fully developed laminar value for a round tube at constant wall temperature"))
        h_c = nu_c * fl.k / s.ref_length
        checks.append(Check("heat transfer correlation", "info",
                            f"{src} gives {fmt(h_c)} W/(m2 K) at Re {fmt(re_)} on the {fmt(s.ref_length * 1e3)} mm "
                            f"hydraulic diameter, and the CFD averages {fmt(h_mean)} W/(m2 K) over the wetted faces, "
                            "which include entrance regions and turns that raise it",
                            {"correlation_W_m2K": h_c, "cfd_W_m2K": h_mean}))
    level, reasons = confidence(checks, [("medium", "convective heat transfer from RANS is commonly within 10 to 25% "
                                                    "of measurements")] if s.turb.model != "laminar" else [])
    tmax, lim = to.values["t_max_C"], to.values["limit_C"]
    how = (f"{fmt(st.speed)} m/s of {fl.name.lower()} past it" if o.plan.mode == "external" else
           f"{o.metrics[1][1]} of {fl.name.lower()} through it" if o.plan.mode != "ducted" else
           f"{o.metrics[1][1]} of {fl.name.lower()} through its duct")
    head = (f"With {how}, the faces shed heat at {fmt(h_mean)} W/(m2 K) on average, and {fmt(watts)} W brings the "
            f"hottest point to {fmt(tmax)} C against a limit of {fmt(lim)} C.")
    metrics = [("max temperature", f"{fmt(tmax)} C"), ("limit", f"{fmt(lim)} C"),
               ("mean h", f"{fmt(h_mean)} W/m2K")] + o.metrics[:2]
    values = {**{k: v for k, v in o.values.items() if not k.startswith("_")}, "t_max_C": tmax, "limit_C": lim,
              "h_mean_W_m2K": h_mean, "fluid_rise_C": rise, "film_C": t_ref, "power_W": watts,
              "faces": {str(k): {"h_W_m2K": v[0], "wetted_mm2": v[1]} for k, v in coeffs.items()},
              "_figures": [["temperature", "The part's temperature with the hottest point marked"],
                           ["pressure", "Pressure on the part's walls"], ["velocity", "Speed through the flow"]]}
    images = {"temperature": to.images.get("temperature", ""), **{k: v for k, v in o.images.items() if k != "error"}}
    assumptions = o.assumptions + [
        f"The fluid enters at {fmt(st.temp)} C" + (f" and warms by {fmt(rise)} C, so the films use {fmt(t_ref)} C" if
                                                     rise else "") + ".",
        "Each face's heat transfer coefficient comes from the CFD with the walls at one uniform temperature, and "
        "faces the flow does not reach get still air at 5 W/(m2 K).",
        "One way coupling, so the fluid's properties do not change with the part's temperature."] + \
        [a for a in to.assumptions if "film" not in a.lower()]
    verdict = to.verdict
    oc = GenericOutcome("thermal", str(Path(st.part).resolve()), st.question, verdict, head, metrics, level, reasons,
                        checks, assumptions, values, images, to.fixes, notes=o.notes + to.notes,
                        solver=f"{o.solver} and {to.solver}", t_total=time.time() - t0)
    if lim is not None:
        oc.criteria = [{"name": "temperature", "value": "t_max_C", "limit": lim, "sense": "max"}]
    _playback(flow_out, out, tmax, watts, h_mean, rise, coeffs, o)
    return oc.write(out)


def _playback(flow_out: Path, out: Path, tmax: float, watts: float, h_mean: float, rise: float, coeffs, o):
    """The flow's playback with the heat answer added: each surface vertex's heat transfer coefficient and the
    temperatures, so a player can colour the part by how hard each face is cooled while the fluid moves."""
    import json

    from . import playback

    src = flow_out / "playback.json"
    if not src.exists():
        return
    doc = json.loads(src.read_text())
    g = o.geometry
    h_vertex = np.zeros(len(g.X))
    for fid, (h, _) in coeffs.items():
        h_vertex[np.unique(g.tris[g.tri_face == fid])] = h
    _, surf = playback.surface_block(g.X, g.tris, {"h": h_vertex}, {"h": "W/m2K"})
    doc["surface"].update(surf)
    doc["kind"] = "cooling"
    doc["about"] += (" surface.h is each face's heat transfer coefficient from the flow, and the scalars give the "
                     "part's hottest point and the heat it sheds.")
    doc["scalars"].update({"hottest": {"value": round(tmax, 4), "unit": "C"},
                           "power": {"value": round(watts, 4), "unit": "W"},
                           "h_mean": {"value": round(h_mean, 4), "unit": "W/m2K"},
                           "fluid_rise": {"value": round(rise, 4), "unit": "C"}})
    (out / "playback.json").write_text(json.dumps(doc, separators=(",", ":")))
    from .scene import write as write_scene
    write_scene(out)
    from ..viz.adapters.thermal import cooled
    try:                 # the part colored by its temperature with the fluid through it, over the flow alone
        cooled(out)
    except Exception:  # noqa: BLE001  a view that fails must not lose the result
        pass
