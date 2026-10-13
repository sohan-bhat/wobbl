"""Heat: steady and transient conduction with convection, radiation, heat sources and fixed temperatures.

"Will this battery pack overheat in a sealed box?" becomes: the cells make
12 W, the box air takes heat away at about 5 W/(m2 K) from every face, the air
is at 35 C. Wobbl solves the temperature field with CalculiX, refines the
mesh until the hottest point stops moving, checks that heat in equals heat out
and compares with a lumped hand calculation.
"""
from __future__ import annotations

import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from ..fe import HeatModel
from ..geometry.select import Selector
from ..materials import resolve_material
from ..mesh.mesher import Sizing, auto_sizing, build_mesh
from ..solvers import base as solvers
from ..units import fmt, parse_quantity
from ..verify.checks import Check, confidence, convergence_check

H_TYPICAL = {"still air in a sealed box": 5.0, "natural convection in open air": 10.0,
             "a small fan": 25.0, "strong forced air": 60.0, "water cooling": 500.0}


@dataclass
class Film:
    h: float          # mW/(mm2 K)
    t_inf: float      # degC
    faces: str = "all"
    text: str = ""


@dataclass
class ThermalStudy:
    part: str
    material: str
    power: float = 0.0                  # mW total
    heat_faces: str = ""                # empty: power is generated through the volume
    films: list[Film] = field(default_factory=list)
    fixed: list[tuple[float, str]] = field(default_factory=list)      # (degC, selector)
    ambient: float = 25.0
    initial: float | None = None
    duration: float = 0.0               # s, 0 = steady state
    limit: float | None = None          # degC
    emissivity: float = 0.0
    question: str = ""
    quality: str = "normal"

    def to_dict(self) -> dict:
        return asdict(self)


def parse_film(text: str, ambient: float) -> Film:
    """'5 W/m2K @ all', '25 W/m2K at 30 C @ top'."""
    t = text.strip()
    faces = "all"
    if "@" in t:
        t, faces = (x.strip() for x in t.rsplit("@", 1))
    tinf = ambient
    m = re.search(r"\bat\s+([-\d.]+\s*[CFK])\b", t, re.I)
    if m:
        tinf = parse_quantity(m.group(1), "temperature").value
        t = (t[:m.start()] + t[m.end():]).strip()
    h = parse_quantity(t.replace(" ", "") if "/" in t else t, "film").value
    return Film(h, tinf, faces, text)


def _tri_weights(mesh, mask):
    """Integration weights of a quadratic field over boundary faces: a third of each face on its midside nodes."""
    _, a, _ = mesh.tri_geometry(mask)
    tris = mesh.tris[mask]
    cols = tris[:, 3:6] if tris.shape[1] == 6 else tris[:, :3]
    w = np.zeros(mesh.n_nodes)
    for k in range(3):
        np.add.at(w, cols[:, k], a / 3.0)
    return w, a.sum()


def build_heat(mesh, study: ThermalStudy, faces, bbox, mat) -> tuple[HeatModel, dict]:
    p = mat.internal()
    sel = Selector(faces, bbox)
    model = HeatModel(mesh, p["conductivity"], p["density"], p["specific_heat"],
                      initial_temp=study.initial if study.initial is not None else study.ambient)
    info = {"film_masks": [], "flux_area": 0.0}
    films = study.films or [Film(H_TYPICAL["natural convection in open air"] * 1e-3, study.ambient, "all", "")]
    for f in films:
        mask = sel.select(f.faces).tri_mask(mesh)
        model.films.append((f.faces, mask, f.h, f.t_inf))
        info["film_masks"].append((mask, f.h, f.t_inf))
        if study.emissivity > 0:
            model.radiation.append((f.faces, mask, study.emissivity, f.t_inf))
    for temp, s in study.fixed:
        nodes = np.unique(mesh.tris[sel.select(s).tri_mask(mesh)].ravel())
        model.fixed_temps.append((s, nodes, temp))
    if study.power:
        if study.heat_faces:
            mask = sel.select(study.heat_faces).tri_mask(mesh)
            _, a, _ = mesh.tri_geometry(mask)
            model.fluxes.append((study.heat_faces, mask, study.power / a.sum()))
            info["flux_area"] = float(a.sum())
        else:
            model.volumetric = study.power / mesh.volume()
    if study.duration > 0:
        model.transient = True
        model.duration = study.duration
        model.time_step = study.duration / 200.0
        model.output_times = list(np.linspace(study.duration / 40, study.duration, 40))
    return model, info


def run_thermal(study: ThermalStudy, out_dir: str | Path, progress=None, pre=None):
    from ..outcome_generic import GenericOutcome
    from ..report.render import line_plot, render_scalar
    from ..structural.run import describe_part

    t0 = time.time()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    mat = resolve_material(study.material)
    adapter = solvers.for_domain("thermal")
    adapter.require()
    say = progress or (lambda a, b: None)
    if pre is None:
        say("geometry", f"reading {Path(study.part).name}")
        pre = describe_part(study.part, out / "work")
    prep, feats = pre
    transient = study.duration > 0
    # a transient run factorizes the matrix at every time step, so it gets a smaller mesh budget
    s0 = auto_sizing(prep, thickness=feats.thickness_typical, fineness=0.8,
                     max_nodes=8000 if transient else 25000)
    factors = [1.0, 0.7] if (study.quality == "fast" or transient) else [1.0, 0.7, 0.5]
    levels, last = [], None
    for k, f in enumerate(factors):
        sz = Sizing(h=s0.h * f)
        say("mesh", f"level {k}: element size {sz.h:.3g} mm")
        mesh = build_mesh(prep, sz)
        model, info = build_heat(mesh, study, feats.faces, feats.bbox, mat)
        say("solve", f"level {k}: {mesh.n_nodes:,} nodes")
        res = adapter.solve_heat(model, out / "work" / f"level{k}")
        T = res.T
        i = int(np.argmax(T))
        levels.append({"level": k, "n_nodes": mesh.n_nodes, "h": sz.h, "t_max": float(T[i]),
                       "t_min": float(T.min()), "xyz": mesh.X[i].tolist(), "t_solve": res.t_solve})
        last = (mesh, model, info, res)
        if k >= 2 and _spread_change(levels, study.ambient) <= 0.02:
            break
    mesh, model, info, res = last
    T = res.T
    np.savez_compressed(out / "fields.npz", X=mesh.X, T=T, tets=mesh.tets, tris=mesh.tris,
                        hist_t=res.history_t, hist_T=res.history_T)
    tmax = float(T.max())
    hot = mesh.X[int(np.argmax(T))]
    limit = study.limit if study.limit is not None else mat.max_service_temp
    # converge whichever end of the temperature range moves more: the hot spot, or the cold face of a part
    # whose hottest point is a held temperature
    hot_r = [L["t_max"] - study.ambient for L in levels]
    cold_r = [L["t_min"] - study.ambient for L in levels]
    use_cold = len(levels) > 1 and _rel(cold_r) > _rel(hot_r)
    checks = [convergence_check(cold_r if use_cold else hot_r, 0.02,
                                "temperature at the coolest point" if use_cold else
                                "temperature rise at the hottest point", "C", "", min_levels=2)]
    extra = []

    # energy balance (steady): heat in = convection + radiation + fixed temperature boundaries
    loss = 0.0
    for mask, h, tinf in info["film_masks"]:
        w, _ = _tri_weights(mesh, mask)
        loss += float((w * h * (T - tinf)).sum())
    if study.emissivity > 0:
        sig = 5.670374e-11
        for mask, h, tinf in info["film_masks"]:
            w, _ = _tri_weights(mesh, mask)
            loss += float((w * study.emissivity * sig * ((T + 273.15) ** 4 - (tinf + 273.15) ** 4)).sum())
    if not model.transient and not model.fixed_temps and study.power:
        err = abs(loss - study.power) / study.power
        st = "pass" if err < 0.01 else ("warn" if err < 0.05 else "fail")
        checks.append(Check("energy balance", st, f"heat in {fmt(study.power / 1000)} W, heat out through the "
                                                    f"surfaces {fmt(loss / 1000)} W ({100 * err:.2g}% apart)",
                            {"in_W": study.power / 1000, "out_W": loss / 1000}))
    elif model.fixed_temps and res.reaction_flux is not None and not model.transient:
        fixed_nodes = np.unique(np.concatenate([n for _, n, _ in model.fixed_temps]))
        q_fixed = float(res.reaction_flux[fixed_nodes].sum())
        supplied = abs(q_fixed + study.power) if q_fixed * study.power >= 0 else abs(abs(q_fixed) - study.power)
        err = abs(supplied - loss) / max(supplied, loss, 1e-12)
        st = "pass" if err < 0.01 else ("warn" if err < 0.05 else "fail")
        checks.append(Check("energy balance", st, f"held temperature faces exchange {fmt(abs(q_fixed) / 1000)} W, "
                                                    f"heat made {fmt(study.power / 1000)} W, surfaces shed "
                                                    f"{fmt(loss / 1000)} W ({100 * err:.2g}% apart)"))
    # lumped hand calculation
    vol = mesh.volume()
    film_area = sum(_tri_weights(mesh, m)[1] for m, _, _ in info["film_masks"])
    hA = sum(_tri_weights(mesh, m)[1] * h for m, h, _ in info["film_masks"])
    lumped = None
    if hA > 0 and study.power and not model.fixed_temps:
        Lc = vol / film_area
        bi = (hA / film_area) * Lc / mat.internal()["conductivity"]
        eps_a = study.emissivity * film_area
        sig = 5.670374e-11
        tinf = study.ambient

        def shed(temp):  # heat leaving a uniform surface at temp, mW (convection plus radiation)
            return hA * (temp - tinf) + eps_a * sig * ((temp + 273.15) ** 4 - (tinf + 273.15) ** 4)

        if model.transient:
            cap = mat.internal()["density"] * mat.internal()["specific_heat"] * vol   # mJ/K
            temp, steps = float(model.initial_temp), 4000
            dt = study.duration / steps
            for _ in range(steps):  # lumped capacitance with radiation, midpoint rule
                k1 = (study.power - shed(temp)) / cap
                k2 = (study.power - shed(temp + 0.5 * dt * k1)) / cap
                temp += dt * k2
            lumped = temp
            mean_T = float(np.average(T, weights=_vol_weights(mesh)))
            err = abs(mean_T - lumped) / max(abs(lumped - tinf), 1e-9)
            tau = cap / hA
            s = (f"lumped model with convection{' and radiation' if eps_a else ''} (Biot {bi:.2g}) gives a mean of "
                 f"{fmt(lumped)} C at the end; FEA mean is {fmt(mean_T)} C ({100 * err:.2g}% of the rise)")
            st = "pass" if (bi < 0.1 and err < 0.05) else ("info" if bi >= 0.1 else "warn")
            checks.append(Check("hand calculation", st, s, {"tau_s": tau, "biot": bi, "lumped_C": lumped}))
        else:
            temp = tinf + study.power / hA
            for _ in range(60):  # Newton on P = h A (T - Tair) + eps sigma A (T^4 - Tair^4)
                f = shed(temp) - study.power
                df = hA + 4 * eps_a * sig * (temp + 273.15) ** 3
                temp -= f / df
            lumped = temp
            surf_mean = 0.0
            for mask, h, tinf_i in info["film_masks"]:
                w, _ = _tri_weights(mesh, mask)
                surf_mean += float((w * T).sum())
            surf_mean = surf_mean / film_area
            err = abs(surf_mean - lumped) / max(abs(lumped - tinf), 1e-9)
            s = (f"heat balance on a uniform skin gives {fmt(lumped)} C; the FEA skin averages {fmt(surf_mean)} C "
                 f"({100 * err:.2g}% of the rise). Biot {bi:.2g}: "
                 + ("the part is nearly isothermal" if bi < 0.1 else "the inside runs hotter than the skin"))
            st = "pass" if err < 0.05 else ("info" if bi >= 0.1 else "warn")
            checks.append(Check("hand calculation", st, s,
                                {"lumped_surface_C": lumped, "fea_surface_C": surf_mean, "biot": bi}))
    if tmax < -273 or tmax > 3000:
        checks.append(Check("units and sanity", "fail", f"max temperature {fmt(tmax)} C is not physical; check "
                                                        "power and film units"))
    else:
        rad_note = ""
        if study.emissivity == 0 and tmax > 70:
            q_rad = 0.9 * 5.670374e-11 * ((tmax + 273.15) ** 4 - (study.ambient + 273.15) ** 4) * film_area
            rad_note = (f"; radiation is off, it could shed up to about {fmt(q_rad / 1000)} W at this temperature "
                        "(pass --radiation 0.9 for painted or plastic surfaces)")
        checks.append(Check("units and sanity", "pass", f"temperatures between {fmt(float(T.min()))} and "
                                                        f"{fmt(tmax)} C{rad_note}"))
    level, reasons = confidence(checks, extra)
    margin = limit - tmax
    verdict = "ok" if margin >= 5 else ("warm" if margin >= 0 else "hot")
    when = ""
    hist = res.history_T
    times = res.history_t
    t_cross = None
    if model.transient and len(times):
        # prepend the starting state so the curve and the crossing time begin at t = 0
        times = np.concatenate([[0.0], np.asarray(times)])
        hist = np.vstack([np.full((1, hist.shape[1]), model.initial_temp), hist])
        mx = hist.max(axis=1)
        over = np.nonzero(mx > limit)[0]
        if len(over):
            k = over[0]
            t_cross = float(times[k])
            if k > 0:  # interpolate between the two output times around the crossing
                t_cross = float(np.interp(limit, [mx[k - 1], mx[k]], [times[k - 1], times[k]]))
            when = f" It passes {fmt(limit)} C after about {fmt(t_cross / 60)} min."
    head = (f"Hottest point reaches {fmt(tmax)} C (limit {fmt(limit)} C) at "
            f"({', '.join(fmt(v) for v in hot)}) mm" + (f" after {fmt(study.duration / 60)} min" if model.transient
                                                         else " at steady state") + "." + when)
    assumptions = [
        f"{mat.name}: conductivity {mat.conductivity:g} W/(m K), specific heat {mat.specific_heat:g} J/(kg K), "
        f"density {mat.density:g} kg/m3.",
        ("Heat is generated evenly through the part." if not study.heat_faces else
         f"Heat enters through {study.heat_faces}."),
        "Convection uses a fixed film coefficient; real coefficients vary with temperature and airflow.",
        f"Surrounding air or coolant at {fmt(study.ambient)} C does not heat up.",
        "Radiation " + (f"with emissivity {study.emissivity:g}." if study.emissivity else "is ignored (conservative)."),
        "Contact resistance between parts and internal airflow are not modeled.",
    ]
    fixes = []
    if verdict != "ok" and hA > 0 and study.power:
        need = (tmax - study.ambient) / max(limit - 5 - study.ambient, 1e-9)
        fixes.append({"kind": "cooling", "title": f"Raise heat transfer about {need:.2g}x",
                      "summary": f"The temperature rise scales with 1/(h A). A small fan takes h from about 5 to 25 "
                                 f"W/(m2 K); fins or a larger case add area. Either way you need about {need:.2g}x "
                                 f"the current {fmt(hA / film_area * 1000)} W/(m2 K) times area.", "auto": False})
        fixes.append({"kind": "power", "title": f"Cut the power to {fmt(study.power / 1000 / need)} W",
                      "summary": "Lower current or duty cycle; the rise is proportional to the heat made.",
                      "auto": False})
    img = {}
    try:
        img["temperature"] = render_scalar(out / "images" / "temperature.png", mesh.X, mesh.tets, mesh.tris, T,
                                           "temperature", "C", "inferno", hot, f"{fmt(tmax)} C")
        if model.transient and len(times):
            img["history"] = line_plot(out / "images" / "history.png", np.asarray(times) / 60,
                                       {"hottest point": hist.max(axis=1), "average": hist.mean(axis=1)},
                                       "minutes", "C", limit, f"limit {fmt(limit)} C")
    except Exception as e:  # noqa: BLE001
        img["error"] = str(e)
    o = GenericOutcome("thermal", str(study.part), study.question, verdict, head,
                       [("max temperature", f"{fmt(tmax)} C"), ("limit", f"{fmt(limit)} C"),
                        ("power", f"{fmt(study.power / 1000)} W"), ("mesh nodes", f"{mesh.n_nodes:,}")],
                       level, reasons, checks, assumptions,
                       {"t_max_C": tmax, "hot_xyz": hot.tolist(), "limit_C": limit, "levels": levels,
                        "lumped_C": lumped, "time_over_limit_s": t_cross,
                        "_figures": [["temperature", "Temperature with the hottest point marked"],
                                     ["history", "Temperature over time"]]},
                       img, fixes, prep.notes, solver=res.solver, t_total=time.time() - t0)
    return o.write(out)


def _rel(vals: list[float]) -> float:
    if len(vals) < 2:
        return 0.0
    return abs(vals[-1] - vals[-2]) / max(abs(vals[-1]), 1e-9)


def _spread_change(levels: list[dict], ambient: float) -> float:
    return max(_rel([L["t_max"] - ambient for L in levels]), _rel([L["t_min"] - ambient for L in levels]))


def _vol_weights(mesh) -> np.ndarray:
    c = mesh.X[mesh.tets[:, :4]]
    v = np.abs(np.einsum("ij,ij->i", c[:, 1] - c[:, 0], np.cross(c[:, 2] - c[:, 0], c[:, 3] - c[:, 0]))) / 6.0
    w = np.zeros(mesh.n_nodes)
    for k in range(mesh.tets.shape[1]):
        np.add.at(w, mesh.tets[:, k], v / mesh.tets.shape[1])
    return w
