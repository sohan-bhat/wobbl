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
        model.ramp_time, info["ramp_reason"] = held_ramp(mesh, model)
        if model.ramp_time:
            model.output_times = early_times(model.ramp_time, model.output_times[0]) + model.output_times
    return model, info


RAMP_FRACTION = 0.1       # of the time heat takes to cross the smallest element at a held face that starts away
RAMP_RUN_FRACTION = 1e-3  # of the run, whichever is shorter
RAMP_SPARE = 4.0          # times the shortest ramp CalculiX's smallest increments can follow
EARLY_RATIO = 1.3         # between saved times from the end of the ramp to the first regular one


def held_ramp(mesh, model: HeatModel) -> tuple[float, str]:
    """How long a held temperature that starts away from the part's temperature takes to reach its target, s, and
    why that long.

    A face held at a new temperature from t = 0 moves its neighbors by a fixed amount however short the time step,
    so CalculiX cannot keep each step under its temperature change limit. It rises linearly instead, over a tenth
    of a2 / alpha for the smallest element touching such a face (a the edge of a regular tetrahedron of the same
    volume, alpha the thermal diffusivity), which is quicker than that mesh can resolve, or over a thousandth of
    the run if that is shorter. CalculiX never takes an increment under a millionth of the run and lets each one
    change a temperature by at most DELTMX (see transient_deltmx), so a big jump over a long run ramps over at least
    four times jump / DELTMX of those increments. Zero when every held temperature starts where the part does.
    """
    from ..solvers.calculix import MIN_INCREMENT, transient_deltmx

    step = [(nodes, temp) for _, nodes, temp in model.fixed_temps if abs(temp - model.initial_temp) > 1e-6]
    if not (model.transient and step):
        return 0.0, ""
    held = np.zeros(mesh.n_nodes, bool)
    held[np.concatenate([n for n, _ in step])] = True
    c = mesh.X[mesh.tets[held[mesh.tets].any(axis=1), :4]]
    v = np.abs(np.einsum("ij,ij->i", c[:, 1] - c[:, 0], np.cross(c[:, 2] - c[:, 0], c[:, 3] - c[:, 0]))) / 6.0
    a = float((6.0 * np.sqrt(2.0) * v.min()) ** (1.0 / 3.0))
    alpha = model.conductivity / (model.density * model.specific_heat)    # mm2/s
    jump = max(abs(temp - model.initial_temp) for _, temp in step)
    options = [(RAMP_FRACTION * a * a / alpha, "a tenth of the time heat takes to cross the smallest element there, "
                                                "quicker than this mesh can resolve"),
               (RAMP_RUN_FRACTION * model.duration, "a thousandth of the run")]
    ramp = min(options)
    floor = RAMP_SPARE * MIN_INCREMENT * model.duration * jump / transient_deltmx(jump)
    if floor > ramp[0]:
        ramp = (floor, f"as quick as CalculiX's shortest time steps can follow for a {fmt(jump)} C jump over a run "
                       "this long")
    return ramp


def _ramp_assumption(model: HeatModel, reason: str) -> str:
    """The result's note on how held temperatures that start away from the part's reach their targets."""
    step = [(name, temp) for name, _, temp in model.fixed_temps if abs(temp - model.initial_temp) > 1e-6]
    start, over = fmt(model.initial_temp), f"in a straight line over the first {fmt(model.ramp_time)} s"
    if len(step) == 1:
        what = f"the held temperature at {step[0][0]} ramps from {start} C to {fmt(step[0][1])} C {over}"
    else:
        names = [f"{name} ({fmt(temp)} C)" for name, temp in step]
        what = (f"the held temperatures at {', '.join(names[:-1])} and {names[-1]} ramp to their targets {over}")
    return f"The part starts at {start} C, and {what} instead of jumping at once, {reason}."


def early_times(ramp: float, first: float) -> list[float]:
    """Saved times through the ramp and the fast response after it, before the first regular saved time.

    Heat through a face that steps to a new temperature is fastest just after the step, and a thin part can take
    in most of its heat before the first regular saved time. The energy balance integrates that flow over the
    saved times, so it gets two points in the ramp and then points spaced by a constant ratio up to the first
    regular time, close enough for a cubic through them to follow it within a fraction of a percent.
    """
    out, t = [0.5 * ramp, ramp], ramp * EARLY_RATIO
    while t < first / EARLY_RATIO ** 0.5:
        out.append(t)
        t *= EARLY_RATIO
    return [x for x in out if x < first]


def run_thermal(study: ThermalStudy, out_dir: str | Path, progress=None, pre=None, sweep: bool = False):
    """Solve the study, refining the mesh until the hottest point settles. With sweep, a model that is not linear
    in its power (radiation, held temperatures, air at other temperatures) is also solved at a few other powers, so
    its view can show the result at any power the designer drags to."""
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
    levels, last, first = [], None, None
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
        first = first or last
        if k >= 2 and _spread_change(levels, study.ambient) <= 0.02:
            break
    mesh, model, info, res = last
    T = res.T
    extra = {}
    if sweep and study.power and not linear_in_power(study):
        say("solve", "the same model at other powers")
        extra = power_sweep(study, first, last, feats, mat, adapter, out / "work")
    np.savez_compressed(out / "fields.npz", X=mesh.X, T=T, tets=mesh.tets, tris=mesh.tris,
                        hist_t=res.history_t, hist_T=res.history_T, **extra)
    _save_study(study, out)
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
    elif model.transient:
        bal = transient_energy_balance(mesh, model, study, info, res)
        if bal is not None:
            checks.append(bal)
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
        ("No heat is made in the part." if not study.power else
         "Heat is generated evenly through the part." if not study.heat_faces else
         f"Heat enters through {study.heat_faces}."),
        "Convection uses a fixed film coefficient; real coefficients vary with temperature and airflow.",
        f"Surrounding air or coolant at {fmt(study.ambient)} C does not heat up.",
        "Radiation " + (f"with emissivity {study.emissivity:g}." if study.emissivity else "is ignored (conservative)."),
        "Contact resistance between parts and internal airflow are not modeled.",
    ]
    if model.ramp_time:
        assumptions.append(_ramp_assumption(model, info["ramp_reason"]))
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


SWEEP_POINTS = 7          # powers from none to the top of the sweep, the asked one added


def linear_in_power(study: ThermalStudy) -> bool:
    """Whether the rise above the air is exactly proportional to the power: no radiation, no held temperatures,
    every film's air at the ambient, and a transient that starts at the ambient."""
    films = study.films or [Film(0.0, study.ambient)]
    return (not study.emissivity and not study.fixed and all(abs(f.t_inf - study.ambient) < 1e-9 for f in films)
            and (study.duration <= 0 or study.initial is None or abs(study.initial - study.ambient) < 1e-9))


def sweep_top(study: ThermalStudy, t_max: float) -> float:
    """How far past the asked power a sweep runs, as a multiple of it: twice the power, or on to where the hottest
    point clearly passes the limit, by the linear estimate, at most five times."""
    rise = t_max - study.ambient
    if study.limit is None or rise <= 0:
        return 2.0
    return float(max(2.0, min(1.3 * (study.limit - study.ambient) / rise, 5.0)))


def power_sweep(study: ThermalStudy, coarse, fine, feats, mat, adapter, work: Path) -> dict:
    """The model solved at other powers on the coarsest mesh, carried to the finest by the difference each makes
    there, so the asked power keeps the converged field. Returns the arrays fields.npz keeps: sweep_W (the powers),
    sweep_T (the field at each, at the end of a transient) and, for a transient, sweep_hot (the hottest point over
    the run's saved times at each power)."""
    from dataclasses import replace

    from scipy.spatial import cKDTree

    mesh0, _, _, res0 = coarse
    mesh, _, _, res = fine
    k = np.unique(np.round(np.r_[np.linspace(0.0, sweep_top(study, float(res.T.max())), SWEEP_POINTS), 1.0], 6))
    near = cKDTree(mesh0.X).query(mesh.X)[1]
    t_f = np.asarray(res.history_t, float)
    hot_f = np.asarray(res.history_T, float).max(1) if len(t_f) else None

    def hot(r):
        return np.interp(t_f, np.asarray(r.history_t, float), np.asarray(r.history_T, float).max(1))

    still = (not study.fixed and all(abs(f.t_inf - study.ambient) < 1e-9 for f in study.films)
             and (study.initial is None or abs(study.initial - study.ambient) < 1e-9))
    fields, hots = [], []
    for i, f in enumerate(k):
        if f == 1.0:
            fields.append(res.T)
            hots.append(hot_f)
            continue
        if f == 0.0 and still:                     # nothing heats it, so it stays at the air's temperature
            Tc, hc = np.full(len(mesh0.X), study.ambient), np.full(len(t_f), study.ambient)
        else:
            model, _ = build_heat(mesh0, replace(study, power=study.power * f), feats.faces, feats.bbox, mat)
            r = adapter.solve_heat(model, work / f"sweep{i}")
            Tc, hc = r.T, (hot(r) if len(t_f) else None)
        fields.append(res.T + (Tc - res0.T)[near])
        if len(t_f):
            hots.append(hot_f + hc - hot(res0))
    d = {"sweep_W": k * study.power / 1000, "sweep_T": np.asarray(fields, np.float32)}
    if len(t_f):
        d["sweep_hot"] = np.asarray(hots, np.float32)
    return d


def _save_study(study: ThermalStudy, out: Path):
    """study.json, so the run can be solved again with other inputs ('wobbl uncertainty', 'wobbl validate-against').
    The part path is relative to the result folder, as a strength study's is."""
    import json

    from ..verify.checks import rel_up

    d = study.to_dict()
    d["part"] = rel_up(d["part"], out)
    (out / "study.json").write_text(json.dumps(d, indent=1))


SIGMA = 5.670374e-11   # Stefan Boltzmann constant, mW/(mm2 K4)


def transient_energy_balance(mesh, model, study: ThermalStudy, info: dict, res) -> Check | None:
    """Energy balance over a transient run, the counterpart of the steady one.

    The heat made over the run, plus whatever flows in through held temperature faces, has to equal the change in
    heat stored in the part plus the heat its surfaces shed over the run. Stored heat uses the exact weights of the
    quadratic temperature field, and the heat flows are integrated over the saved time points through a monotone
    cubic, starting from the uniform initial temperature at t = 0. CalculiX steps backward in time between those
    points, so the two sides drift apart by about the error of its time steps, and the check passes within 2% of
    the largest heat flow in the account.
    """
    if not len(res.history_t) or len(res.history_T) != len(res.history_t):
        return None
    times = np.concatenate([[0.0], np.asarray(res.history_t, float)])
    hist = np.vstack([np.full((1, mesh.n_nodes), float(model.initial_temp)), np.asarray(res.history_T, float)])
    rho_c = model.density * model.specific_heat
    weights = _quad_vol_weights(mesh)
    stored = rho_c * float(weights @ (hist[-1] - hist[0]))           # mJ
    out_rate = np.zeros(len(times))                                   # mW leaving through convection and radiation
    for mask, h, tinf in info["film_masks"]:
        w, _ = _tri_weights(mesh, mask)
        out_rate += h * ((hist - tinf) @ w)
        if study.emissivity > 0:
            out_rate += study.emissivity * SIGMA * (((hist + 273.15) ** 4 - (tinf + 273.15) ** 4) @ w)
    shed = _integrate(out_rate, times, model.ramp_time)
    made = study.power * times[-1]
    faces = []
    if model.fixed_temps:
        if res.reaction_flux_history is None or len(res.reaction_flux_history) != len(res.history_t):
            return None
        faces = _held_inflow(mesh, model, study, info, res.reaction_flux_history, times, hist, weights)
    held = sum(faces)
    # a hot face and a cold one can each move far more heat than their sum, and each carries its own error, so the
    # balance is judged against the largest flow in the account with each held face counted on its own
    scale = max(abs(made), sum(abs(q) for q in faces), abs(stored), abs(shed))
    if scale < 1e-9:   # nothing made, held or exchanged: no heat to account for
        return None
    err = abs(made + held - stored - shed) / scale
    st = "pass" if err < 0.02 else ("warn" if err < 0.05 else "fail")
    src = f"heat in {_energy(made)}"
    if model.fixed_temps:
        src += f" plus {_energy(held)} through held temperature faces"
        gain, loss = sum(q for q in faces if q > 0), -sum(q for q in faces if q < 0)
        if min(gain, loss) > 0.01 * max(gain, loss):
            src += f" ({_energy(gain)} in and {_energy(loss)} out)"
    s = (f"over {fmt(times[-1] / 60)} min, {src}, stored {_energy(stored)}, out through the surfaces "
         f"{_energy(shed)} ({100 * err:.2g}% apart)")
    if st != "pass":
        s += (". The time steps are coarse for how fast this part responds, so treat the temperatures between "
              "them as approximate")
    return Check("energy balance", st, s, {"in_J": made / 1000, "held_J": held / 1000, "stored_J": stored / 1000,
                                           "out_J": shed / 1000, "error": err, "time_points": len(times),
                                           "held_faces_J": [q / 1000 for q in faces]})


def _held_inflow(mesh, model, study, info, rfl_hist, times, hist, weights) -> list[float]:
    """Heat that entered the part through each held temperature face over the run, mJ.

    CalculiX's RFL is the conduction flow at each node, which sums to zero over the part, so the heat a held face
    supplies is its RFL plus what the held nodes exchange themselves: their share of the consistent heat capacity
    and of the convection and radiation on faces that touch them, less the heat made at them. Each face is
    integrated on its own, because a hot face and a cold one push and pull heat at once, and their sum can change
    sign where each flow alone does not.
    """
    held = np.zeros(mesh.n_nodes, bool)
    rho_c, out = model.density * model.specific_heat, []
    for _, nodes, _ in model.fixed_temps:
        face = np.zeros(mesh.n_nodes, bool)
        face[nodes] = True
        face &= ~held      # a node shared by two held faces counts once
        held |= face
        if not face.any():
            out.append(0.0)
            continue
        inflow = _integrate_from_start(rfl_hist[:, face].sum(1), times)
        inflow += rho_c * _held_capacity(mesh, face, hist[-1] - hist[0])
        surf = np.zeros(len(times))
        for mask, h, tinf in info["film_masks"]:
            surf += h * _held_surface(mesh, mask, face, hist - tinf)
            if study.emissivity > 0:
                surf += study.emissivity * SIGMA * _held_surface(mesh, mask, face,
                                                                 (hist + 273.15) ** 4 - (tinf + 273.15) ** 4)
        inflow += _integrate(surf, times, model.ramp_time)
        made_at_face = model.volumetric * weights[face].sum()
        for _, mask, q in model.fluxes:
            w, _ = _tri_weights(mesh, mask)
            made_at_face += q * w[face].sum()
        out.append(inflow - made_at_face * times[-1])
    return out


def _integrate_from_start(rate, times) -> float:
    """Integral from t = 0 of a heat flow saved at times[1:], for flows through held faces.

    Such a flow starts at zero from a uniform part, then can settle within the first saved interval (a thin metal
    part), grow like sqrt(t) (a thick plastic one) or rise through a ramp to a new held temperature, so up to the
    second saved time it follows a power law through the first two saved points instead of a straight line from
    zero. A ramp ends on that second time, so the kink where the flow peaks is never inside a stretch of cubic.
    """
    rate, t = np.asarray(rate, float), np.asarray(times[1:], float)
    if len(rate) < 2:
        return float(rate[0] * t[0] / 2.0)
    p = 1.0
    if rate[0] * rate[1] > 0:
        p = float(np.clip(np.log(rate[1] / rate[0]) / np.log(t[1] / t[0]), -0.9, 3.0))
    return rate[1] * t[1] / (p + 1.0) + _integrate(rate[1:], t[1:])


def _integrate(y, x, knot: float = 0.0) -> float:
    """Integral of a flow saved at times x, through a monotone cubic between the saved points (PCHIP).

    It follows the fast fall of the flow through a face that steps to a new temperature, and the slow changes of a
    long run, without overshooting between points. A knot is a saved time where the flow has a kink, the end of a
    ramp, and the cubic starts afresh there.
    """
    from scipy.interpolate import PchipInterpolator

    y, x = np.asarray(y, float), np.asarray(x, float)
    cut = int(np.argmin(np.abs(x - knot))) if knot > 0 else 0
    if 0 < cut < len(x) - 1 and abs(x[cut] - knot) <= 1e-3 * knot:   # the .frd keeps about six digits of time
        return _integrate(y[:cut + 1], x[:cut + 1]) + _integrate(y[cut:], x[cut:])
    if len(x) < 3:
        return _trapezoid(y, x)
    return float(PchipInterpolator(x, y).integrate(x[0], x[-1]))


_TET_EDGES = [(0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3)]   # midside nodes 4 to 9 of a CalculiX C3D10


def _tet10_mass() -> np.ndarray:
    """Consistent mass matrix of a straight sided 10 node tetrahedron of unit volume and unit density."""
    M = np.full((4, 4), 1.0) + 5.0 * np.eye(4)
    M = np.pad(M, ((0, 6), (0, 6)))
    for k, e in enumerate(_TET_EDGES):
        for c in range(4):
            M[c, 4 + k] = M[4 + k, c] = -4.0 if c in e else -6.0
        for m, f in enumerate(_TET_EDGES):
            M[4 + k, 4 + m] = 32.0 if k == m else (16.0 if set(e) & set(f) else 8.0)
    return M / 420.0


def _tri6_mass() -> np.ndarray:
    """Consistent mass matrix of a 6 node triangle of unit area, midsides on edges 0-1, 1-2 and 2-0."""
    M = np.zeros((6, 6))
    M[:3, :3] = np.full((3, 3), -1.0) + 7.0 * np.eye(3)
    for k, e in enumerate([(0, 1), (1, 2), (2, 0)]):
        for c in range(3):
            M[c, 3 + k] = M[3 + k, c] = 0.0 if c in e else -4.0
    M[3:, 3:] = np.full((3, 3), 16.0) + 16.0 * np.eye(3)
    return M / 180.0


def _held_capacity(mesh, held: np.ndarray, dT: np.ndarray) -> float:
    """Sum over held nodes of the consistent capacity matrix times the temperature change, per unit rho c."""
    if mesh.tets.shape[1] != 10:
        return 0.0
    c = mesh.X[mesh.tets[:, :4]]
    v = np.abs(np.einsum("ij,ij->i", c[:, 1] - c[:, 0], np.cross(c[:, 2] - c[:, 0], c[:, 3] - c[:, 0]))) / 6.0
    touch = held[mesh.tets].any(axis=1)
    phi = held[mesh.tets[touch]].astype(float)
    return float((v[touch] * np.einsum("ei,ij,ej->e", phi, _tet10_mass(), dT[mesh.tets[touch]])).sum())


def _held_surface(mesh, mask: np.ndarray, held: np.ndarray, field: np.ndarray) -> np.ndarray:
    """For each row of field (one per time), the consistent surface integral of it weighted by the held nodes'
    shape functions over the masked faces."""
    tris = mesh.tris[mask]
    touch = held[tris].any(axis=1)
    if not touch.any():
        return np.zeros(len(field))
    _, a, _ = mesh.tri_geometry(mask)
    tris, a = tris[touch], a[touch]
    if tris.shape[1] != 6:   # linear faces: a lumped split is as good as the solver's own
        return (field[:, tris] * (held[tris] * (a / 3.0)[:, None])[None]).sum((1, 2))
    phi = held[tris].astype(float) * a[:, None]
    return np.einsum("ei,ij,tej->t", phi, _tri6_mass(), field[:, tris])


def _trapezoid(y, x) -> float:
    y, x = np.asarray(y, float), np.asarray(x, float)
    return float(np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(x)))


def _energy(mj: float) -> str:
    j = mj / 1000.0
    return f"{fmt(j / 1000)} kJ" if abs(j) >= 1000 else f"{fmt(j)} J"


def _quad_vol_weights(mesh) -> np.ndarray:
    """Exact integration weights of a field over the mesh. For 10 node tetrahedra each corner carries -V/20 and each
    midside node V/5, which integrates a quadratic field exactly, unlike the even split in _vol_weights."""
    c = mesh.X[mesh.tets[:, :4]]
    v = np.abs(np.einsum("ij,ij->i", c[:, 1] - c[:, 0], np.cross(c[:, 2] - c[:, 0], c[:, 3] - c[:, 0]))) / 6.0
    w = np.zeros(mesh.n_nodes)
    quadratic = mesh.tets.shape[1] == 10
    for k in range(mesh.tets.shape[1]):
        share = (-v / 20.0 if k < 4 else v / 5.0) if quadratic else v / 4.0
        np.add.at(w, mesh.tets[:, k], share)
    return w


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
