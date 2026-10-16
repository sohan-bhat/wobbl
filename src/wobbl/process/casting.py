"""Casting: does the part freeze sound, or with a shrinkage pocket inside?

"Will this aluminum bracket cast without porosity in green sand?" becomes: the
cavity fills at once with A356 at 720 C, the sand around it starts at 25 C,
and the metal gives up its superheat and latent heat into the sand. Wobbl
meshes the part with its risers, chills and a block of sand, solves the
freezing with the enthalpy method (process.solidify), and reads off where the
metal freezes last, whether liquid can still reach each place from a riser as
it shrinks, and the Niyama criterion for the porosity left where feeding is
weak. The answer is checked against Chvorinov's rule, refined on a second mesh
and run again at half the time step.
"""
from __future__ import annotations

import math
import shutil
import time
from collections import deque
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

from ..units import fmt
from ..verify.checks import Check, confidence
from . import solidify as S
from .data import Alloy, MoldMaterial, get_alloy, get_mold
from . import budget as bud
from .regions import Chill, RegionMesh, Riser, build, split_interfaces

FEED_TOL = 0.01          # of the casting's freezing time: a path may climb this much and still feed
CONVERGED = 0.03         # change in freezing time between meshes that ends the refinement
CONTACT = 3000.0         # W/(m2 K) across the metal to mold faces: close to perfect contact, since a sand's own
                         # surface conductance falls under 500 W/(m2 K) within the first second
VERIFY = ("mesh convergence", "time step", "energy balance", "hand calculation", "units and sanity")
POCKET_MIN = 0.005       # of the casting volume: smaller unfed spots are mesh noise
SPREAD = 0.2             # how much the latent heat and the sand's heat uptake vary between sources


@dataclass
class CastingStudy:
    part: str
    alloy: str = "a356"
    mold: str = "sand"
    pour: float | None = None          # C, default the alloy's usual pouring temperature
    mold_temp: float = 25.0            # C
    ambient: float = 25.0              # C, air around the mold
    film: float = 10.0                 # W/(m2 K) on the outside of the mold
    contact: float = CONTACT           # W/(m2 K) between metal and mold
    risers: list[Riser] = field(default_factory=list)
    chills: list[Chill] = field(default_factory=list)
    question: str = ""
    quality: str = "normal"
    scale: float = 1.0
    params: dict = field(default_factory=dict)
    auto_risers: bool = False          # place risers over the hot spots when none are given
    mesh_size: float | None = None     # metal element size, mm, instead of the automatic one
    budget: float = bud.DEFAULT_BUDGET  # s, what the mesh levels may be predicted to take (process.budget)


# element size of each level against the base size, coarse to fine. The first two always run, so the mesh
# convergence check always has two meshes to compare, and a third only when the pair has not converged and the
# budget allows it
LEVELS = {"fast": [1.0, 0.72], "normal": [1.0, 0.72, 0.52]}


def effective_heat(alloy: Alloy, pour: float) -> tuple[float, float]:
    """Heat a kilogram gives up to freeze completely, J/kg, and the temperature it gives it up at, C: superheat and
    latent heat, taken out across a mold face held at the middle of the freezing range."""
    tm = 0.5 * (alloy.liquidus + alloy.solidus)
    return alloy.latent + alloy.specific_heat * max(pour - tm, 0.0), tm


def chvorinov_constant(alloy: Alloy, mold: MoldMaterial, pour: float, t0: float, latent: float | None = None):
    """Mold constant B of Chvorinov's rule t = B (V/A)^2, s/mm2.

    A semi infinite mold whose face is held at the metal's freezing temperature Tm takes heat at
    k (Tm - T0) / sqrt(pi alpha t). Over time t that is 2 (Tm - T0) sqrt(k rho c t / pi) per unit area, and equating
    it to the heat a casting of modulus V/A gives up, rho_m dH V/A, gives
    t = (pi / 4) (rho_m dH / (Tm - T0))^2 (V/A)^2 / (k rho c)_mold.
    """
    a = alloy if latent is None else _with_latent(alloy, latent)
    dh, tm = effective_heat(a, pour)
    return math.pi / 4 * (alloy.density * dh / (tm - t0)) ** 2 / mold.effusivity ** 2 * 1e-6


def _with_latent(alloy: Alloy, latent: float) -> Alloy:
    from dataclasses import replace
    return replace(alloy, latent=latent)


def materials_for(alloy: Alloy, mold: MoldMaterial, chills: list[Chill]):
    metal = S.Material(alloy.key, alloy.density, alloy.conductivity, alloy.specific_heat, alloy.latent,
                       alloy.solid_fraction, alloy.liquidus, alloy.solidus)
    mats = [metal, S.Material(mold.key, mold.density, mold.conductivity, mold.specific_heat)]
    for c in chills:
        m = get_mold(c.material)
        mats.append(S.Material(m.key, m.density, m.conductivity, m.specific_heat))
    return mats


def region_materials(rm: RegionMesh) -> np.ndarray:
    """Material index of every region: 0 metal, 1 mold, 2.. chills in order."""
    out, k = [], 2
    for kind in rm.kinds:
        if kind in ("casting", "riser"):
            out.append(0)
        elif kind == "chill":
            out.append(k)
            k += 1
        else:
            out.append(1)
    return np.array(out)


@dataclass
class Solved:
    rm: RegionMesh
    model: S.Model
    result: S.Result
    casting_nodes: np.ndarray
    riser_nodes: np.ndarray
    node_volume: np.ndarray            # mm3 of metal at each node
    base: RegionMesh | None = None     # the mesh before the metal got its own surface nodes
    surface: np.ndarray | None = None  # metal nodes against the mold


def solidify(study: CastingStudy, rm: RegionMesh, alloy: Alloy, mold: MoldMaterial, pour: float, duration: float,
             finer_steps: float = 1.0, progress=None, limit: float | None = None) -> Solved:
    mats = materials_for(alloy, mold, study.chills)
    mat_of_region = region_materials(rm)
    metal_el = np.isin(np.array(rm.kinds)[rm.region], ["casting", "riser"])
    cast_el = np.array(rm.kinds)[rm.region] == "casting"
    riser_el = np.array(rm.kinds)[rm.region] == "riser"
    X2, tets2, (side_metal, side_mold) = split_interfaces(rm.X, rm.tets, metal_el)
    base, rm = rm, replace(rm, X=X2, tets=tets2)
    metal_nodes = np.unique(rm.tets[metal_el])
    T0 = np.full(len(rm.X), float(study.mold_temp))
    T0[metal_nodes] = pour
    t_ny = alloy.solidus + 0.1 * (alloy.liquidus - alloy.solidus)
    model = S.Model(rm.X, rm.tets, mat_of_region[rm.region], mats, T0,
                    contacts=[(side_metal, side_mold, study.contact)],
                    films=[(rm.outer, study.film, study.ambient)] if study.film > 0 else [],
                    duration=duration, watch=metal_nodes,
                    events={"liquidus": alloy.liquidus, "niyama": t_ny, "solidus": alloy.solidus},
                    gradient_event="niyama", stop_below=alloy.solidus - 2.0,
                    dT_max=6.0 / finer_steps, dg_max=0.12 / finer_steps, dt0=0.01, deadline=limit, budget=study.budget)
    res = S.solve(model, progress)
    vol = np.abs(np.linalg.det(np.concatenate([np.ones((len(rm.tets), 4, 1)), rm.X[rm.tets]], axis=2))) / 6
    nv = np.zeros(len(rm.X))
    for c in range(4):
        np.add.at(nv, rm.tets[metal_el, c], vol[metal_el] / 4)
    return Solved(rm, model, res, np.unique(rm.tets[cast_el]), np.unique(rm.tets[riser_el]) if riser_el.any()
                  else np.zeros(0, int), nv, base, np.unique(side_metal))


def _adjacency(tets: np.ndarray, n: int) -> sp.csr_matrix:
    e = np.concatenate([tets[:, [i, j]] for i, j in ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))])
    a = sp.coo_matrix((np.ones(len(e) * 2), (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])), shape=(n, n))
    return a.tocsr()


def feeding(sv: Solved) -> dict:
    """Which parts of the casting liquid metal can still reach as they shrink.

    Metal flows to a freezing spot from places that freeze later. Starting from the risers (or, with none, from the
    last spot to freeze), a node is fed when a neighbor that freezes no earlier is fed, so the fed region grows
    downhill in freezing time. A spot the walk never reaches freezes after everything around it: it is cut off
    with liquid inside, and its shrinkage stays there as a pocket."""
    rm, res = sv.rm, sv.result
    n = len(rm.X)
    t = res.events["solidus"]
    metal_el = np.isin(np.array(rm.kinds)[rm.region], ["casting", "riser"])
    adj = _adjacency(rm.tets[metal_el], n)
    cast = np.zeros(n, bool)
    cast[sv.casting_nodes] = True
    tc = t[sv.casting_nodes]
    t_cast = float(np.nanmax(tc))
    tol = FEED_TOL * t_cast
    seeds = sv.riser_nodes if len(sv.riser_nodes) else sv.casting_nodes[[int(np.nanargmax(tc))]]
    fed = np.zeros(n, bool)
    fed[seeds] = True
    q = deque(int(s) for s in seeds)
    ip, ix = adj.indptr, adj.indices
    tt = np.nan_to_num(t, nan=np.inf)
    while q:
        i = q.popleft()
        for j in ix[ip[i]:ip[i + 1]]:
            if not fed[j] and tt[j] <= tt[i] + tol:
                fed[j] = True
                q.append(j)
    unfed = cast & ~fed
    pockets = []
    if unfed.any():
        sub = adj[unfed][:, unfed]
        ncomp, lab = connected_components(sub, directed=False)
        idx = np.nonzero(unfed)[0]
        vtot = float(sv.node_volume[sv.casting_nodes].sum())
        for c in range(ncomp):
            nodes = idx[lab == c]
            v = float(sv.node_volume[nodes].sum())
            if v < POCKET_MIN * vtot:
                continue
            hot = nodes[int(np.argmax(tt[nodes]))]
            pockets.append({"volume_mm3": v, "hot_xyz": rm.X[hot].tolist(), "t_s": float(tt[hot]),
                            "nodes": nodes})
        pockets.sort(key=lambda p: -p["volume_mm3"])
    last = sv.casting_nodes[int(np.nanargmax(tc))]
    # the time 99% of the casting is solid: the last node alone moves with the mesh where a riser joins
    order = np.argsort(np.nan_to_num(tc, nan=np.inf))
    cum = np.cumsum(sv.node_volume[sv.casting_nodes][order])
    t99 = float(np.nan_to_num(tc, nan=np.inf)[order][min(np.searchsorted(cum, 0.99 * cum[-1]), len(order) - 1)])
    return {"fed": fed, "pockets": pockets, "t_casting_s": t_cast, "t99_s": t99, "last_xyz": rm.X[last].tolist(),
            "t_riser_s": float(np.nanmax(t[sv.riser_nodes])) if len(sv.riser_nodes) else None}


def niyama(alloy: Alloy, G: np.ndarray, rate: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    """The Niyama criterion G / sqrt(dT/dt) in (K s)^0.5/mm, and the shrinkage pore volume percent from Carlson and
    Beckermann's dimensionless form where the alloy has their data."""
    with np.errstate(divide="ignore", invalid="ignore"):
        ny = G / np.sqrt(rate)
    p = alloy.niyama
    if not p:
        return ny, None
    with np.errstate(divide="ignore", invalid="ignore"):
        nys = (p["c_lambda"] * G * 1000.0 * rate ** (-5.0 / 6.0) *
               math.sqrt(p["pressure_drop"] / (p["viscosity"] * p["beta"] * p["freezing_range"])))
    gp = np.full_like(nys, np.nan)
    lo = 0.0
    for top, A, B, C, D in p["fits"]:
        m = (nys > lo) & (nys <= top)
        with np.errstate(divide="ignore", invalid="ignore"):
            gp[m] = -A * np.log10(nys[m]) + B + C * nys[m] ** (-D)
        lo = top
    gp = np.minimum(gp, 100.0 * alloy.shrinkage)
    return ny, gp


def mesh_sizes(feats, quality: str) -> tuple[float, float]:
    """Element size in the metal and the coarsest in the mold, mm."""
    size = float(np.max(np.asarray(feats.bbox[1]) - np.asarray(feats.bbox[0])))
    h = min(feats.thickness_typical / 4.0, size / 25.0)
    if quality == "fast":
        h *= 1.4
    return max(h, 0.6), max(3.0 * h, 6.0)


def riser_for(xyz, t_hot: float, b_eff: float, X: np.ndarray, casting_nodes: np.ndarray) -> Riser:
    """A blind riser over a hot spot that freezes at t_hot: modulus 1.35x the hot spot's (from Chvorinov's rule with
    the mold constant b_eff), height 1.5x the diameter, its foot on the casting's top surface above the spot."""
    m_hot = math.sqrt(t_hot / b_eff)
    d = 5.0 * math.ceil(1.35 * m_hot / 0.2143 / 5.0)              # height 1.5 d gives modulus 0.2143 d
    P = X[casting_nodes]
    near = np.linalg.norm(P[:, :2] - np.asarray(xyz[:2]), axis=1) < max(d / 2, 1e-6)
    top = float(P[near][:, 2].max()) if near.any() else float(xyz[2])
    return Riser(d, 1.5 * d, (round(float(xyz[0]), 1), round(float(xyz[1]), 1), round(top, 2)))


def place_risers(study, prep, alloy, mold, pour, B, modulus, margin, duration, h, h_mold, alpha, say,
                 rounds: int = 2, curvature: int = 12, limit: float | None = None) -> list[Riser]:
    """Risers where the casting needs them. A coarse freeze without risers finds the last spot to freeze and every
    spot cut off from it; each gets a riser sized by the modulus rule. The casting is frozen again with them, and
    any region still cut off gets one more."""
    risers: list[Riser] = []
    for k in range(rounds):
        trial = replace(study, risers=list(risers), auto_risers=False)
        say("risers", f"round {k + 1}: freezing a coarse mesh with {len(risers)} riser(s) to find hot spots")
        rm = build(prep, trial.risers, trial.chills, margin, 1.4 * h, h_mold,
                   0.6 * math.sqrt(alpha * B * modulus ** 2), curvature)
        sv = solidify(trial, rm, alloy, mold, pour, duration, limit=limit)
        fd = feeding(sv)
        t_cast = fd["t_casting_s"]
        if not np.isfinite(t_cast):
            break
        b_eff = B * (t_cast / (B * modulus ** 2))
        spots = [(p["hot_xyz"], p["t_s"]) for p in fd["pockets"]]
        if not risers:
            spots.insert(0, (fd["last_xyz"], t_cast))
        if not spots:
            break
        for xyz, t_hot in spots:
            r = riser_for(xyz, t_hot, b_eff, sv.rm.X, sv.casting_nodes)
            clash = [q for q in risers if np.linalg.norm(np.subtract(q.at[:2], r.at[:2])) < (q.diameter + r.diameter) / 2]
            if not clash:
                risers.append(r)
    return risers


def run_casting(study: CastingStudy, out_dir: str | Path, progress=None, keep: bool = False):
    from ..outcome_generic import GenericOutcome
    from ..structural.run import describe_part
    from ..verify.checks import convergence_check

    t0 = time.time()
    limit = bud.deadline(study.budget)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    say = progress or (lambda a, b: None)
    alloy, mold = get_alloy(study.alloy), get_mold(study.mold)
    pour = float(study.pour if study.pour is not None else alloy.pour)
    if pour <= alloy.liquidus:
        raise ValueError(f"the pouring temperature {fmt(pour)} C is below {alloy.name}'s liquidus "
                         f"({fmt(alloy.liquidus)} C); pour it hotter")
    say("geometry", f"reading {Path(study.part).name}")
    prep, feats = describe_part(study.part, out / "work", study.scale, None, study.params)
    foot = sum(math.pi * r.diameter ** 2 / 4 for r in study.risers)
    modulus = feats.volume / max(feats.surface_area - foot, 1e-9)            # mm, of the casting alone
    B = chvorinov_constant(alloy, mold, pour, study.mold_temp)
    t_chv = B * modulus ** 2
    t_est = B * max([modulus] + [r.modulus for r in study.risers]) ** 2
    alpha = mold.conductivity / (mold.density * mold.specific_heat) * 1e6      # mm2/s
    margin = float(np.clip(3.0 * math.sqrt(alpha * 1.5 * t_est), 15.0, 80.0))
    duration = max(10.0 * t_est, 60.0)
    h, h_mold = mesh_sizes(feats, study.quality)
    if study.mesh_size:
        h, h_mold = float(study.mesh_size), max(3.0 * float(study.mesh_size), 6.0)
    scales = LEVELS["fast" if study.quality == "fast" else "normal"]
    band = 0.6 * math.sqrt(alpha * t_est)
    say("budget", "meshing coarsely to predict the cost of the mesh levels")
    found = bud.pilots(lambda f_p, c_p: build(prep, study.risers, study.chills, margin, h * scales[0] * f_p,
                                             max(h_mold, 3.0 * h * scales[0] * f_p), band, c_p).X)
    searching = study.auto_risers and not study.risers
    # besides its levels the study freezes the first one again at half the step, and finds its risers on two
    # coarse meshes when none were given
    bfit = bud.fit(bud.CASTING, found, h, scales, study.budget, bool(study.mesh_size),
                   extra=lambda n: bud.CASTING.solve(n) * 1.5 + (2 * bud.CASTING.level(n / 2.7) if searching else 0.0))
    say("budget", bud.note(bfit).split(". The budget")[0])
    f_r, c_r = bfit.factor, min(bfit.curvature, 12)
    h_r, h_mold = h * f_r, max(h_mold, 3.0 * h * f_r)
    placed = []
    if searching:
        placed = place_risers(study, prep, alloy, mold, pour, B, modulus, margin, duration, h_r, h_mold, alpha, say,
                              curvature=c_r, limit=limit)
        study = replace(study, risers=placed)
        foot = sum(math.pi * r.diameter ** 2 / 4 for r in study.risers)
        modulus = feats.volume / max(feats.surface_area - foot, 1e-9)
        t_chv = B * modulus ** 2
        t_est = B * max([modulus] + [r.modulus for r in study.risers]) ** 2
        margin = float(np.clip(3.0 * math.sqrt(alpha * 1.5 * t_est), 15.0, 80.0))
        duration = max(10.0 * t_est, 60.0)
    runs = []
    t_levels = t_last = time.time()
    last_s = 0.0
    for k, sc in enumerate(scales):
        h_k = h_r * sc
        if k >= bud.MIN_LEVELS and not bud.room_for(bud.CASTING, len(runs[-1][0].rm.X), h_r * scales[k - 1], c_r, h_k,
                                                    time.time() - t_levels, study.budget, bfit.power, last_s):
            break
        t_last = time.time()
        say("mesh", f"level {k}: metal elements {h_k:.3g} mm, sand up to {h_mold:.3g} mm")
        rm = build(prep, study.risers, study.chills, margin, h_k, h_mold, 0.6 * math.sqrt(alpha * t_est), c_r)
        say("solve", f"level {k}: {len(rm.X):,} nodes, freezing")
        sv = solidify(study, rm, alloy, mold, pour, duration, progress=say, limit=limit)
        runs.append((sv, feeding(sv)))
        last_s = time.time() - t_last
        if k >= 1 and abs(runs[-1][1]["t99_s"] / runs[-2][1]["t99_s"] - 1) < CONVERGED:
            break
    bfit.measured = {"levels": len(runs), "nodes": [len(r[0].rm.X) for r in runs],
                     "seconds": round(time.time() - t_levels, 1)}
    sv, fd = runs[-1]
    say("check", "time step: level 0 again at half the step")
    half = solidify(study, runs[0][0].base, alloy, mold, pour, duration, finer_steps=2.0, limit=limit)
    fd_half = feeding(half)
    res, rm = sv.result, sv.rm
    t_sol = res.events["solidus"]
    cn = sv.casting_nodes
    t_cast = fd["t_casting_s"]
    checks: list[Check] = []
    solid = bool(np.isfinite(t_sol[cn]).all())
    if solid:
        checks.append(convergence_check([r[1]["t99_s"] for r in runs], 0.05,
                                        "time for 99% of the casting to freeze", "s",
                                        "the last level" if len(runs) == len(scales) else "", min_levels=2))
        t_a, t_b = runs[0][1]["t99_s"], fd_half["t99_s"]
        dt_ch = abs(t_b - t_a) / t_a
        st = "pass" if dt_ch < 0.03 else ("warn" if dt_ch < 0.08 else "fail")
        checks.append(Check("time step", st, f"half the time step moves the freezing time from {fmt(t_a)} to "
                                             f"{fmt(t_b)} s ({100 * dt_ch:.2g}%)", {"change": dt_ch}))
    e = res.energy
    st = "pass" if e["error"] < 1e-4 else ("warn" if e["error"] < 1e-2 else "fail")
    checks.append(Check("energy balance", st, f"the cooling metal and sand give up {fmt(e['moved_J'] / 1000)} kJ, "
                                              f"the rest of the mold takes it in and {fmt(e['lost_J'] / 1000)} kJ "
                                              f"leaves its outside, closing to {100 * e['error']:.2g}%", e))
    # Chvorinov's rule for the casting alone: a riser keeps the casting hot where it joins, and chills cool it
    ratio = t_cast / t_chv if solid else float("nan")
    s = (f"Chvorinov's rule with modulus V/A = {fmt(modulus)} mm and mold constant {fmt(B * 100)} s/cm2 gives "
         f"{fmt(t_chv / 60)} min, and the casting freezes in {fmt(t_cast / 60)} min here ({ratio:.2f}x)")
    if study.chills:
        checks.append(Check("hand calculation", "info", s + ", with chills that Chvorinov leaves out cooling it "
                                                            "faster", {"chvorinov_s": t_chv, "ratio": ratio}))
    else:
        st = "pass" if 0.6 <= ratio <= 1.6 else "info"
        checks.append(Check("hand calculation", st, s + ("" if st == "pass" else ", a spread that comes from a "
                            "shape far from compact, which a single modulus cannot capture"),
                            {"chvorinov_s": t_chv, "ratio": ratio}))
    ny, gp = niyama(alloy, res.gradient, res.cooling)
    pockets = fd["pockets"]
    in_pocket = np.zeros(len(rm.X), bool)
    for p in pockets:
        in_pocket[p["nodes"]] = True
    if gp is not None:
        gp = np.where(in_pocket, 100.0 * alloy.shrinkage, gp)
    vol_c = float(sv.node_volume[cn].sum())
    shrink = alloy.shrinkage * vol_c
    if not study.risers:
        checks.append(Check("feeding", "fail", f"there is no riser, so the casting's whole shrinkage, about "
                                               f"{fmt(shrink / 1000)} cm3, collects where it freezes last, at "
                                               f"({_xyz(fd['last_xyz'])}) mm after {fmt(t_cast / 60)} min"))
    elif pockets:
        p = pockets[0]
        many = len(pockets) > 1
        checks.append(Check("feeding", "fail", f"{len(pockets)} region{'s' if many else ''} freeze{'' if many else 's'} "
                            f"cut off from the risers; the largest, {fmt(p['volume_mm3'] / 1000)} cm3 around "
                            f"({_xyz(p['hot_xyz'])}) mm, freezes last at {fmt(p['t_s'] / 60)} min and leaves about "
                            f"{fmt(alloy.shrinkage * p['volume_mm3'] / 1000)} cm3 of shrinkage",
                            {"pockets": [{k: v for k, v in q.items() if k != "nodes"} for q in pockets]}))
    else:
        checks.append(Check("feeding", "pass", "every part of the casting freezes before the metal on its path back "
                                               "to a riser does, so liquid can reach it as it shrinks"))
    t_ratio = None
    if study.risers and fd["t_riser_s"] is not None and solid:
        t_ratio = fd["t_riser_s"] / t_cast
        m_ratio = max(r.modulus for r in study.risers) / modulus
        st = "pass" if t_ratio >= 1.2 else ("warn" if t_ratio >= 1.0 else "fail")
        checks.append(Check("riser", st, f"the risers freeze at {fmt(fd['t_riser_s'] / 60)} min, {t_ratio:.2f}x the "
                                         f"casting, and by hand the riser's modulus is {m_ratio:.2f}x the casting's "
                                         "against the rule of thumb of 1.2x",
                            {"time_ratio": t_ratio, "modulus_ratio": m_ratio}))
    cast_mask = np.zeros(len(rm.X), bool)
    cast_mask[cn] = True
    por_max, por_xyz, por_share = None, None, None
    if gp is not None:
        inner = cast_mask & ~in_pocket
        inner[sv.surface] = False     # porosity forms inside; the skin against the mold is not a Niyama site
        g_c = np.where(inner, gp, np.nan)
        if np.isfinite(g_c).any():
            i = int(np.nanargmax(g_c))
            por_max, por_xyz = float(g_c[i]), rm.X[i].tolist()
            por_share = float(sv.node_volume[np.nan_to_num(g_c) > 0.1].sum()) / vol_c
            st = "pass" if por_max < 0.1 else ("warn" if por_max < 1.0 or por_share < 0.01 else "fail")
            checks.append(Check("porosity", st, f"the dimensionless Niyama criterion gives at most "
                                                f"{fmt(por_max)}% shrinkage porosity, at ({_xyz(por_xyz)}) mm, with "
                                                f"{100 * por_share:.2g}% of the casting over 0.1%",
                                {"max_pct": por_max, "share_over_0.1pct": por_share}))
    else:
        nyc = ny[cn]
        if np.isfinite(nyc).any():
            checks.append(Check("porosity", "info", f"lowest Niyama value {fmt(float(np.nanmin(nyc)))} (K s)^0.5/mm, "
                                                    f"and there is no porosity curve for {alloy.name}"))
    if not solid:
        checks.append(Check("units and sanity", "fail", f"part of the casting was still liquid after "
                                                        f"{fmt(duration / 60)} min"))
    else:
        checks.append(Check("units and sanity", "pass", f"poured at {fmt(pour)} C into {mold.name.lower()} at "
                                                        f"{fmt(study.mold_temp)} C, a {fmt(vol_c / 1000)} cm3 casting "
                                                        f"frozen after {fmt(t_cast / 60)} min"))
    lo_h, hi_h = alloy.latent_range
    spread = 0.0
    if hi_h > lo_h:
        spread = max(abs(chvorinov_constant(alloy, mold, pour, study.mold_temp, x) / B - 1) for x in (lo_h, hi_h))
    extra = [("medium", f"process estimate: freezing times follow the latent heat ({fmt(lo_h / 1000)} to "
                        f"{fmt(hi_h / 1000)} kJ/kg between sources, up to {100 * spread:.0f}% on time) and the "
                        f"mold's heat uptake, which varies about {100 * SPREAD:.0f}% with sand and compaction")]
    # feeding, riser and porosity are the answer itself, so only the checks on the solve set the confidence
    level, reasons = confidence([c for c in checks if c.name in VERIFY], extra)
    fails = (not solid) or (not study.risers) or bool(pockets) or (t_ratio is not None and t_ratio < 1.0) or \
        (por_max is not None and por_max >= 1.0 and por_share is not None and por_share >= 0.01)
    weak = (t_ratio is not None and t_ratio < 1.2) or (por_max is not None and por_max >= 0.1)
    verdict = "fails" if fails else ("marginal" if weak else "ok")
    if not solid:
        head = f"The casting did not finish freezing within {fmt(duration / 60)} min."
    elif not study.risers:
        head = (f"Freezes last at ({_xyz(fd['last_xyz'])}) mm after {fmt(t_cast / 60)} min, and with no riser about "
                f"{fmt(shrink / 1000)} cm3 of shrinkage collects there.")
    elif pockets:
        p = pockets[0]
        head = (f"Shrinkage pocket of about {fmt(alloy.shrinkage * p['volume_mm3'] / 1000)} cm3 near "
                f"({_xyz(p['hot_xyz'])}) mm, which freezes last at {fmt(p['t_s'] / 60)} min cut off from the risers.")
    else:
        head = (f"Freezes sound in {fmt(t_cast / 60)} min, fed throughout by the risers, which freeze at "
                f"{fmt(fd['t_riser_s'] / 60)} min" + (f", with at most {fmt(por_max)}% Niyama porosity."
                                                      if por_max is not None else "."))
    fixes = casting_fixes(study, fd, ratio, B, por_max, por_xyz, t_ratio, rm, cn)
    assumptions = [
        f"{alloy.name}: liquidus {fmt(alloy.liquidus)} C, solidus {fmt(alloy.solidus)} C, latent heat "
        f"{fmt(alloy.latent / 1000)} kJ/kg, {fmt(alloy.conductivity)} W/(m K), {fmt(alloy.specific_heat)} J/(kg K), "
        f"{fmt(alloy.density)} kg/m3, shrinkage on freezing {100 * alloy.shrinkage:.3g}%. Sources: {alloy.source}.",
        f"{mold.name}: {fmt(mold.conductivity)} W/(m K), {fmt(mold.density)} kg/m3, {fmt(mold.specific_heat)} J/(kg K), "
        f"{fmt(margin)} mm thick around the part, its outside cooled at {fmt(study.film)} W/(m2 K) to "
        f"{fmt(study.ambient)} C. Sources: {mold.source}.",
        f"The cavity fills at once with metal at {fmt(pour)} C, so the gating and the cooling while it fills are "
        "not modeled.",
        "Metal and mold touch perfectly throughout. The air gap that opens as the casting shrinks slows cooling "
        "late in freezing, so real freezing times run somewhat longer.",
        "Shrinkage is fed only along paths of metal that freezes later. Gas porosity from hydrogen is not modeled, "
        "which assumes well degassed metal.",
        "Risers are blind cylinders sitting on the casting and poured at the same temperature. The part is cast "
        "the way it is drawn, with +z up.",
        bud.note(bfit),
    ]
    if placed:
        assumptions.insert(0, f"No risers were given, so Wobbl placed {len(placed)} over the hot spots of a coarse "
                              f"freeze without them, each sized by the modulus rule: "
                              f"{'; '.join(r.text() for r in placed)}.")
    if gp is not None:
        assumptions.append("Porosity comes from the dimensionless Niyama criterion of Carlson and Beckermann (2009), "
                           "with the thermal gradient and cooling rate taken 10% of the freezing range above the "
                           "solidus.")
    imgs = casting_images(out, sv, t_sol, gp if gp is not None else ny, gp is not None, fd, por_xyz)
    notes = list(prep.notes)
    try:
        casting_scene(out, sv, fd, verdict, head, study.question, study.risers, por_max, alloy.shrinkage)
    except Exception as e:  # noqa: BLE001  the answer stands without its animation
        notes.append(f"the 3D view was not written: {e}")
    metrics = [("freezes in", f"{fmt(t_cast / 60)} min" if solid else "not frozen"),
               ("shrinkage pockets", str(len(pockets)) if study.risers else "no riser"),
               ("worst porosity", f"{fmt(por_max)}%" if por_max is not None else "n/a"),
               ("riser time ratio", f"{t_ratio:.2f}" if t_ratio is not None else "n/a")]
    values = {"process": "casting", "t_casting_s": t_cast, "t_riser_s": fd["t_riser_s"], "chvorinov_s": t_chv,
              "modulus_mm": modulus, "pour_C": pour, "alloy": alloy.key, "mold": mold.key,
              "risers": [r.text() for r in study.risers], "risers_placed": bool(placed),
              "pockets": [{k: v for k, v in q.items() if k != "nodes"} for q in pockets],
              "porosity_max_pct": por_max, "porosity_xyz": por_xyz, "last_xyz": fd["last_xyz"],
              "budget": bfit.to_dict(),
              "levels": [{"nodes": len(r[0].rm.X), "t_casting_s": r[1]["t_casting_s"], "t99_s": r[1]["t99_s"],
                          "steps": r[0].result.steps} for r in runs],
              "_figures": [["freezing", "Time to freeze, cut through the last spot to freeze"],
                           ["porosity", "Shrinkage porosity from the Niyama criterion, cut through its worst spot"],
                           ["curves", "Solid fraction of the casting and risers over time"]]}
    metal = np.isin(np.array(rm.kinds)[rm.region], ["casting", "riser"])
    used = np.unique(rm.tets[metal])
    field_ = gp if gp is not None else ny
    np.savez_compressed(out / "fields.npz", X=rm.X[used].astype(np.float32), t_solidus=t_sol[used].astype(np.float32),
                        porosity=field_[used].astype(np.float32))
    o = GenericOutcome("casting", str(study.part), study.question, verdict, head, metrics, level, reasons,
                       checks, assumptions, values, imgs, fixes, notes,
                       solver=f"Wobbl enthalpy solver (scipy), {len(rm.X):,} nodes", t_total=time.time() - t0)
    o.write(out)
    if not keep:
        shutil.rmtree(out / "work", ignore_errors=True)
    return o


def _xyz(p) -> str:
    return ", ".join(fmt(float(v)) for v in p)


def casting_fixes(study, fd, ratio, B, por_max, por_xyz, t_ratio, rm, cn) -> list[dict]:
    """Riser size and place from the modulus rule, with the mold constant scaled to agree with the solve."""
    fixes = []
    b_eff = B * (ratio if np.isfinite(ratio) and ratio > 0 else 1.0)
    targets = []
    if not study.risers:
        targets.append((fd["last_xyz"], fd["t_casting_s"], "the last spot to freeze"))
    for p in fd["pockets"][:2]:
        targets.append((p["hot_xyz"], p["t_s"], "the cut off region"))
    for xyz, t_hot, what in targets:
        r = riser_for(xyz, t_hot, b_eff, rm.X, cn)
        at = ",".join(fmt(v) for v in r.at)
        fixes.append({"kind": "riser", "title": f"Add a riser d={fmt(r.diameter)} h={fmt(r.height)} mm over {what}",
                      "summary": f"{what.capitalize()} has a modulus of about {fmt(math.sqrt(t_hot / b_eff))} mm, from "
                                 "its freezing time through Chvorinov's rule scaled to this solve. A blind riser "
                                 f"needs 1.2x that, so d={fmt(r.diameter)} mm and h={fmt(r.height)} mm on top at "
                                 f"({_xyz(r.at)}). Check it with --riser \"d={fmt(r.diameter)} h={fmt(r.height)} at "
                                 f"{at}\", or let Wobbl place risers with --risers auto.", "auto": False})
    for p in fd["pockets"][:1]:
        w = max(10.0, float(np.sort(np.ptp(rm.X[p["nodes"]], axis=0))[-2]))
        fixes.append({"kind": "chill", "title": "Or chill the cut off region",
                      "summary": f"A steel or copper chill against the face nearest ({_xyz(p['hot_xyz'])}) mm, about "
                                 f"{fmt(w)} mm across, makes that region freeze early while the metal behind it can "
                                 "still feed it. Try it with --chill \"steel WxDxH at X,Y,Z\" overlapping that face.",
                      "auto": False})
    if t_ratio is not None and t_ratio < 1.2:
        k = math.sqrt(1.2 / t_ratio)
        fixes.append({"kind": "riser", "title": f"Make the riser about {k:.2g}x wider",
                      "summary": f"The risers freeze at {t_ratio:.2f}x the casting's time and should outlast it by "
                                 f"1.2x. Freezing time grows with the square of the modulus, so a riser about {k:.2g}x "
                                 "the diameter, or an insulating sleeve, gives that margin.", "auto": False})
    if por_max is not None and por_max >= 0.1:
        fixes.append({"kind": "gradient", "title": "Steepen the freezing toward the riser",
                      "summary": f"Porosity of {fmt(por_max)}% at ({_xyz(por_xyz)}) mm comes from a shallow thermal "
                                 "gradient late in freezing. Taper the section so it thickens toward the riser, or "
                                 "chill the far end, so that region freezes in a clear order.", "auto": False})
    return fixes


RISER_SCALE = np.round(np.linspace(0.5, 1.6, 23), 3)       # times the risers' size, on the slider
RISER_RULE = 1.2                                          # a riser should freeze this much later than the casting


def riser_scene(out: Path, V: np.ndarray, F: np.ndarray, tl: np.ndarray, ts: np.ndarray, risers: list[Riser],
                t_cast: float, t_riser: float, por_max: float | None, shrinkage: float, verdict: str, head: str,
                question: str, box=None, hot=None, budget: int = 450_000) -> Path:
    """The metal cut open through its last spot to freeze, with a slider over the risers' size.

    A riser's freezing time goes with the square of its modulus by Chvorinov's rule, and a blind cylinder's modulus
    goes with its size, so each riser scaled k times freezes at k squared times its solved time. The casting keeps
    the times it froze at in the solve. Its cut is colored by when each spot freezes against the riser, past 1 cut
    off from feed and past 1 over 1.2 short of the rule of thumb, the run's lines for fails and marginal. Metal still
    liquid when the risers freeze shrinks with nothing to feed it, so the worst porosity is that unfed fraction of
    the alloy's shrinkage where it is larger than the Niyama porosity the solve found."""
    from ..viz import Scene

    k = RISER_SCALE
    tr = t_riser * k ** 2
    single = len(risers) == 1
    x, unit, label = (k * risers[0].diameter, "mm", "riser diameter") if single else (100 * k, "%", "riser size")
    s = Scene("Casting cut open through its last spot to freeze", domain="casting", question=question,
              source="The casting freezes as the solve found. Each riser's freezing time is its solved one scaled by "
                     "the square of its size, Chvorinov's rule for a riser of the same shape.")
    s.timeline(x, unit=unit, label=label, between="interpolated", kind="input", design=float(x[np.argmin(np.abs(k - 1))]),
               note="the riser scaled as a whole, its height with its diameter")
    if box is not None:
        lo, hi = (np.asarray(v, float) for v in box)
        s.body("box", list(hi - lo), pos=list((lo + hi) / 2), color="#c2a878", opacity=0.12, name="sand mold",
               prop=True)
    # the cut's vertices inside a riser belong to it, not to the casting the riser feeds
    inside = np.zeros(len(V), bool)
    for r in risers:
        ax = "xyz".index(r.axis)
        foot = np.asarray(r.at, float)
        rad = np.linalg.norm(np.delete(V - foot, ax, axis=1), axis=1)
        up = V[:, ax] - foot[ax]
        inside |= (rad <= r.diameter / 2 + 1e-6) & (up > 1e-6) & (up <= r.height + 1e-6)
    keep = ~inside[F].any(axis=1)
    used = np.unique(F[keep])
    lut = np.full(len(V), -1)
    lut[used] = np.arange(len(used))
    Vc, Fc, tsc, tlc = V[used], lut[F[keep]], ts[used], tl[used]
    # the casting is frozen through by t_cast, so a cut vertex shared with the metal of a riser foot counts no later
    ts_fin = np.minimum(np.where(np.isfinite(tsc), tsc, t_cast), t_cast)
    part = s.mesh(Vc, Fc, name="casting, cut open")
    against = ts_fin[None, :] / tr[:, None]
    part.color_by("freezes against the riser", against, unit="x riser", colormap="inferno",
                  range=(0.0, 1.25), limit=1.0, limit_label="riser freezes first",
                  target=1.0 / RISER_RULE)
    # each riser as a cylinder on its foot, growing from it
    for j, r in enumerate(risers):
        ax = "xyz".index(r.axis)
        th = np.linspace(0, 2 * np.pi, 33)[:-1]
        ring = np.c_[np.cos(th), np.sin(th)] * r.diameter / 2
        pts = []
        for hgt in (0.0, r.height):
            q = np.insert(ring, ax, hgt, axis=1)
            pts.append(q)
        X0 = np.vstack(pts + [np.insert(np.zeros((1, 2)), ax, r.height, axis=1)]) + np.asarray(r.at, float)
        n = len(th)
        T = [[i, (i + 1) % n, n + i] for i in range(n)] + [[(i + 1) % n, n + (i + 1) % n, n + i] for i in range(n)]
        T += [[n + i, n + (i + 1) % n, 2 * n] for i in range(n)]
        foot = np.asarray(r.at, float)
        U = (X0 - foot)[None, :, :] * (k - 1)[:, None, None]
        rm_ = s.mesh(X0, np.array(T), name="riser" if j == 0 else f"riser {j + 1}", color="#b9bdc4", opacity=0.55,
                     smooth=False)
        rm_.deform(U)
    ratio = tr / t_cast
    s.readout("riser freezes at", tr / 60.0, unit="min", digits=3)
    s.readout("riser time ratio", ratio, digits=3)
    # metal still liquid when the risers freeze is fed by nothing
    fs = np.clip((tr[:, None] - tlc[None, :]) / np.maximum(ts_fin - tlc, 1e-9)[None, :], 0.0, 1.0)
    unfed = 100.0 * shrinkage * (1.0 - fs.min(axis=1))
    por = np.maximum(unfed, por_max or 0.0)
    s.readout("worst porosity", por, unit="%", digits=3, limit=1.0, target=0.1)
    tmax = float(ts_fin.max())
    for level, name, kind in ((1.0, "riser freezes first", "fail"), (1.0 / RISER_RULE, "riser outlasts it 1.2x",
                                                                     "info")):
        kk = math.sqrt(tmax / (t_riser * level))
        if k[0] <= kk <= k[-1]:
            s.event(float(np.interp(kk, k, x)), name, f"at {fmt(float(np.interp(kk, k, x)))} {unit}", kind=kind)
    if hot is not None:
        s.label("last to freeze", hot, kind="note")
    s.note("The casting's own freezing is kept as solved with the risers at their placed size. A riser this much "
           "larger or smaller also changes how long the metal under it stays hot, which this scaling leaves out.")
    s.verdict(verdict.upper(), head, {"ok": "pass", "marginal": "warn", "fails": "fail"}.get(verdict, "info"))
    return s.save(out, budget=budget)


def riser_scene_from_stored(out: Path) -> Path | None:
    """The riser slider for a run saved before it had one, from its stored scene and outcome: each cut vertex's
    times to start and finish freezing are read back from its solid fraction frames, which grow linearly between
    them."""
    import json

    from ..viz.scene import array, load
    from .regions import parse_riser

    out = Path(out)
    o = json.loads((out / "outcome.json").read_text())
    v = o.get("values") or {}
    risers = [parse_riser(t) for t in v.get("risers") or []]
    if not risers or not v.get("t_riser_s"):
        return None
    h, blob = load((out / "scene.wobbl").read_bytes())
    if h["timeline"].get("kind") == "input":
        return None                                  # already a slider
    mesh = next(ob for ob in h["objects"] if ob["kind"] == "mesh" and "frames" in ob.get("field", {}))
    V, F = array(h, blob, mesh["pos"]), array(h, blob, mesh["tri"]).astype(int)
    fr = array(h, blob, mesh["field"]["frames"])
    t = np.asarray(h["timeline"]["t"], float) * 60.0

    def cross(level):
        """The time each vertex's solid fraction passes level, by straight lines between frames."""
        above = fr >= level
        j = np.where(above.any(axis=0), above.argmax(axis=0), -1)
        out_ = np.full(fr.shape[1], np.inf)
        ok = j > 0
        a, b = fr[j[ok] - 1, np.where(ok)[0]], fr[j[ok], np.where(ok)[0]]
        out_[ok] = t[j[ok] - 1] + (level - a) / np.maximum(b - a, 1e-9) * (t[j[ok]] - t[j[ok] - 1])
        out_[j == 0] = t[0]
        return out_

    ta, tb = cross(0.25), cross(0.75)
    rate = 0.5 / np.maximum(tb - ta, 1e-9)
    tl = np.maximum(ta - 0.25 / rate, 0.0)
    ts = tb + 0.25 / rate
    box = next(([np.asarray(ob["pos"]) - np.asarray(ob["size"]) / 2, np.asarray(ob["pos"]) + np.asarray(ob["size"]) / 2]
                for ob in h["objects"] if ob.get("name") == "sand mold"), None)
    hot = next((lb["at"] for lb in h.get("labels", []) if lb["text"] == "last to freeze"), None)
    head = (h.get("verdict") or {}).get("line") or o.get("headline", "")
    return riser_scene(out, V, F, tl, ts, risers, float(v["t_casting_s"]), float(v["t_riser_s"]),
                       v.get("porosity_max_pct"), get_alloy(v.get("alloy", "a356")).shrinkage, o["verdict"], head,
                       o.get("question", ""), box=box, hot=hot)


def casting_scene(out: Path, sv: Solved, fd: dict, verdict: str, head: str, question: str, risers=(),
                  por_max: float | None = None, shrinkage: float = 0.0) -> Path:
    """The metal cut open through its last spot to freeze. With risers it is a slider over their size (riser_scene),
    without them its solid fraction grows in from the mold walls frame by frame, from each node's own times to start
    and finish freezing."""
    from ..viz import Scene
    from .render import cut_surface, plane_through

    rm, ev = sv.rm, sv.result.events
    metal = np.isin(np.array(rm.kinds)[rm.region], ["casting", "riser"])
    used = np.unique(rm.tets[metal])
    lut = np.full(len(rm.X), -1)
    lut[used] = np.arange(len(used))
    X, tets = rm.X[used], lut[rm.tets[metal]]
    hot = fd["pockets"][0]["hot_xyz"] if fd["pockets"] else fd["last_xyz"]
    cut = plane_through(X, hot)
    V, F, vals = cut_surface(X, tets, {"tl": ev["liquidus"][used], "ts": ev["solidus"][used]}, cut)
    tl, ts = vals["tl"], vals["ts"]
    tl = np.where(tl < 0, 0.0, tl)                 # metal that started below the liquidus starts freezing at once
    ts = np.where(ts < 0, np.inf, ts)              # metal still freezing when the run stopped
    if risers and fd.get("t_riser_s"):
        return riser_scene(out, V, F, tl, ts, list(risers), float(fd["t_casting_s"]), float(fd["t_riser_s"]), por_max,
                           shrinkage, verdict, head, question, box=rm.box, hot=hot)
    t_end = float(np.nanmax(ev["solidus"][used]))
    t = np.linspace(0.0, 1.05 * t_end, 72)
    span = np.maximum(ts - tl, 1e-9)
    fs = np.clip((t[:, None] - tl[None, :]) / span[None, :], 0.0, 1.0)
    w = sv.node_volume[sv.casting_nodes]
    tsc = np.nan_to_num(ev["solidus"][sv.casting_nodes], nan=np.inf)
    solid = np.array([float(w[tsc <= x].sum() / w.sum()) for x in t])
    s = Scene("Casting freezing from the mold walls inward", domain="casting", question=question)
    s.timeline(t / 60.0, unit="min", label="time after pouring", seconds=10, between="interpolated")
    lo, hi = rm.box
    s.body("box", list(hi - lo), pos=list((lo + hi) / 2), color="#c2a878", opacity=0.12, name="sand mold",
           prop=True)
    part = s.mesh(V, F, name="casting, cut open")
    part.color_by("solid fraction", fs, colormap="inferno", range=(0.0, 1.0))
    s.readout("casting solid", 100.0 * solid, unit="%", digits=3)
    t_hot = fd["pockets"][0]["t_s"] if fd["pockets"] else fd["t_casting_s"]
    s.event(t_hot / 60.0, "last spot freezes", f"at ({_xyz(hot)}) mm", kind="fail" if fd["pockets"] else "info")
    s.label("last to freeze", hot, t0=0.6 * t_hot / 60.0, kind="note")
    s.verdict(verdict.upper(), head, {"ok": "pass", "marginal": "warn", "fails": "fail"}.get(verdict, "info"))
    return s.save(out, budget=450_000)


def casting_images(out: Path, sv: Solved, t_sol, field_, is_pct: bool, fd: dict, por_xyz) -> dict:
    from ..report.render import line_plot
    from .render import plane_through, section

    rm = sv.rm
    imgs = {}
    metal = np.isin(np.array(rm.kinds)[rm.region], ["casting", "riser"])
    used = np.unique(rm.tets[metal])
    lut = np.full(len(rm.X), -1)
    lut[used] = np.arange(len(used))
    X, tets = rm.X[used], lut[rm.tets[metal]]
    try:
        hot = fd["pockets"][0]["hot_xyz"] if fd["pockets"] else fd["last_xyz"]
        o, n = plane_through(X, hot)
        imgs["freezing"] = section(out / "images" / "freezing.png", X, tets, t_sol[used] / 60.0, "freezes after",
                                   "min", o, n, hot, f"{fmt(float(np.nanmax(t_sol[sv.casting_nodes])) / 60)} min",
                                   "magma")
        v = field_[used]
        if np.isfinite(v).any():
            mark = por_xyz if por_xyz is not None else hot
            o2, n2 = plane_through(X, mark)
            if is_pct:
                imgs["porosity"] = section(out / "images" / "porosity.png", X, tets, v, "shrinkage porosity", "%",
                                           o2, n2, mark, f"{fmt(float(np.nanmax(v)))}%", "viridis",
                                           (0.0, max(float(np.nanmax(v)), 0.2)))
            else:
                imgs["porosity"] = section(out / "images" / "porosity.png", X, tets, v, "Niyama", "(K s)^0.5/mm",
                                           o2, n2, mark, "", "viridis_r")
        grid = np.linspace(0, float(np.nanmax(t_sol[used])) * 1.05, 120)
        curves = {}
        for name, nodes in (("casting", sv.casting_nodes), ("risers", sv.riser_nodes)):
            if len(nodes):
                w = sv.node_volume[nodes]
                ts = np.nan_to_num(t_sol[nodes], nan=np.inf)
                curves[name] = [float(w[ts <= g].sum() / w.sum()) for g in grid]
        imgs["curves"] = line_plot(out / "images" / "curves.png", grid / 60.0, curves, "minutes", "solid fraction")
    except Exception as e:  # noqa: BLE001
        imgs["error"] = str(e)
    return imgs
