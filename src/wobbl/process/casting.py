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
             finer_steps: float = 1.0, progress=None) -> Solved:
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
                    dT_max=6.0 / finer_steps, dg_max=0.12 / finer_steps, dt0=0.01)
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

