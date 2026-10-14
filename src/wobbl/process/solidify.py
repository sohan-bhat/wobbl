"""Transient heat with latent heat, by the enthalpy method on linear tetrahedra.

A casting gives up its latent heat over a narrow band of temperature, where the
heat it holds jumps by more than a hundred degrees' worth. Wobbl first tried
CalculiX with that latent heat folded into a tabulated specific heat: it needed
microsecond steps to cross the peak and covered 0.04 s of a 4,000 node bar in
ten minutes. So solidification runs here, on the same gmsh meshes, with the
energy of each node written as sensible heat plus the latent heat still held:

    E = C T + Lambda g        (C heat capacity, Lambda latent heat, g liquid fraction)
    (E - E_old) / dt + K T = F

K is the conduction and film matrix. Each step is backward Euler, and the liquid
fraction is found with the source based update of Voller and Swaminathan
(Numerical Heat Transfer B 19, 1991, 175-189): the latent heat enters as a
source linearized about the current liquid fraction, the temperatures are
solved, and the liquid fraction moves by the slope of g(T) times how far the
temperature sits from the one that liquid fraction belongs to. A node that
freezes inside one step gives up exactly its latent heat, energy is conserved
to the iteration tolerance whatever the step size, and the step size is set by
how fast temperatures and liquid fractions move.

Units inside: m, s, J, W, C. Meshes come in mm.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import cg, spsolve


@dataclass
class Material:
    name: str
    density: float                     # kg/m3
    conductivity: float                # W/(m K)
    specific_heat: float               # J/(kg K)
    latent: float = 0.0                # J/kg, 0 for a mold
    solid_fraction: Callable | None = None   # fs(T) of the metal
    liquidus: float = 0.0
    solidus: float = 0.0

    @property
    def metal(self) -> bool:
        return self.latent > 0


@dataclass
class Model:
    X: np.ndarray                      # (n,3) mm
    tets: np.ndarray                   # (m,4) corner nodes
    region: np.ndarray                 # (m,) index into materials
    materials: list[Material]          # at most one metal: the casting and its risers are one pour
    initial: np.ndarray                # (n,) C
    films: list[tuple[np.ndarray, float, float]] = field(default_factory=list)   # (tris (k,3), h W/(m2 K), T_inf)
    fixed: list[tuple[np.ndarray, float]] = field(default_factory=list)          # (nodes, C) held from t = 0
    # (faces on one side (k,3), the matching faces on the other, conductance W/(m2 K)) for split interfaces
    contacts: list[tuple[np.ndarray, np.ndarray, float]] = field(default_factory=list)
    duration: float = 0.0              # s, longest run
    watch: np.ndarray | None = None    # nodes whose events are recorded (the metal), default all
    events: dict[str, float] = field(default_factory=dict)   # name -> temperature crossed while cooling
    gradient_event: str = ""           # event at which the thermal gradient and cooling rate are recorded
    stop_below: float | None = None    # stop once every watched node is below this temperature
    save_times: list[float] = field(default_factory=list)
    dT_max: float = 10.0               # largest temperature change of a node in one step, C
    dg_max: float = 0.2                # largest liquid fraction change of a node in one step
    dt0: float = 1e-3                  # first step, s
    dt_max: float = 0.0                # longest step, s, default a two hundredth of the run


@dataclass
class Result:
    events: dict[str, np.ndarray]      # name -> (n,) time each node crossed it, nan if never
    gradient: np.ndarray               # (n,) K/mm at the gradient event
    cooling: np.ndarray                # (n,) K/s at the gradient event
    T: np.ndarray                      # final temperatures
    g: np.ndarray                      # final liquid fractions
    t_end: float
    steps: int
    saved: dict[float, np.ndarray]     # save time -> temperatures
    history_t: np.ndarray              # step times
    history_fs: np.ndarray             # (steps,) solid fraction of the metal by mass
    energy: dict                       # stored, lost through films, supplied by held faces, error
    wall: float = 0.0
    rejected: int = 0
    stats: dict = field(default_factory=dict)


class LiquidCurve:
    """Liquid fraction g(T) of the metal as a table, with its inverse and slope, both piecewise linear."""

    def __init__(self, mat: Material):
        lo, hi = mat.solidus, mat.liquidus
        T = np.linspace(lo, hi, 801)
        g = 1.0 - np.asarray(mat.solid_fraction(T), float)
        g[0], g[-1] = 0.0, 1.0
        # keep it strictly increasing so the inverse is a function
        g = np.maximum.accumulate(g + np.linspace(0.0, 1e-9, len(g)))
        self.T, self.g = T, g / g[-1]
        self.slope = np.diff(self.g) / np.diff(self.T)

    def of(self, T):
        return np.interp(T, self.T, self.g)

    def temperature(self, g):
        return np.interp(g, self.g, self.T)

    def dgdT(self, g, entering_from_above):
        """Slope of g(T) at liquid fraction g; at g = 0 or 1 the slope on the side the node is entering from."""
        i = np.clip(np.searchsorted(self.g, g, side="right") - 1, 0, len(self.slope) - 1)
        i = np.where(g >= 1.0, len(self.slope) - 1, np.where(g <= 0.0, 0, i))
        return self.slope[i]


def _tet_geometry(X, tets):
    """Shape function gradients (m,4,3) in 1/m and volumes (m,) in m3 of linear tetrahedra."""
    P = X[tets] / 1000.0
    M = np.concatenate([np.ones((len(tets), 4, 1)), P], axis=2)
    det = np.linalg.det(M)
    inv = np.linalg.inv(M)
    return np.transpose(inv[:, 1:, :], (0, 2, 1)), np.abs(det) / 6.0


def assemble(model: Model):
    """Conduction matrix (n,n) W/K, node heat capacity C (J/K) and latent heat Lambda (J), element gradients and
    volumes, and the film diagonal and source."""
    n = len(model.X)
    B, V = _tet_geometry(model.X, model.tets)
    mats = model.materials
    k = np.array([m.conductivity for m in mats])[model.region]
    Ke = (k * V)[:, None, None] * np.einsum("eik,ejk->eij", B, B)
    rows = np.repeat(model.tets, 4, axis=1).ravel()
    cols = np.tile(model.tets, (1, 4)).ravel()
    K = sp.csr_matrix((Ke.ravel(), (rows, cols)), shape=(n, n))
    for ta, tb, h in model.contacts:
        p = model.X[ta] / 1000.0
        g = h * 0.5 * np.linalg.norm(np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), axis=1) / 3.0
        a, b = ta.ravel(), tb.ravel()
        gg = np.repeat(g, 3)
        K = K + sp.csr_matrix((np.r_[gg, gg, -gg, -gg], (np.r_[a, b, a, b], np.r_[a, b, b, a])), shape=(n, n))
    rho = np.array([m.density for m in mats])[model.region]
    cp = np.array([m.specific_heat for m in mats])[model.region]
    lat = np.array([m.latent for m in mats])[model.region]
    C, Lam, metal_mass = np.zeros(n), np.zeros(n), np.zeros(n)
    for c in range(4):
        np.add.at(C, model.tets[:, c], rho * cp * V / 4.0)
        np.add.at(Lam, model.tets[:, c], rho * lat * V / 4.0)
        np.add.at(metal_mass, model.tets[:, c], np.where(lat > 0, rho * V / 4.0, 0.0))
    hdiag, src = np.zeros(n), np.zeros(n)
    for tris, h, tinf in model.films:
        p = model.X[tris[:, :3]] / 1000.0
        a = 0.5 * np.linalg.norm(np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), axis=1)
        for c in range(3):
            np.add.at(hdiag, tris[:, c], h * a / 3.0)
            np.add.at(src, tris[:, c], h * tinf * a / 3.0)
    return K, C, Lam, metal_mass, B, V, hdiag, src


def solve(model: Model, progress=None) -> Result:
    t_wall = time.time()
    say = progress or (lambda a, b: None)
    n = len(model.X)
    K, C, Lam, mmass, B, V, hdiag, src = assemble(model)
    metals = [m for m in model.materials if m.metal]
    curve = LiquidCurve(metals[0]) if metals else None
    A = (K + sp.diags(hdiag)).tocsr()
    T = np.asarray(model.initial, float).copy()
    fixed = np.zeros(n, bool)
    for nodes, temp in model.fixed:
        T[nodes] = temp
        fixed[nodes] = True
    free = np.nonzero(~fixed)[0]
    fx = np.nonzero(fixed)[0]
    Aff = A[free][:, free].tocsr()
    rhs_fix = A[free][:, fixed] @ T[fixed] if fixed.any() else np.zeros(len(free))
    latent = Lam > 0
    g = curve.of(T) if curve is not None else np.zeros(n)
    g = np.where(latent, g, 0.0)
    watch = np.arange(n) if model.watch is None else np.asarray(model.watch)
    wmask = np.zeros(n, bool)
    wmask[watch] = True
    # nodal gradients: volume weighted average of the constant element gradients of the metal elements
    el = np.nonzero(np.array([m.metal for m in model.materials])[model.region])[0]
    if not len(el):
        el = np.arange(len(model.tets))
    G = sp.csr_matrix((np.repeat(V[el], 4), (model.tets[el].ravel(), np.repeat(np.arange(len(el)), 4))),
                      shape=(n, len(el)))
    wsum = np.asarray(G.sum(axis=1)).ravel()
    Gn = sp.diags(np.where(wsum > 0, 1.0 / np.maximum(wsum, 1e-300), 0.0)) @ G

    def node_grad(Tf):
        return np.asarray(Gn @ np.einsum("eik,ei->ek", B[el], Tf[model.tets[el]])) / 1000.0   # K/mm

    def energy(Tf, gf):
        return C * Tf + Lam * gf

    events = {k: np.full(n, np.nan) for k in model.events}
    grad_at, cool_at = np.full(n, np.nan), np.full(n, np.nan)
    saved: dict[float, np.ndarray] = {}
    pending = sorted(t for t in model.save_times if t > 0)
    E0 = energy(T, g)
    lost, supplied = 0.0, 0.0
    msum = max(float(mmass.sum()), 1e-300)
    hist_t, hist_fs = [0.0], [1.0 - float((mmass * g).sum()) / msum]
    t, dt, steps, rejected = 0.0, model.dt0, 0, 0
    stats = {"iterations": 0, "unconverged": 0, "too_big": 0, "direct": 0}
    last_say = time.time()
    while t < model.duration - 1e-12:
        dt = min(dt, model.duration - t, model.dt_max or model.duration / 200.0)
        Tn, gn = T, g
        Tk, gk = Tn.copy(), gn.copy()
        ok = False
        for it in range(30):
            stats["iterations"] += 1
            # latent heat as a source linearized about gk: g ~ gk + s (T - T*(gk)) for nodes that are mushy or
            # crossing into the mushy range, where s is the slope of g(T)
            s = np.zeros(n)
            Tstar = Tk.copy()
            if curve is not None:
                active = latent & (((gk > 0) & (gk < 1)) | ((gk >= 1) & (Tk <= curve.T[-1])) |
                                   ((gk <= 0) & (Tk >= curve.T[0])))
                s[active] = curve.dgdT(gk[active], gk[active] >= 1)
                Tstar[active] = curve.temperature(gk[active])
            diag = (C + Lam * s) / dt
            Mff = (Aff + sp.diags(diag[free])).tocsr()
            rhs = (src + (C * Tn - Lam * (gk - gn) + Lam * s * Tstar) / dt)[free] - rhs_fix
            sol, info = cg(Mff, rhs, x0=Tk[free], rtol=1e-12, atol=0.0, maxiter=3000,
                           M=sp.diags(1.0 / Mff.diagonal()))
            if info != 0:
                stats["direct"] += 1
                sol = spsolve(Mff.tocsc(), rhs)
            Tnew = Tk.copy()
            Tnew[free] = sol
            gnew = gk.copy()
            if curve is not None:
                gnew[latent] = np.clip(gk[latent] + s[latent] * (Tnew[latent] - Tstar[latent]), 0.0, 1.0)
            dT_it = float(np.abs(Tnew - Tk).max())
            dg_it = float(np.abs(gnew - gk).max()) if curve is not None else 0.0
            Tk, gk = Tnew, gnew
            if dT_it < 1e-5 and dg_it < 1e-7:
                ok = True
                break
        dT = float(np.abs(Tk - Tn).max())
        dg = float(np.abs(gk - gn).max()) if curve is not None else 0.0
        too_big = dT > 1.5 * model.dT_max or dg > 1.5 * model.dg_max
        if not ok or too_big:
            stats["unconverged" if not ok else "too_big"] += 1
            dt *= 0.5
            rejected += 1
            if dt < 1e-9:
                raise RuntimeError("the solidification time step collapsed; the model may be badly scaled")
            continue
        t_new = t + dt
        Ek, En = energy(Tk, gk), energy(Tn, gn)
        lost += float((hdiag * Tk - src).sum()) * dt          # W through films at the end of the step
        if len(fx):
            supplied += float(((Ek[fx] - En[fx]) / dt + A[fx] @ Tk - src[fx]).sum()) * dt
        for name, temp in model.events.items():
            ev = events[name]
            cross = wmask & np.isnan(ev) & (Tn > temp) & (Tk <= temp)
            if cross.any():
                th = (Tn[cross] - temp) / np.maximum(Tn[cross] - Tk[cross], 1e-12)
                ev[cross] = t + th * dt
                if name == model.gradient_event:
                    idx = np.nonzero(cross)[0]
                    gr = (1 - th)[:, None] * node_grad(Tn)[idx] + th[:, None] * node_grad(Tk)[idx]
                    grad_at[idx] = np.linalg.norm(gr, axis=1)
                    cool_at[idx] = (Tn[idx] - Tk[idx]) / dt
        while pending and pending[0] <= t_new + 1e-12:
            s_ = pending.pop(0)
            w = (s_ - t) / dt
            saved[s_] = (1 - w) * Tn + w * Tk
        T, g, t = Tk, gk, t_new
        steps += 1
        hist_t.append(t)
        hist_fs.append(1.0 - float((mmass * g).sum()) / msum)
        grow = min(model.dT_max / max(dT, 1e-9), model.dg_max / max(dg, 1e-9))
        dt *= float(np.clip(0.9 * grow, 0.5, 1.5))
        if time.time() - last_say > 5:
            say("solve", f"t = {t:.4g} s, {steps} steps")
            last_say = time.time()
        if model.stop_below is not None and wmask.any() and float(T[wmask].max()) < model.stop_below:
            break
    dE = energy(T, g) - E0
    stored = float(dE.sum())
    # judged against the heat that moved: what the cooling nodes gave up, which is more than the net change when
    # the metal's heat stays in the mold
    moved = max(float(-dE[dE < 0].sum()), abs(lost), abs(supplied), 1e-12)
    err = abs(stored + lost - supplied) / moved
    return Result(events, grad_at, cool_at, T, g, t, steps, saved, np.array(hist_t), np.array(hist_fs),
                  {"stored_J": stored, "lost_J": lost, "supplied_J": supplied, "moved_J": moved, "error": err},
                  time.time() - t_wall, rejected, stats)
