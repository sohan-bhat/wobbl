"""Mold filling: Hele Shaw flow of a shear thinning melt through thin walls, front by front.

A thin wall is a narrow gap, so the melt flows along it and the pressure is the
same across it. Through a gap b, a pressure gradient G drives a flow per unit
width q = S(b, G) G, where the fluidity S = 2 / G^3 times the integral of
tau gamma_dot(tau) from the wall stress G b / 2 down to the middle (for a
Newtonian melt, b^3 / (12 eta)). Wobbl solves this on the part's own linear
tetrahedra: each element conducts S / b per unit area, with b its local wall
thickness, which is the midplane model without extracting a midplane, and
junctions, ribs and bosses come out of the solid mesh itself.

Filling uses control volumes around the nodes (Wang, Hieber and Wang 1986;
Bruschke and Advani 1990). Each step solves the pressure over the filled nodes
with zero pressure at the front, sends the melt that leaves the filled region
into the front nodes it reaches, and advances to fill a few more. The melt
injected always equals the melt in the cavity. With a pressure limit the gate
pressure is capped and the flow falls instead, and with a frozen skin the gap
narrows behind the front as heat leaves into the mold, so a part can stall
short.

Units inside: mm, s, MPa, mm3/s; viscosities from the resin in Pa s.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra
from scipy.sparse.linalg import spsolve
from scipy.special import erfinv



@dataclass
class Viscosity:
    """Cross-WLF (from a Resin) at one melt temperature, or a power law eta = m gamma_dot^(n - 1) for the closed
    form checks."""
    resin: object | None = None
    melt: float = 0.0                  # C
    m: float = 0.0                     # Pa s^n, power law consistency
    n: float = 1.0

    def eta0(self, p_mpa):
        return self.resin.eta0(self.melt, np.asarray(p_mpa) * 1e6)


_CROSS_TABLES: dict[float, tuple[np.ndarray, np.ndarray]] = {}


def cross_integral(n: float) -> tuple[np.ndarray, np.ndarray]:
    """I(s) = integral from 0 to s of s' x(s') ds', tabulated on a log grid of s = tau / tau*, where x = eta0
    gamma_dot / tau* solves s = x / (1 + x^(1 - n)). It depends on n alone, so the fluidity of every element is
    2 tau*^3 I(G b / (2 tau*)) / (eta0 G^3) with one table lookup."""
    if n in _CROSS_TABLES:
        return _CROSS_TABLES[n]
    s = np.geomspace(1e-8, 0.995 if n <= 0 else 1e6, 3000)
    if n <= 0:
        x = s / (1.0 - s)
    else:
        y = np.log(np.maximum(s, s ** (1.0 / n)))
        for _ in range(60):
            x = np.exp(y)
            a = x ** (1.0 - n)
            step = (np.log(x) - np.log1p(a) - np.log(s)) / (1.0 - (1.0 - n) * a / (1.0 + a))
            y -= step
            if np.abs(step).max() < 1e-12:
                break
        x = np.exp(y)
    integrand = s * s * x                       # s x ds = s^2 x d(ln s)
    ls = np.log(s)
    I = np.concatenate([[s[0] ** 3 / 3.0], s[0] ** 3 / 3.0 + np.cumsum(0.5 * (integrand[1:] + integrand[:-1])
                                                                       * np.diff(ls))])
    _CROSS_TABLES[n] = (s, I)
    return s, I


def fluidity(visc: Viscosity, b: np.ndarray, G: np.ndarray, p: np.ndarray) -> np.ndarray:
    """S in mm3/(MPa s) per element for gap b (mm), pressure gradient G (MPa/mm) and local pressure p (MPa)."""
    b = np.asarray(b, float)
    G = np.maximum(np.asarray(G, float), 1e-9)
    tw = G * b / 2.0                                       # MPa, the shear stress at the wall
    if visc.resin is None:
        n, m = visc.n, visc.m * 1e-6                       # m in MPa s^n
        return 2.0 * m ** (-1.0 / n) * tw ** (2.0 + 1.0 / n) / ((2.0 + 1.0 / n) * G ** 3)
    ts = visc.resin.tau * 1e-6                             # MPa
    e0 = np.asarray(visc.eta0(p), float) * 1e-6            # MPa s
    s_tab, I_tab = cross_integral(float(visc.resin.n))
    sw = np.minimum(tw / ts, s_tab[-1])
    low = sw <= s_tab[0]
    I = np.where(low, sw ** 3 / 3.0, np.exp(np.interp(np.log(np.maximum(sw, s_tab[0])), np.log(s_tab),
                                                      np.log(I_tab))))
    return 2.0 * ts ** 3 * I / (e0 * G ** 3)


def newtonian_fluidity(b, eta_pa_s):
    return np.asarray(b, float) ** 3 / (12.0 * eta_pa_s * 1e-6)


@dataclass
class Mesh:
    X: np.ndarray                      # (n,3) mm
    tets: np.ndarray                   # (m,4)
    thickness: np.ndarray              # (m,) mm, the wall each element sits in


@dataclass
class FillSettings:
    gates: list[np.ndarray]            # node index arrays, one per gate
    flow: float                        # mm3/s, set by the fill time
    p_max: float = math.inf            # MPa, the machine's limit at the gate
    t_limit: float = math.inf          # s, give up after this
    freeze: dict | None = None         # {"alpha": mm2/s, "ratio": erfinv factor} for a frozen skin
    step_fill: float = 0.01            # share of the cavity filled per step
    picard: int = 25
    picard_tol: float = 1e-3        # change in gate pressure that ends the Picard iterations of a step


@dataclass
class FillResult:
    t_fill: np.ndarray                 # (n,) s, when each node filled, nan if never
    filled: float                      # share of the cavity volume filled
    times: np.ndarray                  # step end times
    gate_pressure: np.ndarray          # MPa at each step
    flow: np.ndarray                   # mm3/s at each step
    p_end: np.ndarray                  # (n,) pressure at the end of filling
    injected: float                    # mm3
    in_cavity: float                   # mm3
    short: bool
    stalled_by: str = ""
    steps: int = 0
    notes: list[str] = field(default_factory=list)
    share: np.ndarray = field(default_factory=lambda: np.zeros(0))   # share of the cavity full at each step


def geometry(mesh: Mesh):
    P = mesh.X[mesh.tets]
    M = np.concatenate([np.ones((len(mesh.tets), 4, 1)), P], axis=2)
    inv = np.linalg.inv(M)
    B = np.transpose(inv[:, 1:, :], (0, 2, 1))             # (m,4,3) 1/mm
    V = np.abs(np.linalg.det(M)) / 6.0
    return B, V


class _System:
    """The conduction matrix on a fixed pattern: every element's 16 entries land in known slots, so a step only
    rewrites the values."""

    def __init__(self, mesh: Mesh, BB: np.ndarray):
        n = len(mesh.X)
        r = np.repeat(mesh.tets, 4, axis=1).ravel()
        c = np.tile(mesh.tets, (1, 4)).ravel()
        pat = sp.csr_matrix((np.ones(len(r)), (r, c)), shape=(n, n))
        pat.sum_duplicates()
        pat.sort_indices()
        self.K = pat.copy()
        rows = np.repeat(np.arange(n), np.diff(pat.indptr))
        key = rows.astype(np.int64) * n + pat.indices
        self.pos = np.searchsorted(key, r.astype(np.int64) * n + c)
        self.BB = BB.reshape(len(BB), 16)

    def set(self, k: np.ndarray):
        self.K.data = np.bincount(self.pos, (k[:, None] * self.BB).ravel(), minlength=len(self.K.data))

    def solve(self, known: np.ndarray, p_known: np.ndarray, rhs: np.ndarray, x0: np.ndarray) -> np.ndarray:
        """K p = rhs on the unknown nodes, with p = p_known where known, by a sparse factorization: thin walls leave
        the system too badly conditioned for conjugate gradients with a diagonal preconditioner."""
        un = np.nonzero(~known)[0]
        x = np.where(known, p_known, 0.0)
        if len(un):
            Kr = self.K[un]
            x[un] = spsolve(Kr[:, un].tocsc(), rhs[un] - Kr @ x, permc_spec="MMD_AT_PLUS_A")
        return x


def reference_viscosity(visc: Viscosity, rate: float = 1000.0) -> float:
    """Pa s at a typical molding shear rate, for the first guess of a step."""
    if visc.resin is None:
        return visc.m * rate ** (visc.n - 1.0)
    e0 = float(visc.eta0(0.0))
    return e0 / (1.0 + (e0 * rate / visc.resin.tau) ** (1.0 - visc.resin.n))


def fill(mesh: Mesh, visc: Viscosity, s: FillSettings, progress=None) -> FillResult:
    say = progress or (lambda a, b: None)
    n, m = len(mesh.X), len(mesh.tets)
    B, V = geometry(mesh)
    BB = np.einsum("eik,ejk->eij", B, B) * V[:, None, None]  # mm
    sysm = _System(mesh, BB)
    vol = np.zeros(n)
    for c in range(4):
        np.add.at(vol, mesh.tets[:, c], V / 4.0)
    total = float(vol.sum())
    f = np.zeros(n)
    t_fill = np.full(n, np.nan)
    gate = np.unique(np.concatenate(s.gates))
    f[gate] = 1.0
    t_fill[gate] = 0.0
    born = np.full(m, np.nan)                              # when each element first had melt in it
    p = np.zeros(n)
    G_el = np.full(m, np.nan)
    eta_ref = reference_viscosity(visc)
    relax = float(np.clip(visc.n if visc.resin is None else visc.resin.n, 0.3, 1.0))
    t, steps = 0.0, 0
    times, gp, flows, shares = [], [], [], []
    injected = float(vol[gate].sum())
    q_gate = np.zeros(n)
    q_gate[gate] = s.flow * vol[gate] / vol[gate].sum()
    is_gate = np.zeros(n, bool)
    is_gate[gate] = True
    short, why = False, ""
    pressure_mode = False
    say("fill", f"{n:,} nodes, gate of {len(gate)} nodes")
    while f.min() < 1.0:
        full = f >= 1.0 - 1e-12
        active = full[mesh.tets].any(axis=1)
        born[active & np.isnan(born)] = t
        b = mesh.thickness.copy()
        if s.freeze:
            skin = s.freeze["ratio"] * 2.0 * np.sqrt(s.freeze["alpha"] * np.maximum(t - born, 0.0))
            b = np.maximum(b - 2.0 * np.nan_to_num(skin, nan=0.0), 0.02 * mesh.thickness)
        front = np.zeros(n, bool)
        front[np.unique(mesh.tets[active])] = True
        front &= ~full
        if not front.any():
            short, why = True, "no unfilled node touches the melt"
            break
        # a new element starts from the Newtonian gradient at the typical shear rate, the rest from the last step
        guess = np.isnan(G_el) & active
        seen = ~np.isnan(G_el) & active
        if guess.any() and seen.any():
            # elements the melt just reached sit at the front, where the gradient is the low end of the flow's
            G_el = np.where(guess, float(np.percentile(G_el[seen], 25)), G_el)
        elif guess.any():
            sysm.set(np.where(active, newtonian_fluidity(b, eta_ref) / b, 0.0))
            pn = sysm.solve(~full | (pressure_mode & is_gate), np.where(is_gate, s.p_max, 0.0),
                            np.where(pressure_mode, 0.0, q_gate), p)
            Gn = np.linalg.norm(np.einsum("eik,ei->ek", B, pn[mesh.tets]), axis=1)
            G_el = np.where(guess, np.maximum(Gn, 1e-9), G_el)
        G_try = np.where(active, G_el, 1e-9)
        for it in range(s.picard):
            pe = p[mesh.tets].mean(axis=1)
            k = np.zeros(m)
            k[active] = fluidity(visc, b[active], G_try[active], pe[active]) / b[active]
            sysm.set(k)
            known = ~full | (pressure_mode & is_gate)
            p_new = sysm.solve(known, np.where(is_gate & pressure_mode, s.p_max, 0.0),
                               np.where(pressure_mode, 0.0, q_gate), p)
            if not pressure_mode and p_new[gate].max() > s.p_max:
                pressure_mode = True          # the machine holds its limit at the gate and the flow follows
                continue
            G_new = np.maximum(np.linalg.norm(np.einsum("eik,ei->ek", B, p_new[mesh.tets]), axis=1), 1e-9)
            change = abs(p_new[gate].max() - p[gate].max()) / max(p_new[gate].max(), 1e-12)
            p = p_new
            if change < s.picard_tol and it > 0:
                G_try = np.where(active, G_new, G_try)
                break
            # Picard on the shear thinning, relaxed in the logarithm by the power law index: for a power law the
            # gradient a trial gradient gives goes as its -(1/n - 1) power, so that weight lands on the answer at once
            G_try = np.where(active, G_try ** (1.0 - relax) * G_new ** relax, G_try)
        G_el = np.where(active, G_try, G_el)
        # melt leaving the filled region into each front node
        fr = np.nonzero(front)[0]
        inflow = np.maximum(-(sysm.K @ np.where(full, p, 0.0))[fr], 0.0)
        Q = float(inflow.sum())
        if Q <= 1e-9 * s.flow:
            short, why = True, "the melt stopped moving"
            break
        room = (1.0 - f[fr]) * vol[fr]
        t_each = np.where(inflow > 0, room / np.maximum(inflow, 1e-300), np.inf)
        dt = float(np.min(t_each))
        dt = max(dt, s.step_fill * total / Q)            # a share of the cavity per step, at least one node
        if t + dt > s.t_limit:
            dt = s.t_limit - t
        add = inflow * dt
        # pour into the front nodes; what overflows a full node goes to the others in proportion to their inflow
        for _ in range(50):
            over = np.maximum(add - room, 0.0)
            if over.sum() <= 1e-12 * total:
                break
            add = np.minimum(add, room)
            open_ = add < room - 1e-15
            if not open_.any():
                break
            w = inflow * open_
            add = add + over.sum() * w / max(w.sum(), 1e-300)
        add = np.minimum(add, room)
        newly = add >= room - 1e-12 * vol[fr]
        frac = np.where(add > 0, room / np.maximum(add, 1e-300), 1.0)
        t_fill[fr[newly]] = t + np.clip(frac[newly], 0.0, 1.0) * dt
        f[fr] = np.where(newly, 1.0, f[fr] + add / vol[fr])
        injected += float(add.sum())
        t += dt
        steps += 1
        times.append(t)
        gp.append(float(p[gate].max()))
        flows.append(float(add.sum() / dt) if dt > 0 else 0.0)
        shares.append(float((f * vol).sum()) / total)
        if steps % 25 == 0:
            say("fill", f"t = {t:.3g} s, {100 * float((f * vol).sum()) / total:.0f}% full, gate {gp[-1]:.3g} MPa")
        if t >= s.t_limit - 1e-12:
            short, why = True, "the time limit ran out"
            break
        if pressure_mode and flows[-1] < 1e-3 * s.flow:
            short, why = True, "the flow at the pressure limit fell to almost nothing"
            break
    return FillResult(t_fill, float((f * vol).sum()) / total, np.array(times), np.array(gp), np.array(flows), p,
                      injected, float((f * vol).sum()), short, why, steps, share=np.array(shares))


def frozen_skin_ratio(resin, melt: float, mold: float) -> float:
    """erfinv((T_noflow - T_mold) / (T_melt - T_mold)): the frozen skin from one wall reaches 2 r sqrt(alpha t), the
    depth where a semi infinite melt touching a mold at T_mold has cooled to its no flow temperature."""
    x = (resin.no_flow - mold) / max(melt - mold, 1e-9)
    return float(erfinv(np.clip(x, 1e-6, 1 - 1e-9)))


def edge_graph(mesh: Mesh, weight=None) -> sp.csr_matrix:
    e = np.concatenate([mesh.tets[:, [i, j]] for i, j in ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))])
    e = np.unique(np.sort(e, axis=1), axis=0)
    L = np.linalg.norm(mesh.X[e[:, 0]] - mesh.X[e[:, 1]], axis=1)
    w = L if weight is None else L * weight(e)
    n = len(mesh.X)
    return sp.csr_matrix((np.r_[w, w], (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])), shape=(n, n))


def node_thickness(mesh: Mesh) -> np.ndarray:
    bt = np.zeros(len(mesh.X))
    cnt = np.zeros(len(mesh.X))
    for c in range(4):
        np.add.at(bt, mesh.tets[:, c], mesh.thickness)
        np.add.at(cnt, mesh.tets[:, c], 1.0)
    return bt / np.maximum(cnt, 1)


def infer_gate(mesh: Mesh, surface_nodes: np.ndarray, candidates: int = 120) -> int:
    """The surface node from which the melt has the shortest way to the farthest corner, counting a thin wall as
    longer than a thick one in proportion to 1 / thickness^2, the way flow resistance grows. That balances the
    flow paths and puts the gate in a thick section, the usual first choice for a single gate."""
    bt = node_thickness(mesh)
    ref = float(np.median(bt))
    g = edge_graph(mesh, lambda e: (ref / np.maximum(0.5 * (bt[e[:, 0]] + bt[e[:, 1]]), 1e-6)) ** 2)
    rng = np.random.default_rng(1)
    cand = surface_nodes if len(surface_nodes) <= candidates else np.sort(rng.choice(surface_nodes, candidates,
                                                                                     replace=False))
    d = dijkstra(g, indices=cand)
    ecc = d.max(axis=1)
    best = cand[int(np.argmin(ecc))]
    # polish: the neighbors of the best candidate, which the sample may have stepped over
    near = surface_nodes[np.linalg.norm(mesh.X[surface_nodes] - mesh.X[best], axis=1) <
                         3.0 * float(np.median(mesh.thickness)) + 1e-9]
    d2 = dijkstra(g, indices=near)
    return int(near[int(np.argmin(d2.max(axis=1)))])


def weld_lines(mesh: Mesh, t_fill: np.ndarray, angle: float = 135.0) -> np.ndarray:
    """Nodes where two flow fronts meet: the fill time rises toward them from opposite sides, so the fill time
    gradients of the elements around them point against each other by more than the angle."""
    B, _ = geometry(mesh)
    ok = np.isfinite(t_fill[mesh.tets]).all(axis=1)
    g = np.einsum("eik,ei->ek", B, np.nan_to_num(t_fill)[mesh.tets])
    gn = np.linalg.norm(g, axis=1)
    use = np.nonzero(ok & (gn > 1e-12))[0]
    u = g[use] / gn[use, None]
    n = len(mesh.X)
    inc = sp.csr_matrix((np.ones(4 * len(use)), (mesh.tets[use].ravel(), np.repeat(np.arange(len(use)), 4))),
                        shape=(n, len(use)))
    cos_a = math.cos(math.radians(angle))
    weld = []
    for i in range(n):
        els = inc.indices[inc.indptr[i]:inc.indptr[i + 1]]
        if len(els) < 2:
            continue
        d = u[els] @ u[els].T
        if d.min() < cos_a:
            weld.append(i)
    weld = np.array(weld, dtype=int)
    # right at the gate the fill time is nearly flat and the gradients around its corners point every way, which
    # is a melt that has not formed a front yet and not two fronts meeting
    if len(weld):
        weld = weld[t_fill[weld] > 0.02 * np.nanmax(np.where(np.isfinite(t_fill), t_fill, np.nan))]
    return _connected(mesh, weld)


def _connected(mesh: Mesh, weld: np.ndarray, reach: float = 2.5) -> np.ndarray:
    """Keep the weld nodes that have another within a couple of mesh edges. Two fronts meeting leave a line of
    nodes, while one node alone is the noise of a poor element where the fill time is nearly flat, which differs
    from one mesher to the next."""
    if len(weld) < 2:
        return weld
    from scipy.spatial import cKDTree

    t = mesh.tets
    edge = np.median(np.linalg.norm(mesh.X[t[:, 0]] - mesh.X[t[:, 1]], axis=1))
    near = cKDTree(mesh.X[weld]).query_ball_point(mesh.X[weld], reach * edge)
    # a copy of the node on the other face of the plate is not a neighbor along the line
    have = np.array([any(np.linalg.norm(mesh.X[weld[j]][:2] - mesh.X[w][:2]) > 1e-9 for j in nb)
                     for w, nb in zip(weld, near)])
    return weld[have]
