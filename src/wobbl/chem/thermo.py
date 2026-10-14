"""Chemical equilibrium on NASA's thermodynamic data, solved with Cantera.

The products are every gas and condensed species in the NASA Glenn database made only of the reactants' elements,
the same data NASA's CEA program uses (McBride, Gordon and Reno, NASA TM-4513, shipped with Cantera as
nasa_gas.yaml and nasa_condensed.yaml). A condensed species takes part only inside the temperature range its data
covers, as in CEA, so a liquid hands over to its solid at the melting point instead of being extrapolated past it.

Cantera minimizes the Gibbs energy at a fixed temperature and pressure. A fixed enthalpy (the flame and the chamber)
or a fixed entropy (the nozzle) is met by a safeguarded secant on temperature around that, which lets the set of
condensed species change with temperature. When the answer sits on a melting point, the state is the mix of liquid
and solid that matches the target exactly, as the latent heat requires.

Rocket performance follows CEA's infinite area combustor. The chamber is in equilibrium at the reactants'
enthalpy, and the gas expands at the chamber's entropy either staying in equilibrium (shifting) or with the
chamber composition frozen. The throat is where the mass flow per unit area peaks, c* is the chamber pressure over
that peak, and the exit is where the area ratio is reached. Condensed products move with the gas at its velocity
and temperature, and their volume is neglected. Units are SI, with energies per kg of propellant.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

from .install import require

R = 8314.46261815324          # J/(kmol K)
G0 = 9.80665                  # m/s2
P_REF = 101325.0


@dataclass
class Reactants:
    """A propellant or fuel and oxidizer mixture: moles of each element and the enthalpy, both per kg."""
    elements: dict[str, float]     # mol/kg
    h0: float                      # J/kg
    label: str = ""


@dataclass
class State:
    T: float                       # K
    P: float                       # Pa
    h: float                       # J/kg
    s: float                       # J/(kg K)
    n_gas: float                   # kmol of gas per kg
    gas: dict[str, float]          # mole fractions in the gas
    condensed: dict[str, float]    # mass fraction of each condensed product
    cp: float                      # J/(kg K), frozen, gas plus condensed
    moles: np.ndarray | None = None          # gas kmol per kg in the order of the gas phase
    cond_moles: list[tuple[int, float]] = field(default_factory=list)   # (condensed index, kmol/kg)

    @property
    def M(self) -> float:
        """Effective molar mass, kg/kmol: total mass over moles of gas, the one the gas laws use."""
        return 1.0 / self.n_gas

    @property
    def rho(self) -> float:
        """Mass of products per volume of gas, kg/m3."""
        return self.P / (self.n_gas * R * self.T)

    @property
    def gamma(self) -> float:
        """Frozen ratio of specific heats of the gas and condensed mixture."""
        return self.cp / (self.cp - self.n_gas * R)

    @property
    def condensed_fraction(self) -> float:
        return sum(self.condensed.values())

    def major(self, cut: float = 1e-4) -> dict[str, float]:
        return {k: v for k, v in sorted(self.gas.items(), key=lambda kv: -kv[1]) if v >= cut}


@lru_cache(maxsize=1)
def _database():
    ct = require()
    return ct.Species.list_from_file("nasa_gas.yaml"), ct.Species.list_from_file("nasa_condensed.yaml")


def _condensed_phase(ct, sp):
    """A pure condensed species with negligible volume, as CEA treats it (the NASA data has no density)."""
    d = dict(sp.input_data)
    d["equation-of-state"] = {"model": "constant-volume", "molar-volume": 1e-6}
    return ct.Solution(thermo="fixed-stoichiometry", species=[ct.Species.from_dict(d)])


class Equilibrium:
    """Equilibrium states of one set of reactants. Builds the product species once and reuses them."""

    def __init__(self, reactants: Reactants, condensed: bool = True):
        ct = require()
        self.ct = ct
        self.r = reactants
        gas_db, cond_db = _database()
        els = {e.upper() for e in reactants.elements}
        ok = lambda sp: {e.upper() for e in sp.composition} <= els  # noqa: E731
        gas = [sp for sp in gas_db if ok(sp) and sp.charge == 0]
        self.gas = ct.Solution(thermo="ideal-gas", species=gas)
        self.cond = []
        for sp in cond_db if condensed else []:
            if ok(sp):
                self.cond.append((_condensed_phase(ct, sp), sp.thermo.min_temp, sp.thermo.max_temp, sp.name))
        self.names = self.gas.species_names
        self.solves = 0
        # the reactant elements as single atoms in the gas: any start with the right element totals will do
        self._atoms = {}
        for e, n in reactants.elements.items():
            name = next((s for s in self.names if s.upper() == e.upper()), None)
            if name is None:
                raise ValueError(f"the NASA database has no gas phase atom for element {e}")
            self._atoms[name] = n / 1000.0       # kmol/kg
        self.atom_total = sum(self._atoms.values())
        self._edges = sorted({t for _, lo, hi, _ in self.cond for t in (lo, hi)})

    # --- fixed temperature and pressure ---------------------------------------------------------
    def _valid(self, T: float, side: int = 0) -> tuple[int, ...]:
        """Condensed species usable at T. side=-1 or +1 takes the phases just below or above an edge."""
        T2 = T + side * 0.5
        return tuple(i for i, (_, lo, hi, _) in enumerate(self.cond) if lo - 0.01 <= T2 <= hi + 0.01)

    def tp(self, T: float, P: float, side: int = 0) -> State:
        ct = self.ct
        valid = self._valid(T, side)
        self.gas.TPX = T, P, {k: v / self.atom_total for k, v in self._atoms.items()}
        for i in valid:
            self.cond[i][0].TP = T, P
        mix = ct.Mixture([(self.gas, self.atom_total)] + [(self.cond[i][0], 0.0) for i in valid])
        mix.T, mix.P = T, P
        mix.equilibrate("TP", solver="vcs", max_steps=3000)
        self.solves += 1
        cond = [(i, float(mix.phase_moles(k + 1))) for k, i in enumerate(valid)]
        return self._state(T, P, float(mix.phase_moles(0)), self.gas.X.copy(), cond)

    def _state(self, T, P, n_gas, X, cond) -> State:
        g = self.gas
        g.TPX = T, P, X
        h = n_gas * g.enthalpy_mole
        s = n_gas * g.entropy_mole
        cp = n_gas * g.cp_mole
        cmap = {}
        for i, n in cond:
            if n <= 0:
                continue
            ph = self.cond[i][0]
            ph.TP = T, P
            h += n * ph.enthalpy_mole
            s += n * ph.entropy_mole
            cp += n * ph.cp_mole
            cmap[self.cond[i][3]] = cmap.get(self.cond[i][3], 0.0) + n * ph.mean_molecular_weight
        names = self.names
        gas = {names[k]: float(X[k]) for k in range(len(names)) if X[k] > 1e-12}
        return State(T, P, h, s, n_gas, gas, cmap, cp, np.asarray(X) * n_gas,
                     [(i, n) for i, n in cond if n > 0])

    def frozen(self, comp: State, T: float, P: float) -> State:
        """The same composition at another temperature and pressure."""
        return self._state(T, P, comp.n_gas, comp.moles / comp.n_gas, comp.cond_moles)

    # --- fixed enthalpy or entropy --------------------------------------------------------------
    def hp(self, P: float, h: float | None = None, T0: float = 2500.0) -> State:
        h = self.r.h0 if h is None else h
        return self._solve(lambda T, side=0: self.tp(T, P, side), "h", h, T0, P)

    def sp(self, P: float, s: float, T0: float) -> State:
        return self._solve(lambda T, side=0: self.tp(T, P, side), "s", s, T0, P)

    def frozen_sp(self, comp: State, P: float, s: float, T0: float) -> State:
        return self._solve(lambda T, side=0: self.frozen(comp, T, P), "s", s, T0, P, frozen=True)

    def _solve(self, at, key: str, target: float, T0: float, P: float, frozen: bool = False) -> State:
        """Secant on T inside a bracket, for h(T) = target or s(T) = target at fixed P. Both rise with T, with a
        step where a condensed product melts, so a bracket that closes on a melting point is settled there."""
        lo, hi = 200.0, 6000.0
        f_lo = f_hi = None
        T = min(max(T0, lo + 1), hi - 1)
        st = at(T)
        f = getattr(st, key) - target
        slope = st.cp if key == "h" else st.cp / T
        tested = set()
        for _ in range(80):
            if f > 0:
                hi, f_hi = T, f
            else:
                lo, f_lo = T, f
            if abs(f) <= 1e-8 * max(abs(target), st.cp * T) or hi - lo < 1e-5:
                return st
            if not frozen and f_lo is not None and f_hi is not None and hi - lo < 5.0:
                edge = next((e for e in self._edges if lo < e < hi and e not in tested), None)
                if edge is not None:
                    tested.add(edge)
                    a, b = at(edge, -1), at(edge, +1)
                    fa, fb = getattr(a, key) - target, getattr(b, key) - target
                    if fa <= 0 <= fb:
                        return self._blend(a, b, key, target, edge, P)
                    T, st, f = (edge, b, fb) if fb < 0 else (edge, a, fa)
                    continue
            Tn = T - f / slope
            if not lo < Tn < hi:
                if f_lo is not None and f_hi is not None:
                    Tn = 0.5 * (lo + hi)
                else:
                    Tn = max(lo + 1e-3, T - 400) if f > 0 else min(hi - 1e-3, T + 400)
            st_n = at(Tn)
            fn = getattr(st_n, key) - target
            if abs(Tn - T) > 1e-9:
                sl = (fn - f) / (Tn - T)
                slope = sl if sl > 0 else slope
            T, st, f = Tn, st_n, fn
        raise RuntimeError(f"equilibrium temperature did not converge at {P / 1e5:.4g} bar")

    def _blend(self, a: State, b: State, key: str, target: float, edge: float, P: float) -> State:
        """The target falls inside a melting step: the mix of the all solid and all liquid states that meets it."""
        fa, fb = getattr(a, key), getattr(b, key)
        w = 0.0 if abs(fb - fa) < 1e-12 else (target - fa) / (fb - fa)
        n_gas = (1 - w) * a.n_gas + w * b.n_gas
        moles = (1 - w) * a.moles + w * b.moles
        cond = {}
        for i, n in a.cond_moles:
            cond[i] = cond.get(i, 0.0) + (1 - w) * n
        for i, n in b.cond_moles:
            cond[i] = cond.get(i, 0.0) + w * n
        return self._state(edge, P, n_gas, moles / n_gas, list(cond.items()))

    # --- checks ---------------------------------------------------------------------------------
    def element_error(self, st: State) -> float:
        """Largest relative difference between the elements in a state and in the reactants."""
        g = self.gas
        got = {e: 0.0 for e in g.element_names}
        for k, name in enumerate(self.names):
            for e in g.element_names:
                got[e] += st.moles[k] * g.n_atoms(name, e)
        for i, n in st.cond_moles:
            ph = self.cond[i][0]
            for e in ph.element_names:
                got[e] = got.get(e, 0.0) + n * ph.n_atoms(0, e)
        worst = 0.0
        for e, want in self.r.elements.items():
            key = next(k for k in got if k.upper() == e.upper())
            worst = max(worst, abs(got[key] * 1000.0 - want) / want)
        return worst


@dataclass
class Station:
    name: str
    P: float
    T: float
    v: float                # m/s
    area_ratio: float       # A / A_throat
    M: float
    gamma: float
    condensed: float        # mass fraction
    gas: dict[str, float]


@dataclass
class Performance:
    mode: str               # shifting or frozen
    cstar: float            # m/s
    isp_vac: float          # s
    isp_amb: float          # s, at the ambient pressure
    isp_opt: float          # s, exit velocity over g0 (CEA's Isp)
    cf_vac: float
    exit_P: float           # Pa
    stations: list[Station]
    entropy_error: float    # largest relative entropy drift along the nozzle


MODES = ("shifting", "frozen", "frozen-throat")


def rocket(eq: Equilibrium, pc: float, area_ratio: float | None = None, exit_P: float | None = None,
           ambient: float = P_REF, mode: str = "shifting", chamber: State | None = None) -> tuple[State, Performance]:
    """CEA style performance at chamber pressure pc (Pa) and either an exit area ratio or an exit pressure.

    mode is shifting (equilibrium all the way), frozen (the chamber composition all the way) or frozen-throat
    (equilibrium to the throat, then frozen), CEA's three choices.
    """
    if mode not in MODES:
        raise ValueError(f"unknown expansion mode '{mode}' (use {', '.join(MODES)})")
    ch = chamber or eq.hp(pc)
    s_c, h_c = ch.s, ch.h
    froze = {"comp": ch if mode == "frozen" else None}

    def at(P, guess):
        comp = froze["comp"]
        st = eq.frozen_sp(comp, P, s_c, guess) if comp is not None else eq.sp(P, s_c, guess)
        v = math.sqrt(max(2.0 * (h_c - st.h), 0.0))
        return st, v, st.rho * v

    def guess_T(P):
        return ch.T * (P / pc) ** ((ch.gamma - 1) / ch.gamma)

    # throat: the peak of mass flux over pressure, by Brent's bounded search in log pressure. The flux is flat at
    # its peak, so a 0.2% step in pressure moves c* by about a millionth.
    from scipy.optimize import minimize_scalar

    g = ch.gamma
    x0 = math.log((2 / (g + 1)) ** (g / (g - 1)))
    cache = {}

    def neg_flux(x):
        if x not in cache:
            cache[x] = at(pc * math.exp(x), guess_T(pc * math.exp(x)))
        return -cache[x][2]

    minimize_scalar(neg_flux, bounds=(x0 - 0.3, min(x0 + 0.3, -0.01)), method="bounded", options={"xatol": 2e-3})
    xt = max(cache, key=lambda x: cache[x][2])
    th, v_t, G_t = cache[xt]
    cstar = pc / G_t
    if mode == "frozen-throat":
        froze["comp"] = th
    # exit: secant in log pressure on the area ratio, starting from the ideal gas estimate
    if exit_P is None:
        if area_ratio is None or area_ratio < 1:
            raise ValueError("give an exit area ratio of at least 1 or an exit pressure")
        lo_x, hi_x = math.log(1e-7), xt
        x = math.log(_ideal_exit_ratio(area_ratio, ch.gamma))
        x = min(max(x, lo_x + 1e-3), hi_x - 1e-3)
        prev = None
        for _ in range(40):
            ex, v_e, G_e = at(pc * math.exp(x), guess_T(pc * math.exp(x)))
            f = math.log(G_t / G_e) - math.log(area_ratio)
            if abs(f) < 1e-7:
                break
            if f > 0:
                lo_x = x
            else:
                hi_x = x
            if prev is not None and abs(x - prev[0]) > 1e-12:
                step = f * (x - prev[0]) / (f - prev[1])
            else:
                step = -f * max(ch.gamma, 1.01)    # d ln(A/A*) / d ln P is about -1/gamma
            prev = (x, f)
            xn = x - step
            x = xn if lo_x < xn < hi_x else 0.5 * (lo_x + hi_x)
    else:
        ex, v_e, G_e = at(exit_P, guess_T(exit_P))
    eps = G_t / G_e
    isp_opt = v_e / G0
    isp_vac = (v_e + ex.P / G_e) / G0
    isp_amb = (v_e + (ex.P - ambient) / G_e) / G0
    drift = max(abs(s.s - s_c) / abs(s_c) for s in (th, ex))
    stations = [Station("chamber", pc, ch.T, 0.0, float("inf"), ch.M, ch.gamma, ch.condensed_fraction, ch.major()),
                Station("throat", th.P, th.T, v_t, 1.0, th.M, th.gamma, th.condensed_fraction, th.major()),
                Station("exit", ex.P, ex.T, v_e, eps, ex.M, ex.gamma, ex.condensed_fraction, ex.major())]
    return ch, Performance(mode, cstar, isp_vac, isp_amb, isp_opt, isp_vac * G0 / cstar, ex.P, stations, drift)


def _ideal_exit_ratio(eps: float, g: float) -> float:
    """Exit to chamber pressure ratio of an ideal gas nozzle with area ratio eps (supersonic branch)."""
    from scipy.optimize import brentq

    def area(pr):
        t = pr ** ((g - 1) / g)
        m2 = 2 / (g - 1) * (1 / t - 1)
        return math.sqrt(((2 / (g + 1)) * (1 + (g - 1) / 2 * m2)) ** ((g + 1) / (g - 1)) / m2)

    pt = (2 / (g + 1)) ** (g / (g - 1))
    if eps <= 1.0:
        return pt
    return brentq(lambda pr: area(pr) - eps, 1e-12, pt * 0.999)


def cstar_ideal(T: float, M: float, g: float) -> float:
    """c* of an ideal gas with constant properties: sqrt(R T / M) / Gamma(g)."""
    gam = math.sqrt(g) * (2 / (g + 1)) ** ((g + 1) / (2 * (g - 1)))
    return math.sqrt(R * T / M) / gam
