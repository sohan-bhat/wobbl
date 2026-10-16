"""The input a person controls in a chemistry run, and the run's result solved across it.

A run's view is a model the user drives. Drag the input and the whole result follows, each value its own solve.
Every kind of chemistry run has the input a designer would change.

A reaction in water takes the solid dropped in, in grams, against the solution the run assumed, so the gas, the
heat and the temperature change grow until the solution runs out. A reaction of solids or liquids takes its
mixture, the first reactant's share of the charge by mass, and solves the products' temperature at each mix. A
flame takes its equivalence ratio, a crystal its lattice strain through the equation of state, and a molecule its
vibration modes, each swung through its amplitude.
"""
from __future__ import annotations

import dataclasses

import numpy as np

CP_WATER = 4.184            # J/(g K)
EV_A3_GPA = 160.21766208


def _r(x, d: int = 5):
    return None if x is None else float(f"{float(x):.{d}g}")


# --------------------------------------------------------------------------------------------- reactions
def reaction(q, values: dict, T: float, P: float, points: int = 17) -> dict | None:
    """The reaction swept over the input a designer would change, or None when it has none. A reaction with
    anything in water sweeps the grams of the solid added, every other exothermic reaction its mixture."""
    aqueous = any(t.phase == "aq" for t in q.reactants + q.products)
    if aqueous and (values.get("adiabatic") or {}).get("solution_g"):
        return _added(q, values, T, points)
    if not aqueous and values.get("dH_kJ", 0) < 0 and len(q.reactants) >= 2:
        return _mixture(q, values, T, P, max(points - 3, 10))
    return None


def _added(q, values: dict, T: float, points: int) -> dict:
    """Grams of the solid dropped into the solution the run assumed, from none to twice what the solution can
    take. Past that the solution is used up, so the gas and the heat stop growing and the rest sits undissolved."""
    key = next((i for i, t in enumerate(q.reactants) if t.phase == "s"), len(q.reactants) - 1)
    k = q.reactants[key]
    M = k.species.molar_mass
    g0 = k.coef * M                                    # what one mole of reaction takes
    sol = float(values["adiabatic"]["solution_g"])
    dH = float(values["dH_kJ"])
    gas_L = float(values.get("gas_L") or 0.0)
    out = []
    for g in np.linspace(0, 2 * g0, points):
        xi = min(g / g0, 1.0)                          # moles of reaction, the solution limiting past one
        out.append({"x": _r(g, 5), "extent_mol": _r(xi, 5), "heat_kJ": _r(-xi * dH, 5), "gas_L": _r(xi * gas_L, 5),
                    "rise_K": _r(-xi * dH * 1000 / (sol * CP_WATER), 5), "T_K": _r(T - xi * dH * 1000 /
                                                                                  (sol * CP_WATER), 6),
                    "left_g": _r(max(g - xi * g0, 0.0), 5)})
    return {"input": {"what": "added", "name": f"{_called(k)} added", "formula": k.species.formula,
                      "unit": "g", "design": _r(g0, 5), "solution_g": _r(sol, 5)},
            "limit": {"T_K": _r(373.15, 5), "label": "water boils"}, "points": out}


def _mixture(q, values: dict, T: float, P: float, points: int) -> dict:
    """The first reactant's share of the charge by mass, from half to twice what the reaction takes of it, with
    the products' temperature solved at each mix. Whatever is left over takes its share of the heat. Every mix is
    solved the way the run solved its own, the full equilibrium when it converged there and the complete reaction
    with each product in its stable phase when it did not, so the sweep is one model throughout."""
    from .react import _adiabatic, _complete, _data
    fuel = q.reactants[0]
    rest = sum(t.coef * t.species.molar_mass for t in q.reactants[1:])
    ratios = np.unique(np.r_[np.linspace(0.5, 2.0, points - 1), 1.0])
    dH = float(values["dH_kJ"])
    full = ((values.get("adiabatic") or {}).get("kind") or "equilibrium") == "equilibrium"
    out = []
    for r in ratios:
        xi = min(r, 1.0)                               # moles of reaction: the fuel limits when lean, the rest when rich
        reac = [dataclasses.replace(t, coef=t.coef * (r if i == 0 else 1.0)) for i, t in enumerate(q.reactants)]
        prod = [dataclasses.replace(t, coef=t.coef * xi) for t in q.products]
        left = [dataclasses.replace(t, coef=t.coef * (r - xi if i == 0 else 1.0 - xi))
                for i, t in enumerate(q.reactants) if (r - xi if i == 0 else 1.0 - xi) > 1e-9]
        mix = dataclasses.replace(q, reactants=reac, products=prod + left)
        if full:
            ad = _adiabatic(mix, T, P) or {}
        else:
            ad = _complete(mix, T, sum(t.coef * _data(t, T).H(T) for t in reac), "") or {}
        m_fuel = fuel.coef * r * fuel.species.molar_mass
        out.append({"x": _r(100 * m_fuel / (m_fuel + rest), 5), "ratio": _r(r, 4), "extent_mol": _r(xi, 5),
                    "heat_kJ": _r(-xi * dH, 5), "heat_kJ_per_kg": _r(-xi * dH / ((m_fuel + rest) / 1000), 5),
                    "T_K": _r(ad.get("T_K"), 6), "charge_g": _r(m_fuel + rest, 5),
                    "left": {t.species.formula: _r(t.coef * t.species.molar_mass, 4) for t in left},
                    "made": {t.species.formula: _r(t.coef * t.species.molar_mass, 4) for t in prod}})
    m0 = fuel.coef * fuel.species.molar_mass
    return {"input": {"what": "mixture", "name": f"{_called(fuel)} share of the charge",
                      "formula": fuel.species.formula, "unit": "% by mass", "design": _r(100 * m0 / (m0 + rest), 5)},
            "limit": _boils(q.products, T), "points": out}


def _called(t) -> str:
    """A reactant by the name the person used for it, when that is a word and not a bare formula."""
    txt = (t.species.text or "").strip()
    return txt if txt and txt != t.species.formula and not txt[:1].isupper() else t.species.name


def _boils(products, T: float) -> dict | None:
    """The lowest boiling point among the products. Above it part of the charge boils off as it burns."""
    best = None
    for t in products:
        pc = phase_change(t.species, "boils", T)
        if pc and (best is None or pc[0] < best[0]):
            best = (pc[0], t.species.formula)
    return {"T_K": _r(best[0], 5), "label": f"{best[1]} boils"} if best else None


def phase_change(species, kind: str = "boils", T: float = 298.15) -> tuple[float, float] | None:
    """Where a species melts or boils above T at 1 atm, and the heat it takes there in J/mol, from where its next
    phase falls below the one before in Gibbs energy on NASA's data. None when NASA does not list both."""
    from .thermodata import NASA, Data, _counts_key, _nasa
    by = _nasa()
    if by is None:
        return None
    key = _counts_key(species.counts)
    cand = {ph: [Data(species.formula, ph, c.name, NASA, 0, 0, 0, c.thermo.min_temp, c.thermo.max_temp, c)
                 for c in by.get((key, ph), [])] for ph in ("s", "l", "g")}
    lo_ph, hi_ph = (("s", "l"), ("g",)) if kind == "boils" else (("s",), ("l",))

    def low(phs, x):
        c = [d for ph in phs for d in cand[ph] if d.covers(x)]
        return min(c, key=lambda d: d.G(x)) if c else None

    def above(x):
        a, b = low(lo_ph, x), low(hi_ph, x)
        return a is not None and b is not None and b.G(x) < a.G(x)
    for Tb in np.arange(max(T, 200.0), 6000.0, 5.0):
        if above(Tb):
            lo, hi = Tb - 5.0, Tb
            for _ in range(30):
                m = 0.5 * (lo + hi)
                lo, hi = (lo, m) if above(m) else (m, hi)
            a, b = low(lo_ph, lo) or low(lo_ph, hi), low(hi_ph, hi)
            return float(hi), float(b.H(hi) - a.H(hi)) if a is not None and b is not None else 0.0
    return None


# --------------------------------------------------------------------------------------------- crystals
def eos(V, E, n_atoms: int) -> dict:
    """The third order Birch-Murnaghan fit of energy against volume, which is a cubic in V^(-2/3), and from it
    a function giving each volume's pressure and bulk modulus in GPa and energy above the minimum in eV per atom."""
    V, E = np.asarray(V, float), np.asarray(E, float)
    c = np.polyfit(V ** (-2 / 3), E, 3)
    p, d1, d2 = np.poly1d(c), np.poly1d(c).deriv(1), np.poly1d(c).deriv(2)
    vv = np.linspace(V.min(), V.max(), 2001)
    v0 = float(vv[np.argmin(p(vv ** (-2 / 3)))])
    e0 = float(p(v0 ** (-2 / 3)))

    def at(v):
        v = np.asarray(v, float)
        x = v ** (-2 / 3)
        dx, ddx = -2 / 3 * v ** (-5 / 3), 10 / 9 * v ** (-8 / 3)
        dE = d1(x) * dx
        ddE = d2(x) * dx ** 2 + d1(x) * ddx
        return {"pressure_GPa": -dE * EV_A3_GPA, "bulk_GPa": v * ddE * EV_A3_GPA,
                "energy_eV_atom": (p(x) - e0) / n_atoms}
    return {"v0": v0, "e0": e0, "at": at}


def strains(span: float = 0.05, points: int = 17) -> np.ndarray:
    """Linear lattice strains to sweep a crystal through, with zero, its relaxed cell, among them."""
    return np.linspace(-span, span, points)


# --------------------------------------------------------------------------------------------- molecules
def swings(points: int = 9) -> np.ndarray:
    """Amplitudes to swing a normal mode through, as fractions of the drawn swing, from one turning point to the
    other through the rest shape."""
    return np.linspace(-1, 1, points)


def mode_inputs(n_modes: int, amps) -> np.ndarray:
    """The slider value of each frame when a molecule steps through its modes, each swung through amps: the mode's
    number plus a small offset for the swing, so the modes stay apart and in order."""
    amps = np.asarray(amps, float)
    return np.concatenate([k + 1 + 0.4 * amps for k in range(n_modes)])


def flame_limit(limit_K=None) -> tuple[float, str]:
    """The temperature a flame is held to: the one asked for, or where thermal NO starts to form fast."""
    return (float(limit_K), "your limit") if limit_K else (1800.0, "fast thermal NO")


__all__ = ["reaction", "eos", "strains", "swings", "mode_inputs", "flame_limit"]
