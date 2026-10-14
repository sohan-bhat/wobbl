"""Reactions: balancing, predicting products from words, and the salts and ions that take part.

A written reaction is balanced on its elements and charge by the null space of its composition matrix, so
'H2 + O2 -> H2O' becomes 2 H2 + O2 -> 2 H2O and coefficients that do not balance are corrected and said so. When
the user names substances instead ("sodium in water", "baking soda and vinegar"), the products come from the
textbook reaction types: combustion, an active metal in water or acid, an oxide in water, neutralization, an acid
on a carbonate, a more active metal taking another's oxide or salt, precipitation and dissolving. Salts in water
dissolve into their ions when they are soluble by the usual solubility rules.
"""
from __future__ import annotations

from fractions import Fraction
from math import gcd

from . import formula as F
from .resolve import ResolveError, Species, Term
from .substances import ACIDS, ANIONS, CATIONS, STRONG_ACIDS, Ion

# most to least active, the order that decides displacement
ACTIVITY = ["K", "Na", "Li", "Ba", "Sr", "Ca", "Mg", "Al", "Mn", "Zn", "Cr", "Fe", "Ni", "Sn", "Pb", "H", "Cu", "Ag",
            "Hg", "Pt", "Au"]
WATER_METALS = {"Li", "Na", "K", "Rb", "Cs", "Ca", "Sr", "Ba"}
# the oxide each element forms when it burns or rusts
OXIDES = {"H": "H2O", "C": "CO2", "S": "SO2", "P": "P4O10", "Li": "Li2O", "Na": "Na2O", "K": "K2O", "Mg": "MgO",
          "Ca": "CaO", "Sr": "SrO", "Ba": "BaO", "Al": "Al2O3", "Fe": "Fe2O3", "Cu": "CuO", "Zn": "ZnO",
          "Ti": "TiO2", "Si": "SiO2", "Sn": "SnO2", "Pb": "PbO", "Ni": "NiO", "Cr": "Cr2O3", "Mn": "MnO2",
          "B": "B2O3", "Co": "CoO", "Ag": "Ag2O", "N": "N2"}
OTHER_CHARGES = {"Fe": (3, 2), "Cu": (2, 1), "Sn": (2, 4), "Pb": (2, 4), "Mn": (2, 3, 4), "Cr": (3, 2), "Co": (2, 3),
                 "Ti": (4,), "Ni": (2,), "Hg": (2, 1), "Au": (3, 1)}


# ------------------------------------------------------------------------------------------- balancing
def _composition(sp: Species) -> dict:
    d = {k: Fraction(v).limit_denominator(1000) for k, v in sp.counts.items()}
    if sp.charge:
        d["(charge)"] = Fraction(sp.charge)
    return d


def nullspace(rows: list[list[Fraction]]) -> list[list[Fraction]]:
    """Basis of the null space of a rational matrix, by Gauss-Jordan elimination."""
    m = [r[:] for r in rows]
    nrow, ncol = len(m), len(m[0]) if m else 0
    pivots, r = [], 0
    for c in range(ncol):
        p = next((i for i in range(r, nrow) if m[i][c] != 0), None)
        if p is None:
            continue
        m[r], m[p] = m[p], m[r]
        piv = m[r][c]
        m[r] = [v / piv for v in m[r]]
        for i in range(nrow):
            if i != r and m[i][c] != 0:
                f = m[i][c]
                m[i] = [a - f * b for a, b in zip(m[i], m[r])]
        pivots.append(c)
        r += 1
        if r == nrow:
            break
    free = [c for c in range(ncol) if c not in pivots]
    basis = []
    for fc in free:
        v = [Fraction(0)] * ncol
        v[fc] = Fraction(1)
        for i, pc in enumerate(pivots):
            v[pc] = -m[i][fc]
        basis.append(v)
    return basis


def balance(reactants: list[Term], products: list[Term]) -> tuple[list[Term], list[Term], bool]:
    """The smallest whole number coefficients that balance elements and charge. Returns the terms and whether
    the coefficients changed. Raises ResolveError when no balance or more than one exists."""
    terms = reactants + products
    comps = [_composition(t.species) for t in terms]
    keys = sorted({k for c in comps for k in c})
    sign = [1] * len(reactants) + [-1] * len(products)
    rows = [[sign[j] * comps[j].get(k, Fraction(0)) for j in range(len(terms))] for k in keys]
    given = [Fraction(t.coef).limit_denominator(1000) for t in terms]
    if all(sum(r[j] * given[j] for j in range(len(terms))) == 0 for r in rows):
        return reactants, products, False
    basis = nullspace(rows)
    names = " + ".join(t.species.formula for t in reactants) + " -> " + " + ".join(t.species.formula
                                                                                   for t in products)
    if not basis:
        raise ResolveError(f"{names} cannot be balanced, the elements on the two sides differ",
                           "Check every product's atoms appear among the reactants.")
    if len(basis) > 1:
        raise ResolveError(f"{names} balances more than one way, so the amounts are not fixed",
                           "Give the coefficients, or split it into separate reactions.")
    v = basis[0]
    if any(x <= 0 for x in v) and any(x >= 0 for x in v):
        if all(x <= 0 for x in v):
            v = [-x for x in v]
        else:
            raise ResolveError(f"{names} only balances with a substance moved to the other side",
                               "Check which side each substance is on.")
    lcm = 1
    for x in v:
        lcm = lcm * x.denominator // gcd(lcm, x.denominator)
    ints = [int(x * lcm) for x in v]
    g = 0
    for x in ints:
        g = gcd(g, x)
    ints = [x // g for x in ints]
    out = [Term(float(n), t.species, t.phase) for n, t in zip(ints, terms)]
    return out[:len(reactants)], out[len(reactants):], True


def equation(reactants: list[Term], products: list[Term], phases: bool = True) -> str:
    def side(ts):
        parts = []
        for t in ts:
            c = t.coef
            cs = "" if c == 1 else (str(int(c)) if float(c).is_integer() else f"{c:g}")
            ph = f"({t.phase})" if phases and t.phase else ""
            parts.append(f"{cs} {display(t.species)}{ph}".strip())
        return " + ".join(parts)
    return f"{side(reactants)} -> {side(products)}"


def display(sp: Species) -> str:
    """How the formula reads in an equation: the conventional order for salts and acids, Hill order otherwise."""
    from .substances import CONVENTIONAL
    if sp.formula in CONVENTIONAL:
        return CONVENTIONAL[sp.formula]
    ions = sp.ions or (ions_of(sp) if sp.kind in ("salt", "oxide") else None)
    if ions:
        return salt_formula(*ions)
    if sp.kind == "ion":
        body = sp.formula.split("^")[0].rstrip("+-")
        return body + ("" if abs(sp.charge) == 1 else f"^{abs(sp.charge)}") + ("+" if sp.charge > 0 else "-")
    return sp.formula


# ------------------------------------------------------------------------------------------- salts and ions
def ion_species(ion: Ion, phase: str = "aq") -> Species:
    counts, _ = F.parse(ion.formula)
    return Species(ion.formula, ion.name, F.hill(counts, ion.charge), dict(counts), ion.charge, ion.smiles, "ion",
                   phase, "Wobbl's table of ions")


def salt_formula(cat: tuple, an: tuple) -> str:
    (nc, cf, _), (na, af, _) = cat, an

    def part(f, n):
        poly = len(F.parse(f)[0]) > 1 or sum(F.parse(f)[0].values()) > 1
        return (f"({f}){n}" if poly else f"{f}{n}") if n > 1 else f
    if af == "CH3COO":
        return part(cf, nc) + part(af, na) if nc > 1 else f"CH3COO{cf}" if na == 1 else part(cf, nc) + part(af, na)
    return part(cf, nc) + part(af, na)


def salt(cat: Ion, an: Ion, phase: str = "") -> Species:
    g = gcd(cat.charge, -an.charge)
    nc, na = -an.charge // g, cat.charge // g
    text = salt_formula((nc, cat.formula, cat.charge), (na, an.formula, an.charge))
    counts, _ = F.parse(text)
    name = f"{cat.name} {an.name}"
    from .substances import lookup
    known = lookup(name) or lookup(name.replace("(II)", "").replace("(ii)", ""))
    sp = Species(text, known.name if known else name, F.hill(counts), dict(counts), 0,
                 known.smiles if known else "", "oxide" if an.formula == "O" else "salt", phase,
                 "built from its ions")
    sp.ions = ((nc, cat.formula, cat.charge), (na, an.formula, an.charge))
    if known:
        sp.crystal, sp.lattice = known.crystal, known.lattice
    return sp


def ions_of(sp: Species) -> tuple | None:
    """The cation and anion a salt or oxide is made of, as ((n, formula, charge), (n, formula, charge))."""
    if sp.ions:
        return sp.ions
    counts = {k: Fraction(v) for k, v in sp.counts.items()}
    cations = [(c, z) for c in CATIONS for z in ((CATIONS[c].charge,) + OTHER_CHARGES.get(c, ()))]
    cations += [(c, z) for c, zs in OTHER_CHARGES.items() if c not in CATIONS for z in zs]
    for cf, cz in cations:
        cc, _ = F.parse(cf)
        for an in ANIONS.values():
            ac, _ = F.parse(an.formula)
            g = gcd(cz, -an.charge)
            nc, na = -an.charge // g, cz // g
            for k in (1, 2, 3, 4):
                want: dict = {}
                for e, n in cc.items():
                    want[e] = want.get(e, 0) + n * nc * k
                for e, n in ac.items():
                    want[e] = want.get(e, 0) + n * na * k
                if want == counts:
                    return ((nc * k, cf, cz), (na * k, an.formula, an.charge))
    return None


def soluble(cat: str, an: str) -> bool:
    """The usual solubility rules for a salt in water at room temperature."""
    if cat in ("Li", "Na", "K", "Rb", "Cs", "NH4"):
        return True
    if an in ("NO3", "CH3COO", "ClO", "HCO3", "HSO4"):
        return True
    if an in ("Cl", "Br", "I"):
        return cat not in ("Ag", "Pb", "Hg")
    if an == "F":
        return cat not in ("Ca", "Mg", "Sr", "Ba", "Pb")
    if an == "SO4":
        return cat not in ("Ba", "Sr", "Pb", "Ca", "Ag")
    if an == "OH":
        return cat in ("Ba", "Sr")
    return False


def _cation(sym: str) -> Ion:
    if sym in CATIONS:
        return CATIONS[sym]
    z = OTHER_CHARGES.get(sym, (2,))[0]
    return Ion(sym, z, F.element(sym)["name"].lower(), f"[{sym}+{z}]")


def _sp(formula: str, phase: str = "") -> Species:
    from .resolve import species
    sp = species(formula, allow_online=False) if formula not in ("H2O",) else species("water", allow_online=False)
    if phase:
        sp.phase = phase
    return sp


# ------------------------------------------------------------------------------------------- prediction
def _acid(sp: Species):
    from .resolve import Species as _S  # noqa: F401
    key = F.hill(sp.counts)
    if key in ACIDS:
        an, n = ACIDS[key]
        return ANIONS[an], n, key in STRONG_ACIDS
    return None


def _is_organic(sp: Species) -> bool:
    return "C" in sp.counts and "H" in sp.counts and sp.kind == "molecule"


def _combustion(fuel: Species) -> list[str] | None:
    elems = set(fuel.counts)
    if fuel.kind in ("metal", "element") and len(elems) == 1:
        sym = next(iter(elems))
        ox = OXIDES.get(sym)
        return [ox] if ox and ox != fuel.formula else None
    if elems <= {"C", "H", "O", "N", "S"} and elems & {"C", "H", "S"}:
        out = []
        if "C" in elems:
            out.append("CO2")
        if "H" in elems:
            out.append("H2O")
        if "N" in elems:
            out.append("N2")
        if "S" in elems:
            out.append("SO2")
        return out
    return None


def predict(sps: list[Species], burning: bool = False, in_water: bool = False):
    """(reactants, products, rule) for substances named together, or None when no rule fits."""
    by = {s.formula: s for s in sps}
    o2 = by.get("O2")
    water = by.get("H2O")
    others = [s for s in sps if s.formula not in ("O2", "H2O")]

    def done(rs: list[Species], ps: list[tuple[str, str]], rule: str):
        wet = aqueous or any(w in rule for w in ("acid", "neutraliz", "displacing", "precipitation", "water"))
        r_terms = []
        for s in rs:
            ph = s.phase or _default_phase(s, aqueous)
            if wet and (_acid(s) or (s.kind == "salt" and (ions_of(s) and soluble(ions_of(s)[0][1],
                                                                                   ions_of(s)[1][1])))):
                ph = "aq" if not (s.kind == "salt" and "carbonate" in rule and ph == "s") else ph
            r_terms.append(Term(1.0, s, ph))
        p_terms = []
        for f, ph in ps:
            p = f if isinstance(f, Species) else _sp(f)
            p_terms.append(Term(1.0, p, ph or _default_phase(p, aqueous)))
        r_terms, p_terms, _ = balance(r_terms, p_terms)
        return r_terms, p_terms, rule

    aqueous = bool(water) or any(s.phase == "aq" or s.solution[0] for s in sps)
    # burning: anything with oxygen
    if (burning or o2) and len(others) == 1 and o2:
        prods = _combustion(others[0])
        if prods:
            word = "rusting" if others[0].formula == "Fe" and not burning else "combustion"
            return done([others[0], o2], [(p, "g" if p in ("CO2", "H2O", "N2", "SO2") else "s") for p in prods],
                        f"{word} of {others[0].name}")
    if len(others) == 1 and water:
        s = others[0]
        sym = next(iter(s.counts)) if len(s.counts) == 1 else ""
        if s.kind in ("metal",) and sym in WATER_METALS:
            cat = _cation(sym)
            base = salt(cat, ANIONS["OH"], "aq" if soluble(sym, "OH") else "s")
            return done([s, water], [(base, ""), ("H2", "g")], "an active metal in water")
        if s.kind == "oxide":
            ions = ions_of(s)
            if ions and ions[0][1] in WATER_METALS | {"Mg"}:
                base = salt(_cation(ions[0][1]), ANIONS["OH"], "aq" if soluble(ions[0][1], "OH") else "s")
                return done([s, water], [(base, "")], "a metal oxide in water")
        if s.formula in ("CO2", "SO2", "O3S", "O2S"):
            acid = {"CO2": "H2CO3", "O2S": "H2SO3", "O3S": "H2SO4"}.get(s.formula)
            if acid:
                return done([s, water], [(acid, "aq")], "a nonmetal oxide in water")
        if s.kind in ("salt",):
            ions = ions_of(s)
            if ions and soluble(ions[0][1], ions[1][1]):
                c = ion_species(_ion(ions[0]))
                a = ion_species(_ion(ions[1]))
                r, p, _ = balance([Term(1.0, s, "s")], [Term(1.0, c, "aq"), Term(1.0, a, "aq")])
                return r, p, f"{s.name} dissolving in water"
            if ions:
                raise ResolveError(f"{s.name} barely dissolves in water, so nothing much happens",
                                   "Ask about it on its own, or with an acid.")
        if s.kind == "metal":
            raise ResolveError(f"{s.name} does not react with water at room temperature",
                               "Try an acid, oxygen or a salt of a less active metal.")
        if s.kind == "molecule":
            raise ResolveError(f"{s.name} and water do not react, it just mixes or dissolves",
                               f"Ask about '{s.name}' with --solvent water for its solvation.")
    if len(others) == 2:
        a, b = others
        for x, y in ((a, b), (b, a)):
            acid = _acid(x)
            if not acid:
                continue
            an, nh, strong = acid
            yi = ions_of(y) if y.kind in ("salt", "oxide") else None
            if yi and yi[1][1] == "OH":
                cat = _cation(yi[0][1])
                return done([x, y], [(salt(cat, an, "aq" if soluble(cat.formula, an.formula) else "s"), ""),
                                     ("H2O", "l")], "neutralization of an acid by a base")
            if yi and yi[1][1] in ("CO3", "HCO3"):
                cat = _cation(yi[0][1])
                prod = salt(cat, an, "aq" if soluble(cat.formula, an.formula) else "s")
                return done([x, y], [(prod, ""), ("H2O", "l"), ("CO2", "g")], "an acid on a carbonate")
            if yi and yi[1][1] == "O":
                cat = _cation(yi[0][1])
                return done([x, y], [(salt(cat, an, "aq" if soluble(cat.formula, an.formula) else "s"), ""),
                                     ("H2O", "l")], "an acid on a metal oxide")
            if y.kind == "metal" and len(y.counts) == 1:
                sym = next(iter(y.counts))
                if sym in ACTIVITY and ACTIVITY.index(sym) < ACTIVITY.index("H"):
                    cat = _cation(sym)
                    return done([y, x], [(salt(cat, an, "aq" if soluble(sym, an.formula) else "s"), ""),
                                         ("H2", "g")], "a metal more active than hydrogen in acid")
                raise ResolveError(f"{y.name} is less active than hydrogen, so {x.name} does not dissolve it",
                                   "Copper, silver and gold need an oxidizing acid such as nitric acid.")
        for x, y in ((a, b), (b, a)):
            if x.kind == "metal" and len(x.counts) == 1 and y.kind in ("oxide", "salt"):
                sym = next(iter(x.counts))
                yi = ions_of(y)
                if not yi or sym not in ACTIVITY or yi[0][1] not in ACTIVITY:
                    continue
                if ACTIVITY.index(sym) < ACTIVITY.index(yi[0][1]):
                    other = _sp(yi[0][1])
                    an = _ion(yi[1])
                    if an.formula == "O":
                        return done([x, y], [(OXIDES.get(sym, ""), "s"), (other, "l" if burning else "s")],
                                    f"{x.name} taking the oxygen from {y.name}, a thermite type reaction")
                    cat = _cation(sym)
                    return done([x, y], [(salt(cat, an, "aq"), ""), (other, "s")],
                                f"{x.name} displacing {other.name} from its salt")
                raise ResolveError(f"{x.name} is less active than {yi[0][1]}, so it cannot take its place",
                                   "The more active metal has to be the free one.")
        ai, bi = ions_of(a) if a.kind == "salt" else None, ions_of(b) if b.kind == "salt" else None
        if ai and bi:
            c1, a1, c2, a2 = ai[0][1], ai[1][1], bi[0][1], bi[1][1]
            p1 = salt(_cation(c1), _ion(bi[1]), "aq" if soluble(c1, a2) else "s")
            p2 = salt(_cation(c2), _ion(ai[1]), "aq" if soluble(c2, a1) else "s")
            if "s" in (p1.phase, p2.phase):
                for x in (a, b):
                    x.phase = x.phase if x.phase not in ("", "s") else "aq"
                return done([a, b], [(p1, ""), (p2, "")], "precipitation of an insoluble salt")
            raise ResolveError(f"{a.name} and {b.name} stay dissolved as ions, so nothing happens",
                               "Two soluble salts only react when one product does not dissolve.")
    if len(others) == 1 and _is_organic(others[0]) and o2:
        prods = _combustion(others[0])
        return done([others[0], o2], [(p, "g") for p in prods], f"combustion of {others[0].name}")
    return None


def _ion(t: tuple) -> Ion:
    _, f, z = t
    for d in (CATIONS, ANIONS):
        if f in d and d[f].charge == z:
            return d[f]
    if z > 0:
        return Ion(f, z, F.element(f)["name"].lower() if f in F.table() else f, f"[{f}+{z}]")
    return Ion(f, z, f, "")


def _default_phase(sp: Species, aqueous: bool) -> str:
    if sp.phase:
        return sp.phase
    if sp.formula == "H2O":
        return "l"
    if sp.kind == "ion":
        return "aq"
    if sp.kind in ("metal", "oxide", "network"):
        return "s"
    if sp.kind == "salt":
        ions = ions_of(sp)
        if aqueous and ions and soluble(ions[0][1], ions[1][1]):
            return "aq"
        return "s"
    return "aq" if aqueous else ""
