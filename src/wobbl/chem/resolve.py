"""Read anything chemical and say what it is.

The text can be a name ("caffeine", "table salt", "rust"), a formula (C8H10N4O2, NaCl, SO4^2-), SMILES or InChI,
an element, a crystal (a CIF file or "fcc copper"), a mixture ("methane and air at 1 atm"), a reaction written out
("2 H2 + O2 -> 2 H2O") or in words ("sodium in water", "baking soda and vinegar"), a rocket propellant or a battery
cell. Conditions written into the text (at 500 K, at 2 atm, in water) are read out of it. Names go to the offline
table first, then to PubChem, and the result says where each name came from.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from . import formula as F
from . import pubchem
from .substances import COMMON_FORMULAS, Substance, lookup

METALS = {"Alkali metal", "Alkaline earth metal", "Transition metal", "Post-transition metal", "Lanthanide",
          "Actinide"}
DIATOMIC = {"H", "N", "O", "F", "Cl", "Br", "I"}
GASES_AT_ROOM = {"H2", "N2", "O2", "F2", "Cl2", "He", "Ne", "Ar", "Kr", "Xe", "Rn"}


class ResolveError(ValueError):
    def __init__(self, msg: str, hint: str = ""):
        super().__init__(msg)
        self.hint = hint


@dataclass
class Species:
    text: str                    # as the user wrote it
    name: str
    formula: str                 # Hill order, with the charge
    counts: dict
    charge: int = 0
    smiles: str = ""
    kind: str = "molecule"       # molecule, ion, element, salt, oxide, network, metal
    phase: str = ""              # g, l, s or aq when known
    source: str = ""             # how it was resolved
    cid: int | None = None
    iupac: str = ""
    multiplicity: int = 0        # 0 lets the structure code work it out
    crystal: str = ""            # structure prototype, for solids with one
    lattice: float = 0.0         # measured lattice constant, angstrom
    solution: tuple[str, float] = ("", 0.0)
    note: str = ""
    ions: tuple = ()             # (count, ion formula, charge) for a salt, so it can dissolve into its ions

    @property
    def molar_mass(self) -> float:
        return F.molar_mass(self.counts)

    @property
    def atoms(self) -> int:
        return int(sum(self.counts.values()))

    def label(self) -> str:
        return self.name if self.name.lower() == self.formula.lower() else f"{self.name} ({self.formula})"

    def to_dict(self) -> dict:
        return {"text": self.text, "name": self.name, "formula": self.formula, "charge": self.charge,
                "smiles": self.smiles, "kind": self.kind, "phase": self.phase, "source": self.source,
                "cid": self.cid, "molar_mass_g_mol": round(self.molar_mass, 4)}


@dataclass
class Term:
    coef: float
    species: Species
    phase: str = ""


@dataclass
class Query:
    text: str
    kind: str                    # molecule, ion, element, crystal, salt, mixture, reaction, propellant, cell
    species: list[Species] = field(default_factory=list)
    reactants: list[Term] = field(default_factory=list)
    products: list[Term] = field(default_factory=list)
    reaction_source: str = ""    # written, or the rule or named reaction that supplied it
    amounts: dict = field(default_factory=dict)   # mixture mole numbers by formula
    temperature: float | None = None     # K, as written in the text
    pressure: float | None = None        # Pa
    solvent: str = ""
    phase: str = ""
    path: Path | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def main(self) -> Species | None:
        return self.species[0] if self.species else None

    def resolved(self) -> list[str]:
        """One line per name saying what it was taken to be and from where."""
        out, low = [], self.text.lower()
        for s in self.species:
            if s.text and s.source and s.text.lower() in low:
                line = f"'{s.text}' is {s.label()}, from {s.source}"
                if line not in out:
                    out.append(line)
        return out + self.notes


# ------------------------------------------------------------------------- conditions in the text
_T = r"(-?\d+(?:\.\d+)?)\s*(?:°\s*)?(K|C|F|kelvin|celsius|fahrenheit)\b"
_P = r"(\d+(?:\.\d+)?(?:e[+-]?\d+)?)\s*(atm|bar|mbar|kpa|mpa|pa|psi|psia|torr|mmhg)\b"


def conditions(text: str) -> tuple[str, dict]:
    """Take 'at 500 K', 'at 2 atm', '25 C' and 'in water' out of the text."""
    found: dict = {}
    t = text

    def temp(m):
        v, u = float(m.group(1)), m.group(2).lower()
        found["temperature"] = v if u in ("k", "kelvin") else v + 273.15 if u in ("c", "celsius") else \
            (v - 32) * 5 / 9 + 273.15
        return " "

    def pres(m):
        v, u = float(m.group(1)), m.group(2).lower()
        scale = {"atm": 101325.0, "bar": 1e5, "mbar": 100.0, "kpa": 1e3, "mpa": 1e6, "pa": 1.0, "psi": 6894.757,
                 "psia": 6894.757, "torr": 133.322, "mmhg": 133.322}[u]
        found["pressure"] = v * scale
        return " "
    t = re.sub(r"(?:\bat\s+|,\s*|\band\s+)?" + _T, temp, t, flags=re.I)
    t = re.sub(r"(?:\bat\s+|,\s*|\band\s+)?" + _P, pres, t, flags=re.I)
    t = re.sub(r"\b(?:at\s+)?(?:room temperature|rtp)\b", lambda m: (found.setdefault("temperature", 298.15), " ")[1],
               t, flags=re.I)
    t = re.sub(r"\b(?:at\s+)?(?:stp)\b", lambda m: (found.setdefault("temperature", 273.15),
                                                    found.setdefault("pressure", 101325.0), " ")[2], t, flags=re.I)
    t = re.sub(r"\s+", " ", t).strip(" ,.;?")
    return t, found


# ------------------------------------------------------------------------- one substance
def _from_substance(s: Substance, text: str, source: str) -> Species:
    counts, charge = F.parse(s.formula)
    return Species(text, s.name, F.hill(counts, charge), dict(counts), charge, s.smiles, s.kind, s.phase, source,
                   multiplicity=s.multiplicity, crystal=s.crystal, lattice=s.lattice, solution=s.solution,
                   note=s.note)


def _element(sym: str, text: str, source: str) -> Species:
    e = F.element(sym)
    block = e.get("block") or ""
    kind = "metal" if block in METALS else "element"
    return Species(text, e["name"].lower(), sym, {sym: 1}, 0, f"[{sym}]", kind, _phase(e), source)


def _phase(e: dict) -> str:
    st = (e.get("state") or "").lower()
    return "g" if st.startswith("gas") else "l" if st.startswith("liquid") else "s" if st else ""


def _classify(counts: dict, charge: int, smiles: str = "") -> str:
    if charge:
        return "ion"
    syms = set(counts)
    table = F.table()
    metals = {s for s in syms if (table[s].get("block") or "") in METALS}
    if len(syms) == 1:
        return "metal" if metals else "element"
    if counts.get("C", 0) > 4 and "H" in syms:      # organic, metal complexes included, like chlorophyll
        return "molecule"
    if metals and metals == syms:
        return "metal"
    if smiles and "." in smiles and ("+" in smiles or "-" in smiles):
        return "oxide" if syms - metals == {"O"} else "salt"
    if metals and not ("C" in syms and "H" in syms):
        if smiles and "." not in smiles and "+" not in smiles:
            return "molecule"
        return "oxide" if syms - metals == {"O"} else "salt"
    return "molecule"


def _from_pubchem(p: dict, text: str, how: str) -> Species:
    counts, _ = F.parse(p["MolecularFormula"].replace("+", "").replace("-", ""))
    charge = int(p.get("Charge") or 0)
    smiles = p.get("SMILES") or p.get("ConnectivitySMILES") or ""
    name = p.get("Title") or p.get("IUPACName") or text
    kind = _classify(counts, charge, smiles)
    return Species(text, name.lower() if name.isupper() else name, F.hill(counts, charge), dict(counts), charge,
                   smiles, kind, "", f"PubChem CID {p['CID']} ({how})", cid=p.get("CID"),
                   iupac=p.get("IUPACName") or "")


def looks_like_smiles(t: str) -> bool:
    if not re.fullmatch(r"[A-Za-z0-9@+\-\[\]()=#$/\\%.:*]+", t):
        return False
    if F.is_formula(t) and not re.search(r"([A-Z][a-z]?)\1|[a-z]{2}|[=#()\[\]@/\\]", t):
        return False
    if re.fullmatch(r"[a-z]+", t) and not re.fullmatch(r"[cnosp]+\d?", t):
        return False
    letters = re.sub(r"[^A-Za-z]", "", re.sub(r"\[[^\]]*\]", "", t))
    organic = re.sub(r"Cl|Br|[BCNOPSFIcnosp]", "", letters)
    return organic == ""


def species(text: str, allow_online: bool = True) -> Species:
    """One substance from a name, formula, SMILES, InChI or element."""
    t = text.strip()
    if not t:
        raise ResolveError("nothing to resolve")
    low = t.lower()
    if t.startswith("InChI="):
        smi = _inchi_to_smiles(t)
        if smi:
            sp = _from_smiles(smi, allow_online)
            sp.text, sp.source = t, sp.source.replace("the SMILES itself", "the InChI itself")
            return sp
        p = _try(lambda: pubchem.by_inchi(t)) if allow_online else None
        if p:
            return _from_pubchem(p, t, "by InChI")
        raise ResolveError(f"could not read the InChI '{t}'", "Check it starts with InChI=1S/.")
    if re.fullmatch(r"[A-Z][a-z]?", t) and t in F.table():
        return _element(t, t, "the periodic table")
    if low in F.by_name():
        s = lookup(low)
        if s and s.formula not in (F.by_name()[low],):    # 'oxygen' means O2, 'iron' means Fe
            return _from_substance(s, t, "Wobbl's table of common substances")
        return _element(F.by_name()[low], t, "the periodic table")
    s = lookup(low)
    if s:
        return _from_substance(s, t, "Wobbl's table of common substances")
    if looks_like_smiles(t):
        return _from_smiles(t, allow_online)
    if F.is_formula(t) and re.search(r"[A-Z]", t):
        return _from_formula(t, allow_online)
    if allow_online:
        p = _try(lambda: pubchem.by_name(t))
        if p:
            return _from_pubchem(p, t, "by name")
    raise ResolveError(f"could not tell what '{t}' is",
                       "Give a formula like C8H10N4O2, a SMILES like CCO or a common name. Names beyond Wobbl's "
                       "table need PubChem, which needs the internet." if allow_online and pubchem.online() else
                       "Give a formula or a SMILES. PubChem is off, so only common names are known.")


_OFFLINE_SEEN: list[str] = []


def _try(fn):
    try:
        return fn()
    except pubchem.Offline as e:
        _OFFLINE_SEEN.append(str(e))
        return None


def _inchi_to_smiles(inchi: str) -> str:
    try:
        from .install import load_pkg
        rd = load_pkg("rdkit")
        if rd is None:
            return ""
        from rdkit import Chem
        m = Chem.MolFromInchi(inchi)
        return Chem.MolToSmiles(m) if m is not None else ""
    except Exception:  # noqa: BLE001
        return ""


def _smiles_counts(smi: str) -> tuple[dict, int] | None:
    """Element counts and charge of a SMILES, with RDKit when it is installed, else None."""
    try:
        from .install import load_pkg
        if load_pkg("rdkit") is None:
            return None
        from rdkit import Chem
        m = Chem.MolFromSmiles(smi)
        if m is None:
            return None
        m = Chem.AddHs(m)
        counts: dict = {}
        for a in m.GetAtoms():
            counts[a.GetSymbol()] = counts.get(a.GetSymbol(), 0) + 1
        return counts, Chem.GetFormalCharge(m)
    except Exception:  # noqa: BLE001
        return None


def _from_smiles(smi: str, allow_online: bool) -> Species:
    hit = _smiles_counts(smi)
    p = _try(lambda: pubchem.by_smiles(smi)) if allow_online else None
    if p:
        sp = _from_pubchem(p, smi, "by SMILES")
        sp.smiles = smi
        return sp
    if hit is None:
        from .substances import ANIONS, CATIONS, by_smiles
        known = by_smiles(smi)
        if known:
            return _from_substance(known, smi, "Wobbl's table of common substances, by its SMILES")
        for ion in (*CATIONS.values(), *ANIONS.values()):
            if ion.smiles == smi:
                c, _ = F.parse(ion.formula)
                return Species(smi, ion.name, F.hill(c, ion.charge), dict(c), ion.charge, smi, "ion", "aq",
                               "Wobbl's table of ions, by its SMILES")
        raise ResolveError(f"could not read the SMILES '{smi}'",
                           "RDKit reads SMILES and installs on first use, or PubChem reads it online.")
    counts, charge = hit
    formula = F.hill(counts, charge)
    if charge:
        from .substances import ANIONS, CATIONS
        for ion in (*CATIONS.values(), *ANIONS.values()):
            if ion.charge == charge and dict(F.parse(ion.formula)[0]) == dict(counts):
                return Species(smi, ion.name, formula, counts, charge, smi, "ion", "aq", "Wobbl's table of ions")
    known = COMMON_FORMULAS.get(formula)
    name = known.name if known and known.smiles == smi else formula
    return Species(smi, name, formula, counts, charge, smi, _classify(counts, charge, smi), "", "the SMILES itself")


def _from_formula(text: str, allow_online: bool) -> Species:
    from .substances import CONDENSED, by_smiles
    if text.strip() in CONDENSED:
        name, smi = CONDENSED[text.strip()]
        known = by_smiles(smi) or lookup(name)
        if known and known.smiles == smi:
            return _from_substance(known, text, "Wobbl's table of common substances")
        counts, charge = F.parse(text)
        return Species(text, name, F.hill(counts, charge), dict(counts), charge, smi,
                       _classify(dict(counts), charge, smi), "", "its condensed formula")
    counts, charge = F.parse(text)
    hill = F.hill(counts, charge)
    known = COMMON_FORMULAS.get(hill)
    if known:
        sp = _from_substance(known, text, "Wobbl's table of common substances")
        return sp
    if len(counts) == 1 and not charge:
        sym, n = next(iter(counts.items()))
        if n == 1:
            return _element(sym, text, "the periodic table")
        if sym in DIATOMIC and n == 2:
            pass
    kind = _classify(counts, charge)
    if charge:
        from .substances import ANIONS, CATIONS
        for ion in (*CATIONS.values(), *ANIONS.values()):
            c2, _ = F.parse(ion.formula)
            if dict(c2) == dict(counts) and ion.charge == charge:
                return Species(text, ion.name, hill, dict(counts), charge, ion.smiles, "ion", "aq",
                               "Wobbl's table of ions")
        return Species(text, hill, hill, dict(counts), charge, "", "ion", "aq", "the formula itself")
    if kind in ("salt", "oxide", "metal"):
        return Species(text, hill, hill, dict(counts), 0, "", kind, "s", "the formula itself")
    if allow_online:
        hits = _try(lambda: pubchem.by_formula(hill.replace("^", "")))
        if hits:
            sp = _from_pubchem(hits[0], text, "by formula")
            if len(hits) > 1:
                sp.note = (f"{hill} has {len(hits)} or more isomers in PubChem, and Wobbl took the most cited, "
                           f"{sp.name}. Give a name or SMILES for another")
            return sp
    return Species(text, hill, hill, dict(counts), 0, "", kind, "", "the formula itself, with no structure known")


# ------------------------------------------------------------------------- whole queries
NAMED = {
    "thermite": "2 Al + Fe2O3 -> Al2O3 + 2 Fe",
    "photosynthesis": "6 CO2 + 6 H2O -> C6H12O6 + 6 O2",
    "respiration": "C6H12O6 + 6 O2 -> 6 CO2 + 6 H2O",
    "cellular respiration": "C6H12O6 + 6 O2 -> 6 CO2 + 6 H2O",
    "haber process": "N2 + 3 H2 -> 2 NH3",
    "haber bosch": "N2 + 3 H2 -> 2 NH3",
    "ammonia synthesis": "N2 + 3 H2 -> 2 NH3",
    "water gas shift": "CO + H2O -> CO2 + H2",
    "steam reforming": "CH4 + H2O -> CO + 3 H2",
    "rusting": "4 Fe + 3 O2 -> 2 Fe2O3",
    "rusting iron": "4 Fe + 3 O2 -> 2 Fe2O3",
    "iron rusting": "4 Fe + 3 O2 -> 2 Fe2O3",
    "electrolysis of water": "2 H2O -> 2 H2 + O2",
    "water electrolysis": "2 H2O -> 2 H2 + O2",
    "limestone calcination": "CaCO3 -> CaO + CO2",
    "calcination": "CaCO3 -> CaO + CO2",
    "lime slaking": "CaO + H2O -> Ca(OH)2",
    "slaking lime": "CaO + H2O -> Ca(OH)2",
    "hydrogen peroxide decomposition": "2 H2O2 -> 2 H2O + O2",
    "elephant toothpaste": "2 H2O2 -> 2 H2O + O2",
    "ozone decomposition": "2 O3 -> 3 O2",
    "contact process": "2 SO2 + O2 -> 2 SO3",
    "fermentation": "C6H12O6 -> 2 C2H5OH + 2 CO2",
    "hydrogen fuel cell": "2 H2 + O2 -> 2 H2O",
    "fuel cell": "2 H2 + O2 -> 2 H2O",
}
# a bare = needs spaces around it, so a SMILES keeps its double bonds
ARROW = re.compile(r"\s*(?:<=>|<->|->|→|=>|⇌|⟶)\s*|\s+=\s+")


def resolve(text: str, temperature: float | None = None, pressure: float | None = None, solvent: str = "",
            phase: str = "", allow_online: bool = True) -> Query:
    """What the text names, as one Query. temperature in K and pressure in Pa override what the text says."""
    raw = (text or "").strip()
    if not raw:
        raise ResolveError("nothing to simulate", "Give a name, formula, SMILES, equation or a few words.")
    _OFFLINE_SEEN.clear()
    p = Path(raw).expanduser()
    if p.suffix.lower() in (".cif", ".xyz", ".mol", ".sdf", ".pdb", ".poscar", ".vasp") and p.exists():
        q = Query(raw, "crystal" if p.suffix.lower() in (".cif", ".poscar", ".vasp") else "molecule", path=p)
    else:
        body, cond = conditions(raw)
        q = _query(body, raw, allow_online)
        q.temperature = cond.get("temperature")
        q.pressure = cond.get("pressure")
    if temperature is not None:
        q.temperature = temperature
    if pressure is not None:
        q.pressure = pressure
    if solvent:
        q.solvent = solvent
    if phase:
        q.phase = phase
    if _OFFLINE_SEEN and allow_online:
        q.notes.append(f"PubChem was not reachable ({_OFFLINE_SEEN[0]}), so only Wobbl's own table was used")
    return q


def _query(body: str, raw: str, allow_online: bool) -> Query:
    low = body.lower().strip()
    from .propellants import INGREDIENTS, PROPELLANTS
    # a propellant preset or an ingredient composition
    if re.match(r"^[\w-]+\s*:\s*\d", body) and all(k.strip().lower() in INGREDIENTS
                                                   for k in re.findall(r"([\w-]+)\s*:", body)):
        return Query(raw, "propellant", notes=[f"'{body}' is a propellant composition by mass"])
    if low in PROPELLANTS or any(low in pp.aliases for pp in PROPELLANTS.values()):
        return Query(raw, "propellant", notes=[f"'{body}' is a rocket propellant Wobbl has published data for"])
    # a battery cell
    if re.fullmatch(r"(?:(?:li(?:thium)?[- ]?ion|licoo2|lco)\s+)?(?:18650|21700|26650|14500)(?:\s+cell)?|"
                    r"(?:a\s+)?(?:li(?:thium)?[- ]?ion|licoo2|lco)\s+(?:cell|battery)(?:\s+cell)?", low):
        fmt = re.search(r"18650|21700|26650|14500", low)
        return Query(raw, "cell", notes=[f"'{body}' is a lithium ion cell, format {fmt.group(0) if fmt else '18650'}"])
    # a reaction written out
    if ARROW.search(body) and not re.fullmatch(r"[^=]*=\s*\d.*", body):
        return _written(body, raw, allow_online)
    named = NAMED.get(low) or NAMED.get(re.sub(r"^(?:the|a)\s+", "", low))
    if named:
        q = _written(named, raw, allow_online)
        q.reaction_source = f"the named reaction '{low}'"
        return q
    if air := (low in ("air", "dry air")):
        return _mixture([("N2", 0.7808), ("O2", 0.2095), ("Ar", 0.0093)], raw, allow_online, note="dry air")
    del air
    # one substance written as a formula, SMILES or ion, plus signs included
    if F.is_formula(body) or (looks_like_smiles(body) and " " not in body):
        return _single(body, raw, allow_online)
    # a crystal named by its structure, as 'fcc copper' or 'copper crystal'
    m = re.match(rf"^(?:(?:{STRUCTURE_WORDS})\s+)+(.+)$|^(.+?)(?:\s+(?:{STRUCTURE_WORDS}))+$", body, re.I)
    if m:
        what = (m.group(1) or m.group(2)).strip()
        word = re.search(STRUCTURE_WORDS, body, re.I).group(0).lower()
        q = _single(what, raw, allow_online)
        proto = PROTOTYPE_WORDS.get(word.replace(" ", ""), "")
        if proto:
            q.main.crystal = proto
        q.kind = "crystal" if q.kind in ("element", "crystal", "salt") else q.kind
        return q
    # reactions and mixtures in words
    m = re.match(r"^(?:burning|combustion of|burn|oxidation of|igniting)\s+(.+)$|^(.+?)\s+(?:burning|combustion|"
                 r"burns|flame)$", body, re.I)
    if m:
        fuel = (m.group(1) or m.group(2)).strip()
        return _pair([fuel, "air"], raw, allow_online, burning=True)
    if "+" in body and not ARROW.search(body):
        try:
            q = _written(body, raw, allow_online)
            return _pair([t.species for t in q.reactants], raw, allow_online)
        except ResolveError:
            pass
    m = re.match(r"^(?:mixing\s+|mix\s+|reacting\s+|react\s+)?(.+?)\s+(?:and|with|plus|\+|in|into|added to|"
                 r"dropped in|dropped into|reacting with|reacts with|mixed with)\s+(.+)$", body, re.I)
    if m and not _is_one(body):
        parts = [m.group(1), m.group(2)]
        more = re.split(r"\s+(?:and|with|plus|\+)\s+", parts[1], flags=re.I)
        parts = [parts[0], *more]
        verb = re.search(r"\s(in|into|dropped in|dropped into|added to)\s", f" {low} ")
        try:
            return _pair(parts, raw, allow_online, solvent_word=bool(verb))
        except ResolveError as e:
            # 'glucose in water' is glucose dissolved in water, not a reaction
            if verb and len(parts) == 2 and "do not react" in str(e):
                q = _single(parts[0], raw, allow_online)
                q.solvent = parts[1].strip().lower()
                q.notes.append(f"{q.main.name} dissolved in {q.solvent}, with the solvent as a continuum")
                return q
            raise
    return _single(body, raw, allow_online)


STRUCTURE_WORDS = (r"fcc|bcc|hcp|sc|simple cubic|face centered cubic|body centered cubic|hexagonal close packed|"
                   r"diamond cubic|rock ?salt|zinc ?blende|wurtzite|fluorite|rutile|corundum|cesium chloride|"
                   r"crystal|lattice|bulk|solid")
PROTOTYPE_WORDS = {"fcc": "fcc", "facecenteredcubic": "fcc", "bcc": "bcc", "bodycenteredcubic": "bcc", "hcp": "hcp",
                   "hexagonalclosepacked": "hcp", "sc": "sc", "simplecubic": "sc", "diamondcubic": "diamond",
                   "rocksalt": "rocksalt", "zincblende": "zincblende", "wurtzite": "wurtzite",
                   "fluorite": "fluorite", "rutile": "rutile", "corundum": "corundum",
                   "cesiumchloride": "cesiumchloride"}


def _single(body: str, raw: str, allow_online: bool) -> Query:
    sp = species(body, allow_online)
    kind = sp.kind
    if kind in ("metal", "element"):
        kind = "element"
    if sp.crystal and kind not in ("element",):
        kind = "crystal"
    q = Query(raw, kind, [sp])
    if sp.note:
        q.notes.append(sp.note)
    return q


def _is_one(body: str) -> bool:
    """True when the whole text is one known name, like 'sodium bicarbonate' or 'iron(iii) oxide'."""
    low = body.lower()
    if lookup(low) or low in F.by_name():
        return True
    return False


def _written(body: str, raw: str, allow_online: bool) -> Query:
    sides = ARROW.split(body.strip())
    if len(sides) > 2:
        raise ResolveError(f"'{body}' has more than one arrow", "Write one reaction, like 2 H2 + O2 -> 2 H2O.")
    q = Query(raw, "reaction", reaction_source="written")
    for i, side in enumerate(sides):
        terms = []
        for term in re.split(r"\s+\+\s+|\s*\+\s*(?=[A-Za-z\d(\[])", side.strip()):
            if not term.strip():
                continue
            m = re.match(r"^(\d+(?:\.\d+)?(?:/\d+)?)?\s*(.+?)(?:\s*\((g|l|s|aq)\))?$", term.strip())
            coef = m.group(1) or "1"
            num = float(coef.split("/")[0]) / (float(coef.split("/")[1]) if "/" in coef else 1.0)
            sp = species(m.group(2), allow_online)
            terms.append(Term(num, sp, m.group(3) or ""))
        if i == 0:
            q.reactants = terms
        else:
            q.products = terms
    q.species = [t.species for t in q.reactants + q.products]
    if not q.reactants:
        raise ResolveError(f"'{body}' has no reactants")
    if q.products:
        from .reaction import balance, equation
        q.reactants, q.products, changed = balance(q.reactants, q.products)
        if changed:
            q.notes.append(f"balanced as {equation(q.reactants, q.products, phases=False)}")
    return q


def _pair(parts: list[str], raw: str, allow_online: bool, burning: bool = False,
          solvent_word: bool = False) -> Query:
    """Two or more substances named together: a mixture when they are all gases, else a reaction to predict."""
    from .reaction import predict
    sps = []
    for p in parts:
        if isinstance(p, Species):
            sps.append(None if p.name == "air" else p)
            continue
        p = re.sub(r"^(?:a|an|the|some)\s+", "", p.strip(), flags=re.I)
        if p.lower() in ("air", "the air"):
            sps.append(None)
            continue
        sps.append(species(p, allow_online))
    has_air = any(s is None for s in sps)
    real = [s for s in sps if s is not None]
    gases = [s for s in real if _gaseous(s)]
    if (has_air or any(s.formula == "O2" for s in real)) and len(gases) == len(real) and real:
        fuel = [s for s in real if s.formula not in ("O2", "N2")]
        if fuel:
            q = Query(raw, "mixture", real, notes=[])
            q.amounts = {"_fuel": [s.formula for s in fuel], "_oxidizer": "air" if has_air else "O2"}
            return q
    if len(gases) == len(real) and not has_air and len(real) > 1:
        q = Query(raw, "mixture", real)
        q.amounts = {s.formula: 1.0 for s in real}
        return q
    if has_air:
        real.append(species("oxygen", allow_online))
    q = Query(raw, "reaction", real)
    pred = predict(real, burning=burning, in_water=solvent_word and any(s.formula == "H2O" for s in real))
    if pred is None:
        names = " and ".join(s.label() for s in real)
        raise ResolveError(f"Wobbl could not predict what {names} make together",
                           "Write the reaction out, like 'NaHCO3 + CH3COOH -> NaCH3COO + H2O + CO2'.")
    q.reactants, q.products, q.reaction_source = pred
    q.species = [t.species for t in q.reactants + q.products]
    return q


def _gaseous(s: Species) -> bool:
    if s.phase:
        return s.phase == "g"
    return s.formula in GASES_AT_ROOM


def _mixture(parts: list[tuple[str, float]], raw: str, allow_online: bool, note: str = "") -> Query:
    sps = [species(f, allow_online) for f, _ in parts]
    q = Query(raw, "mixture", sps, notes=[note] if note else [])
    q.amounts = {s.formula: x for s, (_, x) in zip(sps, parts)}
    return q
