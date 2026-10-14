"""Chemistry for wobbl simulate: anything chemical, as a name, formula, SMILES, InChI, element, crystal, mixture,
reaction, propellant or battery cell, with the physics picked from the question.

The domain recognizes its things with chem/resolve.py, offline, so recognition never waits on PubChem. Every
analysis runs through chem/run.py's job_entry, which records each guess and stops at job.ready() under --plan.
"""
from __future__ import annotations

import re

from ..simulate import Analysis, Domain, Option, register
from ..simulate.things import recognize_chemistry

ENTRY = "wobbl.chem.run:job_entry"
MOLECULAR = ("molecule", "ion")
SOLIDS = ("element", "crystal", "salt", "material")
FLAME_WORDS = r"flame|\bburn|combust|adiabatic|ignit|in air\b|with air\b|in oxygen"


def recognize(thing: str, question: str):
    """(score, kind) for a chemical thing, from its shape: an equation, a composition, a cell, an element, a
    formula, a SMILES or a name in Wobbl's own table. A molecule asked about burning is a fuel."""
    t = (thing or "").strip()
    if t:
        try:
            from .resolve import resolve
            q = resolve(t, allow_online=False)
        except Exception:  # noqa: BLE001  not something the table or the text's shape can say
            q = None
        if q is not None:
            kind = {"ion": "molecule", "oxide": "crystal", "metal": "element", "network": "crystal"}.get(q.kind, q.kind)
            if kind == "molecule" and re.search(FLAME_WORDS, question or "", re.I):
                kind = "fuel"
            if kind in ("molecule", "element") and len(t.split()) > 3:
                return None
            return {"reaction": 9, "propellant": 9, "cell": 8, "mixture": 8}.get(kind, 7), kind
        try:
            from ..materials import LIBRARY
            low = t.lower()
            for m in LIBRARY.values():
                if low in (m.key, m.name.lower(), *m.aliases):
                    return 6, "material"
        except Exception:  # noqa: BLE001
            pass
    return recognize_chemistry(thing, question)


COMMON = (Option("temperature", "Temperature, e.g. '500 K' or '25 C'", "298.15 K"),
          Option("pressure", "Pressure, e.g. '2 atm', or a rocket's chamber pressure", "1 atm"),
          Option("solvent", "Solvent for a molecule, e.g. 'water' or 'toluene'"),
          Option("accuracy", "fast (GFN2-xTB only), normal (with DFT checks on small molecules) or high (DFT)",
                 "normal"))

register(Domain(
    "chemistry", "Chemistry: a molecule, ion, element, crystal, salt, mixture, reaction, propellant or battery cell, "
                 "as a name, formula, SMILES, InChI, equation or a few words",
    recognize=recognize,
    words=(r"\bmolecules?\b", r"\bcompounds?\b", r"\breactions?\b", r"\breacts?\b", r"\bcombust", r"\bflames?\b",
           r"\bisp\b", r"specific impulse", r"\bequilibrium\b", r"\benthalpy\b", r"\bentropy\b", r"\bgibbs\b",
           r"\bpropellants?\b", r"\bfuels?\b", r"\boxidi[sz]er\b", r"stoichiometric", r"equivalence ratio",
           r"thermal runaway", r"\bli-?ion\b", r"molar mass", r"\bir spectrum\b", r"\binfrared\b", r"\bdipole\b",
           r"\bhomo\b|\blumo\b", r"\bconformers?\b", r"\bcrystal\b", r"lattice constant", r"\bsmiles\b",
           r"\binchi\b", r"\bmolecular dynamics\b", r"\bbalance the\b", r"\bdissolv", r"\bfizz"),
    kinds={"reaction": r"->|→|=>|<=>", "cell": r"18650|21700|26650|14500|li-?ion|lithium ion|\bcells?\b",
           "propellant": r"\bpropellants?\b|\bisp\b|c\*|specific impulse|\bkn(?:sb|dx|su)\b",
           "fuel": r"methane|propane|butane|hydrogen|ethane|ethylene|acetylene|methanol|natural gas|\blpg\b",
           "crystal": r"\bcrystal\b|lattice|unit cell", "mixture": r"\bmixture\b|\bmixed with\b"},
    options=COMMON, default_kind="molecule", results=("chemistry", "combustion", "battery"),
    example='wobbl simulate caffeine "what does its IR spectrum look like?"'))

_A = [
    ("properties", "Structure, energy, dipole, charges and orbitals of a molecule, with its IR spectrum, "
                   "thermochemistry, conformers, solvation and motion", MOLECULAR, True,
     (r"\bdipole\b", r"\bpolar", r"\bcharges?\b", r"\bhomo\b", r"\blumo\b", r"\bgap\b", r"\borbital", r"solvat",
      r"dissolv", r"solub", r"\benergy\b", r"\bproperties\b", r"\bstable\b"), ("structure",), ("molecule",)),
    ("structure", "3D structure and conformers of a molecule, refined with GFN2-xTB and checked with DFT", MOLECULAR,
     False, (r"geometr", r"bond lengths?", r"bond angles?", r"\bshape\b", r"3d structure", r"\bconformers?\b",
             r"\bstructure\b"), (), ("geometry", "conformers")),
    ("spectrum", "Infrared spectrum and vibrations of a molecule, each band named", MOLECULAR, False,
     (r"\bir\b", r"infrared", r"spectr", r"vibrat", r"frequenc", r"wavenumber"), ("structure", "properties", "thermo"),
     ("ir", "vibrations", "frequencies")),
    ("thermo", "Enthalpy, entropy, Gibbs energy and heat capacity of a molecule, with its tabulated phase changes",
     MOLECULAR, False, (r"entropy", r"enthalpy", r"gibbs", r"heat capacity", r"thermochem", r"\bboil", r"\bmelt",
                        r"heat of formation"), ("structure", "properties"), ("thermochemistry",)),
    ("dynamics", "Molecular dynamics at a temperature, with energy conservation and time step checks",
     MOLECULAR + SOLIDS, False, (r"\bdynamic", r"\bmd\b", r"trajector", r"\bmoving\b", r"jiggl", r"diffus"), (),
     ("md", "molecular-dynamics")),
    ("reaction", "A reaction balanced, with its heat, Gibbs energy, equilibrium constant, gas made and temperature "
                 "rise", ("reaction",), True,
     (r"enthalpy", r"\bheat\b", r"exotherm", r"endotherm", r"spontan", r"gibbs", r"release", r"\bproducts?\b",
      r"balance", r"fizz", r"\bgas\b", r"temperature rise", r"feasib", r"\bgo\b", r"happen"), (),
     ("thermochemistry-of-reaction",)),
    ("equilibrium", "Equilibrium composition of a reaction or gas mixture at a temperature and pressure",
     ("reaction", "mixture"), False,
     (r"equilibrium", r"\byield", r"conversion", r"how far", r"\bcomposition\b"), ("reaction",), ()),
    ("path", "Reaction path and barrier between a reactant and product, by nudged elastic band with GFN2-xTB",
     ("reaction",), False, (r"barrier", r"transition state", r"activation energy", r"mechanism", r"reaction path",
                            r"\bneb\b"), (), ("barrier", "neb")),
    ("combustion", "Adiabatic flame temperature and equilibrium products of a fuel or gas mixture",
     ("fuel", "mixture", "reaction"), True, (r"flame", r"\bburn", r"combust", r"adiabatic", r"ignit", r"how hot"),
     ("reaction",), ("flame",)),
    ("propellant", "Rocket propellant c* and Isp, shifting and frozen, CEA style",
     ("propellant", "mixture", "reaction"), True, (r"\bisp\b", r"specific impulse", r"c\*", r"rocket", r"nozzle"),
     ("combustion", "reaction"), ()),
    ("crystal", "Crystal structure, lattice constant, density, cohesive energy and bulk modulus of an element or "
                "solid, with its atomic data", SOLIDS, True,
     (r"lattice", r"crystal", r"density", r"bulk modulus", r"\bstiff", r"cohesive", r"unit cell", r"elastic",
      r"atomic", r"electronegativ", r"ionization"), (), ("lattice", "solid", "element")),
    ("runaway", "Lithium ion cell thermal runaway onset, by Semenov's criterion and a transient", ("cell",), True,
     (r"runaway", r"run away", r"self.?heat", r"\babuse\b", r"\boven\b"), (), ()),
]

EXTRA = {"combustion": (Option("phi", "Equivalence ratio, fuel over stoichiometric", "1"),
                        Option("oxidizer", "air or oxygen", "air"),
                        Option("limit", "Temperature the flame must stay under, e.g. '1800 C'")),
         "propellant": (Option("area-ratio", "Nozzle exit to throat area ratio"),
                        Option("ambient", "Pressure outside the nozzle", "1 atm")),
         "runaway": (Option("ambient", "Temperature around the cell", "25 C"),
                     Option("film", "Cooling, e.g. '25 W/m2K' or 'sealed box'", "7.17 W/m2K"),
                     Option("from", "A heat result folder whose hottest temperature surrounds the cell")),
         "dynamics": (Option("md-time", "Length of the run in fs", "1000"),)}

for name, summary, kinds, default, keywords, covers, aliases in _A:
    register(Analysis("chemistry", name, summary, ENTRY, keywords=keywords, options=EXTRA.get(name, ()),
                      aliases=aliases, kinds=kinds, default=default, covers=covers, dry=True,
                      example={"properties": 'wobbl simulate caffeine',
                               "spectrum": 'wobbl simulate CCO "what does its IR spectrum look like?"',
                               "reaction": 'wobbl simulate "baking soda and vinegar"',
                               "crystal": 'wobbl simulate copper "how stiff is it?"',
                               "combustion": 'wobbl simulate methane "how hot is its flame in air?"',
                               "equilibrium": 'wobbl simulate "N2 + 3 H2 <=> 2 NH3 at 700 K and 200 bar"',
                               "propellant": 'wobbl simulate knsb "what Isp at 5 MPa with an area ratio of 8?"',
                               "runaway": 'wobbl simulate 18650 "can it run away at 60 C?"'}.get(name, "")))
