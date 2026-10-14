"""Common substances by everyday and chemical name, so names resolve offline and household words mean something.

PubChem resolves any name online. This table covers the substances people name most, the household words PubChem
does not know as chemicals (table salt, rust, baking soda, vinegar, lye, chalk), and what PubChem cannot say, such
as which crystal structure a solid takes and which of an acid's protons are acidic. Each entry gives a formula, a
SMILES for molecules, its kind (molecule, salt, metal oxide, network solid) and its phase at room temperature.
Vinegar and the like are solutions, so they carry what is dissolved and how much.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Substance:
    name: str
    formula: str
    smiles: str = ""
    kind: str = "molecule"       # molecule, salt, oxide, network, metal
    phase: str = ""              # g, l or s at 25 C and 1 atm
    aliases: tuple[str, ...] = ()
    crystal: str = ""            # a structure prototype in material.py
    lattice: float = 0.0         # measured lattice constant, angstrom, for the check
    solution: tuple[str, float] = ("", 0.0)   # (solute, mass fraction) for solutions like vinegar
    note: str = ""
    multiplicity: int = 0        # spin multiplicity when the SMILES does not say it, as for triplet oxygen


S = Substance
TABLE: list[Substance] = [
    # gases
    S("water", "H2O", "O", phase="l", aliases=("h2o", "ice", "steam")),
    S("hydrogen", "H2", "[H][H]", phase="g", aliases=("h2", "hydrogen gas")),
    S("oxygen", "O2", "O=O", phase="g", aliases=("o2", "oxygen gas", "lox"), multiplicity=3),
    S("nitrogen", "N2", "N#N", phase="g", aliases=("n2", "nitrogen gas")),
    S("ozone", "O3", "[O-][O+]=O", phase="g", aliases=("o3",)),
    S("carbon dioxide", "CO2", "O=C=O", phase="g", aliases=("co2", "dry ice")),
    S("carbon monoxide", "CO", "[C-]#[O+]", phase="g", aliases=("co",)),
    S("methane", "CH4", "C", phase="g", aliases=("ch4", "natural gas")),
    S("ethane", "C2H6", "CC", phase="g", aliases=("c2h6",)),
    S("propane", "C3H8", "CCC", phase="g", aliases=("c3h8", "lpg")),
    S("butane", "C4H10", "CCCC", phase="g", aliases=("c4h10", "n-butane")),
    S("ethylene", "C2H4", "C=C", phase="g", aliases=("ethene", "c2h4")),
    S("acetylene", "C2H2", "C#C", phase="g", aliases=("ethyne", "c2h2")),
    S("ammonia", "NH3", "N", phase="g", aliases=("nh3",)),
    S("nitric oxide", "NO", "[N]=O", phase="g", aliases=("nitrogen monoxide",)),
    S("nitrogen dioxide", "NO2", "N(=O)[O]", phase="g", aliases=("no2",)),
    S("nitrous oxide", "N2O", "[N-]=[N+]=O", phase="g", aliases=("laughing gas", "n2o")),
    S("sulfur dioxide", "SO2", "O=S=O", phase="g", aliases=("so2",)),
    S("hydrogen sulfide", "H2S", "S", phase="g", aliases=("h2s",)),
    S("hydrogen chloride", "HCl", "Cl", phase="g", aliases=("hcl",)),
    S("hydrogen fluoride", "HF", "F", phase="g", aliases=("hf",)),
    S("hydrogen cyanide", "HCN", "C#N", phase="g", aliases=("hcn", "prussic acid")),
    S("chlorine", "Cl2", "ClCl", phase="g", aliases=("cl2",)),
    S("fluorine", "F2", "FF", phase="g", aliases=("f2",)),
    S("formaldehyde", "CH2O", "C=O", phase="g", aliases=("methanal",)),
    # liquids and molecular solids
    S("methanol", "CH4O", "CO", phase="l", aliases=("methyl alcohol", "wood alcohol")),
    S("ethanol", "C2H6O", "CCO", phase="l", aliases=("ethyl alcohol", "alcohol", "grain alcohol")),
    S("isopropanol", "C3H8O", "CC(C)O", phase="l", aliases=("isopropyl alcohol", "rubbing alcohol", "2-propanol")),
    S("acetone", "C3H6O", "CC(C)=O", phase="l", aliases=("propanone", "nail polish remover")),
    S("acetic acid", "C2H4O2", "CC(=O)O", phase="l", aliases=("ethanoic acid", "glacial acetic acid")),
    S("vinegar", "C2H4O2", "CC(=O)O", phase="aq", solution=("acetic acid", 0.05),
      note="household vinegar, about 5% acetic acid in water"),
    S("formic acid", "CH2O2", "C(=O)O", phase="l", aliases=("methanoic acid",)),
    S("benzene", "C6H6", "c1ccccc1", phase="l"),
    S("toluene", "C7H8", "Cc1ccccc1", phase="l", aliases=("methylbenzene",)),
    S("phenol", "C6H6O", "Oc1ccccc1", phase="s", aliases=("carbolic acid",)),
    S("octane", "C8H18", "CCCCCCCC", phase="l", aliases=("n-octane",)),
    S("isooctane", "C8H18", "CC(C)CC(C)(C)C", phase="l", aliases=("2,2,4-trimethylpentane", "gasoline", "petrol")),
    S("hexane", "C6H14", "CCCCCC", phase="l", aliases=("n-hexane",)),
    S("cyclohexane", "C6H12", "C1CCCCC1", phase="l"),
    S("hydrogen peroxide", "H2O2", "OO", phase="l", aliases=("h2o2", "peroxide")),
    S("hydrazine", "H4N2", "NN", phase="l", aliases=("n2h4",)),
    S("glycerol", "C3H8O3", "OCC(O)CO", phase="l", aliases=("glycerin", "glycerine")),
    S("urea", "CH4N2O", "NC(N)=O", phase="s", aliases=("carbamide",)),
    S("glucose", "C6H12O6", "OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O", phase="s", aliases=("dextrose",
                                                                                           "blood sugar")),
    S("sucrose", "C12H22O11", "OC[C@H]1O[C@@](CO)(O[C@H]2O[C@H](CO)[C@@H](O)[C@H](O)[C@H]2O)[C@@H](O)[C@@H]1O",
      phase="s", aliases=("sugar", "table sugar", "cane sugar")),
    S("caffeine", "C8H10N4O2", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C", phase="s"),
    S("aspirin", "C9H8O4", "CC(=O)Oc1ccccc1C(=O)O", phase="s", aliases=("acetylsalicylic acid",)),
    S("paracetamol", "C8H9NO2", "CC(=O)Nc1ccc(O)cc1", phase="s", aliases=("acetaminophen", "tylenol")),
    S("ibuprofen", "C13H18O2", "CC(C)Cc1ccc(cc1)C(C)C(=O)O", phase="s"),
    S("nicotine", "C10H14N2", "CN1CCC[C@H]1c1cccnc1", phase="l"),
    S("citric acid", "C6H8O7", "OC(=O)CC(O)(CC(=O)O)C(=O)O", phase="s"),
    S("lactic acid", "C3H6O3", "CC(O)C(=O)O", phase="l"),
    S("naphthalene", "C10H8", "c1ccc2ccccc2c1", phase="s", aliases=("mothballs",)),
    S("glycine", "C2H5NO2", "NCC(=O)O", phase="s"),
    S("alanine", "C3H7NO2", "C[C@H](N)C(=O)O", phase="s", aliases=("l-alanine",)),
    S("adenine", "C5H5N5", "Nc1ncnc2[nH]cnc12", phase="s"),
    S("vanillin", "C8H8O3", "COc1cc(C=O)ccc1O", phase="s"),
    S("menthol", "C10H20O", "CC(C)[C@@H]1CC[C@@H](C)C[C@H]1O", phase="s"),
    # acids and bases in water
    S("sulfuric acid", "H2SO4", "OS(=O)(=O)O", phase="l", aliases=("h2so4", "battery acid", "oil of vitriol")),
    S("nitric acid", "HNO3", "O[N+](=O)[O-]", phase="l", aliases=("hno3",)),
    S("hydrochloric acid", "HCl", "Cl", phase="aq", solution=("hydrogen chloride", 0.10),
      aliases=("muriatic acid",), note="taken as 10% hydrogen chloride in water"),
    S("phosphoric acid", "H3PO4", "OP(=O)(O)O", phase="s", aliases=("h3po4",)),
    S("carbonic acid", "H2CO3", "OC(=O)O", phase="aq"),
    S("sodium hydroxide", "NaOH", "[Na+].[OH-]", kind="salt", phase="s", aliases=("lye", "caustic soda", "naoh")),
    S("potassium hydroxide", "KOH", "[K+].[OH-]", kind="salt", phase="s", aliases=("caustic potash", "koh")),
    S("calcium hydroxide", "CaO2H2", "[Ca+2].[OH-].[OH-]", kind="salt", phase="s",
      aliases=("slaked lime", "hydrated lime", "ca(oh)2")),
    # salts and minerals
    S("sodium chloride", "NaCl", "[Na+].[Cl-]", kind="salt", phase="s", crystal="rocksalt", lattice=5.640,
      aliases=("table salt", "salt", "halite", "rock salt", "nacl")),
    S("potassium chloride", "KCl", "[K+].[Cl-]", kind="salt", phase="s", crystal="rocksalt", lattice=6.293,
      aliases=("sylvite", "kcl")),
    S("lithium fluoride", "LiF", "[Li+].[F-]", kind="salt", phase="s", crystal="rocksalt", lattice=4.027),
    S("magnesium oxide", "MgO", "[Mg+2].[O-2]", kind="oxide", phase="s", crystal="rocksalt", lattice=4.212,
      aliases=("magnesia", "periclase")),
    S("calcium oxide", "CaO", "[Ca+2].[O-2]", kind="oxide", phase="s", crystal="rocksalt", lattice=4.811,
      aliases=("quicklime", "lime", "burnt lime")),
    S("cesium chloride", "CsCl", "[Cs+].[Cl-]", kind="salt", phase="s", crystal="cesiumchloride", lattice=4.123,
      aliases=("caesium chloride",)),
    S("zinc sulfide", "SZn", "[Zn+2].[S-2]", kind="salt", phase="s", crystal="zincblende", lattice=5.409,
      aliases=("sphalerite", "zinc blende", "zns")),
    S("gallium arsenide", "AsGa", "[Ga]#[As]", kind="network", phase="s", crystal="zincblende", lattice=5.653,
      aliases=("gaas",)),
    S("silicon carbide", "CSi", "[Si]#[C]", kind="network", phase="s", crystal="zincblende", lattice=4.360,
      aliases=("carborundum", "sic", "moissanite")),
    S("calcium fluoride", "CaF2", "[Ca+2].[F-].[F-]", kind="salt", phase="s", crystal="fluorite", lattice=5.463,
      aliases=("fluorite", "fluorspar")),
    S("calcium carbonate", "CCaO3", "[Ca+2].[O-]C([O-])=O", kind="salt", phase="s",
      aliases=("chalk", "limestone", "calcite", "marble", "caco3")),
    S("sodium bicarbonate", "CHNaO3", "[Na+].OC([O-])=O", kind="salt", phase="s",
      aliases=("baking soda", "sodium hydrogen carbonate", "bicarbonate of soda", "nahco3")),
    S("sodium carbonate", "CNa2O3", "[Na+].[Na+].[O-]C([O-])=O", kind="salt", phase="s",
      aliases=("washing soda", "soda ash", "na2co3")),
    S("sodium acetate", "C2H3NaO2", "CC(=O)[O-].[Na+]", kind="salt", phase="s"),
    S("sodium hypochlorite", "ClNaO", "[Na+].[O-]Cl", kind="salt", phase="aq", aliases=("bleach",)),
    S("potassium nitrate", "KNO3", "[K+].[O-][N+](=O)[O-]", kind="salt", phase="s", aliases=("saltpeter",
                                                                                          "saltpetre")),
    S("calcium sulfate", "CaO4S", "[Ca+2].[O-]S([O-])(=O)=O", kind="salt", phase="s", aliases=("gypsum",
                                                                                            "plaster")),
    S("copper sulfate", "CuO4S", "[Cu+2].[O-]S([O-])(=O)=O", kind="salt", phase="s",
      aliases=("copper(ii) sulfate", "blue vitriol")),
    S("silver nitrate", "AgNO3", "[Ag+].[O-][N+](=O)[O-]", kind="salt", phase="s"),
    S("silver chloride", "AgCl", "[Ag+].[Cl-]", kind="salt", phase="s", crystal="rocksalt", lattice=5.549),
    S("iron(iii) oxide", "Fe2O3", "[O-2].[O-2].[O-2].[Fe+3].[Fe+3]", kind="oxide", phase="s", crystal="corundum",
      aliases=("rust", "hematite", "ferric oxide", "fe2o3")),
    S("aluminium oxide", "Al2O3", "[O-2].[O-2].[O-2].[Al+3].[Al+3]", kind="oxide", phase="s", crystal="corundum",
      aliases=("aluminum oxide", "alumina", "corundum", "sapphire", "ruby", "al2o3")),
    S("titanium dioxide", "O2Ti", "O=[Ti]=O", kind="oxide", phase="s", crystal="rutile", lattice=4.594,
      aliases=("titania", "rutile", "tio2")),
    S("zinc oxide", "OZn", "[Zn]=O", kind="oxide", phase="s", crystal="wurtzite", lattice=3.250,
      aliases=("zno", "zincite")),
    S("silicon dioxide", "O2Si", "O=[Si]=O", kind="network", phase="s", aliases=("silica", "quartz", "sand",
                                                                               "sio2")),
    S("diamond", "C", "", kind="network", phase="s", crystal="diamond", lattice=3.567),
    S("graphite", "C", "", kind="network", phase="s", note="carbon's stable form at room temperature"),
]


def _key(formula: str) -> str:
    from .formula import hill, parse
    counts, charge = parse(formula)
    return hill(counts, charge)


COMMON_FORMULAS: dict[str, Substance] = {}
for _s in TABLE:
    if _s.kind == "molecule" and not _s.solution[0] or _s.crystal:
        COMMON_FORMULAS.setdefault(_key(_s.formula), _s)
_INDEX: dict[str, Substance] = {}
for _s in TABLE:
    for _k in (_s.name, *_s.aliases):
        _INDEX.setdefault(_k.lower(), _s)


_BY_SMILES = {s.smiles: s for s in TABLE if s.smiles and not s.solution[0]}


def lookup(name: str) -> Substance | None:
    return _INDEX.get(name.strip().lower())


def by_smiles(smiles: str) -> Substance | None:
    return _BY_SMILES.get(smiles.strip())


@dataclass(frozen=True)
class Ion:
    formula: str
    charge: int
    name: str
    smiles: str = ""


CATIONS = {
    "Li": Ion("Li", 1, "lithium", "[Li+]"), "Na": Ion("Na", 1, "sodium", "[Na+]"),
    "K": Ion("K", 1, "potassium", "[K+]"), "Rb": Ion("Rb", 1, "rubidium", "[Rb+]"),
    "Cs": Ion("Cs", 1, "cesium", "[Cs+]"), "Ag": Ion("Ag", 1, "silver", "[Ag+]"),
    "NH4": Ion("NH4", 1, "ammonium", "[NH4+]"), "Mg": Ion("Mg", 2, "magnesium", "[Mg+2]"),
    "Ca": Ion("Ca", 2, "calcium", "[Ca+2]"), "Sr": Ion("Sr", 2, "strontium", "[Sr+2]"),
    "Ba": Ion("Ba", 2, "barium", "[Ba+2]"), "Zn": Ion("Zn", 2, "zinc", "[Zn+2]"),
    "Cu": Ion("Cu", 2, "copper(II)", "[Cu+2]"), "Fe": Ion("Fe", 2, "iron(II)", "[Fe+2]"),
    "Ni": Ion("Ni", 2, "nickel", "[Ni+2]"), "Pb": Ion("Pb", 2, "lead", "[Pb+2]"),
    "Sn": Ion("Sn", 2, "tin", "[Sn+2]"), "Mn": Ion("Mn", 2, "manganese", "[Mn+2]"),
    "Co": Ion("Co", 2, "cobalt", "[Co+2]"), "Al": Ion("Al", 3, "aluminium", "[Al+3]"),
    "Cr": Ion("Cr", 3, "chromium", "[Cr+3]"),
}
ANIONS = {
    "F": Ion("F", -1, "fluoride", "[F-]"), "Cl": Ion("Cl", -1, "chloride", "[Cl-]"),
    "Br": Ion("Br", -1, "bromide", "[Br-]"), "I": Ion("I", -1, "iodide", "[I-]"),
    "OH": Ion("OH", -1, "hydroxide", "[OH-]"), "NO3": Ion("NO3", -1, "nitrate", "[O-][N+](=O)[O-]"),
    "CH3COO": Ion("CH3COO", -1, "acetate", "CC(=O)[O-]"), "HCO3": Ion("HCO3", -1, "bicarbonate", "OC([O-])=O"),
    "HSO4": Ion("HSO4", -1, "hydrogen sulfate", "OS(=O)(=O)[O-]"), "ClO": Ion("ClO", -1, "hypochlorite", "[O-]Cl"),
    "SO4": Ion("SO4", -2, "sulfate", "[O-]S([O-])(=O)=O"), "CO3": Ion("CO3", -2, "carbonate", "[O-]C([O-])=O"),
    "O": Ion("O", -2, "oxide", "[O-2]"), "S": Ion("S", -2, "sulfide", "[S-2]"),
    "PO4": Ion("PO4", -3, "phosphate", "[O-]P([O-])([O-])=O"),
}
# acids by Hill formula: the anion each gives up and how many protons
ACIDS = {_key(k): v for k, v in {"HCl": ("Cl", 1), "HBr": ("Br", 1), "HI": ("I", 1), "HF": ("F", 1),
                                 "HNO3": ("NO3", 1), "H2SO4": ("SO4", 2), "H3PO4": ("PO4", 3),
                                 "C2H4O2": ("CH3COO", 1), "H2CO3": ("CO3", 2), "CH2O2": ("HCOO", 1)}.items()}
STRONG_ACIDS = {_key(k) for k in ("HCl", "HBr", "HI", "HNO3", "H2SO4")}
ANIONS["HCOO"] = Ion("HCOO", -1, "formate", "[O-]C=O")
# how people write these in equations, where Hill order would hide the chemistry
CONVENTIONAL = {_key(k): v for k, v in {
    "C2H4O2": "CH3COOH", "C2H6O": "C2H5OH", "CH4O": "CH3OH", "H2SO4": "H2SO4", "HCl": "HCl", "NH3": "NH3",
    "H2CO3": "H2CO3", "HNO3": "HNO3", "H3PO4": "H3PO4", "CH2O2": "HCOOH", "H2O2": "H2O2", "NH4+": "NH4+",
    "C6H12O6": "C6H12O6", "C3H8O3": "C3H5(OH)3", "H2S": "H2S", "HF": "HF", "HCN": "HCN"}.items()}
