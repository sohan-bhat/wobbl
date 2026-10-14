"""Reading anything chemical, offline: names, formulas, SMILES, ions, elements, crystals, mixtures and reactions."""
import pytest

from wobbl.chem import formula as F
from wobbl.chem.reaction import balance, equation, ions_of
from wobbl.chem.resolve import ResolveError, conditions, resolve, species


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setenv("WOBBL_OFFLINE", "1")


@pytest.mark.parametrize("text,counts,charge,hill", [
    ("Ca(OH)2", {"Ca": 1, "O": 2, "H": 2}, 0, "CaH2O2"),
    ("CuSO4·5H2O", {"Cu": 1, "S": 1, "O": 9, "H": 10}, 0, "CuH10O9S"),
    ("SO4^2-", {"S": 1, "O": 4}, -2, "O4S^2-"),
    ("SO42-", {"S": 1, "O": 4}, -2, "O4S^2-"),
    ("NO3-", {"N": 1, "O": 3}, -1, "NO3-"),
    ("Fe3+", {"Fe": 1}, 3, "Fe^3+"),
    ("[Fe(CN)6]4-", {"Fe": 1, "C": 6, "N": 6}, -4, "C6FeN6^4-"),
    ("C8H10N4O2", {"C": 8, "H": 10, "N": 4, "O": 2}, 0, "C8H10N4O2"),
])
def test_formulas_with_brackets_hydrates_and_charges(text, counts, charge, hill):
    c, q = F.parse(text)
    assert dict(c) == counts and q == charge and F.hill(c, q) == hill


def test_molar_mass_and_bad_formulas():
    assert F.molar_mass(F.parse("H2O")[0]) == pytest.approx(18.015, abs=0.002)
    assert F.molar_mass(F.parse("C8H10N4O2")[0]) == pytest.approx(194.19, abs=0.01)
    with pytest.raises(F.FormulaError):
        F.parse("Xq2")
    assert not F.is_formula("caffeine")


@pytest.mark.parametrize("text,name,kind", [
    ("caffeine", "caffeine", "molecule"), ("table salt", "sodium chloride", "crystal"),
    ("rust", "iron(iii) oxide", "crystal"), ("C8H10N4O2", "caffeine", "molecule"), ("CCO", "ethanol", "molecule"),
    ("c1ccccc1", "benzene", "molecule"), ("Cu", "copper", "element"), ("argon", "argon", "element"),
    ("oxygen", "oxygen", "molecule"), ("SO4^2-", "sulfate", "ion"), ("NH4+", "ammonium", "ion"),
    ("NaHCO3", "sodium bicarbonate", "salt"), ("knsb", None, "propellant"), ("18650 cell", None, "cell"),
    ("fcc copper", "copper", "crystal"), ("diamond", "diamond", "crystal"),
])
def test_single_things(text, name, kind):
    q = resolve(text)
    assert q.kind == kind
    if name:
        assert q.main.name == name
        assert q.resolved()


def test_conditions_come_out_of_the_text():
    body, c = conditions("water at 500 K and 10 bar")
    assert body == "water" and c == {"temperature": 500.0, "pressure": 1e6}
    q = resolve("glucose at 37 C in water", solvent="water")
    assert q.temperature == pytest.approx(310.15) and q.solvent == "water" and q.main.name == "glucose"
    assert resolve("methane and air at 1 atm").pressure == 101325.0


@pytest.mark.parametrize("text,eq,rule", [
    ("sodium in water", "2 Na(s) + 2 H2O(l) -> 2 NaOH(aq) + H2(g)", "active metal in water"),
    ("baking soda and vinegar", "CH3COOH(aq) + NaHCO3(s) -> CH3COONa(aq) + H2O(l) + CO2(g)", "carbonate"),
    ("zinc and hydrochloric acid", "Zn(s) + 2 HCl(aq) -> ZnCl2(aq) + H2(g)", "more active than hydrogen"),
    ("HCl + NaOH", "HCl(aq) + NaOH(aq) -> NaCl(aq) + H2O(l)", "neutralization"),
    ("silver nitrate and sodium chloride", "AgNO3(aq) + NaCl(aq) -> AgCl(s) + NaNO3(aq)", "precipitation"),
    ("burning magnesium", "2 Mg(s) + O2(g) -> 2 MgO(s)", "combustion"),
    ("aluminium and iron(iii) oxide", "2 Al(s) + Fe2O3(s) -> Al2O3(s) + 2 Fe(s)", "thermite"),
    ("zinc and copper sulfate", "Zn(s) + CuSO4(aq) -> ZnSO4(aq) + Cu(s)", "displacing"),
    ("CaO in water", "CaO(s) + H2O(l) -> Ca(OH)2(s)", "oxide in water"),
    ("ethanol and oxygen", "CH3CH2OH", "combustion"),
])
def test_reactions_in_words_are_predicted_and_balanced(text, eq, rule):
    q = resolve(text)
    assert q.kind == "reaction" and rule in q.reaction_source
    if "CH3CH2OH" not in eq:
        assert equation(q.reactants, q.products) == eq
    else:
        assert equation(q.reactants, q.products).endswith("-> 2 CO2(g) + 3 H2O(g)")


def test_written_reactions_are_balanced_and_named_ones_known():
    q = resolve("N2 + H2 -> NH3")
    assert equation(q.reactants, q.products, False) == "N2 + 3 H2 -> 2 NH3" and "balanced as" in q.notes[0]
    q = resolve("2 H2 + O2 -> 2 H2O")
    assert not q.notes and [t.coef for t in q.reactants] == [2, 1]
    t = resolve("thermite")
    assert equation(t.reactants, t.products, phases=False) == "2 Al + Fe2O3 -> Al2O3 + 2 Fe"
    with pytest.raises(ResolveError, match="cannot be balanced"):
        resolve("H2 + O2 -> CO2")


def test_mixtures_and_refusals():
    q = resolve("methane and air")
    assert q.kind == "mixture" and q.amounts["_oxidizer"] == "air"
    assert resolve("hydrogen and oxygen").kind == "mixture"
    with pytest.raises(ResolveError, match="less active than hydrogen"):
        resolve("gold and hydrochloric acid")
    with pytest.raises(ResolveError, match="could not tell"):
        resolve("unobtainium sludge")


def test_salts_split_into_their_ions():
    assert ions_of(species("NaHCO3")) == ((1, "Na", 1), (1, "HCO3", -1))
    assert ions_of(species("Fe2O3")) == ((2, "Fe", 3), (3, "O", -2))
    r, p, changed = balance(resolve("H2 + O2 -> H2O").reactants, resolve("H2 + O2 -> H2O").products)
    assert not changed and [t.coef for t in r + p] == [2, 1, 2]
