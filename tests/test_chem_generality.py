"""Generality: the chemistry pipeline over 60 varied substances and reactions it was not tuned for, organics,
inorganics, radicals, ions, salts, metals, noble gases, biomolecule fragments, mixtures and reactions.

The fast test resolves and plans every item offline. The slow one runs them all at fast accuracy and writes the
count to docs/chem-generality.json, the number the README quotes.
"""
import json
from pathlib import Path

import pytest

from wobbl.chem import install
from wobbl.chem.generality import ITEMS, run
from wobbl.chem.resolve import resolve
from wobbl.chem.run import plan

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setenv("WOBBL_OFFLINE", "1")


def test_the_list_is_varied():
    kinds = {what for _, _, what in ITEMS}
    assert len(ITEMS) >= 40
    assert kinds >= {"organic", "inorganic", "radical", "ion", "salt", "metal", "noble gas", "biomolecule", "mixture",
                     "reaction", "gas"}


@pytest.mark.parametrize("thing,question,what", ITEMS, ids=[t for t, _, _ in ITEMS])
def test_every_item_resolves_and_plans_offline(thing, question, what):
    if not install.available_pkg("rdkit") and what in ("organic", "biomolecule", "ion") and "(" in thing:
        pytest.skip("reading this SMILES needs RDKit")
    q = resolve(thing)
    assert plan(q, question) in ("properties", "crystal", "reaction", "combustion", "equilibrium", "thermo")


def test_the_published_count_holds_up():
    p = ROOT / "docs" / "chem-generality.json"
    if not p.exists():
        pytest.skip("the generality run has not been published yet")
    d = json.loads(p.read_text())
    assert d["items"] >= 40 and d["succeeded"] / d["items"] >= 0.9


@pytest.mark.slow
@pytest.mark.skipif(not (install.available_pkg("tblite") and install.available()), reason="chemistry not installed")
def test_every_item_runs(tmp_path):
    res = run(tmp_path, log=lambda m: None)
    failed = [r for r in res["rows"] if not r["ok"]]
    assert res["succeeded"] / res["items"] >= 0.9, failed
