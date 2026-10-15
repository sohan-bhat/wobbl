"""The generality check: run the chemistry pipeline over a varied list of substances and reactions it was not
tuned for, and count how many give a checked answer.

    python -m wobbl.chem.generality [out_dir]

Each item runs at fast accuracy (GFN2-xTB only, no images or dynamics), offline from Wobbl's own table, formulas
and SMILES, so the count does not depend on the network. The results go to docs/chem-generality.json.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# (thing, question, what it is)
ITEMS = [
    # organics
    ("caffeine", "", "organic"), ("CCO", "", "organic"), ("c1ccccc1", "", "organic"), ("aspirin", "", "organic"),
    ("acetone", "", "organic"), ("toluene", "", "organic"), ("phenol", "", "organic"), ("urea", "", "organic"),
    ("CC(=O)O", "", "organic"), ("naphthalene", "", "organic"), ("CC(C)CC(C)(C)C", "", "organic"),
    ("C1CCCCC1", "", "organic"),
    # small inorganics and gases
    ("water", "", "inorganic"), ("NH3", "", "inorganic"), ("CO2", "", "inorganic"), ("H2SO4", "", "inorganic"),
    ("SO2", "", "inorganic"), ("O3", "", "inorganic"), ("HCN", "", "inorganic"), ("H2O2", "", "inorganic"),
    ("NO2", "", "radical"), ("oxygen", "", "gas"), ("nitrogen", "", "gas"), ("methane", "", "gas"),
    # ions
    ("NH4+", "", "ion"), ("[OH-]", "", "ion"), ("NO3-", "", "ion"), ("SO4^2-", "", "ion"), ("CC(=O)[O-]", "", "ion"),
    # salts, metals and noble gases
    ("table salt", "", "salt"), ("KCl", "", "salt"), ("MgO", "", "salt"), ("calcium fluoride", "", "salt"),
    ("NaHCO3", "", "salt"), ("copper", "", "metal"), ("gold", "", "metal"), ("aluminium", "", "metal"),
    ("Fe", "", "metal"), ("Ni", "", "metal"), ("argon", "", "noble gas"),
    # biomolecule fragments
    ("glycine", "", "biomolecule"), ("alanine", "", "biomolecule"), ("adenine", "", "biomolecule"),
    ("NCC(=O)NCC(=O)O", "", "biomolecule"), ("OC[C@H]1OC(O)[C@H](O)[C@@H]1O", "", "biomolecule"),
    # mixtures and reactions
    ("methane and air", "", "mixture"), ("hydrogen and oxygen", "", "mixture"),
    ("2 H2 + O2 -> 2 H2O", "", "reaction"), ("baking soda and vinegar", "", "reaction"),
    ("sodium in water", "", "reaction"), ("thermite", "", "reaction"),
    ("N2 + 3 H2 <=> 2 NH3 at 700 K and 200 bar", "", "reaction"), ("HCl + NaOH", "", "reaction"),
    ("CaCO3 -> CaO + CO2", "", "reaction"), ("2 H2O2 -> 2 H2O + O2", "", "reaction"),
    ("photosynthesis", "", "reaction"), ("silver nitrate and sodium chloride", "", "reaction"),
    ("zinc and hydrochloric acid", "", "reaction"), ("burning magnesium", "", "reaction"),
    ("C3H8 + O2 -> CO2 + H2O", "", "reaction"),
]


def run(out_dir: str | Path, items=ITEMS, log=print) -> dict:
    import os
    from .run import simulate
    os.environ.setdefault("WOBBL_OFFLINE", "1")
    out = Path(out_dir)
    rows = []
    for k, (thing, question, what) in enumerate(items):
        t0 = time.time()
        row = {"thing": thing, "question": question, "what": what}
        try:
            o = simulate(thing, question, accuracy="fast", quick=True, render=False, out=out / f"{k:02d}")
            row.update(ok=True, confidence=o.confidence, verdict=o.verdict, headline=o.headline[:200],
                       physics=o.values.get("physics") or "", checks={c.name: c.status for c in o.checks})
        except Exception as e:  # noqa: BLE001  a failure is a result here, counted and explained
            row.update(ok=False, error=f"{type(e).__name__}: {e}"[:300])
        row["seconds"] = round(time.time() - t0, 1)
        rows.append(row)
        log(f"{'ok  ' if row['ok'] else 'FAIL'} {row['seconds']:6.1f}s {thing[:40]:40} "
            f"{row.get('confidence', row.get('error', ''))[:80]}")
    ok = sum(r["ok"] for r in rows)
    return {"items": len(rows), "succeeded": ok, "rows": rows}


if __name__ == "__main__":
    import tempfile
    root = Path(__file__).resolve().parents[3]
    with tempfile.TemporaryDirectory() as t:
        res = run(sys.argv[1] if len(sys.argv) > 1 else t)
    (root / "docs" / "chem-generality.json").write_text(json.dumps(res, indent=1))
    print(f"{res['succeeded']} of {res['items']} succeeded")
