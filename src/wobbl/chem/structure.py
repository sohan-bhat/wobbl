"""Three dimensional structures: from SMILES through RDKit, from PubChem's 3D records, from files, or one atom.

RDKit embeds a molecule with ETKDG distance geometry and relaxes each conformer with the MMFF94 force field
(UFF when MMFF lacks parameters), which gives sensible starting geometries for the quantum method that refines
them. Spin comes from unpaired electrons, RDKit's radicals for molecules and Hund's rule for atoms, unless the
substance says otherwise (oxygen is a triplet).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import formula as F


class StructureError(ValueError):
    def __init__(self, msg: str, hint: str = ""):
        super().__init__(msg)
        self.hint = hint


@dataclass
class Structure:
    symbols: list[str]
    positions: np.ndarray                     # angstrom
    bonds: list[tuple[int, int, float]] = field(default_factory=list)
    charge: int = 0
    multiplicity: int = 1
    source: str = ""
    conformers: list[tuple[float, np.ndarray]] = field(default_factory=list)   # (MMFF kcal/mol, positions)
    smiles: str = ""

    @property
    def numbers(self) -> list[int]:
        return [F.element(s)["z"] for s in self.symbols]

    @property
    def masses(self) -> np.ndarray:
        return np.array([F.mass(s) for s in self.symbols])

    @property
    def uhf(self) -> int:
        return max(self.multiplicity - 1, 0)

    def heavy(self) -> int:
        return sum(1 for s in self.symbols if s != "H")

    def xyz(self, comment: str = "") -> str:
        lines = [str(len(self.symbols)), comment]
        lines += [f"{s:2} {x:12.6f} {y:12.6f} {z:12.6f}" for s, (x, y, z) in zip(self.symbols, self.positions)]
        return "\n".join(lines) + "\n"


def _rdkit():
    from .install import require_pkg
    require_pkg("rdkit")
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    return Chem


def unpaired_atom(symbol: str) -> int:
    """Unpaired electrons of a free atom in its ground state, from its electron configuration."""
    cfg = F.element(symbol).get("config") or ""
    tail = re.findall(r"(\d)([spdf])(\d+)", cfg.replace(" ", ""))
    if not tail:
        return 0
    last = {}
    for n, l, occ in tail:
        last[(n, l)] = int(occ)
    unpaired = 0
    for (n, l), occ in last.items():
        orbitals = {"s": 1, "p": 3, "d": 5, "f": 7}[l]
        unpaired += occ if occ <= orbitals else 2 * orbitals - occ
    return unpaired


def from_species(sp, conformers: int = 20, seed: int = 7) -> Structure:
    """A structure and its conformers for a resolved species."""
    if sp.kind in ("element", "metal") and len(sp.counts) == 1 and sum(sp.counts.values()) == 1:
        sym = next(iter(sp.counts))
        return Structure([sym], np.zeros((1, 3)), [], sp.charge, unpaired_atom(sym) + 1, "a single atom")
    if not sp.smiles:
        if sp.cid:
            st = from_pubchem(sp.cid)
            if st:
                return st
        raise StructureError(f"no structure is known for {sp.label()}, only its formula",
                             "Give a SMILES or a name PubChem knows, so Wobbl can build it in 3D.")
    st = from_smiles(sp.smiles, conformers, seed)
    if sp.multiplicity:
        st.multiplicity = sp.multiplicity
    return st


def from_smiles(smiles: str, conformers: int = 20, seed: int = 7) -> Structure:
    Chem = _rdkit()
    from rdkit.Chem import AllChem
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        raise StructureError(f"RDKit could not read the SMILES '{smiles}'", "Check the SMILES, or give a name.")
    m = Chem.AddHs(m)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    params.pruneRmsThresh = 0.25
    params.useRandomCoords = m.GetNumAtoms() > 40
    ids = list(AllChem.EmbedMultipleConfs(m, numConfs=max(conformers, 1), params=params))
    if not ids:
        params.useRandomCoords = True
        ids = list(AllChem.EmbedMultipleConfs(m, numConfs=max(conformers, 1), params=params))
    if not ids:
        raise StructureError(f"RDKit could not place {smiles} in 3D", "Try a name, so PubChem's 3D record is used.")
    energies = [0.0] * len(ids)
    how = "ETKDG distance geometry"
    try:
        if AllChem.MMFFHasAllMoleculeParams(m):
            res = AllChem.MMFFOptimizeMoleculeConfs(m, maxIters=2000)
            energies = [e for _, e in res]
            how += " relaxed with MMFF94"
        else:
            res = AllChem.UFFOptimizeMoleculeConfs(m, maxIters=2000)
            energies = [e for _, e in res]
            how += " relaxed with UFF"
    except Exception:  # noqa: BLE001  a force field that fails leaves the embedded coordinates
        pass
    confs = sorted(((energies[k], m.GetConformer(cid).GetPositions()) for k, cid in enumerate(ids)),
                   key=lambda c: c[0])
    symbols = [a.GetSymbol() for a in m.GetAtoms()]
    bonds = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx(), b.GetBondTypeAsDouble()) for b in m.GetBonds()]
    mult = _radicals(m) + 1
    return Structure(symbols, np.array(confs[0][1]), bonds, Chem.GetFormalCharge(m), mult,
                     f"RDKit {how}, the lowest of {len(confs)} conformers", confs, smiles)


def from_pubchem(cid: int) -> Structure | None:
    from . import pubchem
    try:
        sdf = pubchem.sdf3d(cid)
    except pubchem.Offline:
        return None
    if not sdf:
        return None
    Chem = _rdkit()
    m = Chem.MolFromMolBlock(sdf, removeHs=False)
    if m is None:
        return None
    return Structure([a.GetSymbol() for a in m.GetAtoms()], m.GetConformer().GetPositions(),
                     [(b.GetBeginAtomIdx(), b.GetEndAtomIdx(), b.GetBondTypeAsDouble()) for b in m.GetBonds()],
                     Chem.GetFormalCharge(m), _radicals(m) + 1, f"PubChem's 3D record of CID {cid}")


def _radicals(m) -> int:
    return sum(a.GetNumRadicalElectrons() for a in m.GetAtoms())


def from_file(path: str | Path) -> Structure:
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix in (".mol", ".sdf"):
        Chem = _rdkit()
        m = Chem.MolFromMolFile(str(p), removeHs=False)
        if m is None:
            raise StructureError(f"could not read {p.name}")
        return Structure([a.GetSymbol() for a in m.GetAtoms()], m.GetConformer().GetPositions(),
                         [(b.GetBeginAtomIdx(), b.GetEndAtomIdx(), b.GetBondTypeAsDouble()) for b in m.GetBonds()],
                         Chem.GetFormalCharge(m), 1, f"the file {p.name}")
    from .install import require_pkg
    require_pkg("ase")
    from ase.io import read
    a = read(str(p))
    syms = a.get_chemical_symbols()
    pos = a.get_positions()
    return Structure(syms, pos, perceive_bonds(syms, pos), 0, _guess_multiplicity(syms, 0), f"the file {p.name}")


def perceive_bonds(symbols: list[str], positions: np.ndarray, scale: float = 1.2) -> list[tuple[int, int, float]]:
    """Bonds between atoms closer than the sum of their covalent radii times a margin."""
    from .formula import element
    rad = []
    for s in symbols:
        r = COVALENT.get(s, (element(s).get("radius_pm") or 150) / 100 * 0.6)
        rad.append(r)
    out = []
    n = len(symbols)
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(positions[i] - positions[j])
            if d < scale * (rad[i] + rad[j]):
                out.append((i, j, 1.0))
    return out


def _guess_multiplicity(symbols: list[str], charge: int) -> int:
    ne = sum(F.element(s)["z"] for s in symbols) - charge
    return 1 if ne % 2 == 0 else 2


# covalent radii in angstrom (Cordero et al., Dalton Trans. 2008) for the common elements
COVALENT = {"H": 0.31, "He": 0.28, "Li": 1.28, "Be": 0.96, "B": 0.84, "C": 0.76, "N": 0.71, "O": 0.66, "F": 0.57,
            "Ne": 0.58, "Na": 1.66, "Mg": 1.41, "Al": 1.21, "Si": 1.11, "P": 1.07, "S": 1.05, "Cl": 1.02, "Ar": 1.06,
            "K": 2.03, "Ca": 1.76, "Ti": 1.60, "Cr": 1.39, "Mn": 1.39, "Fe": 1.32, "Co": 1.26, "Ni": 1.24,
            "Cu": 1.32, "Zn": 1.22, "Ga": 1.22, "Ge": 1.20, "As": 1.19, "Se": 1.20, "Br": 1.20, "Kr": 1.16,
            "Ag": 1.45, "Sn": 1.39, "I": 1.39, "Xe": 1.40, "Pt": 1.36, "Au": 1.36, "Hg": 1.32, "Pb": 1.46}
