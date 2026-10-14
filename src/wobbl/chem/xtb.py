"""GFN2-xTB and GFN1-xTB through tblite: energies, forces, charges, dipoles, orbitals and bond orders.

Extended tight binding is a semiempirical quantum method parameterized for elements up to radon. It gets
geometries to about 0.01 to 0.02 angstrom, frequencies to about 5 to 10%, and dipoles to about 0.3 D, in a
fraction of a second for a small molecule, which is why every molecule and crystal run starts with it. Solvents
are the ALPB continuum model. Units here are angstrom and eV, converted from tblite's atomic units.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

# tblite parallelizes small matrices badly, and a busy machine makes it worse, so one thread unless asked
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

BOHR = 0.529177210903          # angstrom
HARTREE = 27.211386245988      # eV
DEBYE_PER_EBOHR = 2.541746473
METHOD_NOTE = {"GFN2-xTB": "GFN2-xTB (Bannwarth, Ehlert and Grimme 2019)",
               "GFN1-xTB": "GFN1-xTB (Grimme, Bannwarth and Shushkov 2017)"}
# solvents ALPB knows, by the names people use
SOLVENTS = {"water": "water", "h2o": "water", "methanol": "methanol", "ethanol": "ethanol", "acetone": "acetone",
            "acetonitrile": "acetonitrile", "benzene": "benzene", "toluene": "toluene", "hexane": "hexane",
            "n-hexane": "hexane", "chloroform": "chloroform", "dichloromethane": "ch2cl2", "ch2cl2": "ch2cl2",
            "dmso": "dmso", "dimethyl sulfoxide": "dmso", "dmf": "dmf", "thf": "thf", "tetrahydrofuran": "thf",
            "ether": "ether", "diethyl ether": "ether", "octanol": "octanol", "carbon disulfide": "cs2",
            "ethyl acetate": "ethylacetate", "phenol": "phenol", "aniline": "aniline", "furan": "furane"}


class SCFError(RuntimeError):
    pass


@dataclass
class Result:
    energy: float                       # eV
    forces: np.ndarray                  # eV/angstrom, (n, 3)
    dipole: np.ndarray                  # debye, (3,)
    charges: np.ndarray = field(default_factory=lambda: np.zeros(0))
    orbitals: np.ndarray = field(default_factory=lambda: np.zeros(0))       # eV
    occupations: np.ndarray = field(default_factory=lambda: np.zeros(0))
    bond_orders: np.ndarray | None = None
    stress: np.ndarray | None = None    # eV/angstrom^3, Voigt order, periodic only

    def homo_lumo(self) -> tuple[float | None, float | None]:
        if not len(self.orbitals):
            return None, None
        occ = self.occupations
        filled = np.where(occ > 0.5)[0] if occ.max() > 1.5 else np.where(occ > 0.25)[0]
        homo = self.orbitals[filled.max()] if len(filled) else None
        empty = np.where(occ < (1.0 if occ.max() > 1.5 else 0.5))[0]
        empty = empty[empty > (filled.max() if len(filled) else -1)]
        lumo = self.orbitals[empty.min()] if len(empty) else None
        return homo, lumo


def tblite():
    from .install import require_pkg
    return require_pkg("tblite")


class XTB:
    """One structure's calculator. compute() takes new positions and reuses the last wavefunction as its guess."""

    def __init__(self, numbers, charge: int = 0, uhf: int = 0, method: str = "GFN2-xTB", solvent: str = "",
                 cell: np.ndarray | None = None, pbc=None, etemp: float = 300.0, accuracy: float = 1.0):
        tblite()
        self.numbers = np.asarray(numbers, int)
        self.charge, self.uhf, self.method = int(charge), int(uhf), method
        self.solvent = SOLVENTS.get(solvent.strip().lower(), solvent.strip().lower()) if solvent else ""
        self.cell = None if cell is None else np.asarray(cell, float)
        self.pbc = None if pbc is None else np.asarray(pbc, bool)
        self.etemp, self.accuracy = etemp, accuracy
        self._calc = None
        self._res = None
        self.calls = 0

    @property
    def label(self) -> str:
        return self.method + (f" in {self.solvent} (ALPB)" if self.solvent else "")

    def _build(self, pos_bohr):
        from tblite.interface import Calculator
        lat = None if self.cell is None else self.cell / BOHR
        calc = Calculator(self.method, self.numbers, pos_bohr, charge=float(self.charge), uhf=self.uhf,
                          lattice=lat, periodic=self.pbc)
        calc.set("verbosity", 0)
        calc.set("accuracy", self.accuracy)
        calc.set("temperature", self.etemp * 3.166808578545117e-06)    # kelvin to hartree
        calc.set("max-iter", 300)
        if self.solvent:
            calc.add("alpb-solvation", self.solvent)
        if self.cell is None:
            calc.add("molecular-multipoles")
            calc.add("bond-orders")
        return calc

    def compute(self, positions, cell=None) -> Result:
        pos = np.asarray(positions, float) / BOHR
        if cell is not None:
            self.cell = np.asarray(cell, float)
            self._calc = None
        if self._calc is None:
            self._calc = self._build(pos)
            self._res = None
        else:
            self._calc.update(pos, None if self.cell is None else self.cell / BOHR)
        try:
            res = self._calc.singlepoint(self._res) if self._res is not None else self._calc.singlepoint()
        except Exception as e:  # noqa: BLE001  tblite raises its own error types
            self._res = None
            raise SCFError(f"{self.method} did not converge ({str(e).strip()[:160]})") from None
        self._res = res
        self.calls += 1
        g = res.get("gradient")
        out = Result(energy=float(res.get("energy")) * HARTREE, forces=-np.asarray(g) * HARTREE / BOHR,
                     dipole=np.asarray(res.get("dipole")) * DEBYE_PER_EBOHR, charges=np.asarray(res.get("charges")),
                     orbitals=np.asarray(res.get("orbital-energies")) * HARTREE,
                     occupations=np.asarray(res.get("orbital-occupations")))
        try:
            out.bond_orders = np.asarray(res.get("bond-orders"))
        except Exception:  # noqa: BLE001
            out.bond_orders = None
        if self.cell is not None:
            vir = np.asarray(res.get("virial")) * HARTREE
            vol = abs(np.linalg.det(self.cell))
            s = vir / vol
            out.stress = np.array([s[0, 0], s[1, 1], s[2, 2], s[1, 2], s[0, 2], s[0, 1]])
        return out

    def ase(self):
        """The same calculator as an ASE calculator, for optimizers, dynamics and reaction paths."""
        return ase_calculator(self)


def ase_calculator(engine):
    from .install import require_pkg
    require_pkg("ase")
    from ase.calculators.calculator import Calculator, all_changes

    class _Wrap(Calculator):
        implemented_properties = ["energy", "forces", "stress", "dipole"]

        def __init__(self):
            super().__init__()
            self.engine = engine

        def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            cell = np.array(self.atoms.cell) if self.atoms.pbc.any() else None
            r = self.engine.compute(self.atoms.get_positions(), cell)
            self.results = {"energy": r.energy, "free_energy": r.energy, "forces": r.forces,
                            "dipole": r.dipole / 4.80320471}    # ASE keeps dipoles in e*angstrom
            if r.stress is not None:
                self.results["stress"] = r.stress
            self.last = r

    return _Wrap()
