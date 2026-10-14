"""Density functional theory through PySCF, the higher rung of the method ladder for small molecules.

B3LYP with the def2-SVP basis (normal accuracy) or def2-TZVP (high accuracy), with density fitting for speed,
restricted for closed shells and unrestricted otherwise. It gets bond lengths to about 0.01 angstrom and scaled
frequencies to about 3 to 4%, and costs minutes where GFN2-xTB costs seconds, so Wobbl uses it to check GFN2-xTB
on molecules small enough and says how far the two methods differ.
"""
from __future__ import annotations

import numpy as np

from .xtb import BOHR, HARTREE, Result, SCFError

BASIS = {"normal": "def2-svp", "high": "def2-tzvp"}


def pyscf():
    from .install import require_pkg
    return require_pkg("pyscf")


class DFT:
    def __init__(self, symbols: list[str], charge: int = 0, multiplicity: int = 1, accuracy: str = "normal",
                 xc: str = "b3lyp", solvent: str = ""):
        pyscf()
        self.symbols = list(symbols)
        self.charge, self.spin = int(charge), max(int(multiplicity) - 1, 0)
        self.basis = BASIS.get(accuracy, accuracy)
        self.xc = xc
        self.solvent = solvent
        self._dm = None
        self.calls = 0
        self.converged = True

    @property
    def label(self) -> str:
        return f"{self.xc.upper()}/{self.basis.replace('def2-svp', 'def2-SVP').replace('def2-tzvp', 'def2-TZVP')}"

    def _mol(self, positions):
        from pyscf import gto
        atoms = [(s, tuple(p)) for s, p in zip(self.symbols, np.asarray(positions, float))]
        return gto.M(atom=atoms, basis=self.basis, charge=self.charge, spin=self.spin, unit="Angstrom", verbose=0)

    def compute(self, positions, cell=None) -> Result:
        from pyscf import dft
        mol = self._mol(positions)
        mf = (dft.RKS(mol) if self.spin == 0 else dft.UKS(mol)).density_fit()
        mf.xc = self.xc
        mf.conv_tol = 1e-9
        mf.max_cycle = 150
        if self.solvent:
            from pyscf import solvent as _solv
            mf = _solv.PCM(mf)
            mf.with_solvent.eps = 78.36 if self.solvent == "water" else 20.0
        e = mf.kernel(dm0=self._dm) if self._dm is not None else mf.kernel()
        if not mf.converged:
            mf = mf.newton()
            e = mf.kernel(dm0=mf.make_rdm1())
        if not mf.converged:
            self.converged = False
            raise SCFError(f"{self.label} did not converge")
        self._dm = mf.make_rdm1()
        g = mf.nuc_grad_method().kernel()
        self.calls += 1
        occ = np.asarray(mf.mo_occ)
        ens = np.asarray(mf.mo_energy)
        if occ.ndim == 2:          # unrestricted: put both spins in one list, each occupation out of one
            ens, occ = np.concatenate(ens), np.concatenate(occ)
            order = np.argsort(ens)
            ens, occ = ens[order], occ[order]
        dip = np.asarray(mf.dip_moment(unit="Debye", verbose=0))
        try:
            pop = mf.mulliken_pop(verbose=0)[1]
        except Exception:  # noqa: BLE001
            pop = np.zeros(len(self.symbols))
        self.mf = mf
        return Result(energy=float(e) * HARTREE, forces=-np.asarray(g) * HARTREE / BOHR, dipole=dip,
                      charges=np.asarray(pop), orbitals=ens * HARTREE, occupations=occ)

    def ase(self):
        from .xtb import ase_calculator
        return ase_calculator(self)
