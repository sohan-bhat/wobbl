"""Elements and crystals: atomic data, crystal structure, lattice constant, density, cohesive energy and bulk
modulus, and their atoms moving at a temperature.

The structure comes from the element's reference state (ASE's table of measured structures), from a crystal
prototype for compounds Wobbl knows (rock salt, zinc blende, fluorite, rutile, corundum and the rest), or from a CIF
file. The energy comes from the method that fits the material: effective medium theory for the fcc metals it was
fitted to (Al, Ni, Cu, Pd, Ag, Pt, Au), Lennard-Jones for the noble gases, and periodic DFT (PBE with GTH
pseudopotentials through PySCF, with k-points) for everything else. The energy at nine volumes around the measured
one gives a Birch-Murnaghan equation of state, and from it the lattice constant, density and bulk modulus. The
cohesive energy is the isolated atoms' energy less the crystal's, per atom.
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np

from ..outcome_generic import GenericOutcome
from ..units import fmt
from ..verify.checks import Check, confidence
from . import formula as F
from .resolve import Query, ResolveError

EV_A3_GPA = 160.21766208
EMT_METALS = {"Al", "Ni", "Cu", "Pd", "Ag", "Pt", "Au"}
# Lennard-Jones well depth (K) and size (A) for the noble gases, from second virial coefficients (Hirschfelder,
# Curtiss and Bird)
LJ = {"Ne": (35.6, 2.75), "Ar": (119.8, 3.405), "Kr": (171.0, 3.60), "Xe": (221.0, 4.10), "He": (10.22, 2.556)}
# measured bulk modulus (GPa) and cohesive energy (eV/atom) at low temperature, Kittel's Introduction to Solid State
# Physics, chapter 3
MEASURED = {"Cu": (137, 3.49), "Al": (72.2, 3.39), "Ag": (100.7, 2.95), "Au": (173.2, 3.81), "Ni": (186, 4.44),
            "Pd": (180.8, 3.89), "Pt": (278.3, 5.84), "Fe": (168.3, 4.28), "W": (323.2, 8.90), "Si": (98.8, 4.63),
            "Ge": (77.2, 3.85), "C": (442, 7.37), "Na": (6.8, 1.113), "K": (3.2, 0.934), "Li": (11.6, 1.63),
            "Mg": (35.4, 1.51), "Zn": (59.8, 1.35), "Ti": (105.1, 4.85), "Cr": (190.1, 4.10), "Mo": (272.5, 6.82),
            "Pb": (43.0, 2.03), "Ar": (2.7, 0.080), "Ne": (1.1, 0.020), "Kr": (3.4, 0.116), "Xe": (3.6, 0.16),
            "ClNa": (24.0, None), "ClK": (17.4, None), "MgO": (160, None), "FLi": (67, None)}


def _ase():
    from .install import require_pkg
    return require_pkg("ase")


# ------------------------------------------------------------------------------------------- structures
def build(q: Query):
    """(ase Atoms in its conventional cell, description, measured lattice constant or None, the species)."""
    _ase()
    from ase.build import bulk
    from ase.data import atomic_numbers, reference_states
    if q.path is not None:
        from ase.io import read
        a = read(str(q.path))
        return a, f"the file {q.path.name}", None, None
    sp = q.main
    if sp is None:
        raise ResolveError("no solid named")
    if sp.kind == "material":
        sp = _base_metal(sp)
    proto = sp.crystal
    if len(sp.counts) == 1:
        sym = next(iter(sp.counts))
        ref = reference_states[atomic_numbers[sym]]
        if ref is None:
            raise ResolveError(f"no measured crystal structure for {sym}")
        sym_struct = proto or ref.get("symmetry")
        a = ref.get("a")
        if sym_struct in ("fcc", "bcc", "sc", "diamond"):
            return bulk(sym, sym_struct, a=a, cubic=True), f"{sym_struct} {sp.name}", a, sp
        if sym_struct == "hcp":
            ca = ref.get("c/a", 1.633)
            return bulk(sym, "hcp", a=a, c=a * ca), f"hcp {sp.name} with c/a {ca:.3f}", a, sp
        raise ResolveError(f"{sp.name}'s solid is {sym_struct}, a structure Wobbl does not build yet",
                           "Give a CIF file of it instead.")
    if not proto:
        return None, "", None, sp
    a = sp.lattice
    ions = _formula_parts(sp)
    name = "".join(e * int(sp.counts[e]) for e in ions)        # 'CaFF' for fluorite, as ASE's builder wants
    where = f"{sp.name} in the {PRETTY.get(proto, proto)} structure"
    if proto in ("rocksalt", "cesiumchloride", "zincblende", "fluorite"):
        return bulk(name, proto, a=a, cubic=True), where, a, sp
    if proto == "wurtzite":
        return bulk(name, "wurtzite", a=a, c=a * 1.602), where, a, sp
    if proto == "rutile":
        from ase.spacegroup import crystal
        m, x = ions
        return crystal([m, x], [(0, 0, 0), (0.305, 0.305, 0)], spacegroup=136, cellpar=[a, a, a * 0.644, 90, 90, 90]), \
            where, a, sp
    if proto == "corundum":
        from ase.spacegroup import crystal
        m, x = ions
        c_a = 2.73
        return crystal([m, x], [(0, 0, 0.352), (0.306, 0, 0.25)], spacegroup=167,
                       cellpar=[a or 4.76, a or 4.76, (a or 4.76) * c_a, 90, 90, 120]), where, a, sp
    raise ResolveError(f"no builder for the {proto} structure")


PRETTY = {"rocksalt": "rock salt", "cesiumchloride": "cesium chloride", "zincblende": "zinc blende"}
# measured lattice energies from the Born-Haber cycle, kJ/mol (CRC Handbook)
LATTICE_ENERGY = {"ClNa": 787, "ClK": 717, "FLi": 1037, "MgO": 3850, "CaO": 3460, "ClCs": 659, "CaF2": 2651,
                  "AgCl": 915, "BrNa": 747, "INa": 704}


def _formula_parts(sp) -> list[str]:
    """The cation and anion symbols of a binary solid, metal first."""
    syms = sorted(sp.counts, key=lambda s: (F.element(s).get("electronegativity") or 4.0))
    return syms


def _base_metal(sp):
    """An engineering alloy as its base metal, the crystal that sets its lattice."""
    from .resolve import species
    low = sp.name.lower()
    for word, sym in (("alumin", "Al"), ("steel", "Fe"), ("iron", "Fe"), ("copper", "Cu"), ("brass", "Cu"),
                      ("bronze", "Cu"), ("titan", "Ti"), ("nickel", "Ni"), ("inconel", "Ni"), ("magnes", "Mg"),
                      ("zinc", "Zn"), ("tungsten", "W"), ("silver", "Ag"), ("gold", "Au")):
        if word in low:
            base = species(sym, allow_online=False)
            base.note = f"{sp.name} as its base metal, {base.name}"
            return base
    raise ResolveError(f"{sp.name} is an engineering material, not one crystal",
                       "Its strength and stiffness come from the part domain, as wobbl simulate part.step.")


# ------------------------------------------------------------------------------------------- energies
class Engine:
    name = ""
    accuracy: dict = {}

    def energy(self, atoms) -> float:
        raise NotImplementedError

    def atom_energy(self, sym: str) -> float:
        raise NotImplementedError


class EMTEngine(Engine):
    name = "EMT"

    def __init__(self):
        from ase.calculators.emt import EMT
        self.calc = EMT

    def energy(self, atoms) -> float:
        a = atoms.copy()
        a.calc = self.calc()
        return a.get_potential_energy()

    def atom_energy(self, sym: str) -> float:
        from ase import Atoms
        a = Atoms(sym, positions=[[0, 0, 0]], cell=[12, 12, 12], pbc=False)
        a.calc = self.calc()
        return a.get_potential_energy()

    def ase(self):
        return self.calc()


class LJEngine(Engine):
    """Lennard-Jones with the force shifted to zero at 3 sigma, so energy is conserved in dynamics. All pairs by
    the minimum image, which suits the few hundred atoms Wobbl runs."""
    name = "Lennard-Jones"

    def __init__(self, sym: str):
        eps_k, sigma = LJ[sym]
        self.eps = eps_k * 8.617333262e-5
        self.sigma = sigma
        self.rc = 3.0 * sigma

    def ase(self):
        from ase.calculators.calculator import Calculator, all_changes
        eng = self

        class _LJ(Calculator):
            implemented_properties = ["energy", "free_energy", "forces"]

            def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
                super().calculate(atoms, properties, system_changes)
                e, f = eng.pairs(self.atoms.get_positions(), np.array(self.atoms.cell))
                self.results = {"energy": e, "free_energy": e, "forces": f}
        return _LJ()

    def pairs(self, pos: np.ndarray, cell: np.ndarray) -> tuple[float, np.ndarray]:
        inv = np.linalg.inv(cell)
        d = pos[:, None, :] - pos[None, :, :]
        frac = d @ inv
        d = (frac - np.round(frac)) @ cell
        r2 = (d ** 2).sum(-1)
        n = len(pos)
        iu = np.triu_indices(n, 1)
        rr = np.sqrt(r2[iu])
        mask = rr < self.rc
        rr, dv = rr[mask], d[iu][mask]
        s6 = (self.sigma / rr) ** 6
        sc6 = (self.sigma / self.rc) ** 6
        v = 4 * self.eps * (s6 * s6 - s6)
        vc = 4 * self.eps * (sc6 * sc6 - sc6)
        dv_dr = 4 * self.eps * (-12 * s6 * s6 + 6 * s6) / rr
        dvc = 4 * self.eps * (-12 * sc6 * sc6 + 6 * sc6) / self.rc
        e = float(np.sum(v - vc - (rr - self.rc) * dvc))
        fmag = -(dv_dr - dvc)
        fv = (fmag / rr)[:, None] * dv
        f = np.zeros_like(pos)
        i, j = iu[0][mask], iu[1][mask]
        np.add.at(f, i, fv)
        np.add.at(f, j, -fv)
        return e, f

    def energy(self, atoms) -> float:
        a = atoms
        reps = max(1, int(np.ceil(2 * self.rc / min(np.linalg.norm(atoms.cell, axis=1)))))
        if reps > 1:             # the minimum image needs a box at least twice the cutoff
            a = atoms.repeat((reps, reps, reps))
        e, _ = self.pairs(a.get_positions(), np.array(a.cell))
        return e * len(atoms) / len(a)

    def atom_energy(self, sym: str) -> float:
        return 0.0


# Shannon's ionic radii (Acta Cryst. A32, 751, 1976), angstrom, for six fold coordination unless the key says
SHANNON = {("Li", 1): 0.76, ("Na", 1): 1.02, ("K", 1): 1.38, ("Rb", 1): 1.52, ("Cs", 1): 1.67, ("Ag", 1): 1.15,
           ("Mg", 2): 0.72, ("Ca", 2): 1.00, ("Sr", 2): 1.18, ("Ba", 2): 1.35, ("Ni", 2): 0.69, ("Mn", 2): 0.83,
           ("Fe", 2): 0.78, ("Co", 2): 0.745, ("Cd", 2): 0.95, ("Pb", 2): 1.19, ("F", -1): 1.33, ("Cl", -1): 1.81,
           ("Br", -1): 1.96, ("I", -1): 2.20, ("O", -2): 1.40, ("S", -2): 1.84, ("Se", -2): 1.98,
           ("Cs", 1, 8): 1.74, ("Ca", 2, 8): 1.12, ("Sr", 2, 8): 1.26, ("Ba", 2, 8): 1.42, ("Cl", -1, 8): 1.81,
           ("Br", -1, 8): 1.96, ("F", -1, 4): 1.31, ("O", -2, 4): 1.38}
COORDINATION = {"rocksalt": (6, 6), "cesiumchloride": (8, 8), "fluorite": (8, 4)}
COULOMB = 14.399645          # e^2/(4 pi eps0), eV A
RHO = 0.345                  # A, the Born-Mayer hardness most ionic crystals share


def ewald(positions: np.ndarray, charges: np.ndarray, cell: np.ndarray, accuracy: float = 1e-8) -> float:
    """Coulomb energy of a periodic array of point charges, eV, by Ewald summation."""
    vol = abs(np.linalg.det(cell))
    n = len(charges)
    alpha = np.sqrt(np.pi) * (n / vol ** 2) ** (1 / 6)
    rcut = np.sqrt(-np.log(accuracy)) / alpha
    kcut = 2 * alpha * np.sqrt(-np.log(accuracy))
    from math import erfc
    reps = [int(np.ceil(rcut / np.linalg.norm(cell[i]))) + 1 for i in range(3)]
    e_real = 0.0
    shifts = np.array([(i, j, k) for i in range(-reps[0], reps[0] + 1) for j in range(-reps[1], reps[1] + 1)
                       for k in range(-reps[2], reps[2] + 1)]) @ cell
    for a in range(n):
        d = positions[a] - positions[None, :, :] + shifts[:, None, :]
        r = np.linalg.norm(d, axis=2)
        mask = (r > 1e-10) & (r < rcut)
        e_real += 0.5 * charges[a] * np.sum((charges[None, :] * np.vectorize(erfc)(alpha * r) / np.where(mask, r, 1))
                                            [mask])
    rec = 2 * np.pi * np.linalg.inv(cell).T
    kreps = [int(np.ceil(kcut / np.linalg.norm(rec[i]))) for i in range(3)]
    e_rec = 0.0
    for i in range(-kreps[0], kreps[0] + 1):
        for j in range(-kreps[1], kreps[1] + 1):
            for k in range(-kreps[2], kreps[2] + 1):
                if i == j == k == 0:
                    continue
                kv = i * rec[0] + j * rec[1] + k * rec[2]
                k2 = kv @ kv
                if k2 > kcut ** 2:
                    continue
                sk = np.sum(charges * np.exp(1j * positions @ kv))
                e_rec += np.exp(-k2 / (4 * alpha ** 2)) / k2 * abs(sk) ** 2
    e_rec *= 2 * np.pi / vol
    e_self = -alpha / np.sqrt(np.pi) * np.sum(charges ** 2)
    return float((e_real + e_rec + e_self) * COULOMB)


class IonicEngine(Engine):
    """The Born-Mayer ionic model: formal charges summed by Ewald, and a repulsion between touching cation and anion
    that falls off as exp(-r/0.345 A), its strength set so the ions touch at the sum of Shannon's radii."""
    name = "Born-Mayer ionic model"

    def __init__(self, atoms, proto: str, charges: dict):
        self.charges = charges
        cn = COORDINATION[proto]
        syms = atoms.get_chemical_symbols()
        cat = next(s for s in syms if charges[s] > 0)
        an = next(s for s in syms if charges[s] < 0)
        rc = SHANNON.get((cat, charges[cat], cn[0])) or SHANNON.get((cat, charges[cat]))
        ra = SHANNON.get((an, charges[an], cn[1])) or SHANNON.get((an, charges[an]))
        if rc is None or ra is None:
            raise ResolveError(f"no ionic radius for {cat} or {an}")
        self.r0 = rc + ra
        self.cat, self.an = cat, an
        q = np.array([charges[s] for s in syms], float)
        pos = atoms.get_positions()
        d = self._nearest(atoms)
        s0 = self.r0 / d             # the cell scale where the ions touch
        self.ref = atoms.copy()
        self.ref.set_cell(np.array(atoms.cell) * s0, scale_atoms=True)
        h = 1e-4
        e = [ewald(self.ref.get_positions() * (1 + x), q, np.array(self.ref.cell) * (1 + x)) for x in (-h, h)]
        r = [self._rep(self.ref, 1 + x, 1.0) for x in (-h, h)]
        self.b = -(e[1] - e[0]) / (r[1] - r[0])
        self.q = q
        _ = pos

    def _nearest(self, atoms) -> float:
        syms = atoms.get_chemical_symbols()
        d = atoms.get_all_distances(mic=True)
        return float(min(d[i, j] for i in range(len(syms)) for j in range(len(syms))
                         if syms[i] == self.cat and syms[j] == self.an))

    def _rep(self, atoms, scale: float, b: float) -> float:
        a = atoms.copy()
        a.set_cell(np.array(atoms.cell) * scale, scale_atoms=True)
        syms = a.get_chemical_symbols()
        d = a.get_all_distances(mic=True)
        tot = 0.0
        for i in range(len(syms)):
            for j in range(len(syms)):
                if syms[i] == self.cat and syms[j] == self.an and d[i, j] < 1.3 * self.r0 * scale:
                    tot += np.exp(-d[i, j] / RHO)
        return b * tot

    def energy(self, atoms) -> float:
        s = (atoms.get_volume() / self.ref.get_volume()) ** (1 / 3)
        return ewald(self.ref.get_positions() * s, self.q, np.array(self.ref.cell) * s) + self._rep(self.ref, s,
                                                                                                     self.b)

    def atom_energy(self, sym: str) -> float:
        return 0.0        # free ions, so the cohesive energy here is the lattice energy per ion


class DFTEngine(Engine):
    """PBE with GTH pseudopotentials and plane wave quality Gaussian basis sets through PySCF, with a k-point
    mesh. Slow, and the honest choice for solids no simpler method was fitted to."""

    def __init__(self, accuracy: str = "normal"):
        from .install import require_pkg
        require_pkg("pyscf")
        self.basis = "gth-dzvp" if accuracy == "high" else "gth-szv"
        self.kmesh = 3 if accuracy == "high" else 2
        self.name = f"PBE/{self.basis.upper().replace('GTH-', 'GTH-')} with {self.kmesh}x{self.kmesh}x{self.kmesh} " \
                    "k-points"

    def _cell(self, atoms):
        from pyscf.pbc import gto
        c = gto.Cell()
        c.atom = [(s, tuple(p)) for s, p in zip(atoms.get_chemical_symbols(), atoms.get_positions())]
        c.a = np.array(atoms.cell)
        c.basis = self.basis
        c.pseudo = "gth-pbe"
        c.verbose = 0
        c.spin = int(sum(F.element(s)["z"] for s in atoms.get_chemical_symbols()) % 2)
        c.build()
        return c

    def energy(self, atoms) -> float:
        from pyscf.pbc import dft
        cell = self._cell(atoms)
        mf = (dft.KRKS if cell.spin == 0 else dft.KUKS)(cell, cell.make_kpts([self.kmesh] * 3)).density_fit()
        mf.xc = "pbe"
        mf.smearing = None
        e = mf.kernel()
        if not mf.converged:
            raise RuntimeError("the periodic SCF did not converge")
        return float(e) * 27.211386245988

    def atom_energy(self, sym: str) -> float:
        from pyscf.pbc import dft, gto
        c = gto.Cell()
        c.atom = [(sym, (0, 0, 0))]
        c.a = np.eye(3) * 10.0
        c.basis = self.basis
        c.pseudo = "gth-pbe"
        c.verbose = 0
        from .structure import unpaired_atom
        c.spin = unpaired_atom(sym)
        c.build()
        mf = dft.UKS(c).density_fit()
        mf.xc = "pbe"
        return float(mf.kernel()) * 27.211386245988


def engine_for(symbols: set[str], accuracy: str, atoms=None, sp=None, desc: str = "") -> Engine:
    if symbols <= EMT_METALS:
        return EMTEngine()
    if len(symbols) == 1 and next(iter(symbols)) in LJ:
        return LJEngine(next(iter(symbols)))
    proto = getattr(sp, "crystal", "") if sp is not None else ""
    if proto in COORDINATION and atoms is not None and accuracy != "high":
        from .reaction import ions_of
        ions = ions_of(sp)
        if ions:
            try:
                return IonicEngine(atoms, proto, {ions[0][1]: ions[0][2], ions[1][1]: ions[1][2]})
            except ResolveError:
                pass
    if accuracy == "high":
        return DFTEngine(accuracy)
    raise ResolveError(f"{desc or 'this solid'} has no fitted potential in Wobbl, only periodic DFT, which takes "
                       "tens of minutes", "Run it with --accuracy high to compute it with PBE in PySCF, or ask about "
                       "an EMT metal (Al, Ni, Cu, Pd, Ag, Pt, Au), a noble gas or an ionic crystal.")


# ------------------------------------------------------------------------------------------- the run
def run_material(q: Query, phys: str, opts, out_dir, log=None, options: dict | None = None) -> GenericOutcome:
    t0 = time.time()
    log = log or (lambda stage, msg="": None)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    options = options or {}
    atoms, desc, a_ref, sp = build(q)
    if atoms is None:
        return _salt(q, sp, opts, out, t0)
    syms = set(atoms.get_chemical_symbols())
    try:
        eng = engine_for(syms, opts.accuracy, atoms, sp, desc)
    except ResolveError as e:
        return _measured_only(q, sp, atoms, desc, a_ref, opts, out, t0, str(e))
    values: dict = {"structure": desc, "method": eng.name, "atoms_in_cell": len(atoms),
                    "atomic": _atomic(syms)}
    checks: list[Check] = []
    if phys == "dynamics":
        return _dynamics_run(q, atoms, desc, eng, opts, out, log, t0, options)
    if opts.accuracy == "fast" and isinstance(eng, DFTEngine):
        raise ResolveError(f"{desc} needs periodic DFT, which fast accuracy skips",
                           "Run it at normal accuracy, which takes a few minutes for a small cell.")
    log("equation of state", f"{desc} with {eng.name}")
    n_pts = 9 if not isinstance(eng, DFTEngine) else 5
    scales = np.linspace(0.94, 1.06, n_pts) if not isinstance(eng, DFTEngine) else np.linspace(0.95, 1.05, n_pts)
    V, E, frames = [], [], []
    for s in scales:
        b = atoms.copy()
        b.set_cell(np.array(atoms.cell) * s, scale_atoms=True)
        e = eng.energy(b)
        V.append(b.get_volume())
        E.append(e)
        frames.append(b)
    from ase.eos import EquationOfState
    eos = EquationOfState(V, E, eos="birchmurnaghan")
    v0, e0, B = eos.fit()
    s0 = (v0 / atoms.get_volume()) ** (1 / 3)
    a_conv = float(np.linalg.norm(atoms.cell[0])) * s0
    if not atoms.cell.orthorhombic and atoms.cell.cellpar()[5] == 60.0:
        a_conv = a_conv * math.sqrt(2)
    mass = sum(F.mass(s) for s in atoms.get_chemical_symbols()) * 1.66053906660e-24
    density = mass / (v0 * 1e-24)
    n = len(atoms)
    e_atoms = sum(eng.atom_energy(s) for s in atoms.get_chemical_symbols())
    coh = (e_atoms - e0) / n
    i_min = int(np.argmin(E))
    inside = 0 < i_min < len(E) - 1
    resid = float(np.sqrt(np.mean((np.array(E) - eos.func(np.array(V), *eos.eos_parameters)) ** 2)))
    checks.append(Check("equation of state", "pass" if inside and resid < 1e-3 * n else "warn",
                        f"{len(V)} volumes from {100 * (scales[0] ** 3 - 1):+.0f}% to "
                        f"{100 * (scales[-1] ** 3 - 1):+.0f}% "
                        f"around the measured one bracket the minimum, and the Birch-Murnaghan fit misses them by "
                        f"{resid * 1000:.2f} meV" if inside else
                        "the lowest energy sits at the edge of the volumes tried, so the minimum is extrapolated",
                        {"rms_eV": resid}))
    values.update({"lattice_A": a_conv, "volume_A3": v0, "density_g_cm3": density, "bulk_modulus_GPa": B * EV_A3_GPA,
                   "cohesive_eV_atom": coh, "energy_eV": e0, "eos": {"V": V, "E": E}})
    measured = _measured(sp, q, a_ref)
    if measured:
        checks.append(_reference(values, measured, eng))
        values["measured"] = measured
    level, reasons = confidence([c for c in checks if c.status != "info"])
    name = sp.name if sp is not None else desc
    ionic = isinstance(eng, IonicEngine)
    if ionic:
        per_fu = (e_atoms - e0) / (n / sum(sp.counts.values())) * 96.485
        values["lattice_energy_kJ_mol"] = per_fu
        if sp.formula in LATTICE_ENERGY:
            measured_le = LATTICE_ENERGY[sp.formula]
            values["measured_lattice_energy_kJ_mol"] = measured_le
            err = abs(per_fu - measured_le) / measured_le
            checks.append(Check("lattice energy", "pass" if err < 0.08 else "warn",
                                f"the ionic model's lattice energy is {per_fu:.0f} kJ/mol against {measured_le} "
                                f"from the Born-Haber cycle ({100 * err:.1f}% apart)", {"error": err}))
            level, reasons = confidence([c for c in checks if c.status != "info"])
    tail = (f"lattice energy {values['lattice_energy_kJ_mol']:.0f} kJ/mol" if ionic else
            f"cohesive energy {coh:.2f} eV per atom")
    head = (f"{desc[0].upper() + desc[1:]} by {eng.name}: lattice constant {a_conv:.3f} A, density "
            f"{density:.3g} g/cm3, bulk modulus {B * EV_A3_GPA:.0f} GPa and {tail}.")
    if measured.get("lattice_A"):
        head += f" Measured, the lattice constant is {measured['lattice_A']:.3f} A."
    metrics = [("lattice constant", f"{a_conv:.3f} A"), ("density", f"{density:.3g} g/cm3"),
               ("bulk modulus", f"{B * EV_A3_GPA:.0f} GPa"),
               ("lattice energy", f"{values['lattice_energy_kJ_mol']:.0f} kJ/mol") if ionic else
               ("cohesive energy", f"{coh:.2f} eV/atom")]
    imgs = {}
    if opts.render:
        try:
            from .charts import sweep_chart
            imgs["eos"] = sweep_chart(out / "images" / "eos.png", np.array(V) / n, (np.array(E) - e0) / n,
                                      "volume per atom, A^3", "energy per atom above the minimum, eV",
                                      (v0 / n, 0.0))
            values["_figures"] = [["eos", "Energy against volume, the equation of state"]]
        except Exception as e:  # noqa: BLE001
            imgs["error"] = str(e)
    from . import playback
    values["files"] = {"playback": playback.crystal(out, name, atoms, desc, frames, list(V), list(E), values)}
    from .reference import ACCURACY
    acc = ACCURACY.get(eng.name.split()[0], {})
    assumptions = [f"{desc} at 0 K with the measured shape held as the volume changes, by {eng.name}"
                   + (f", good to about {100 * acc['lattice']:.1f}% in lattice constant" if "lattice" in acc else "")
                   + (f" and {100 * acc['bulk']:.0f}% in bulk modulus" if "bulk" in acc else "") + ".",
                   "Zero point motion and thermal expansion are left out, which together make room temperature "
                   "lattices about 0.5% larger."]
    if sp is not None and sp.note:
        assumptions.insert(0, sp.note + ".")
    o = GenericOutcome("chemistry", name, opts.question, "ok" if level != "low" else "marginal", head, metrics,
                       level, reasons, checks, assumptions, values, imgs, solver=f"{eng.name} with ASE",
                       t_total=time.time() - t0)
    return o.write(out, [_atomic_table(values["atomic"])])


def _measured(sp, q, a_ref) -> dict:
    out = {}
    if sp is not None and sp.lattice:
        out["lattice_A"] = sp.lattice
    elif a_ref:
        out["lattice_A"] = a_ref
    key = sp.formula if sp is not None else ""
    if key in MEASURED:
        b, c = MEASURED[key]
        out["bulk_modulus_GPa"] = b
        if c:
            out["cohesive_eV_atom"] = c
    if sp is not None and len(sp.counts) == 1:
        d = F.element(next(iter(sp.counts))).get("density_g_cm3")
        if d and d > 0.5:
            out["density_g_cm3"] = d
    return out


def _reference(v: dict, m: dict, eng) -> Check:
    from .reference import ACCURACY
    acc = ACCURACY.get(eng.name.split()[0], {"lattice": 0.03, "bulk": 0.3, "cohesive": 0.3})
    parts, worst = [], 0.0
    for key, label, unit, tol in (("lattice_A", "lattice constant", "A", acc.get("lattice", 0.03)),
                                  ("bulk_modulus_GPa", "bulk modulus", "GPa", acc.get("bulk", 0.3)),
                                  ("cohesive_eV_atom", "cohesive energy", "eV/atom", acc.get("cohesive", 0.3))):
        if key in m and v.get(key):
            err = abs(v[key] - m[key]) / m[key]
            worst = max(worst, err / tol)
            parts.append(f"{label} {v[key]:.3g} {unit} against {m[key]:.3g} ({100 * err:.1f}% apart)")
    status = "pass" if worst <= 2 else "warn" if worst <= 4 else "fail"
    return Check("reference data", status, "against measurements, " + "; ".join(parts) +
                 f" (the largest miss is {worst:.1f} times {eng.name}'s typical error)", {"worst": worst})


def _atomic(syms: set[str]) -> list[dict]:
    keys = ("symbol", "name", "z", "mass", "config", "electronegativity", "ionization_eV", "electron_affinity_eV",
            "radius_pm", "melting_K", "boiling_K", "density_g_cm3", "block", "oxidation_states")
    return [{k: F.element(s).get(k) for k in keys} for s in sorted(syms, key=lambda s: F.element(s)["z"])]


def _atomic_table(rows: list[dict]) -> str:
    from ..report.html import table
    head = ["", *[r["symbol"] for r in rows]]
    lines = []
    for k, label in (("name", "name"), ("z", "atomic number"), ("mass", "atomic mass"), ("config", "configuration"),
                     ("electronegativity", "electronegativity (Pauling)"), ("ionization_eV", "first ionization, eV"),
                     ("radius_pm", "van der Waals radius, pm"), ("melting_K", "melts, K"), ("boiling_K", "boils, K"),
                     ("block", "group")):
        lines.append([label, *[str(r.get(k) if r.get(k) is not None else "") for r in rows]])
    return "<h2>Atomic data (PubChem, from NIST)</h2>" + table(head, lines)


def _measured_only(q, sp, atoms, desc, a_ref, opts, out, t0, why) -> GenericOutcome:
    """The structure and measured values of a solid no fast method in Wobbl covers, saying so plainly."""
    m = _measured(sp, q, a_ref)
    syms = set(atoms.get_chemical_symbols())
    values = {"structure": desc, "atomic": _atomic(syms), "measured": m, "computed": False}
    mass = sum(F.mass(x) for x in atoms.get_chemical_symbols()) * 1.66053906660e-24
    values["density_g_cm3"] = mass / (atoms.get_volume() * 1e-24)
    parts = [f"lattice constant {m['lattice_A']:.3f} A" if m.get("lattice_A") else "",
             f"bulk modulus {m['bulk_modulus_GPa']:.0f} GPa" if m.get("bulk_modulus_GPa") else "",
             f"cohesive energy {m['cohesive_eV_atom']:.2f} eV per atom" if m.get("cohesive_eV_atom") else ""]
    parts = [x for x in parts if x]
    head = (f"{desc[0].upper() + desc[1:]}: measured " + ", ".join(parts) + "." if parts else f"{desc}.") + \
        f" Wobbl did not compute these, since {why.split(',')[0]}."
    checks = [Check("method", "info", why)]
    from . import playback
    values["files"] = {"playback": playback.crystal(out, sp.name if sp else desc, atoms, desc, None, None, None,
                                                    values)}
    o = GenericOutcome("chemistry", sp.name if sp else desc, opts.question, "ok", head,
                       [(k.replace("_", " "), f"{v:.4g}") for k, v in m.items()], "medium",
                       ["values are measured, not computed"], checks,
                       ["Measured values from ASE's table of reference structures and Kittel's Introduction to Solid "
                        "State Physics."], values, {}, solver="tables", t_total=time.time() - t0)
    return o.write(out, [_atomic_table(values["atomic"])])


def _salt(q, sp, opts, out, t0) -> GenericOutcome:
    """A solid with no structure on file: its tabulated thermochemistry and what dissolving it does."""
    from .reaction import _default_phase, _ion, ion_species, ions_of, soluble
    from .resolve import Term
    from .thermodata import lookup
    d = lookup(sp, "s", opts.temperature)
    values = {"formula": sp.formula, "molar_mass_g_mol": sp.molar_mass, "atomic": _atomic(set(sp.counts))}
    checks = []
    head = f"{sp.name} ({sp.formula}, {sp.molar_mass:.2f} g/mol) has no crystal structure on file in Wobbl."
    if d is not None:
        values.update({"H_formation_kJ_mol": d.h298 / 1000, "S_J_molK": d.s298, "Cp_J_molK": d.cp298,
                       "source": d.source})
        head += (f" Its heat of formation is {d.h298 / 1000:.1f} kJ/mol and its entropy {d.s298:.1f} J/(mol K) "
                 f"({d.source.split(' (')[0]}).")
    ions = ions_of(sp)
    if ions and soluble(ions[0][1], ions[1][1]):
        from .react import thermo
        qq = Query(f"{sp.formula} in water", "reaction")
        qq.reactants = [Term(1.0, sp, "s")]
        qq.products = [Term(ions[0][0], ion_species(_ion(ions[0])), "aq"),
                       Term(ions[1][0], ion_species(_ion(ions[1])), "aq")]
        try:
            th = thermo(qq, opts.temperature)
            values["dissolving"] = {"dH_kJ": th["dH"] / 1000, "dG_kJ": th["dG"] / 1000}
            word = "warms" if th["dH"] < 0 else "cools"
            head += (f" Dissolving it in water takes {th['dH'] / 1000:+.1f} kJ/mol, so it {word} the water, with "
                     f"dG = {th['dG'] / 1000:.1f} kJ/mol.")
            checks.append(Check("data", "pass", "the solid and its ions come from NASA's and the NBS tables"))
        except ResolveError as e:
            checks.append(Check("data", "skip", str(e)))
    _ = _default_phase
    level, reasons = confidence(checks)
    o = GenericOutcome("chemistry", sp.name, opts.question, "ok", head, [("molar mass", f"{sp.molar_mass:.2f} g/mol")],
                       level, reasons, checks, ["Standard state tables at 298.15 K."], values, {},
                       solver="Wobbl thermochemistry tables", t_total=time.time() - t0)
    return o.write(out, [_atomic_table(values["atomic"])])


# ------------------------------------------------------------------------------------------- dynamics
def _dynamics_run(q, atoms, desc, eng, opts, out, log, t0, options) -> GenericOutcome:
    """Atoms of the solid, or of its liquid above the melting point, moving at the temperature: a Langevin run to
    reach it, then constant energy runs at two time steps whose energy drift checks the integration."""
    from ase import units
    from ase.md.langevin import Langevin
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
    from ase.md.verlet import VelocityVerlet
    if isinstance(eng, DFTEngine):
        raise ResolveError(f"dynamics of {desc} needs a fitted potential, and Wobbl has one only for the EMT metals "
                           "and the noble gases", "Ask for its crystal properties instead.")
    T = opts.temperature
    sym = atoms.get_chemical_symbols()[0]
    e = F.element(sym)
    liquid = e.get("melting_K") and T > e["melting_K"] and (not e.get("boiling_K") or T < e["boiling_K"] * 1.05)
    reps = max(1, int(np.ceil((2.0 * (3 * LJ[sym][1] if sym in LJ else 5.0) + 0.5) / np.linalg.norm(atoms.cell[0]))))
    sc = atoms.repeat((reps, reps, reps))
    if liquid:
        rho = {"Ar": 1.40, "Ne": 1.21, "Kr": 2.41, "Xe": 2.94}.get(sym)     # g/cm3 near the triple point
        if rho:
            m = sum(F.mass(s) for s in sc.get_chemical_symbols()) * 1.66053906660e-24
            s = ((m / rho) / (sc.get_volume() * 1e-24)) ** (1 / 3)
            sc.set_cell(np.array(sc.cell) * s, scale_atoms=True)
    sc.calc = eng.ase()
    rng = np.random.default_rng(5)
    MaxwellBoltzmannDistribution(sc, temperature_K=T, rng=rng)
    Stationary(sc)
    dt = 5.0 if sym in LJ else 2.0
    log("dynamics", f"{len(sc)} atoms of {'liquid' if liquid else 'solid'} {e['name'].lower()} at {fmt(T)} K")
    lang = Langevin(sc, timestep=dt * units.fs, temperature_K=T, friction=0.01 / units.fs, rng=rng, fixcm=False)
    lang.run(300)
    start = (sc.get_positions().copy(), sc.get_velocities().copy())
    runs, frames = {}, []
    for d in (dt, dt / 2):
        b = sc.copy()
        b.set_positions(start[0])
        b.set_velocities(start[1])
        b.calc = eng.ase()
        vv = VelocityVerlet(b, timestep=d * units.fs)
        steps = int(1000.0 / d)
        es, ts = [], []
        every = max(1, steps // 60)
        for k in range(steps):
            vv.run(1)
            es.append(b.get_potential_energy() + b.get_kinetic_energy())
            ts.append(b.get_temperature())
            if d == dt and k % every == 0:
                frames.append(b.get_positions().round(2).tolist())
        es = np.array(es)
        runs[d] = {"drift_meV_atom": float((es[-1] - es[0]) / len(b) * 1000),
                   "spread_meV_atom": float((es.max() - es.min()) / len(b) * 1000), "T_mean": float(np.mean(ts))}
    a, b2 = runs[dt], runs[dt / 2]
    kt = units.kB * T * 1000
    ratio = a["spread_meV_atom"] / max(b2["spread_meV_atom"], 1e-12)
    checks = [Check("energy conservation", "pass" if abs(a["drift_meV_atom"]) < 0.01 * kt * 10 else "warn",
                    f"over 1 ps at constant energy the total energy drifts {a['drift_meV_atom']:+.4f} meV per atom and "
                    f"wobbles within {a['spread_meV_atom']:.4f} meV, against kT = {kt:.2f} meV", a),
              Check("time step convergence", "pass" if ratio > 2.0 else "warn",
                    f"halving the {dt:g} fs step shrinks the wobble {ratio:.1f} times, near the 4 times a second "
                    "order integrator gives", {"ratio": ratio})]
    level, reasons = confidence(checks)
    phase = "liquid" if liquid else "solid"
    head = (f"{len(sc)} atoms of {phase} {e['name'].lower()} at {fmt(T)} K by {eng.name}: the run holds "
            f"{fmt(a['T_mean'])} K, and its energy drifts {a['drift_meV_atom']:+.4f} meV per atom over 1 ps at "
            "constant energy.")
    values = {"atoms": len(sc), "phase": phase, "T_K": T, "nve": {str(k): v for k, v in runs.items()}, "dt_fs": dt,
              "method": eng.name, "drift_meV_atom": a["drift_meV_atom"], "ratio": ratio}
    from . import playback
    values["files"] = {"playback": playback.crystal(out, e["name"].lower(), sc, f"{phase} {e['name'].lower()}",
                                                    None, None, None, values, md=frames, md_fs=1000.0)}
    o = GenericOutcome("chemistry", e["name"].lower(), opts.question, "ok" if level != "low" else "marginal", head,
                       [("atoms", str(len(sc))), ("temperature", f"{fmt(a['T_mean'])} K"),
                        ("energy drift", f"{a['drift_meV_atom']:+.4f} meV/atom")], level, reasons, checks,
                       [f"{eng.name} for {e['name'].lower()} in a periodic box, a Langevin start at {fmt(T)} K then "
                        "constant energy."], values, {}, solver=f"{eng.name} with ASE", t_total=time.time() - t0)
    return o.write(out)
