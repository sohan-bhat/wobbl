"""Chemistry's Python libraries, each installed the first time a run needs it.

Wobbl's base install stays small, so the chemistry libraries arrive only when a run uses them, or with
'wobbl setup --only chemistry'.

  cantera   equilibrium and flames on NASA's thermodynamic data, for mixtures, reactions and motors
  rdkit     reading SMILES and InChI and building 3D structures and conformers
  tblite    GFN2-xTB and GFN1-xTB, the semiempirical quantum method behind every molecule and crystal run
  ase       optimizers, normal modes, molecular dynamics, reaction paths and equations of state
  pyscf     density functional theory, the higher rung of the method ladder for small molecules (with h5py)

Each is a Python library rather than a program, so instead of a conda environment Wobbl reads the pinned release
from PyPI, takes the wheel built for this Python and platform, checks its SHA-256 against the index and unpacks it
into ~/.wobbl/py/<name>. It then removes what Wobbl never uses (tests, examples, type stubs, data it never reads)
and deletes the download, so the pruned library is all that stays on disk. A library that is already importable,
for example from 'pip install rdkit', is used as it is and never touched.

Installs on first use follow the solver registry in setup/installer.py, so WOBBL_AUTO_INSTALL=0 turns them off.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import os
import shutil
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from ..solvers.base import SolverNotFound
from ..solvers.locate import wobbl_home

PYPI_JSON = "https://pypi.org/pypi/{name}/{v}/json"
Log = Callable[[str], None]


@dataclass(frozen=True)
class Package:
    name: str                              # the import name and the folder under ~/.wobbl/py
    version: str
    domains: tuple[str, ...]               # the setup registry's domains that install it on first use
    what: str                              # one line for setup and doctor
    prune: tuple[str, ...] = ()            # paths under the unpacked wheel to delete, fnmatch patterns allowed
    keep_data: tuple[str, ...] = ()        # for Cantera, the data files Wobbl reads
    needs: tuple[str, ...] = ()            # other packages that come with it

    @property
    def pinned(self) -> str:
        return os.environ.get(f"WOBBL_{self.name.upper()}_VERSION", self.version)


COMMON_PRUNE = ("**/tests", "**/test", "**/examples", "**/__pycache__", "*-stubs")
PACKAGES: dict[str, Package] = {p.name: p for p in [
    Package("cantera", "3.2.0", ("combustion", "motor", "mixture", "reaction", "chemistry"),
            "equilibrium and flames on NASA's thermodynamic data",
            prune=("cantera/examples", "cantera/test", "cantera/with_units", "cantera/ck2yaml.py",
                   "cantera/cti2yaml.py", "cantera/ctml2yaml.py", "cantera/lxcat2yaml.py", "cantera/yaml2ck.py"),
            keep_data=("nasa_gas.yaml", "nasa_condensed.yaml", "gri30.yaml")),
    # 2025.9.3 is the newest release with wheels for every platform Wobbl runs on, Intel Macs included
    Package("rdkit", "2025.9.3", ("molecule", "chemistry"), "SMILES, InChI, 3D structures and conformers",
            prune=("rdkit/Contrib", "rdkit/Docs", "rdkit/include", "rdkit/share", "rdkit/Chem/Draw/*.js",
                   "rdkit/Chem/Pharm2D", "rdkit/Chem/Pharm3D", "rdkit/Chem/Fraggle",
                   "rdkit/Dbase", "rdkit/sping", "rdkit/VLib", "rdkit/SimDivFilters", "rdkit/Avalon",
                   "rdkit/Chem/Subshape", "rdkit/Data/Pains", "rdkit/Data/Fragments", "rdkit/Data/SmartsLib",
                   "rdkit/Data/NCI", "rdkit/Data/Pubchem*", "rdkit/Data/RECAP*", "rdkit/Data/*.sdf",
                   "rdkit/Data/*.smi", "rdkit/Data/*.csv")),
    Package("tblite", "0.7.0", ("molecule", "crystal", "chemistry"), "GFN2-xTB and GFN1-xTB quantum chemistry"),
    Package("ase", "3.29.0", ("molecule", "crystal", "chemistry"),
            "optimizers, normal modes, dynamics, reaction paths and equations of state",
            prune=("ase/gui", "ase/test", "ase/visualize/*.py.bak", "ase/db/static", "ase/db/templates",
                   "ase/utils/sphinx.py", "ase/cli/completion.py")),
    Package("h5py", "3.16.0", ("dft",), "HDF5 files for PySCF"),
    Package("pyscf", "2.14.0", ("dft", "chemistry"), "density functional theory for the method ladder",
            needs=("h5py",),
            prune=("pyscf/lib/deps/include", "pyscf/lib/deps/share", "pyscf/*/test", "pyscf/*/*/test",
                   "pyscf/pbc/*/test", "pyscf/tools/test", "pyscf/lib/*.a")),
]}

# Cantera's original names, which the motor, flame and setup code use
CANTERA_VERSION = PACKAGES["cantera"].pinned
DOMAINS = ("combustion", "motor")
KEEP_DATA = set(PACKAGES["cantera"].keep_data)
PRUNE_PATHS = list(PACKAGES["cantera"].prune)
PRUNE_SUFFIXES = (".pyi", ".pxd")
HINT = "Run 'wobbl setup --only cantera' (about 6 MB), or 'pip install cantera' into Wobbl's Python."


def pkg_dir(name: str) -> Path:
    return wobbl_home() / "py" / name


def lib_dir() -> Path:
    return pkg_dir("cantera")


def load_pkg(name: str):
    """The module, from the Python environment or from Wobbl's own folder, or None."""
    for dep in PACKAGES[name].needs if name in PACKAGES else ():
        _add_path(dep)
    try:
        return importlib.import_module(name)
    except ImportError:
        pass
    if not _add_path(name):
        return None
    importlib.invalidate_caches()
    try:
        return importlib.import_module(name)
    except ImportError:
        return None


def _add_path(name: str) -> bool:
    d = pkg_dir(name)
    if not (d / name).is_dir():
        return False
    if str(d) not in sys.path:
        sys.path.append(str(d))
    return True


def available_pkg(name: str) -> bool:
    return load_pkg(name) is not None


def require_pkg(name: str):
    """Return the module, installing it first when this is the first run on this machine that needs it."""
    mod = load_pkg(name)
    if mod is not None:
        return mod
    p = PACKAGES[name]
    hint = f"Run 'wobbl setup --only {name}', or 'pip install {name}=={p.pinned}' into Wobbl's Python."
    from ..setup.installer import SetupError, install_on_demand
    try:
        install_on_demand((name,))
    except SetupError as e:
        raise SolverNotFound(f"{name} is not installed and installing it failed ({e})", hint) from None
    mod = load_pkg(name)
    if mod is None:
        raise SolverNotFound(f"{name} is not installed", hint)
    return mod


# Cantera keeps its own entry points
def load():
    return load_pkg("cantera")


def available() -> bool:
    return load() is not None


def require():
    """Return cantera, installing it first when this is the first chemistry study on this machine."""
    ct = load()
    if ct is not None:
        return ct
    from ..setup.installer import SetupError, install_on_demand
    try:
        install_on_demand(DOMAINS)
    except SetupError as e:
        raise SolverNotFound(f"Cantera is not installed and installing it failed ({e})", HINT) from None
    ct = load()
    if ct is None:
        raise SolverNotFound("Cantera is not installed", HINT)
    return ct


def pick_wheel(files: list[dict], tags=None) -> dict | None:
    """The wheel in a PyPI release that this interpreter prefers, following packaging's tag order."""
    if tags is None:
        from packaging.tags import sys_tags
        tags = [str(t) for t in sys_tags()]
    rank = {t: i for i, t in enumerate(tags)}
    best, best_rank = None, len(rank)
    for f in files:
        name = f.get("filename", "")
        if f.get("packagetype") != "bdist_wheel" or not name.endswith(".whl"):
            continue
        py, abi, plats = name[:-4].split("-")[-3:]
        for p in py.split("."):
            for a in abi.split("."):
                for pl in plats.split("."):
                    r = rank.get(f"{p}-{a}-{pl}")
                    if r is not None and r < best_rank:
                        best, best_rank = f, r
    return best


def tree_size(p: Path) -> int:
    total = 0
    for root, _, files in os.walk(p):
        for f in files:
            try:
                total += os.lstat(os.path.join(root, f)).st_size
            except OSError:
                pass
    return total


def _drop(p: Path) -> int:
    if p.is_symlink() or p.is_file():
        n = p.lstat().st_size
        p.unlink()
        return n
    if p.is_dir():
        n = tree_size(p)
        shutil.rmtree(p, ignore_errors=True)
        return n
    return 0


def prune_pkg(name: str, root: Path) -> int:
    """Delete what Wobbl never uses from an unpacked wheel; return the bytes removed."""
    p = PACKAGES[name]
    removed = 0
    for pat in p.prune + (COMMON_PRUNE if name != "cantera" else ()):
        for hit in sorted(root.glob(pat), key=lambda q: -len(q.parts)):
            removed += _drop(hit)
    for q in list(root.iterdir()):
        if q.name.endswith(".data"):          # console scripts
            removed += _drop(q)
    if p.keep_data:
        data = root / name / "data"
        if data.is_dir():
            for q in list(data.iterdir()):
                if q.name not in p.keep_data and q.name != "__init__.py":
                    removed += _drop(q)
    for q in list((root / name).rglob("*")) if (root / name).is_dir() else []:
        if q.is_file() and (q.suffix in PRUNE_SUFFIXES or q.name == "py.typed"):
            removed += _drop(q)
    for q in list(root.rglob("__pycache__")):
        removed += _drop(q)
    return removed


def prune(root: Path) -> int:
    return prune_pkg("cantera", root)


def install_pkg(name: str, log: Log = print, force: bool = False) -> dict:
    """Fetch, verify, unpack and prune one library into ~/.wobbl/py/<name>, with the libraries it needs."""
    from ..setup.installer import SetupError, download

    p = PACKAGES[name]
    deps = [install_pkg(d, log, force) for d in p.needs]
    dest = pkg_dir(name)
    marker = dest / f"wobbl-{name}.json"
    if (dest / name).is_dir() and marker.exists() and not force:
        return {**json.loads(marker.read_text()), "already": True}
    tmp = Path(tempfile.mkdtemp(prefix=f"wobbl-{name}-"))
    try:
        meta = download(PYPI_JSON.format(name=name, v=p.pinned), tmp / "release.json", log)
        f = pick_wheel(json.loads(meta.read_text()).get("urls", []))
        if f is None:
            raise SetupError(f"{name} {p.pinned} has no wheel for Python {sys.version_info.major}."
                             f"{sys.version_info.minor} on this platform, and 'pip install {name}' may build one")
        log(f"downloading {f['filename']} ({f['size'] / 1e6:.1f} MB)")
        whl = download(f["url"], tmp / f["filename"], log)
        want = (f.get("digests") or {}).get("sha256", "")
        if want and hashlib.sha256(whl.read_bytes()).hexdigest() != want:
            raise SetupError(f"the {name} download does not match PyPI's checksum, so run the install again")
        stage = dest.with_name(f"{name}.partial")
        shutil.rmtree(stage, ignore_errors=True)
        with zipfile.ZipFile(whl) as z:
            z.extractall(stage)
            for info in z.infolist():          # keep the executable bits of bundled programs
                mode = (info.external_attr >> 16) & 0o777
                if mode:
                    os.chmod(stage / info.filename, mode)
        unpacked = tree_size(stage)
        prune_pkg(name, stage)
        info = {"name": name, "version": p.pinned, "wheel": f["filename"], "download_bytes": int(f["size"]),
                "unpacked_bytes": unpacked, "installed_bytes": tree_size(stage), "path": str(dest)}
        (stage / marker.name).write_text(json.dumps(info, indent=1))
        shutil.rmtree(dest, ignore_errors=True)
        stage.rename(dest)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)    # the download is not kept
    log(f"{name} {p.pinned}: {info['download_bytes'] / 1e6:.1f} MB download, "
        f"{info['unpacked_bytes'] / 1e6:.1f} MB unpacked, {info['installed_bytes'] / 1e6:.1f} MB after pruning")
    if deps:
        info["with"] = deps
    return info


def install(log: Log = print, force: bool = False) -> dict:
    return install_pkg("cantera", log, force)


def installed_bytes(name: str) -> int:
    p = PACKAGES[name]
    return sum(tree_size(pkg_dir(n)) for n in (name, *p.needs) if pkg_dir(n).is_dir())


def _probe(name: str):
    """Run a tiny problem with the library and return a status line, or raise."""
    if name == "cantera":
        ct = load_pkg("cantera")
        g = ct.Solution("gri30.yaml")
        g.TPX = 300.0, 101325.0, "H2:2, O2:1"
        g.equilibrate("HP")
        if not 2900 < g.T < 3300:
            raise RuntimeError(f"hydrogen and oxygen came out at {g.T:.0f} K")
    elif name == "rdkit":
        from rdkit import Chem
        from rdkit.Chem import AllChem
        m = Chem.AddHs(Chem.MolFromSmiles("CCO"))
        if AllChem.EmbedMolecule(m, randomSeed=7) != 0:
            raise RuntimeError("could not embed ethanol in 3D")
    elif name == "tblite":
        import numpy as np
        from tblite.interface import Calculator
        calc = Calculator("GFN2-xTB", np.array([8, 1, 1]),
                          np.array([[0.0, 0.0, 0.22], [0.0, 1.43, -0.89], [0.0, -1.43, -0.89]]))
        calc.set("verbosity", 0)
        e = calc.singlepoint().get("energy")
        if not -5.2 < e < -5.0:
            raise RuntimeError(f"water's GFN2-xTB energy came out at {e:.4f} Eh")
    elif name == "ase":
        from ase.build import bulk
        from ase.calculators.emt import EMT
        a = bulk("Cu", "fcc", a=3.6)
        a.calc = EMT()
        a.get_potential_energy()
    elif name == "pyscf":
        from pyscf import gto, scf
        mf = scf.RHF(gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0))
        e = mf.kernel()
        if not -1.2 < e < -1.0:
            raise RuntimeError(f"H2's Hartree-Fock energy came out at {e:.4f} Eh")
    elif name == "h5py":
        import h5py  # noqa: F401


def check_pkg(name: str, log: Log = print) -> dict | str:
    mod = load_pkg(name)
    if mod is None:
        return f"{name} is not installed, and it installs the first time a chemistry run needs it"
    ok, err = True, ""
    try:
        _probe(name)
    except Exception as e:  # noqa: BLE001
        ok, err = False, str(e)[:200]
    where = Path(mod.__file__).parent
    mine = where.resolve().is_relative_to(pkg_dir(name).resolve())
    size = f" ({installed_bytes(name) / 1e6:.1f} MB)" if mine else ""
    version = getattr(mod, "__version__", PACKAGES[name].pinned)
    log(f"{name} {version}{size} {'works' if ok else 'FAILED ' + err}")
    return {"path": str(where) + size, "version": version, "works": ok, "error": err}


def check(log: Log = print) -> dict | str:
    """Solve one small equilibrium, hydrogen burning in oxygen, the way the setup registry checks a solver."""
    r = check_pkg("cantera", log)
    if isinstance(r, str):
        return "Cantera is not installed, and it installs the first time you run a combustion or motor study"
    return r
