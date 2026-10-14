"""Cantera, the chemistry solver, installed the first time someone needs it.

Wobbl's base install stays small, so Cantera only arrives with the first combustion or motor study, or with
'wobbl setup --only cantera'. It is a Python library rather than a program, so instead of a conda environment
Wobbl reads the pinned release from PyPI, takes the wheel built for this Python and platform, checks its SHA-256
against the index and unpacks it into ~/.wobbl/py/cantera. It then removes what it never uses (the other
mechanisms, the example data, the file converters, the tests and the type stubs) and deletes the download, so the
pruned library is all that stays on disk. A Cantera that is already importable, for example from
'pip install wobbl[chem]', is used as it is and never touched.

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
from pathlib import Path
from typing import Callable

from ..solvers.base import SolverNotFound
from ..solvers.locate import wobbl_home

CANTERA_VERSION = os.environ.get("WOBBL_CANTERA_VERSION", "3.2.0")
PYPI_JSON = "https://pypi.org/pypi/cantera/{v}/json"
DOMAINS = ("combustion", "motor")
# the only data files Wobbl reads: NASA's gas and condensed species, and GRI-Mech 3.0 for the flame cross check
KEEP_DATA = {"nasa_gas.yaml", "nasa_condensed.yaml", "gri30.yaml"}
PRUNE_PATHS = ["cantera/examples", "cantera/test", "cantera/with_units", "cantera/ck2yaml.py", "cantera/cti2yaml.py",
               "cantera/ctml2yaml.py", "cantera/lxcat2yaml.py", "cantera/yaml2ck.py"]
PRUNE_SUFFIXES = (".pyi", ".pxd")
HINT = "Run 'wobbl setup --only cantera' (about 6 MB), or 'pip install cantera' into Wobbl's Python."

Log = Callable[[str], None]


def lib_dir() -> Path:
    return wobbl_home() / "py" / "cantera"


def load():
    """The cantera module, from the Python environment or from Wobbl's own folder, or None."""
    try:
        return importlib.import_module("cantera")
    except ImportError:
        pass
    d = lib_dir()
    if not (d / "cantera").is_dir():
        return None
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))
    try:
        return importlib.import_module("cantera")
    except ImportError:
        return None


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


def prune(root: Path) -> int:
    """Delete what Wobbl never uses from an unpacked Cantera wheel; return the bytes removed."""
    removed = 0

    def drop(p: Path):
        nonlocal removed
        if p.is_dir():
            removed += tree_size(p)
            shutil.rmtree(p, ignore_errors=True)
        elif p.exists():
            removed += p.stat().st_size
            p.unlink()

    for rel in PRUNE_PATHS:
        drop(root / rel)
    for p in list(root.iterdir()):
        if p.name.endswith(".data"):      # console scripts for the file converters
            drop(p)
    data = root / "cantera" / "data"
    if data.is_dir():
        for p in list(data.iterdir()):
            if p.name not in KEEP_DATA and p.name != "__init__.py":
                drop(p)
    for p in list((root / "cantera").rglob("*")):
        if p.is_file() and (p.suffix in PRUNE_SUFFIXES or p.name == "py.typed"):
            drop(p)
    for p in list(root.rglob("__pycache__")):
        drop(p)
    return removed


def tree_size(p: Path) -> int:
    total = 0
    for root, _, files in os.walk(p):
        for f in files:
            try:
                total += os.lstat(os.path.join(root, f)).st_size
            except OSError:
                pass
    return total


def install(log: Log = print, force: bool = False) -> dict:
    """Fetch, verify, unpack and prune Cantera into ~/.wobbl/py/cantera. Returns what it did, with sizes."""
    from ..setup.installer import SetupError, download

    dest = lib_dir()
    marker = dest / "wobbl-cantera.json"
    if (dest / "cantera").is_dir() and marker.exists() and not force:
        return {**json.loads(marker.read_text()), "already": True}
    tmp = Path(tempfile.mkdtemp(prefix="wobbl-cantera-"))
    try:
        meta = download(PYPI_JSON.format(v=CANTERA_VERSION), tmp / "release.json", log)
        f = pick_wheel(json.loads(meta.read_text()).get("urls", []))
        if f is None:
            raise SetupError(f"Cantera {CANTERA_VERSION} has no wheel for Python {sys.version_info.major}."
                             f"{sys.version_info.minor} on this platform, and 'pip install cantera' may build one")
        log(f"downloading {f['filename']} ({f['size'] / 1e6:.1f} MB)")
        whl = download(f["url"], tmp / f["filename"], log)
        want = (f.get("digests") or {}).get("sha256", "")
        if want and hashlib.sha256(whl.read_bytes()).hexdigest() != want:
            raise SetupError("the Cantera download does not match PyPI's checksum, so run the install again")
        stage = dest.with_name("cantera.partial")
        shutil.rmtree(stage, ignore_errors=True)
        with zipfile.ZipFile(whl) as z:
            z.extractall(stage)
        unpacked = tree_size(stage)
        prune(stage)
        info = {"version": CANTERA_VERSION, "wheel": f["filename"], "download_bytes": int(f["size"]),
                "unpacked_bytes": unpacked, "installed_bytes": tree_size(stage), "path": str(dest)}
        (stage / "wobbl-cantera.json").write_text(json.dumps(info, indent=1))
        shutil.rmtree(dest, ignore_errors=True)
        stage.rename(dest)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)    # the download is not kept
    log(f"Cantera {CANTERA_VERSION}: {info['download_bytes'] / 1e6:.1f} MB download, "
        f"{info['unpacked_bytes'] / 1e6:.1f} MB unpacked, {info['installed_bytes'] / 1e6:.1f} MB after pruning")
    return info


def check(log: Log = print) -> dict | str:
    """Solve one small equilibrium, hydrogen burning in oxygen, the way the setup registry checks a solver."""
    ct = load()
    if ct is None:
        return "Cantera is not installed, and it installs the first time you run a combustion or motor study"
    ok, err = False, ""
    try:
        g = ct.Solution("gri30.yaml")
        g.TPX = 300.0, 101325.0, "H2:2, O2:1"
        g.equilibrate("HP")
        ok = 2900 < g.T < 3300
        err = "" if ok else f"hydrogen and oxygen came out at {g.T:.0f} K"
    except Exception as e:  # noqa: BLE001
        err = str(e)[:200]
    where = Path(ct.__file__).parent
    size = f" ({tree_size(lib_dir()) / 1e6:.1f} MB)" if where.resolve().is_relative_to(lib_dir().resolve()) else ""
    log(f"Cantera {ct.__version__}{size} {'works' if ok else 'FAILED ' + err}")
    return {"path": str(where) + size, "version": ct.__version__, "works": ok, "error": err}
