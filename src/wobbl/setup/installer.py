"""Install the solvers into ~/.wobbl without admin rights.

Wobbl downloads a pinned micromamba binary, then creates a private conda-forge
environment holding the solvers. Nothing touches system folders, and
'wobbl uninstall --all' removes it by deleting ~/.wobbl.

Every solver registers a Recipe below: the domains it serves, its conda-forge
packages, the executables Wobbl runs and a check on a tiny problem. 'wobbl
setup' installs the core recipes, and any other recipe is installed the first
time its domain is used.

Platform notes (conda-forge availability):
  CalculiX  osx-64, osx-arm64, linux-64, linux-aarch64, win-64
  ngspice   osx-64, linux-64, win-64; on Apple Silicon the Intel build runs under Rosetta 2;
            not packaged for linux-aarch64
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import ssl
import stat
import subprocess
import sys
import tempfile
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from ..solvers.locate import env_prefix, platform_tag, wobbl_home

MICROMAMBA_VERSION = os.environ.get("WOBBL_MICROMAMBA_VERSION", "2.9.0-0")
MM_URL = "https://github.com/mamba-org/micromamba-releases/releases/download/{v}/micromamba-{p}"

Log = Callable[[str], None]


class SetupError(RuntimeError):
    pass


@dataclass
class Recipe:
    """One solver: the conda-forge packages for its domains, what of them to keep and how to check it works.

    ``keep`` matches the names of the executables Wobbl runs (fnmatch patterns, without .exe). ``check`` runs the
    solver on a tiny problem and returns {"path", "version", "works", "error"}, or a string saying why it is
    missing. Core recipes are installed by 'wobbl setup', the others the first time one of their domains is used.
    """
    name: str
    domains: tuple[str, ...]
    packages: tuple[str, ...]
    keep: tuple[str, ...] = ()
    platforms: tuple[str, ...] = ()       # conda platforms with a native build, empty for all
    rosetta: bool = False                 # on Apple Silicon, install the Intel build into env-x86
    core: bool = False
    env: str = "env"
    missing: str = ""                     # what to say on a platform without a build
    check: Callable[[Log], dict | str] | None = None
    # a solver that is not a conda package (Cantera is a Python wheel) brings its own install and presence test
    installer: Callable[[Log, bool], dict] | None = None
    present: Callable[[], bool] | None = None

    def package_names(self) -> list[str]:
        return [re.split(r"[\s=<>!~]", p, maxsplit=1)[0] for p in self.packages]


RECIPES: dict[str, Recipe] = {}


def register(recipe: Recipe) -> Recipe:
    RECIPES[recipe.name] = recipe
    return recipe


def recipes_for(domain: str) -> list[Recipe]:
    return [r for r in RECIPES.values() if domain in r.domains]


def target(recipe: Recipe, plat: str | None = None) -> tuple[str, str | None] | None:
    """The environment a recipe goes into on this platform and the conda platform to force there, or None when
    conda-forge has no build for it."""
    plat = plat or platform_tag()
    if not recipe.platforms or plat in recipe.platforms:
        return recipe.env, None
    if plat == "osx-arm64" and recipe.rosetta:
        return "env-x86", "osx-64"
    return None


def installed(recipe: Recipe) -> bool:
    if recipe.present is not None:
        return recipe.present()
    t = target(recipe)
    meta = env_prefix(t[0]) / "conda-meta" if t else None
    if meta is None or not meta.is_dir():
        return False
    have = {f.name.rsplit("-", 2)[0] for f in meta.glob("*.json")}
    return all(n in have for n in recipe.package_names())


def _ssl_context():
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:  # noqa: BLE001
        return ssl.create_default_context()


def download(url: str, dest: Path, log: Log = print) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "wobbl-setup"})
        with urllib.request.urlopen(req, context=_ssl_context(), timeout=120) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f)
    except Exception as e:  # noqa: BLE001  fall back to curl, which uses the system trust store
        curl = shutil.which("curl")
        if not curl:
            raise SetupError(f"could not download {url}: {e}") from e
        r = subprocess.run([curl, "-fsSL", "-o", str(tmp), url], capture_output=True, text=True)
        if r.returncode != 0:
            raise SetupError(f"could not download {url}: {e}; curl said {r.stderr.strip()}") from e
    tmp.replace(dest)
    return dest


def micromamba_path() -> Path:
    exe = "micromamba.exe" if sys.platform == "win32" else "micromamba"
    return wobbl_home() / "bin" / exe


def ensure_micromamba(log: Log = print, force: bool = False) -> Path:
    mm = micromamba_path()
    if mm.exists() and not force:
        return mm
    plat = platform_tag()
    name = plat + (".exe" if plat.startswith("win") else "")
    url = MM_URL.format(v=MICROMAMBA_VERSION, p=name)
    log(f"downloading micromamba {MICROMAMBA_VERSION} for {plat}")
    download(url, mm, log)
    try:
        sha_file = Path(tempfile.mkdtemp()) / "mm.sha256"
        download(MM_URL.format(v=MICROMAMBA_VERSION, p=plat) + ".sha256", sha_file, log)
        want = sha_file.read_text().split()[0].strip().lower()
        got = hashlib.sha256(mm.read_bytes()).hexdigest()
        if want and want != got:
            mm.unlink()
            raise SetupError("micromamba checksum mismatch; the download was corrupted, run 'wobbl setup' again")
    except SetupError:
        raise
    except Exception:  # noqa: BLE001  checksum file unavailable; the HTTPS download is still trusted
        log("checksum file unavailable, continuing")
    mm.chmod(mm.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return mm


def _mm(mm: Path, args: list[str], log: Log, logfile: Path) -> None:
    env = dict(os.environ)
    env["MAMBA_ROOT_PREFIX"] = str(wobbl_home() / "mamba")
    env["CONDA_PKGS_DIRS"] = str(wobbl_home() / "mamba" / "pkgs")
    cmd = [str(mm), *args]
    with open(logfile, "a") as lf:
        lf.write(f"\n$ {' '.join(cmd)}\n")
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
        last = time.time()
        for line in p.stdout:
            lf.write(line)
            s = line.strip()
            if s and (s.startswith(("+", "Transaction", "Linking", "Downloading")) or "Done" in s) and \
                    time.time() - last > 0.5:
                log("  " + s[:100])
                last = time.time()
        p.wait()
    if p.returncode != 0:
        raise SetupError(f"micromamba failed (exit {p.returncode}); see {logfile}")


def _chosen(names: list[str] | None) -> list[Recipe]:
    """The core recipes, or those named by recipe or domain."""
    if not names:
        return [r for r in RECIPES.values() if r.core]
    want = [n.strip().lower() for n in names if n.strip()]
    out = [r for r in RECIPES.values() if any(r.name.startswith(n) or n in r.domains for n in want)]
    if not out:
        raise SetupError(f"no solver called {', '.join(want)}; choose from {', '.join(RECIPES)}")
    return out


def plan(plat: str | None = None, names: list[str] | None = None) -> dict:
    """What an install puts where: each environment with the packages it gets, and what this platform lacks."""
    plat = plat or platform_tag()
    out = {"platform": plat, "env": [], "env_x86": [], "unavailable": {}, "groups": [], "custom": []}
    groups: dict[tuple, dict] = {}
    for r in _chosen(names):
        if r.installer is not None:
            out["custom"].append(r.name)
            continue
        t = target(r, plat)
        if t is None:
            out["unavailable"][r.name] = r.missing or f"conda-forge has no {r.name} build for {plat}."
            continue
        g = groups.setdefault(t, {"env": t[0], "platform": t[1], "specs": [], "recipes": []})
        g["specs"] += list(r.packages)
        g["recipes"].append(r.name)
        out["env_x86" if t[0] == "env-x86" else "env"] += list(r.packages)
    out["groups"] = list(groups.values())
    return out


def install(log: Log = print, force: bool = False, only: list[str] | None = None) -> dict:
    home = wobbl_home()
    home.mkdir(parents=True, exist_ok=True)
    logfile = home / "setup.log"
    p = plan(names=only)
    unavailable = dict(p["unavailable"])
    if p["groups"]:
        mm = ensure_micromamba(log, force)
        base = ["-y", "-r", str(home / "mamba"), "-c", "conda-forge", "--override-channels"]
    for g in p["groups"]:
        env = env_prefix(g["env"])
        verb = "install" if (env / "conda-meta").exists() and not force else "create"
        names = ", ".join(g["recipes"])
        if g["platform"]:
            log(f"installing {names} (Intel build, runs under Rosetta 2)")
            try:
                _mm(mm, [verb, *base, "--platform", g["platform"], "-p", str(env), *g["specs"]], log, logfile)
            except SetupError as e:
                unavailable.update({n: f"Intel {n} could not be installed: {e}" for n in g["recipes"]})
        else:
            log(f"installing {names} into {env}")
            _mm(mm, [verb, *base, "-p", str(env), *g["specs"]], log, logfile)
    for name in p["custom"]:
        log(f"installing {name}")
        try:
            RECIPES[name].installer(log, force)
        except SetupError as e:
            unavailable[name] = str(e)
    status = verify(log)
    status["platform"] = p["platform"]
    status["unavailable"] = {**status.get("unavailable", {}), **unavailable}
    (home / "config.json").write_text(json.dumps(status, indent=1))
    return status


def install_on_demand(domains: tuple[str, ...] | list[str], log: Log | None = None) -> bool:
    """Install the solvers of a domain the first time it is used. True when something was installed.

    WOBBL_AUTO_INSTALL=0 turns this off. Progress goes to stderr, since an MCP server's stdout carries the
    protocol.
    """
    if os.environ.get("WOBBL_AUTO_INSTALL", "1").strip().lower() in ("0", "no", "false", "off"):
        return False
    todo = list({r.name: r for d in domains for r in recipes_for(d) if target(r) and not installed(r)})
    if not todo:
        return False
    log = log or (lambda m: print(f"wobbl: {m}", file=sys.stderr, flush=True))
    log(f"first {domains[0]} study on this machine, installing {', '.join(todo)} into {wobbl_home()}")
    install(log, only=todo)
    return True


# ------------------------------------------------------------------- checking
TEST_INP = """*NODE, NSET=NALL
1,0,0,0
2,1,0,0
3,0,1,0
4,0,0,1
*ELEMENT, TYPE=C3D4, ELSET=EALL
1,1,2,3,4
*MATERIAL, NAME=M
*ELASTIC
1000,0.3
*SOLID SECTION, ELSET=EALL, MATERIAL=M
*BOUNDARY
1,1,3
2,2,3
3,3,3
*STEP
*STATIC
*CLOAD
4,3,1.
*NODE FILE
U
*END STEP
"""


def check_calculix(log: Log) -> dict | str:
    from ..solvers import base
    from ..solvers.locate import tool_env

    ccx = base.get("calculix")
    path = ccx.locate()
    if path is None:
        return "CalculiX was not found after install; see ~/.wobbl/setup.log"
    ok, msg = False, ""
    with tempfile.TemporaryDirectory() as tmp:
        Path(tmp, "t.inp").write_text(TEST_INP)
        try:
            r = subprocess.run([str(path), "-i", "t"], cwd=tmp, capture_output=True, text=True, timeout=60,
                               env=tool_env(path))
            ok = Path(tmp, "t.frd").exists() and r.returncode == 0
            msg = "" if ok else (r.stdout + r.stderr)[-300:]
        except Exception as e:  # noqa: BLE001
            msg = str(e)
    log(f"CalculiX {ccx.version()} {'works' if ok else 'FAILED: ' + msg}")
    return {"path": str(path), "version": ccx.version(), "works": ok, "error": msg}


def check_ngspice(log: Log) -> dict | str:
    from ..solvers import base

    sp = base.get("ngspice")
    path = sp.locate()
    if path is None:
        return "ngspice is not installed"
    v = sp.version()
    ok, err = False, ""
    try:
        with tempfile.TemporaryDirectory() as tmp:
            data, _, _ = sp.run("* check\nV1 a 0 DC 1\nR1 a 0 1k\n.dc V1 1 1 1\n.end\n", tmp, ["i(v1)"], "t",
                                timeout=60)
            ok = abs(float(data["i(v1)"][-1]) + 1e-3) < 1e-6
    except Exception as e:  # noqa: BLE001
        err = str(e)[:200]
    if not ok and sys.platform == "darwin":
        err = "the Intel ngspice needs Rosetta 2: run 'softwareupdate --install-rosetta --agree-to-license'"
    log(f"ngspice {v} {'works' if ok else 'FAILED ' + err}")
    return {"path": str(path), "version": v, "works": ok, "error": err}


def verify(log: Log = print) -> dict:
    """Run each solver on a tiny problem and report what works."""
    out = {"solvers": {}, "unavailable": {}}
    for r in RECIPES.values():
        if r.check is None or not (r.core or installed(r)):
            continue
        got = r.check(log)
        if isinstance(got, str):
            out["unavailable"].setdefault(r.name, got)
        else:
            out["solvers"][r.name] = got
    return out


register(Recipe("calculix", ("structural", "thermal"), ("calculix>=2.21",), keep=("ccx*", "calculix*"), core=True,
                check=check_calculix))
register(Recipe("ngspice", ("circuit",), ("ngspice>=37",), keep=("ngspice*",), core=True,
                platforms=("osx-64", "linux-64", "win-64"), rosetta=True, check=check_ngspice,
                missing="conda-forge has no ngspice build for this platform. Install ngspice with your package "
                        "manager (for example 'apt install ngspice') and Wobbl will find it on PATH, or set "
                        "WOBBL_NGSPICE."))


def _chem(name: str):
    """A function of wobbl.chem.install, imported only when the registry calls it."""
    def call(*args):
        from ..chem import install as chem_install
        return getattr(chem_install, name)(*args)
    return call


register(Recipe("cantera", ("combustion", "motor"), (), check=_chem("check"), installer=_chem("install"),
                present=_chem("available")))


def remove_all(log: Log = print) -> None:
    home = wobbl_home()
    if home.exists():
        shutil.rmtree(home)
        log(f"removed {home}")


def disk_usage() -> int:
    total = 0
    for root, _, files in os.walk(wobbl_home()):
        for f in files:
            try:
                total += os.lstat(os.path.join(root, f)).st_size
            except OSError:
                pass
    return total
