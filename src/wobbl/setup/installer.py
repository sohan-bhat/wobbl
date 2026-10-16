"""Install the solvers into ~/.wobbl without admin rights.

Wobbl downloads a pinned micromamba binary, then creates a private conda-forge
environment holding the solvers. Nothing touches system folders, and
'wobbl uninstall --all' removes it by deleting ~/.wobbl.

Every solver registers a Recipe below: the domains it serves, its conda-forge
packages, the executables Wobbl runs and a check on a tiny problem. 'wobbl
setup' installs the core recipes, and any other recipe is installed the first
time its domain is used. After each install Wobbl deletes micromamba's download
cache and prunes the environment of headers, static libraries, docs and
executables no solver runs, and puts them back if a solver stops working
without them.

Platform notes (conda-forge availability):
  CalculiX  osx-64, osx-arm64, linux-64, linux-aarch64, win-64
  ngspice   osx-64, linux-64, win-64; on Apple Silicon the Intel build runs under Rosetta 2;
            not packaged for linux-aarch64
"""
from __future__ import annotations

import fnmatch
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

from ..solvers.locate import bin_dirs, env_prefix, platform_tag, wobbl_home
from . import progress

MICROMAMBA_VERSION = os.environ.get("WOBBL_MICROMAMBA_VERSION", "2.9.0-0")
MM_URL = "https://github.com/mamba-org/micromamba-releases/releases/download/{v}/micromamba-{p}"

Log = Callable[[str], None]

# what no solver run needs, relative to an environment: headers, static and import libraries, build metadata,
# documentation, translations, test suites and the libraries only the gettext tools load
PRUNE = ("include", "share/doc", "share/man", "share/info", "share/locale", "share/gettext", "share/gtk-doc",
         "lib/*.a", "lib/*.la", "lib/pkgconfig", "lib/cmake", "lib/libgettext*", "lib/libtextstyle*",
         "lib/libasprintf*", "lib/python*/test", "lib/python*/site-packages/*/tests",
         "Library/include", "Library/share/doc", "Library/share/man", "Library/share/info", "Library/share/locale",
         "Library/lib/*.lib", "Library/lib/*.a", "Library/lib/pkgconfig", "Library/lib/cmake")


class SetupError(RuntimeError):
    def __init__(self, msg: str = "", tail: str = ""):
        super().__init__(msg)
        self.tail = tail     # the end of the raw log, shown when an install fails


@dataclass
class Recipe:
    """One solver: the conda-forge packages for its domains, what of them to keep and how to check it works.

    ``keep`` matches the names of the executables Wobbl runs (fnmatch patterns, without .exe). Every other
    executable in the environment's bin folder is pruned, so a recipe whose solver starts helper programs or a
    Python must name them too. ``prune`` adds paths to delete, relative to the environment. ``check`` runs the
    solver on a tiny problem and returns {"path", "version", "works", "error"}, or a string saying why it is
    missing. Core recipes are installed by 'wobbl setup', the others the first time one of their domains is used.
    """
    name: str
    domains: tuple[str, ...]
    packages: tuple[str, ...]
    keep: tuple[str, ...] = ()
    prune: tuple[str, ...] = ()
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


SHOWN_BYTES = 1_000_000     # a download this big or bigger gets its own progress line


def download(url: str, dest: Path, log: Log = print, label: str | None = None) -> Path:
    """Fetch url into dest. A big download shows its bytes and percent on the current install's progress."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    name = label or dest.name
    stage = None
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "wobbl-setup"})
        with urllib.request.urlopen(req, context=_ssl_context(), timeout=120) as r, open(tmp, "wb") as f:
            total = int(r.headers.get("Content-Length") or 0)
            if total >= SHOWN_BYTES:
                stage = progress.current().stage(f"download {name}", total=total, unit="bytes")
            while True:
                chunk = r.read(1 << 16)
                if not chunk:
                    break
                f.write(chunk)
                if stage is not None:
                    stage.update(advance=len(chunk))
    except Exception as e:  # noqa: BLE001  fall back to curl, which uses the system trust store
        if stage is not None:
            stage.update(completed=0, detail="retrying with curl")
        curl = shutil.which("curl")
        if not curl:
            if stage is not None:
                stage.fail()
            raise SetupError(f"could not download {url}: {e}") from e
        r = subprocess.run([curl, "-fsSL", "-o", str(tmp), url], capture_output=True, text=True)
        if r.returncode != 0:
            if stage is not None:
                stage.fail()
            raise SetupError(f"could not download {url}: {e}; curl said {r.stderr.strip()}") from e
    tmp.replace(dest)
    if stage is not None:
        stage.done("")
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
    download(url, mm, log, label=f"micromamba {MICROMAMBA_VERSION}")
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


def log_path(what: str) -> Path:
    """A new file under ~/.wobbl/logs for the raw output of one install step."""
    d = wobbl_home() / "logs"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{what}-{time.strftime('%Y%m%d-%H%M%S')}.log"


def log_tail(path: Path, lines: int = 12) -> str:
    try:
        return "\n".join(path.read_text(errors="replace").strip().splitlines()[-lines:])
    except OSError:
        return ""


def _archive(item: dict, pkgs: Path) -> list[Path]:
    """Where micromamba puts a package's archive while and after it downloads."""
    from urllib.parse import urlparse
    u = urlparse(item.get("url", ""))
    fn = item.get("fn") or Path(u.path).name
    return [pkgs / u.scheme / u.netloc / u.path.lstrip("/"), pkgs / fn] if u.scheme else [pkgs / fn]


def _downloaded(fetch: list[dict], pkgs: Path) -> int:
    """Bytes of the planned packages on disk so far, partial downloads included."""
    listing: dict[Path, list[str]] = {}
    got = 0
    for item in fetch:
        size = int(item.get("size") or 0)
        best = 0
        for path in _archive(item, pkgs):
            if path.parent not in listing:
                try:
                    listing[path.parent] = os.listdir(path.parent)
                except OSError:
                    listing[path.parent] = []
            for n in listing[path.parent]:
                if n.startswith(path.name):
                    q = path.parent / n
                    try:
                        best = max(best, size if q.is_dir() else q.stat().st_size)
                    except OSError:
                        pass
        got += min(best, size) if size else 0
    return got


def _linked(env: Path, since: float) -> int:
    meta = env / "conda-meta"
    try:
        return sum(1 for f in meta.glob("*.json") if f.stat().st_mtime >= since)
    except OSError:
        return 0


def _plan_transaction(mm: Path, args: list[str], env: dict, logfile: Path) -> tuple[list[dict], list[dict]] | None:
    """The packages a micromamba command would download and link, from a dry run, or None when it cannot say."""
    with open(logfile, "a") as lf:
        lf.write(f"\n$ {mm} {' '.join(args)} --dry-run --json\n")
        lf.flush()
        try:
            r = subprocess.run([str(mm), *args, "--dry-run", "--json"], stdout=subprocess.PIPE, stderr=lf, text=True,
                               env=env, stdin=subprocess.DEVNULL, timeout=900)
        except (subprocess.TimeoutExpired, OSError) as e:
            lf.write(f"dry run gave up: {e}\n")
            return None
    try:
        d = json.loads(r.stdout)
    except ValueError:
        d = None
    if r.returncode != 0:
        with open(logfile, "a") as lf:
            if d is None:
                lf.write(r.stdout)
            else:
                lf.write(json.dumps(d, indent=1) + "\n")
                for h in d.get("log_history") or []:
                    if isinstance(h, dict) and h.get("message"):
                        lf.write(f"{h.get('level', 'error')}: {h['message']}\n")
                for prob in d.get("solver_problems") or []:
                    lf.write(f"problem: {prob}\n")
        raise SetupError(f"micromamba could not work out the packages (exit {r.returncode}). Its log is {logfile}",
                         log_tail(logfile))
    acts = (d or {}).get("actions", {})
    return (list(acts.get("FETCH", [])), list(acts.get("LINK", []))) if d is not None else None


def run_micromamba(mm: Path, args: list[str], what: str, prefix: Path, pkgs: Path | None = None) -> Path:
    """Run a micromamba create or install with a progress line for each of its stages: working out the packages,
    downloading them with bytes and percent, and installing them by count. Its raw output goes to a file under
    ~/.wobbl/logs, which a failure names and quotes the end of."""
    pkgs = pkgs or wobbl_home() / "mamba" / "pkgs"
    env = dict(os.environ, MAMBA_ROOT_PREFIX=str(wobbl_home() / "mamba"), CONDA_PKGS_DIRS=str(pkgs))
    logfile = log_path(f"micromamba-{re.sub(r'[^a-z0-9]+', '-', what.lower()).strip('-')}")
    rep = progress.current()
    with rep.stage(f"resolve {what}") as st:
        st.detail = "could not work out the packages"
        planned = _plan_transaction(mm, args, env, logfile)
        st.detail = ""
        if planned is not None:
            fetch, link = planned
            size = sum(int(x.get("size") or 0) for x in fetch)
            st.detail = f"{len(link)} packages" + (f", {progress.nbytes(size)} to download" if fetch else "")
    fetch, link = planned if planned is not None else ([], [])
    total = sum(int(x.get("size") or 0) for x in fetch)
    dl = rep.stage(f"download {what}", total=total, unit="bytes") if total else None
    inst = None
    t0 = time.time()
    with open(logfile, "a") as lf:
        lf.write(f"\n$ {mm} {' '.join(args)}\n")
        lf.flush()
        p = subprocess.Popen([str(mm), *args], stdout=lf, stderr=subprocess.STDOUT, env=env, stdin=subprocess.DEVNULL)
        try:
            while True:
                code = p.poll()
                if dl is not None and dl.finished is None:
                    dl.update(completed=_downloaded(fetch, pkgs))
                n = _linked(prefix, t0 - 1)
                if inst is None and (n or code is not None or (dl is not None and dl.completed >= total)
                                     or dl is None):
                    if dl is not None and code in (None, 0):
                        dl.done()
                    inst = rep.stage(f"install {what}", total=len(link) or None, unit="packages")
                if inst is not None and link:
                    inst.update(completed=min(n, len(link)))
                if code is not None:
                    break
                time.sleep(0.2)
        except BaseException:
            p.kill()
            p.wait()
            for s in (dl, inst):
                if s is not None:
                    s.fail("stopped")
            raise
    if p.returncode != 0:
        for s in (dl, inst):
            if s is not None:
                s.fail()
        raise SetupError(f"micromamba could not install {what} (exit {p.returncode}). Its log is {logfile}",
                         log_tail(logfile))
    if dl is not None:
        dl.done()
    if inst is not None:
        inst.done(f"{len(link)} packages" if link else "")
    return logfile


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
    """Install the core solvers, or those named, with a progress line for every step."""
    with progress.reporting(log) as rep:
        return _install(rep, force, only)


def _install(rep: progress.Reporter, force: bool, only: list[str] | None) -> dict:
    home = wobbl_home()
    home.mkdir(parents=True, exist_ok=True)
    note = rep.note
    p = plan(names=only)
    unavailable = dict(p["unavailable"])
    if p["groups"]:
        mm = ensure_micromamba(note, force)
        base = ["-y", "-r", str(home / "mamba"), "-c", "conda-forge", "--override-channels"]
    for g in p["groups"]:
        env = env_prefix(g["env"])
        verb = "install" if (env / "conda-meta").exists() and not force else "create"
        names = ", ".join(g["recipes"])
        if g["platform"]:
            try:
                run_micromamba(mm, [verb, *base, "--platform", g["platform"], "-p", str(env), *g["specs"]],
                               f"{names} (Intel build for Rosetta 2)", env)
            except SetupError as e:
                unavailable.update({n: f"Intel {n} could not be installed: {e}" for n in g["recipes"]})
        else:
            run_micromamba(mm, [verb, *base, "-p", str(env), *g["specs"]], names, env)
    for name in p["custom"]:
        with rep.stage(f"install {name}") as st:
            try:
                RECIPES[name].installer(note, force)
            except SetupError as e:
                unavailable[name] = str(e)
                st.fail(str(e)[:100])
    chosen = [r.name for r in _chosen(only)]
    rep.checks = chosen
    status = verify(note)
    if p["groups"]:
        with rep.stage("prune files no solver needs") as st:
            status = prune(status, note)
        with rep.stage("clear the download cache") as st:
            status["cache_freed"] = clean_cache(note)
            if status["cache_freed"]:
                st.detail = f"{progress.nbytes(status['cache_freed'])} freed"
    status["chosen"] = chosen
    status["platform"] = p["platform"]
    status["unavailable"] = {**status.get("unavailable", {}), **unavailable}
    status["size"] = sizes()
    status["seconds"] = round(time.time() - rep.t0, 1)
    (home / "config.json").write_text(json.dumps(status, indent=1))
    return status


def install_on_demand(domains: tuple[str, ...] | list[str], log: Log | None = None) -> bool:
    """Install the solvers of a domain the first time it is used. True when something was installed.

    WOBBL_AUTO_INSTALL=0 turns this off. Plain progress goes to stderr, since an MCP server's stdout carries
    the protocol, and the command line draws it as a live display instead.
    """
    if os.environ.get("WOBBL_AUTO_INSTALL", "1").strip().lower() in ("0", "no", "false", "off"):
        return False
    todo = list({r.name: r for d in domains for r in recipes_for(d) if target(r) and not installed(r)})
    if not todo:
        return False
    with progress.reporting(log) as rep:
        rep.note(f"first {domains[0]} study on this machine, so Wobbl installs {', '.join(todo)} into "
                 f"{wobbl_home()} once")
        install(rep.note, only=todo)
    return True


# -------------------------------------------------------------------- pruning
def _envs() -> list[Path]:
    names = {r.env for r in RECIPES.values()} | {"env-x86"}
    return [env_prefix(n) for n in sorted(names) if (env_prefix(n) / "conda-meta").is_dir()]


def _keeps(name: str) -> bool:
    stem = name[:-4] if name.lower().endswith(".exe") else name
    pats = [p for r in RECIPES.values() for p in r.keep]
    if sys.platform == "win32":
        stem, pats = stem.lower(), [p.lower() for p in pats]
    return any(fnmatch.fnmatchcase(stem, p) for p in pats)


def prunable(env: Path) -> list[Path]:
    """Paths in an environment that no solver run needs: the PRUNE patterns, each recipe's own, and every
    executable no recipe keeps."""
    found: list[Path] = []
    for pat in PRUNE + tuple(p for r in RECIPES.values() for p in r.prune):
        found += sorted(env.glob(pat))
    for d in bin_dirs(env):
        if d == env or not d.is_dir():
            continue
        for f in sorted(d.iterdir()):
            exe = f.suffix.lower() == ".exe" if sys.platform == "win32" else not f.is_dir()
            if exe and not _keeps(f.name):
                found.append(f)
    out: list[Path] = []
    for p in found:   # a file inside a folder that goes anyway is not moved twice
        if (p.is_symlink() or p.exists()) and not any(q == p or q in p.parents for q in out):
            out.append(p)
    return out


def prune(before: dict, log: Log = print) -> dict:
    """Delete what no solver run needs, then check every solver again and put it all back if one stopped working.

    Files are moved aside first, a rename on the same disk, so putting them back costs nothing.
    """
    trash = wobbl_home() / "pruned"
    shutil.rmtree(trash, ignore_errors=True)
    moved = []
    for env in _envs():
        for path in prunable(env):
            dest = trash / env.name / path.relative_to(env)
            dest.parent.mkdir(parents=True, exist_ok=True)
            os.replace(path, dest)
            moved.append((path, dest))
    if not moved:
        return before
    rep = progress.current() if progress._current else None
    keep = getattr(rep, "checks", None)
    if rep is not None:
        rep.checks = []          # the second round of checks is silent
    try:
        after = verify(lambda m: None)
    finally:
        if rep is not None:
            rep.checks = keep
    broke = [n for n, s in before["solvers"].items()
             if s.get("works") and not after["solvers"].get(n, {}).get("works")]
    if broke:
        for path, dest in reversed(moved):
            path.parent.mkdir(parents=True, exist_ok=True)
            os.replace(dest, path)
        log(f"kept every file, since {', '.join(broke)} stopped working without them")
        after = before
    shutil.rmtree(trash, ignore_errors=True)
    return after


def _links_into(env: Path, folder: Path) -> bool:
    root = folder.resolve()
    for d, dirs, files in os.walk(env):
        for f in dirs + files:
            p = Path(d, f)
            if p.is_symlink() and root in p.resolve().parents:
                return True
    return False


def clean_cache(log: Log = print) -> int:
    """Delete micromamba's package and index cache inside Wobbl's home and return the bytes freed.

    The environments keep working without it, since micromamba links packages in by hard link or copy. Setting
    WOBBL_KEEP_CACHE keeps it, for CI runners that cache downloads between runs.
    """
    pkgs = wobbl_home() / "mamba" / "pkgs"
    if os.environ.get("WOBBL_KEEP_CACHE") or not pkgs.exists():
        return 0
    if any(_links_into(env, pkgs) for env in _envs()):
        log("kept the download cache, since the solver environment links into it")
        return 0
    freed = disk_usage(pkgs, shared=False)
    shutil.rmtree(pkgs, ignore_errors=True)
    return freed


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
        return "CalculiX was not found after install; the install log is in ~/.wobbl/logs"
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
    """Run each solver on a tiny problem and report what works. During an install each check the install asked
    for (the reporter's checks, or all when it names none) gets its own progress line."""
    out = {"solvers": {}, "unavailable": {}}
    rep = progress.current() if progress._current else None
    show = getattr(rep, "checks", None)
    for r in RECIPES.values():
        if r.check is None or not (r.core or installed(r)):
            continue
        shown = rep is not None and (show is None or r.name in show)
        st = rep.stage(f"check {r.name}") if shown else None
        quiet = rep is not None and st is None
        got = r.check((lambda m: None) if quiet else log if st is None else
                      (lambda m, st=st: setattr(st, "detail", m)))
        if isinstance(got, str):
            out["unavailable"].setdefault(r.name, got)
        else:
            out["solvers"][r.name] = got
        if st is not None:
            ok = not isinstance(got, str) and got.get("works")
            st.done() if ok else st.fail(got if isinstance(got, str) else (got.get("error") or "")[:100])
    return out


# CalculiX factors its matrices with SPOOLES, which does not call BLAS, so the reference BLAS gives the same answers
# as OpenBLAS (within CalculiX's own run to run noise) at the same speed, and saves 60 MB
register(Recipe("calculix", ("structural", "thermal"), ("calculix>=2.21", "libblas=*=*netlib"),
                keep=("ccx*", "calculix*"), core=True, check=check_calculix))
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


register(Recipe("cantera", ("combustion", "motor", "mixture", "reaction", "chemistry"), (), check=_chem("check"),
                installer=_chem("install"), present=_chem("available")))


def _chem_pkg(fn: str, name: str):
    """wobbl.chem.install's function for one chemistry library, imported only when the registry calls it."""
    def call(*args):
        from ..chem import install as chem_install
        return getattr(chem_install, fn)(name, *args)
    return call


# the other chemistry libraries are Python wheels too, each installed when a run first needs it
for _name, _domains in (("rdkit", ("molecule", "chemistry")), ("tblite", ("molecule", "crystal", "chemistry")),
                        ("ase", ("molecule", "crystal", "chemistry")), ("pyscf", ("dft", "chemistry"))):
    register(Recipe(_name, (_name, *_domains), (), check=_chem_pkg("check_pkg", _name),
                    installer=_chem_pkg("install_pkg", _name), present=_chem_pkg("available_pkg", _name)))


def _mech(name: str):
    """A function of wobbl.mechanism.install, imported only when the registry calls it."""
    def call(*args):
        from ..mechanism import install as mech_install
        return getattr(mech_install, name)(*args)
    return call


register(Recipe("mujoco", ("mechanism",), (), check=_mech("check"), installer=_mech("install"),
                present=_mech("available")))


def _getdp(name: str):
    """A function of wobbl.solvers.getdp, imported only when the registry calls it."""
    def call(*args):
        from ..solvers import getdp
        return getattr(getdp, name)(*args)
    return call


# GetDP is one self contained binary from getdp.info rather than a conda package
register(Recipe("getdp", ("magnetic", "electric", "board", "antenna", "sound"), (), check=_getdp("check"),
                installer=_getdp("install"), present=_getdp("available")))


def _flow(name: str):
    """A function of wobbl.flow.install, imported only when the registry calls it."""
    def call(*args):
        from ..flow import install as flow_install
        return getattr(flow_install, name)(*args)
    return call


register(Recipe("openfoam", ("flow",), (), platforms=("osx-64", "osx-arm64", "linux-64"), check=_flow("check"),
                installer=_flow("install"), present=_flow("available"),
                missing="OpenFOAM has no build Wobbl can install here. Run Wobbl inside WSL on Windows, or point "
                        "WOBBL_OPENFOAM at an OpenFOAM v2412 folder. Propellers, hulls, pipes and drag curves work "
                        "without it."))


# ---------------------------------------------------------------------- disk
def remove_all(log: Log = print) -> None:
    home = wobbl_home()
    if home.exists():
        shutil.rmtree(home)
        log(f"removed {home}")


def disk_usage(path: Path | None = None, shared: bool = True) -> int:
    """Bytes a folder takes, each hard linked file counted once. shared=False skips files with other links, which
    deleting the folder would not free."""
    total, seen = 0, set()
    for root, _, files in os.walk(path or wobbl_home()):
        for f in files:
            try:
                st = os.lstat(os.path.join(root, f))
            except OSError:
                continue
            if (st.st_dev, st.st_ino) in seen or (not shared and st.st_nlink > 1):
                continue
            seen.add((st.st_dev, st.st_ino))
            total += st.st_size
    return total


def sizes() -> dict:
    """Bytes each solver environment, the download cache and Wobbl's whole home take."""
    home = wobbl_home()
    out = {e.name: disk_usage(e) for e in _envs()}
    out["cache"] = disk_usage(home / "mamba")
    out["total"] = disk_usage(home)
    return out
