"""Install the solvers into ~/.wobbl without admin rights.

Wobbl downloads a pinned micromamba binary, then creates a private conda-forge
environment holding CalculiX and ngspice. Nothing touches system folders, and
'wobbl uninstall --all' removes it by deleting ~/.wobbl.

Platform notes (conda-forge availability):
  CalculiX  osx-64, osx-arm64, linux-64, linux-aarch64, win-64
  ngspice   osx-64, linux-64, win-64; on Apple Silicon the Intel build runs under Rosetta 2;
            not packaged for linux-aarch64
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import ssl
import stat
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Callable

from ..solvers.locate import env_prefix, platform_tag, wobbl_home

MICROMAMBA_VERSION = os.environ.get("WOBBL_MICROMAMBA_VERSION", "2.9.0-0")
MM_URL = "https://github.com/mamba-org/micromamba-releases/releases/download/{v}/micromamba-{p}"
NATIVE_NGSPICE = {"osx-64", "linux-64", "win-64"}
PACKAGES = {"calculix": "calculix>=2.21", "ngspice": "ngspice>=37"}

Log = Callable[[str], None]


class SetupError(RuntimeError):
    pass


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


def plan(plat: str | None = None) -> dict:
    plat = plat or platform_tag()
    out = {"platform": plat, "env": [PACKAGES["calculix"]], "env_x86": [], "unavailable": {}}
    if plat in NATIVE_NGSPICE:
        out["env"].append(PACKAGES["ngspice"])
    elif plat == "osx-arm64":
        out["env_x86"].append(PACKAGES["ngspice"])
    else:
        out["unavailable"]["ngspice"] = (f"conda-forge has no ngspice build for {plat}. Install ngspice with your "
                                         "package manager (for example 'apt install ngspice') and Wobbl will find "
                                         "it on PATH, or set WOBBL_NGSPICE.")
    return out


def install(log: Log = print, force: bool = False, only: list[str] | None = None) -> dict:
    home = wobbl_home()
    home.mkdir(parents=True, exist_ok=True)
    logfile = home / "setup.log"
    p = plan()
    if only:
        p["env"] = [s for s in p["env"] if any(s.startswith(o) for o in only)]
        p["env_x86"] = [s for s in p["env_x86"] if any(s.startswith(o) for o in only)]
    mm = ensure_micromamba(log, force)
    base = ["-y", "-r", str(home / "mamba"), "-c", "conda-forge", "--override-channels"]
    if p["env"]:
        env = env_prefix("env")
        verb = "install" if (env / "conda-meta").exists() and not force else "create"
        log(f"installing {', '.join(s.split('>')[0] for s in p['env'])} into {env}")
        _mm(mm, [verb, *base, "-p", str(env), *p["env"]], log, logfile)
    if p["env_x86"]:
        env = env_prefix("env-x86")
        verb = "install" if (env / "conda-meta").exists() and not force else "create"
        log(f"installing {', '.join(s.split('>')[0] for s in p['env_x86'])} (Intel build, runs under Rosetta 2)")
        try:
            _mm(mm, [verb, *base, "--platform", "osx-64", "-p", str(env), *p["env_x86"]], log, logfile)
        except SetupError as e:
            p["unavailable"]["ngspice"] = f"Intel ngspice could not be installed: {e}"
    status = verify(log)
    status["platform"] = p["platform"]
    status["unavailable"] = {**p["unavailable"], **status.get("unavailable", {})}
    (home / "config.json").write_text(json.dumps(status, indent=1))
    return status


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


def verify(log: Log = print) -> dict:
    """Run each solver on a tiny problem and report what works."""
    from ..solvers import base

    out = {"solvers": {}, "unavailable": {}}
    ccx = base.get("calculix")
    path = ccx.locate()
    if path is None:
        out["unavailable"]["calculix"] = "CalculiX was not found after install; see ~/.wobbl/setup.log"
    else:
        ok, msg = False, ""
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "t.inp").write_text(TEST_INP)
            from ..solvers.locate import tool_env
            try:
                r = subprocess.run([str(path), "-i", "t"], cwd=tmp, capture_output=True, text=True, timeout=60,
                                   env=tool_env(path))
                ok = Path(tmp, "t.frd").exists() and r.returncode == 0
                msg = "" if ok else (r.stdout + r.stderr)[-300:]
            except Exception as e:  # noqa: BLE001
                msg = str(e)
        out["solvers"]["calculix"] = {"path": str(path), "version": ccx.version(), "works": ok, "error": msg}
        log(f"CalculiX {ccx.version()} {'works' if ok else 'FAILED: ' + msg}")
    sp = base.get("ngspice")
    path = sp.locate()
    if path is None:
        out["unavailable"].setdefault("ngspice", "ngspice is not installed")
    else:
        v = sp.version()
        ok = bool(v)
        err = ""
        if not ok and sys.platform == "darwin":
            err = "the Intel ngspice needs Rosetta 2: run 'softwareupdate --install-rosetta --agree-to-license'"
        out["solvers"]["ngspice"] = {"path": str(path), "version": v, "works": ok, "error": err}
        log(f"ngspice {v} {'works' if ok else 'FAILED ' + err}")
    return out


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
