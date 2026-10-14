"""Install OpenFOAM for flow studies, on demand and as small as it can be.

Flow is the one domain whose solver is large, so Wobbl fetches it only when a flow study first needs it or when
'wobbl setup flow' asks for it, and keeps only what it runs. A full OpenFOAM v2412 build is about a gigabyte, with
tutorials, sources, headers, an MPI and over a hundred libraries. Wobbl runs every case as one process and calls a
handful of programs, so it keeps those programs, the libraries they load (read from each binary's own list of
needed libraries), the serial stand in for MPI and OpenFOAM's etc folder, and deletes the rest with the download.

  macOS    the native build packaged by openfoam-app (github.com/gerlero/openfoam-app, release v2.1.2) for Intel and
           Apple Silicon. Its disk image is mounted read only just long enough to copy the kept files out.
  Linux    conda-forge's openfoam 2412 for x86_64, through the micromamba that installs CalculiX, into its own
           environment with its own package cache, which is deleted afterwards.
  Windows  no build, so CFD studies say so. Propellers, hulls and the pipe and drag correlations still run.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import struct
import subprocess
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Callable

from ..setup.installer import SetupError
from ..solvers.locate import platform_tag, wobbl_home

API = "2412"
APP_RELEASE = "v2.1.2"
APP_URL = "https://github.com/gerlero/openfoam-app/releases/download/{release}/{name}"
APP_ASSETS = {
    "osx-64": ("openfoam2412-app-x86_64.zip", "4059d8d2cf3620947a4c10838a9f79958185483ac1a54fe4f0b336457961abc5"),
    "osx-arm64": ("openfoam2412-app-arm64.zip", "0611ebfaf09c2585fd58d6363b986b9d762f7ebd6cd544b36eadb6ffd8b5683d"),
}
CONDA_SPEC = "openfoam=2412"
# the programs flow studies call, and nothing else
APPS = ("blockMesh", "checkMesh", "simpleFoam", "pimpleFoam", "postProcess", "snappyHexMesh",
        "surfaceFeatureExtract", "topoSet")
# libraries OpenFOAM opens by name when a case asks for them (function objects), which no binary lists as needed
RUNTIME_LIBS = ("libfieldFunctionObjects", "libforces", "libutilityFunctionObjects", "libsolverFunctionObjects")
MARKER = "wobbl-openfoam.json"
TRANSIENT_MB = {"osx-64": 1700, "osx-arm64": 1700, "linux-64": 1400}   # disk a lean install needs on the way

Log = Callable[[str], None]


class FlowSetupError(SetupError):
    pass


def openfoam_home() -> Path:
    return Path(os.environ["WOBBL_OPENFOAM"]).expanduser() if os.environ.get("WOBBL_OPENFOAM") else \
        wobbl_home() / "openfoam"


def unavailable_reason(plat: str | None = None) -> str:
    """Why this machine has no OpenFOAM build Wobbl can install, or '' when it has one."""
    plat = plat or platform_tag()
    if plat in APP_ASSETS or plat == "linux-64":
        return ""
    if plat.startswith("win"):
        return ("OpenFOAM has no native Windows build Wobbl can install. Run Wobbl inside WSL for CFD; propellers, "
                "hulls and the pipe and drag correlations work here as they are.")
    return (f"conda-forge has no OpenFOAM build for {plat}. Install OpenFOAM v2412 yourself and set WOBBL_OPENFOAM "
            "to its folder.")


def installed() -> dict | None:
    """The marker of a finished install, or None."""
    p = openfoam_home() / MARKER
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def install(log: Log = print, force: bool = False) -> dict:
    """Download OpenFOAM, keep what Wobbl runs and delete the rest. Returns the marker, with sizes in MB."""
    from ..setup.installer import download

    plat = platform_tag()
    why = unavailable_reason(plat)
    if why:
        raise FlowSetupError(why)
    home = openfoam_home()
    if installed() and not force:
        return installed()
    home.parent.mkdir(parents=True, exist_ok=True)
    free_mb = shutil.disk_usage(home.parent).free / 1e6
    if free_mb < TRANSIENT_MB[plat] + 300:
        raise FlowSetupError(f"installing OpenFOAM needs about {TRANSIENT_MB[plat]} MB free on the way (it keeps "
                             f"under 200 MB), and {free_mb:.0f} MB is free")
    stage = Path(tempfile.mkdtemp(prefix="openfoam-", dir=home.parent))
    t0 = time.time()
    try:
        if plat in APP_ASSETS:
            name, sha = APP_ASSETS[plat]
            url = APP_URL.format(release=APP_RELEASE, name=name)
            log(f"downloading OpenFOAM v{API} for macOS ({name}, about 250 MB, once)")
            zpath = download(url, stage / name, log)
            got = _sha256(zpath)
            if got != sha:
                raise FlowSetupError(f"{name} does not match its published checksum; run 'wobbl setup flow' again")
            full_mb, source = _from_app(zpath, stage, log), f"openfoam-app {APP_RELEASE}"
        else:
            full_mb, source = _from_conda(stage, log), f"conda-forge {CONDA_SPEC}"
        root = stage / "root"
        info = {"api": API, "source": source, "platform": plat, "full_MB": round(full_mb),
                "kept_MB": round(dir_size(root) / 1e6), "apps": list(APPS), "seconds": round(time.time() - t0)}
        (root / MARKER).write_text(json.dumps(info, indent=1))
        if home.exists():
            shutil.rmtree(home)
        root.replace(home)
        log(f"OpenFOAM v{API} is ready in {home}, keeping {info['kept_MB']} MB of the {info['full_MB']} MB full "
            "build")
        return info
    finally:
        shutil.rmtree(stage, ignore_errors=True)


def available() -> bool:
    """An OpenFOAM Wobbl can run is here, its own or one the user pointed it at."""
    from ..solvers.base import get

    return get("openfoam").locate() is not None


def check(log: Log = print) -> dict | str:
    """Mesh and solve a tiny channel, the registry's test that the install works."""
    from ..solvers.base import get
    from . import foam

    of = get("openfoam")
    path = of.locate()
    if path is None:
        return unavailable_reason() or "OpenFOAM is not installed; run 'wobbl setup --only openfoam'"
    ok, err = False, ""
    with tempfile.TemporaryDirectory() as tmp:
        try:
            case = foam.Case(Path(tmp) / "check")
            turb = foam.Turbulence()
            foam.channel(case, 1.0, 0.1, 10, 4)
            foam.write_fields(case, (1.0, 0.0, 0.0), turb)
            foam.write_physics(case, 0.01, turb)
            foam.write_steady(case, turb, 20, "")
            case.run("blockMesh")
            ok = "End" in case.run("simpleFoam")
        except Exception as e:  # noqa: BLE001
            err = str(e)[:300]
    log(f"OpenFOAM {of.version()} {'works' if ok else 'FAILED: ' + err}")
    return {"path": str(path), "version": of.version(), "works": ok, "error": err}


def ensure(log: Log | None = None) -> Path:
    """The OpenFOAM a flow study runs, installed first through the solver registry, with a one line notice, when
    this is the first flow study on the machine. WOBBL_AUTO_INSTALL=0 turns that off."""
    from ..setup.installer import install_on_demand
    from ..solvers.base import SolverNotFound, get

    found = get("openfoam").locate()
    if found is not None:
        return found
    why = unavailable_reason()
    if why:
        raise SolverNotFound("OpenFOAM is not available here", why)
    install_on_demand(("flow",), log)
    found = get("openfoam").locate()
    if found is None:
        raise SolverNotFound("OpenFOAM is not installed",
                             "Run 'wobbl setup --only openfoam' (under 200 MB kept), or unset WOBBL_AUTO_INSTALL.")
    return found


# --- macOS: copy the kept files out of openfoam-app's disk image ------------------------------------------------
def _from_app(zpath: Path, stage: Path, log: Log) -> float:
    """Unpack the disk image, mount it read only and copy the kept files to stage/root. Returns the full size, MB."""
    dmg = stage / "openfoam.dmg"
    with zipfile.ZipFile(zpath) as z:
        member = next(n for n in z.namelist() if n.endswith(".dmg"))
        with z.open(member) as src, open(dmg, "wb") as dst:
            shutil.copyfileobj(src, dst, 1 << 22)
    zpath.unlink()
    mnt = stage / "mnt"
    mnt.mkdir()
    r = subprocess.run(["hdiutil", "attach", "-readonly", "-nobrowse", "-noautoopen", "-noverify", "-mountpoint",
                        str(mnt), str(dmg)], capture_output=True, text=True)
    if r.returncode != 0:
        raise FlowSetupError(f"could not mount the OpenFOAM disk image: {(r.stderr or r.stdout).strip()[:200]}")
    try:
        full = dir_size(mnt) / 1e6
        plat_dir = next(p for p in (mnt / "platforms").iterdir() if (p / "bin" / "simpleFoam").exists())
        libdirs = [plat_dir / "lib", plat_dir / "lib" / "dummy", mnt / "env" / "lib"]
        keep = closure([plat_dir / "bin" / a for a in APPS], libdirs)
        log(f"keeping {len(APPS)} programs and {len(keep) - len(APPS)} libraries")
        root = stage / "root"
        for p in keep:
            dest = root / p.relative_to(mnt)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dest)
        shutil.copytree(mnt / "etc", root / "etc", symlinks=True)
        for f in ("api-info", "build-info"):
            if (mnt / "META-INFO" / f).exists():
                (root / "META-INFO").mkdir(exist_ok=True)
                shutil.copy2(mnt / "META-INFO" / f, root / "META-INFO" / f)
    finally:
        subprocess.run(["hdiutil", "detach", "-force", str(mnt)], capture_output=True)
    dmg.unlink()
    return full


# --- Linux: conda-forge into its own environment, pruned in place ----------------------------------------------
def _from_conda(stage: Path, log: Log) -> float:
    from ..setup.installer import ensure_micromamba

    mm = ensure_micromamba(log)
    root, pkgs = stage / "root", stage / "pkgs"
    env = dict(os.environ, MAMBA_ROOT_PREFIX=str(wobbl_home() / "mamba"), CONDA_PKGS_DIRS=str(pkgs))
    log(f"installing {CONDA_SPEC} from conda-forge (about 250 MB to download)")
    logfile = wobbl_home() / "setup.log"
    with open(logfile, "a") as lf:
        cmd = [str(mm), "create", "-y", "-c", "conda-forge", "--override-channels", "-p", str(root), CONDA_SPEC]
        lf.write(f"\n$ {' '.join(cmd)}\n")
        lf.flush()
        r = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env)
    if r.returncode != 0:
        raise FlowSetupError(f"micromamba could not install OpenFOAM (exit {r.returncode}); see {logfile}")
    shutil.rmtree(pkgs, ignore_errors=True)
    full = dir_size(root) / 1e6
    keep = closure([root / "bin" / a for a in APPS], [root / "lib", root / "lib" / "dummy"])
    log(f"keeping {len(APPS)} programs and {len(keep) - len(APPS)} libraries")
    prune(root, keep, keep_dirs=("etc",))
    return full


def prune(root: Path, keep: set[Path], keep_dirs: tuple[str, ...] = ()) -> None:
    """Delete every file under root that is not kept, along with symlinks that do not lead to a kept file, then the
    folders this empties. Folders named in keep_dirs stay whole."""
    real = {p.resolve() for p in keep}
    for d, dirs, files in os.walk(root, topdown=True):
        here = Path(d)
        if here != root and here.relative_to(root).parts[0] in keep_dirs:
            dirs[:] = []
            continue
        for f in files:
            p = here / f
            if p in keep or (p.exists() and p.resolve() in real):
                continue
            p.unlink()
    for d, _, _ in sorted(os.walk(root), key=lambda t: -len(t[0])):
        if Path(d) != root and not any(Path(d).iterdir()):
            Path(d).rmdir()


# --- which libraries a program needs -----------------------------------------------------------------------------
def closure(programs: list[Path], libdirs: list[Path]) -> set[Path]:
    """The programs, plus every library they need found in libdirs, followed through the libraries' own needs, plus
    the libraries OpenFOAM opens by name. The first libdir holding a name wins, the order the binaries' run paths
    search."""
    index: dict[str, Path] = {}
    for d in libdirs:
        if d.is_dir():
            for p in sorted(d.iterdir()):
                if p.is_file() and (".so" in p.name or p.suffix == ".dylib"):
                    index.setdefault(p.name, p)
    missing = [p for p in programs if not p.exists()]
    if missing:
        raise FlowSetupError(f"this OpenFOAM build lacks {', '.join(p.name for p in missing)}")
    todo = list(programs) + [index[n] for n in index if n.split(".")[0] in RUNTIME_LIBS]
    keep: set[Path] = set()
    while todo:
        p = todo.pop()
        if p in keep:
            continue
        keep.add(p)
        for name in needed_libs(p):
            base = name.rsplit("/", 1)[-1]
            if base in index and index[base] not in keep:
                todo.append(index[base])
    return keep


def needed_libs(path: Path) -> list[str]:
    """The shared libraries a Mach-O or ELF binary names as needed, as written in the file."""
    with open(path, "rb") as f:
        data = f.read()
    if data[:4] == b"\x7fELF":
        return _elf_needed(data)
    magic = struct.unpack("<I", data[:4])[0]
    if magic in (0xcafebabe, 0xbebafeca):        # universal binary: every slice needs the same libraries
        return _macho_needed(data[struct.unpack(">I", data[16:20])[0]:])
    return _macho_needed(data)


_LC_DYLIB = {0xC, 0x18 | 0x80000000, 0x1F | 0x80000000, 0x20, 0x23 | 0x80000000}


def _macho_needed(data: bytes) -> list[str]:
    magic = struct.unpack("<I", data[:4])[0]
    if magic not in (0xfeedfacf, 0xfeedface):
        return []
    ncmds = struct.unpack("<I", data[16:20])[0]
    off = 32 if magic == 0xfeedfacf else 28
    out = []
    for _ in range(ncmds):
        cmd, size = struct.unpack("<II", data[off:off + 8])
        if cmd in _LC_DYLIB:
            name_off = struct.unpack("<I", data[off + 8:off + 12])[0]
            out.append(data[off + name_off:off + size].split(b"\0", 1)[0].decode())
        off += size
    return out


def _elf_needed(data: bytes) -> list[str]:
    if data[4] != 2 or data[5] != 1:              # only 64 bit little endian builds are installed
        return []
    shoff, = struct.unpack("<Q", data[0x28:0x30])
    shentsize, shnum = struct.unpack("<HH", data[0x3A:0x3E])
    sections = [struct.unpack("<IIQQQQIIQQ", data[shoff + i * shentsize:shoff + (i + 1) * shentsize])
                for i in range(shnum)]
    out = []
    for s in sections:
        if s[1] != 6:                              # SHT_DYNAMIC
            continue
        strtab = sections[s[6]]
        for k in range(s[5] // 16):
            tag, val = struct.unpack("<qQ", data[s[4] + 16 * k:s[4] + 16 * k + 16])
            if tag == 0:
                break
            if tag == 1:                           # DT_NEEDED
                start = strtab[4] + val
                out.append(data[start:data.index(b"\0", start)].decode())
    return out


def dir_size(path: Path) -> int:
    total = 0
    for d, _, files in os.walk(path):
        for f in files:
            try:
                total += os.lstat(os.path.join(d, f)).st_size
            except OSError:
                pass
    return total


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()
