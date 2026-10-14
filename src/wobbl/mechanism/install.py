"""MuJoCo, the rigid body solver, installed the first time someone needs it.

Wobbl's base install stays small, so MuJoCo only arrives with the first mechanism study, or with
'wobbl setup --only mujoco'. It is a Python library rather than a program, so Wobbl reads the pinned release from
PyPI, takes the wheel built for this Python and platform, checks its SHA-256 against the index and unpacks into
~/.wobbl/py/mujoco only what a headless solver loads, leaving out the C headers, the plugins, the viewer and its
app bundle, the OpenGL renderers, USD export, system identification, the code generators and the tests. It checks
that the pruned copy imports and steps a model, and deletes the download. A MuJoCo that is already importable,
for example from 'pip install wobbl[mechanism]', is used as it is and never touched.

3.10.0 is pinned because it is the newest release with Intel Mac wheels, and the one Wobbl's checks were run on.
Installs on first use follow the solver registry in setup/installer.py, so WOBBL_AUTO_INSTALL=0 turns them off.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Callable

from ..solvers.base import SolverNotFound
from ..solvers.locate import wobbl_home

MUJOCO_VERSION = os.environ.get("WOBBL_MUJOCO_VERSION", "3.10.0")
PYPI_JSON = "https://pypi.org/pypi/mujoco/{v}/json"
DOMAINS = ("mechanism",)
# folders and files inside the mujoco package that a headless simulation never imports
PRUNE_DIRS = ["include", "plugin", "sysid", "usd", "experimental", "testdata", "introspect", "cgl", "egl", "osmesa",
              "glfw", "rendering", "MuJoCo_(mjpython).app", "mjx"]
PRUNE_FILES = ["viewer.py", "rollout.py", "minimize.py", "msh2obj.py", "mjpython"]
PRUNE_PREFIXES = ("_simulate", "_rollout", "_render")
PRUNE_SUFFIXES = ("_test.py", ".pyi", ".lib", ".a")
HINT = "Run 'wobbl setup --only mujoco' (about 8 MB), or 'pip install mujoco' into Wobbl's Python."
SMOKE = ("import sys; sys.path.insert(0, sys.argv[1]); import mujoco; "
         "m = mujoco.MjModel.from_xml_string('<mujoco><worldbody><body><joint/><geom size=\"0.1\"/></body>"
         "</worldbody></mujoco>'); d = mujoco.MjData(m); mujoco.mj_step(m, d); print(mujoco.__version__)")

Log = Callable[[str], None]


def lib_dir() -> Path:
    return wobbl_home() / "py" / "mujoco"


WARNINGS: list[str] = []      # what MuJoCo warned about since the last clear, instead of MUJOCO_LOG.TXT in the cwd


def load():
    """The mujoco module, from the Python environment or from Wobbl's own folder, or None."""
    try:
        return _quiet(importlib.import_module("mujoco"))
    except ImportError:
        pass
    d = lib_dir()
    if not (d / "mujoco").is_dir():
        return None
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))
    try:
        return _quiet(importlib.import_module("mujoco"))
    except ImportError:
        return None


def _quiet(mj):
    """Collect MuJoCo's warnings for the checks rather than letting it write a log file wherever Wobbl runs."""
    if not getattr(mj, "_wobbl_quiet", False):
        mj.set_mju_user_warning(lambda msg: WARNINGS.append(str(msg).strip()))
        mj._wobbl_quiet = True
    return mj


def available() -> bool:
    return load() is not None


def require():
    """Return mujoco, installing it first when this is the first mechanism study on this machine."""
    mj = load()
    if mj is not None:
        return mj
    from ..setup.installer import SetupError, install_on_demand
    try:
        install_on_demand(DOMAINS)
    except SetupError as e:
        raise SolverNotFound(f"MuJoCo is not installed and installing it failed ({e})", HINT) from None
    mj = load()
    if mj is None:
        raise SolverNotFound("MuJoCo is not installed", HINT)
    return mj


def unused(name: str) -> bool:
    """Whether a path inside the wheel is one Wobbl never loads, so it is neither unpacked nor kept."""
    parts = name.replace("\\", "/").strip("/").split("/")
    base = parts[-1]
    if "__pycache__" in parts or parts[0].endswith(".dist-info"):     # pip's records, which nothing here reads
        return True
    if parts[0] != "mujoco":
        return False
    return ((len(parts) > 2 and parts[1] in PRUNE_DIRS) or (len(parts) == 2 and base in PRUNE_FILES)
            or base.startswith(PRUNE_PREFIXES) or base.endswith(PRUNE_SUFFIXES))


def prune(root: Path) -> int:
    """Delete what Wobbl never uses from an unpacked MuJoCo wheel; return the bytes removed."""
    removed = 0
    for p in sorted(root.rglob("*"), reverse=True):
        rel = p.relative_to(root).as_posix()
        if p.is_file() and unused(rel):
            removed += p.stat().st_size
            p.unlink()
        elif p.is_dir() and not any(p.iterdir()):
            p.rmdir()
    return removed


def _smoke(stage: Path) -> str:
    """Import the unpacked copy in a fresh Python and step a one body model; '' when it works, else the error."""
    try:
        r = subprocess.run([sys.executable, "-I", "-c", SMOKE, str(stage)], capture_output=True, text=True,
                           timeout=120)
    except Exception as e:  # noqa: BLE001
        return str(e)
    return "" if r.returncode == 0 else (r.stderr.strip().splitlines() or ["failed"])[-1][:200]


def install(log: Log = print, force: bool = False) -> dict:
    """Fetch, verify, unpack and prune MuJoCo into ~/.wobbl/py/mujoco. Returns what it did, with sizes."""
    from ..chem.install import pick_wheel, tree_size
    from ..setup.installer import SetupError, download

    dest = lib_dir()
    marker = dest / "wobbl-mujoco.json"
    if (dest / "mujoco").is_dir() and marker.exists() and not force:
        return {**json.loads(marker.read_text()), "already": True}
    tmp = Path(tempfile.mkdtemp(prefix="wobbl-mujoco-"))
    try:
        meta = download(PYPI_JSON.format(v=MUJOCO_VERSION), tmp / "release.json", log)
        f = pick_wheel(json.loads(meta.read_text()).get("urls", []))
        if f is None:
            raise SetupError(f"MuJoCo {MUJOCO_VERSION} has no wheel for Python {sys.version_info.major}."
                             f"{sys.version_info.minor} on this platform, and 'pip install mujoco' may find a newer "
                             "one")
        log(f"downloading {f['filename']} ({f['size'] / 1e6:.1f} MB)")
        whl = download(f["url"], tmp / f["filename"], log)
        want = (f.get("digests") or {}).get("sha256", "")
        if want and hashlib.sha256(whl.read_bytes()).hexdigest() != want:
            raise SetupError("the MuJoCo download does not match PyPI's checksum, so run the install again")
        stage = dest.with_name("mujoco.partial")
        shutil.rmtree(stage, ignore_errors=True)
        done = False
        try:
            # the parts Wobbl prunes are never written, which keeps the peak on disk near the final size
            with zipfile.ZipFile(whl) as z:
                files = [i for i in z.infolist() if not i.is_dir()]
                unpacked = sum(i.file_size for i in files)
                for i in files:
                    if not unused(i.filename):
                        z.extract(i, stage)
            err = _smoke(stage)
            pruned = not err
            if err:   # a later release may need a part Wobbl prunes; keep it whole rather than broken
                log(f"the pruned MuJoCo did not import ({err}), so Wobbl keeps the whole wheel")
                with zipfile.ZipFile(whl) as z:
                    z.extractall(stage)
                err = _smoke(stage)
                if err:
                    raise SetupError(f"MuJoCo {MUJOCO_VERSION} does not import on this machine ({err})")
            info = {"version": MUJOCO_VERSION, "wheel": f["filename"], "download_bytes": int(f["size"]),
                    "unpacked_bytes": unpacked, "installed_bytes": tree_size(stage), "pruned": pruned,
                    "path": str(dest)}
            (stage / "wobbl-mujoco.json").write_text(json.dumps(info, indent=1))
            shutil.rmtree(dest, ignore_errors=True)
            stage.rename(dest)
            done = True
        finally:
            if not done:     # a full disk or a failed check leaves nothing half unpacked behind
                shutil.rmtree(stage, ignore_errors=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)    # the download is not kept
    log(f"MuJoCo {MUJOCO_VERSION}: {info['download_bytes'] / 1e6:.1f} MB download, "
        f"{info['unpacked_bytes'] / 1e6:.1f} MB unpacked, {info['installed_bytes'] / 1e6:.1f} MB after pruning")
    return info


def check(log: Log = print) -> dict | str:
    """Swing a one meter pendulum through a quarter period, the way the setup registry checks a solver."""
    from ..chem.install import tree_size

    mj = load()
    if mj is None:
        return "MuJoCo is not installed, and it installs the first time you run a mechanism study"
    ok, err = False, ""
    try:
        m = mj.MjModel.from_xml_string(
            "<mujoco><option timestep='0.0005'/><worldbody><body><joint axis='0 1 0'/>"
            "<geom type='sphere' size='0.001' pos='0 0 -1' mass='1'/></body></worldbody></mujoco>")
        d = mj.MjData(m)
        d.qpos[0] = 0.05
        while d.time < 0.25 * 2.00641:     # a quarter of the small swing period of a 1 m pendulum
            mj.mj_step(m, d)
        ok = bool(abs(d.qpos[0]) < 0.002)
        err = "" if ok else f"a 1 m pendulum was at {d.qpos[0]:.4f} rad after a quarter swing, not 0"
    except Exception as e:  # noqa: BLE001
        err = str(e)[:200]
    where = Path(mj.__file__).parent
    mine = where.resolve().is_relative_to(lib_dir().resolve())
    size = f" ({tree_size(lib_dir()) / 1e6:.1f} MB)" if mine else ""
    log(f"MuJoCo {mj.__version__}{size} {'works' if ok else 'FAILED ' + err}")
    return {"path": str(where) + size, "version": mj.__version__, "works": ok, "error": err}
