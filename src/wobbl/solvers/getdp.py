"""GetDP adapter: finite element field solves for magnetic, electric, radio and sound studies.

GetDP (getdp.info, GPL) is the solver that comes with gmsh. It handles static
and time harmonic Maxwell problems, the Helmholtz equation and eigenvalue
problems on gmsh meshes. Wobbl downloads one pinned, self contained binary the
first time a field study runs, keeps only that binary and its license, and
deletes the archive, so nothing is installed up front.
"""
from __future__ import annotations

import hashlib
import os
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import time
import zipfile
from pathlib import Path

from .base import SolverAdapter, SolverError, register
from .locate import find_tool, platform_tag, tool_env, wobbl_home

VERSION = "3.5.0"
URL = "https://getdp.info/bin/{}"
# complex arithmetic builds (time harmonic and eigen solves need them), checksums of the archives as published
BUILDS = {
    "osx-64": ("MacOSX/getdp-3.5.0-MacOSXc.tgz",
               "ba514f4d6009b385ff27f6083f2843ff11a4cb1f3a829fdc0a5f8a2613646077", "getdp-3.5.0-MacOSX/bin/getdp"),
    "linux-64": ("Linux/getdp-3.5.0-Linux64c.tgz",
                 "d3c28fa18f20d6147b4c7367d4dd802e9f7ddb58c608688bbb71919dbca8041d", "getdp-3.5.0-Linux64/bin/getdp"),
    "win-64": ("Windows/getdp-3.5.0-Windows64c.zip",
               "9b07461586f669ed847bff70a70223d385add769e78c29a6f7bc6c7de5f858c3", "getdp-3.5.0-Windows64/getdp.exe"),
}
BUILDS["osx-arm64"] = BUILDS["osx-64"]  # the Intel build runs under Rosetta 2, like ngspice
DOMAINS = ("magnetic", "electric", "board", "antenna", "sound")
THREADS = 2


def home() -> Path:
    return wobbl_home() / "getdp"


class GetDP(SolverAdapter):
    name = "GetDP"
    domains = DOMAINS
    install_hint = "Run 'wobbl setup magnetic' (or any field domain) to fetch GetDP, about 11 MB."

    def locate(self) -> Path | None:
        return find_tool(["getdp"], "WOBBL_GETDP", prefixes=("getdp",))

    def version(self) -> str:
        p = self.locate()
        if not p:
            return ""
        try:
            r = subprocess.run([str(p), "--version"], capture_output=True, text=True, timeout=20,
                               stdin=subprocess.DEVNULL, env=tool_env(p, THREADS))
            m = re.search(r"(\d+\.\d+\.\d+)", r.stdout + r.stderr)
            return m.group(1) if m else ""
        except Exception:  # noqa: BLE001
            return ""

    def require(self) -> Path:
        """The binary, fetched on first use unless WOBBL_NO_AUTO_INSTALL is set."""
        p = self.locate()
        if p is None and not os.environ.get("WOBBL_NO_AUTO_INSTALL"):
            size = install(lambda m: None, notice=True)
            p = self.locate() if size.get("installed_bytes") else None
        if p is None:
            return super().require()
        return p

    def run(self, pro: str, mesh: Path, workdir: str | Path, resolution: str, postop: str, name: str = "model",
            timeout: float = 600, verbosity: int = 3) -> str:
        """Write the .pro next to the mesh, solve and post process. Returns the log."""
        exe = self.require()
        work = Path(workdir)
        work.mkdir(parents=True, exist_ok=True)
        (work / f"{name}.pro").write_text(pro)
        msh = os.path.relpath(Path(mesh).resolve(), work.resolve())
        cmd = [str(exe), f"{name}.pro", "-msh", msh, "-solve", resolution, "-pos", postop, "-v", str(verbosity)]
        env = tool_env(exe, THREADS)
        env["OMP_NUM_THREADS"] = str(THREADS)
        try:
            r = subprocess.run(cmd, cwd=work, capture_output=True, text=True, timeout=timeout, env=env,
                               stdin=subprocess.DEVNULL)
        except subprocess.TimeoutExpired:
            raise SolverError(f"GetDP did not finish within {timeout:.0f} s",
                              "Use a coarser mesh or fewer sweep points") from None
        log = (r.stdout or "") + (r.stderr or "")
        errs = [ln.split(":", 1)[-1].strip() for ln in log.splitlines() if ln.startswith("Error")]
        if r.returncode != 0 or errs:
            raise SolverError("GetDP failed: " + (" ".join(errs[:3]) or log[-400:]), log=log[-4000:])
        return log


def install(log=print, notice: bool = False, force: bool = False) -> dict:
    """Fetch the pinned binary into ~/.wobbl/getdp, keep only it and its license, and report the sizes."""
    plat = platform_tag()
    if plat not in BUILDS:
        raise SolverError(f"GetDP publishes no build for {plat}",
                          "Build GetDP from getdp.info and set WOBBL_GETDP to its path")
    rel, sha, member = BUILDS[plat]
    dest = home()
    exe = dest / "bin" / Path(member).name
    if exe.exists() and not force:
        return {"installed_bytes": _size(dest), "path": str(exe), "already": True}
    if notice:
        print(f"wobbl: fetching GetDP {VERSION} for field studies (about 11 MB, once) into {dest}", file=sys.stderr)
    from ..setup.installer import download

    tmp = dest / ".download"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True, exist_ok=True)
    archive = tmp / Path(rel).name
    t0 = time.time()
    log(f"downloading GetDP {VERSION} for {plat}")
    try:
        download(URL.format(rel), archive, log)
        fetched = archive.stat().st_size
        got = hashlib.sha256(archive.read_bytes()).hexdigest()
        if got != sha:
            raise SolverError("the GetDP download does not match its published checksum", "Run 'wobbl setup' again")
        full = _extract(archive, member, exe)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)  # the archive and anything half written
    exe.chmod(exe.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    out = {"download_bytes": fetched, "unpacked_bytes": full,
           "installed_bytes": _size(dest), "path": str(exe), "seconds": round(time.time() - t0, 1)}
    log(f"GetDP {VERSION}: {full / 1e6:.0f} MB unpacked, {out['installed_bytes'] / 1e6:.0f} MB kept after pruning "
        "the docs, examples and templates")
    return out


def _extract(archive: Path, member: str, exe: Path) -> int:
    """Unpack only the binary and the license. Returns what a full unpack would have taken, in bytes."""
    exe.parent.mkdir(parents=True, exist_ok=True)
    lic = exe.parent.parent / "LICENSE.txt"
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as z:
            full = sum(i.file_size for i in z.infolist())
            with z.open(member) as src, open(exe, "wb") as dst:
                shutil.copyfileobj(src, dst)
            lic_name = next((n for n in z.namelist() if n.endswith("LICENSE.txt")), None)
            if lic_name:
                lic.write_bytes(z.read(lic_name))
        return full
    with tarfile.open(archive, "r:gz") as t:
        members = t.getmembers()
        full = sum(m.size for m in members if m.isfile())
        src = t.extractfile(member)
        if src is None:
            raise SolverError(f"{member} is missing from the GetDP archive")
        with open(exe, "wb") as dst:
            shutil.copyfileobj(src, dst)
        lic_m = next((m for m in members if m.name.endswith("LICENSE.txt")), None)
        if lic_m is not None:
            lic.write_bytes(t.extractfile(lic_m).read())
    return full


def _size(path: Path) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += os.lstat(os.path.join(root, f)).st_size
            except OSError:
                pass
    return total


def uninstall() -> None:
    shutil.rmtree(home(), ignore_errors=True)


register(GetDP())
