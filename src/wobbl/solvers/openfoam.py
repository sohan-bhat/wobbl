"""OpenFOAM adapter: runs blockMesh and the incompressible solvers on a case folder Wobbl wrote, always as a single
process, and hands back the log.

Wobbl installs a lean OpenFOAM v2412 the first time a flow study needs it (wobbl.flow.install). An OpenFOAM the
user already has is used instead when WOBBL_OPENFOAM points at its folder, or when its environment is sourced
(WM_PROJECT_DIR). Each program is started directly with OpenFOAM's environment variables set, so no shell has to
source etc/bashrc, and the binaries find their libraries through their own run paths.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from .base import SolverAdapter, SolverError, register


class OpenFOAM(SolverAdapter):
    name = "OpenFOAM"
    domains = ("flow", "cfd")
    install_hint = ("Run 'wobbl setup --only openfoam' (under 200 MB kept), or let the first flow study install it. "
                    "WOBBL_OPENFOAM can point at an OpenFOAM you already have.")

    def root(self) -> Path | None:
        from ..flow.install import openfoam_home

        cands = [openfoam_home()]
        if os.environ.get("WM_PROJECT_DIR"):
            cands.append(Path(os.environ["WM_PROJECT_DIR"]))
        for c in cands:
            if bin_dir(c) is not None:
                return c
        return None

    def locate(self) -> Path | None:
        r = self.root()
        return bin_dir(r) / "simpleFoam" if r else None

    def version(self) -> str:
        r = self.root()
        if r is None:
            return ""
        for f in (r / "wobbl-openfoam.json",):
            try:
                return "v" + json.loads(f.read_text())["api"]
            except (OSError, ValueError, KeyError):
                pass
        try:
            m = re.search(r"api=(\d+)", (r / "META-INFO" / "api-info").read_text())
            return "v" + m.group(1) if m else ""
        except OSError:
            return ""

    def env(self) -> dict[str, str]:
        r = self.root()
        env = dict(os.environ)
        env.update({"WM_PROJECT_DIR": str(r), "WM_PROJECT": "OpenFOAM", "FOAM_SIGFPE": "1",
                    "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"})
        if self.version():
            env["WM_PROJECT_VERSION"] = self.version()
        if sys.platform.startswith("linux"):
            libs = [str(r / "lib"), str(r / "lib" / "dummy")]
            env["LD_LIBRARY_PATH"] = os.pathsep.join(libs + [env.get("LD_LIBRARY_PATH", "")])
        return env

    def run(self, app: str, case: str | Path, args: tuple[str, ...] = (), timeout: float = 1800,
            log_name: str = "", check: bool = True) -> str:
        """Run one OpenFOAM program on a case and return its log. A non zero exit or a FOAM FATAL ERROR raises,
        quoting the error, unless check is False (checkMesh exits non zero when the grid fails a check)."""
        exe = bin_dir(self.root()) / app if self.root() else None
        if exe is None or not exe.exists():
            self.require()
            raise SolverError(f"this OpenFOAM has no {app}", "Run 'wobbl setup --only openfoam --force'.")
        case = Path(case)
        log_path = case / f"log.{log_name or app}"
        t0 = time.time()
        with open(log_path, "w") as lf:
            try:
                p = subprocess.run([str(exe), "-case", str(case), *args], stdout=lf, stderr=subprocess.STDOUT,
                                   timeout=timeout, env=self.env(), stdin=subprocess.DEVNULL)
            except subprocess.TimeoutExpired:
                raise SolverError(f"{app} did not finish within {timeout:.0f} s",
                                  "Use a coarser grid or a shorter run") from None
        log = log_path.read_text(errors="replace")
        if check and (p.returncode != 0 or "FOAM FATAL" in log):
            m = re.search(r"--> FOAM FATAL.*?(?=\n\n|\Z)", log, re.S)
            msg = " ".join((m.group(0) if m else log[-600:]).split())[:500]
            raise SolverError(f"{app} failed: {msg}", log=log[-4000:])
        self.last_seconds = time.time() - t0
        return log


def bin_dir(root: Path | None) -> Path | None:
    """The folder of OpenFOAM programs under an install: bin for conda-forge's layout, platforms/<build>/bin for
    OpenFOAM's own."""
    if root is None:
        return None
    if (root / "bin" / "simpleFoam").is_file():
        return root / "bin"
    plats = root / "platforms"
    if plats.is_dir():
        for p in sorted(plats.iterdir()):
            if (p / "bin" / "simpleFoam").is_file():
                return p / "bin"
    return None


register(OpenFOAM())
