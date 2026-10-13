"""ngspice adapter: runs SPICE netlists in batch mode and reads the vectors back."""
from __future__ import annotations

import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from .base import SolverAdapter, SolverError, register
from .locate import find_tool, tool_env


class NgSpice(SolverAdapter):
    name = "ngspice"
    domains = ("circuit",)

    def locate(self) -> Path | None:
        # on Windows ngspice.exe is the GUI build; the console build answers batch runs and -v promptly
        names = ["ngspice_con", "ngspice"] if sys.platform == "win32" else ["ngspice", "ngspice_con"]
        return find_tool(names, "WOBBL_NGSPICE", prefixes=("env", "env-x86"))

    def version(self) -> str:
        p = self.locate()
        if not p:
            return ""
        try:
            r = subprocess.run([str(p), "-v"], capture_output=True, text=True, timeout=20, env=tool_env(p),
                               stdin=subprocess.DEVNULL)
            m = re.search(r"ngspice[- ]?(\d+)", r.stdout + r.stderr, re.I)
            return m.group(1) if m else ""
        except Exception:  # noqa: BLE001
            return ""

    def run(self, netlist: str, workdir: str | Path, vectors: list[str], analysis_name: str = "run",
            timeout: float = 300) -> tuple[dict[str, np.ndarray], str, float]:
        """Run a netlist (without .control) and return the requested vectors from the last analysis."""
        exe = self.require()
        work = Path(workdir)
        work.mkdir(parents=True, exist_ok=True)
        data = work / f"{analysis_name}.txt"
        data.unlink(missing_ok=True)
        ctrl = ["", ".control", "set noaskquit", "set wr_singlescale", "set wr_vecnames", "set filetype=ascii",
                "run", f"wrdata {data.name} {' '.join(vectors)}", "quit", ".endc"]
        body = netlist.rstrip()
        if body.lower().endswith(".end"):
            body = body[: -4].rstrip()
        cir = work / f"{analysis_name}.cir"
        cir.write_text(body + "\n" + "\n".join(ctrl) + "\n.end\n")
        t0 = time.time()
        try:
            r = subprocess.run([str(exe), "-b", cir.name], cwd=work, capture_output=True, text=True,
                               timeout=timeout, env=tool_env(exe), stdin=subprocess.DEVNULL)
        except subprocess.TimeoutExpired:
            raise SolverError(f"ngspice did not finish within {timeout:.0f} s",
                              "Shorten the simulated time or raise the time step") from None
        log = (r.stdout or "") + (r.stderr or "")
        if not data.exists():
            errs = [ln for ln in log.splitlines() if "error" in ln.lower()]
            raise SolverError("ngspice failed: " + (" ".join(errs[:4]) or log[-400:]), log=log[-3000:])
        return _read_wrdata(data), log, time.time() - t0


def _read_wrdata(path: Path) -> dict[str, np.ndarray]:
    lines = path.read_text().splitlines()
    header = lines[0].split()
    rows = [list(map(float, ln.split())) for ln in lines[1:] if ln.strip()]
    arr = np.array(rows) if rows else np.zeros((0, len(header)))
    return {name: arr[:, i] for i, name in enumerate(header) if i < arr.shape[1]}


register(NgSpice())
