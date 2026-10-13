"""Find the solver binaries Wobbl installed (or ones already on the machine)."""
from __future__ import annotations

import os
import platform
import shutil
import sys
from pathlib import Path


def wobbl_home() -> Path:
    return Path(os.environ.get("WOBBL_HOME", "~/.wobbl")).expanduser()


def env_prefix(name: str = "env") -> Path:
    return wobbl_home() / name


def bin_dirs(prefix: Path) -> list[Path]:
    if sys.platform == "win32":
        return [prefix / "Library" / "bin", prefix / "Scripts", prefix, prefix / "bin"]
    return [prefix / "bin"]


def find_tool(names: list[str], env_var: str | None = None, prefixes: tuple[str, ...] = ("env",)) -> Path | None:
    """Look in $env_var, then Wobbl's own environments, then PATH."""
    if env_var and os.environ.get(env_var):
        p = Path(os.environ[env_var]).expanduser()
        if p.exists():
            return p
    exe = ".exe" if sys.platform == "win32" else ""
    for pre in prefixes:
        for d in bin_dirs(env_prefix(pre)):
            for n in names:
                for cand in (d / n, d / (n + exe)):
                    if cand.is_file() and os.access(cand, os.X_OK):
                        return cand
    for n in names:
        w = shutil.which(n)
        if w:
            return Path(w)
    return None


def tool_env(tool: Path, threads: int | None = None) -> dict[str, str]:
    """Environment for running a tool from a conda prefix: its library dirs on PATH, thread counts set."""
    env = dict(os.environ)
    d = tool.parent
    extra = [str(d)]
    if sys.platform == "win32":
        root = d.parent if d.name.lower() == "bin" and d.parent.name.lower() == "library" else d
        prefix = root.parent if root.name.lower() == "library" else root
        extra += [str(prefix / "Library" / "bin"), str(prefix / "Library" / "mingw-w64" / "bin"), str(prefix)]
    env["PATH"] = os.pathsep.join(extra + [env.get("PATH", "")])
    n = str(threads or min(4, os.cpu_count() or 1))
    for k in ("OMP_NUM_THREADS", "CCX_NPROC_EQUATION_SOLVER", "CCX_NPROC_RESULTS", "CCX_NPROC_STIFFNESS",
              "OPENBLAS_NUM_THREADS"):
        env.setdefault(k, n)
    return env


def platform_tag() -> str:
    """conda platform string for this machine."""
    m = platform.machine().lower()
    if sys.platform == "darwin":
        return "osx-arm64" if m in ("arm64", "aarch64") else "osx-64"
    if sys.platform == "win32":
        return "win-64"
    if m in ("aarch64", "arm64"):
        return "linux-aarch64"
    if m in ("ppc64le",):
        return "linux-ppc64le"
    return "linux-64"
