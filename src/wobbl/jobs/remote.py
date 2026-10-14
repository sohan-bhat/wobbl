"""Machines the user owns, reached over SSH: a workstation, a lab server or a rented box.

Wobbl keeps a name, a host as ssh understands it and a folder for jobs in ~/.wobbl/remotes.json, and nothing
else. Keys, passwords and agents stay with ssh itself, so Wobbl never sees or stores a credential, and every
command runs with BatchMode so ssh fails rather than prompting.

A job on a remote needs only a POSIX shell and what it runs there, CalculiX for a deck or Wobbl itself for any
Wobbl command of any domain. Wobbl sends the input files, starts the run under nohup so it outlives the
connection and the laptop's sleep, reads the small status files back when asked and fetches the compact results
at the end, leaving the large solver files on the remote until they are wanted.

LocalTransport runs every command in a folder on this machine the way ssh runs it on the remote, with the same
shell, paths and copies. The tests drive the whole remote flow through it, and 'wobbl remote add NAME
local:/some/folder' gives the same flow without a second machine.
"""
from __future__ import annotations

import json
import shlex
import shutil
import subprocess
import tarfile
from abc import ABC, abstractmethod
from pathlib import Path

from ..solvers.locate import wobbl_home

PROBE = ("echo os=$(uname -s); "
         "echo cores=$(nproc 2>/dev/null || sysctl -n hw.physicalcpu 2>/dev/null || echo 1); "
         "echo ram=$( (awk '/MemTotal/ {print $2*1024}' /proc/meminfo 2>/dev/null) || sysctl -n hw.memsize "
         "2>/dev/null || echo 0); "
         "echo ccx=$(command -v ccx || command -v ccx_2.23 || ls $HOME/.wobbl/env/bin/ccx 2>/dev/null); "
         "echo python=$(command -v python3); "
         "echo wobbl=$(command -v wobbl || ls $HOME/.local/bin/wobbl 2>/dev/null); "
         "echo free=$(df -Pk . | awk 'NR==2 {print $4*1024}')")


class RemoteError(RuntimeError):
    pass


class Transport(ABC):
    """Run shell commands and copy files under one root folder, here or on another machine."""
    name: str = ""

    @abstractmethod
    def run(self, command: str, timeout: float = 60.0) -> subprocess.CompletedProcess: ...

    @abstractmethod
    def put(self, local: Path, remote: str): ...

    @abstractmethod
    def get(self, remote: str, local: Path): ...

    def get_dir(self, remote: str, local: Path, exclude: tuple[str, ...] = ()):
        """A whole folder, leaving out the named entries at its top, streamed as one tar archive."""
        ex = " ".join(f"--exclude={shlex.quote('./' + e)}" for e in exclude)
        p = subprocess.Popen(self.argv(f"tar cf - {ex} -C {shlex.quote(remote)} ."), stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE)
        Path(local).mkdir(parents=True, exist_ok=True)
        try:
            with tarfile.open(fileobj=p.stdout, mode="r|") as tf:
                try:
                    tf.extractall(local, filter="data")
                except TypeError:  # Python before 3.12 has no extraction filter, so check each member by hand
                    for m in tf:
                        if m.name.startswith(("/", "..")) or ".." in Path(m.name).parts or m.issym() or m.islnk():
                            continue
                        tf.extract(m, local)
        except tarfile.ReadError:
            pass
        err = p.stderr.read().decode(errors="replace") if p.stderr else ""
        if p.wait() != 0:
            raise RemoteError(f"copying {remote} from {self.name} failed ({err.strip()[:300]})")

    def argv(self, command: str) -> list[str]:
        return ["sh", "-c", command]

    def read(self, remote: str, tail: int = 0) -> str:
        """A small text file, or its last ``tail`` bytes. Empty when it does not exist yet."""
        q = shlex.quote(remote)
        r = self.run(f"tail -c {int(tail)} {q} 2>/dev/null" if tail else f"cat {q} 2>/dev/null")
        return r.stdout if r.returncode == 0 else ""

    def check(self, command: str, timeout: float = 60.0) -> str:
        r = self.run(command, timeout)
        if r.returncode != 0:
            raise RemoteError(f"'{command}' failed on {self.name} ({(r.stderr or r.stdout).strip()[:300]})")
        return r.stdout


class LocalTransport(Transport):
    """A folder on this machine standing in for a remote root."""

    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()
        self.name = f"local:{self.root}"

    def run(self, command: str, timeout: float = 60.0) -> subprocess.CompletedProcess:
        # a session of its own, as an SSH connection has, so a job's process group never holds this process
        self.root.mkdir(parents=True, exist_ok=True)
        return subprocess.run(["sh", "-c", command], cwd=self.root, capture_output=True, text=True, timeout=timeout,
                              start_new_session=True)

    def get_dir(self, remote: str, local: Path, exclude: tuple[str, ...] = ()):
        src = self.root / remote
        if not src.is_dir():
            raise RemoteError(f"{remote} is not a folder on {self.name}")
        shutil.copytree(src, local, dirs_exist_ok=True,
                        ignore=lambda d, names: [n for n in names if Path(d) == src and n in exclude])

    def put(self, local: Path, remote: str):
        dest = self.root / remote
        dest.parent.mkdir(parents=True, exist_ok=True)
        if Path(local).is_dir():
            shutil.copytree(local, dest, dirs_exist_ok=True)
        else:
            shutil.copy2(local, dest)

    def get(self, remote: str, local: Path):
        src = self.root / remote
        Path(local).parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(src, local, dirs_exist_ok=True)
        else:
            shutil.copy2(src, local)


class SSHTransport(Transport):
    """ssh, and rsync or scp, with the user's own ssh setup."""

    def __init__(self, host: str, root: str = "wobbl-jobs", port: int | None = None):
        self.host, self.port = host, port
        self.root = root[2:] if root.startswith("~/") else root      # relative paths start at the home folder
        self.name = host

    def _opts(self) -> list[str]:
        return ["-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "-o", "ServerAliveInterval=30"]

    def argv(self, command: str) -> list[str]:
        return self.ssh_argv(command)

    def ssh_argv(self, command: str) -> list[str]:
        root = shlex.quote(self.root)
        return ["ssh", *self._opts(), *(["-p", str(self.port)] if self.port else []), self.host,
                f"mkdir -p {root} && cd {root} && {command}"]

    def run(self, command: str, timeout: float = 60.0) -> subprocess.CompletedProcess:
        try:
            return subprocess.run(self.ssh_argv(command), capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            raise RemoteError(f"{self.host} did not answer within {timeout:.0f} s") from None

    def _remote(self, path: str) -> str:
        return f"{self.host}:{self.root}/{path}"

    def copy_argv(self, src: str, dst: str, folder: bool) -> list[str]:
        if shutil.which("rsync"):
            ssh = " ".join(["ssh", *self._opts(), *(["-p", str(self.port)] if self.port else [])])
            return ["rsync", "-a", "--partial", "-e", ssh, src + ("/" if folder else ""), dst]
        return ["scp", *self._opts(), *(["-P", str(self.port)] if self.port else []), *(["-r"] if folder else []),
                src, dst]

    def put(self, local: Path, remote: str):
        parent = str(Path(remote).parent)
        if parent not in ("", "."):
            self.check(f"mkdir -p {shlex.quote(parent)}")
        r = subprocess.run(self.copy_argv(str(local), self._remote(remote), Path(local).is_dir()),
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise RemoteError(f"copying {local} to {self.host} failed ({r.stderr.strip()[:300]})")

    def get(self, remote: str, local: Path):
        Path(local).parent.mkdir(parents=True, exist_ok=True)
        r = subprocess.run(self.copy_argv(self._remote(remote), str(local), False), capture_output=True, text=True)
        if r.returncode != 0:
            raise RemoteError(f"copying {remote} from {self.host} failed ({r.stderr.strip()[:300]})")


# ------------------------------------------------------------------------------------------------- config
def config_path() -> Path:
    return wobbl_home() / "remotes.json"


def load() -> dict:
    try:
        return json.loads(config_path().read_text())
    except (OSError, ValueError):
        return {}


def save(cfg: dict):
    p = config_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(cfg, indent=1))


def transport(entry: dict) -> Transport:
    host = entry["host"]
    if host.startswith("local:"):
        return LocalTransport(host[len("local:"):])
    return SSHTransport(host, entry.get("root", "wobbl-jobs"), entry.get("port"))


def probe(t: Transport) -> dict:
    """Cores, memory, free disk and solvers on the other side, read once when the remote is added."""
    out = {}
    for line in t.check(PROBE, timeout=30).splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()

    def num(k):
        try:
            return float(out.get(k) or 0)
        except ValueError:
            return 0.0
    return {"os": out.get("os", ""), "cores": max(1, int(num("cores"))), "ram": num("ram"), "free": num("free"),
            "ccx": out.get("ccx", ""), "python": out.get("python", ""), "wobbl": out.get("wobbl", "")}


def add(name: str, host: str, root: str = "wobbl-jobs", port: int | None = None, ccx: str = "",
        check: bool = True, wobbl: str = "") -> dict:
    """Register a remote, probing it so estimates know its cores and memory."""
    if not name.replace("-", "").replace("_", "").isalnum():
        raise RemoteError("a remote name is letters, digits, - and _")
    entry = {"host": host, "root": root, **({"port": port} if port else {})}
    if check:
        entry.update(probe(transport(entry)))
    if ccx:
        entry["ccx"] = ccx
    if wobbl:
        entry["wobbl"] = wobbl
    cfg = load()
    cfg[name] = entry
    save(cfg)
    return entry


def remove(name: str) -> bool:
    cfg = load()
    if cfg.pop(name, None) is None:
        return False
    save(cfg)
    return True


def get(name: str) -> dict:
    cfg = load()
    if name not in cfg:
        known = ", ".join(cfg) or "none yet"
        raise RemoteError(f"no remote named '{name}' (the known ones are {known}). Add one with wobbl remote add {name} "
                          "user@host")
    return cfg[name]


def machines() -> dict:
    """Each configured remote as an estimate.Machine, with the cores and memory its probe found."""
    from ..estimate import Machine

    out = {}
    for name, e in load().items():
        cores = int(e.get("cores") or 1)
        out[name] = Machine(name, cores, cores, float(e.get("ram") or 8e9), float(e.get("free") or float("inf")))
    return out
