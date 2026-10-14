"""Solver adapter interface and registry.

A solver adapter knows how to turn a solver neutral model (``wobbl.fe``) into
input files, run the external program and read the results back. Domains
(structural, thermal, circuits, and later CFD and motion) ask the registry for
an adapter instead of importing one directly, so a new solver is one new file.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


class SolverError(RuntimeError):
    """The external solver failed. ``hint`` says what the user can do about it."""

    def __init__(self, message: str, hint: str = "", log: str = ""):
        super().__init__(message)
        self.hint = hint
        self.log = log

    def __str__(self) -> str:
        base = super().__str__()
        return f"{base}. {self.hint}" if self.hint else base


class SolverNotFound(SolverError):
    pass


@dataclass
class SolverInfo:
    name: str
    domains: tuple[str, ...]
    path: Path | None
    version: str = ""
    note: str = ""

    @property
    def available(self) -> bool:
        return self.path is not None


class SolverAdapter(ABC):
    name: str = ""
    domains: tuple[str, ...] = ()
    install_hint: str = "Run 'wobbl setup' to install the solvers."

    @abstractmethod
    def locate(self) -> Path | None: ...

    def version(self) -> str:
        return ""

    def info(self) -> SolverInfo:
        p = self.locate()
        return SolverInfo(self.name, self.domains, p, self.version() if p else "", "" if p else self.install_hint)

    def require(self) -> Path:
        """The solver's path, installing it first when this is the first study of its domain on this machine."""
        p = self.locate()
        if p is None and self.domains:
            from ..setup.installer import SetupError, install_on_demand
            try:
                if install_on_demand(self.domains):
                    p = self.locate()
            except SetupError as e:
                raise SolverNotFound(f"{self.name} is not installed and installing it failed ({e})",
                                     self.install_hint) from None
        if p is None:
            raise SolverNotFound(f"{self.name} is not installed", self.install_hint)
        return p


class PlannedAdapter(SolverAdapter):
    """A domain on the roadmap: registered so 'wobbl doctor' and agents can see it is coming."""

    def __init__(self, name: str, domains: tuple[str, ...], version: str):
        self.name = name
        self.domains = domains
        self.install_hint = f"{name} support is planned for Wobbl {version}."

    def locate(self) -> Path | None:
        return None


_REGISTRY: dict[str, SolverAdapter] = {}


def register(adapter: SolverAdapter) -> SolverAdapter:
    _REGISTRY[adapter.name.lower()] = adapter
    return adapter


def adapters() -> list[SolverAdapter]:
    _load_builtin()
    return list(_REGISTRY.values())


def for_domain(domain: str) -> SolverAdapter:
    """First available adapter for a domain, else the first registered one (whose require() explains)."""
    _load_builtin()
    cands = [a for a in _REGISTRY.values() if domain in a.domains]
    if not cands:
        raise SolverNotFound(f"no solver handles '{domain}'")
    for a in cands:
        if a.locate() is not None:
            return a
    return cands[0]


def get(name: str) -> SolverAdapter:
    _load_builtin()
    return _REGISTRY[name.lower()]


def _load_builtin():
    if _REGISTRY:
        return
    from . import calculix, ngspice  # noqa: F401  (they register themselves)
    register(PlannedAdapter("OpenFOAM", ("cfd",), "0.3"))
    register(PlannedAdapter("MuJoCo", ("mechanisms",), "0.4"))
    register(PlannedAdapter("Elmer", ("multiphysics",), "0.3"))
