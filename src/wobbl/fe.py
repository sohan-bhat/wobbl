"""Solver neutral finite element models and results.

The domain layer (structural, thermal) turns a question into one of these
models; a solver adapter (CalculiX today) turns the model into numbers. Keeping
this boundary explicit is what lets new solvers plug in without touching the
verification and reporting code.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .mesh.mesher import Mesh


@dataclass
class NodeConstraint:
    """Zero displacement for some degrees of freedom (1=x, 2=y, 3=z) of a node set."""
    name: str
    nodes: np.ndarray
    dofs: tuple[int, ...] = (1, 2, 3)


@dataclass
class NormalConstraint:
    """Zero displacement along a direction (a roller or symmetry plane that is not axis aligned)."""
    name: str
    nodes: np.ndarray
    normal: np.ndarray


@dataclass
class RigidCoupling:
    """Nodes that move together as a rigid body, driven by a force and moment at a reference point."""
    name: str
    nodes: np.ndarray
    ref_point: np.ndarray
    force: np.ndarray = field(default_factory=lambda: np.zeros(3))
    moment: np.ndarray = field(default_factory=lambda: np.zeros(3))


@dataclass
class StaticModel:
    mesh: Mesh
    E: float                      # MPa
    nu: float
    density: float                # t/mm3
    constraints: list[NodeConstraint] = field(default_factory=list)
    normal_constraints: list[NormalConstraint] = field(default_factory=list)
    forces: np.ndarray | None = None          # (n,3) nodal forces, N
    gravity: np.ndarray | None = None         # body acceleration vector, mm/s2
    rigid: list[RigidCoupling] = field(default_factory=list)
    label: str = "static"

    def nodal_forces(self) -> np.ndarray:
        return self.forces if self.forces is not None else np.zeros((self.mesh.n_nodes, 3))

    def applied_total(self) -> tuple[np.ndarray, np.ndarray]:
        """Total applied force (N) and moment about the origin (N mm), including gravity and rigid loads."""
        F = self.nodal_forces()
        X = self.mesh.X
        f = F.sum(0)
        m = np.cross(X, F).sum(0)
        for r in self.rigid:
            f = f + r.force
            m = m + np.cross(r.ref_point, r.force) + r.moment
        if self.gravity is not None:
            mass, com = mass_properties(self.mesh, self.density)
            f = f + mass * self.gravity
            m = m + np.cross(com, mass * self.gravity)
        return f, m


@dataclass
class StaticResult:
    U: np.ndarray                 # (n,3) displacement, mm
    S: np.ndarray                 # (n,6) stress xx,yy,zz,xy,yz,zx, MPa (nodal, averaged)
    RF: np.ndarray                # (n,3) reaction forces at constrained nodes, N
    extra_U: dict[str, np.ndarray] = field(default_factory=dict)   # rigid body reference node motions
    t_solve: float = 0.0
    log: str = ""
    n_dof: int = 0
    solver: str = ""

    def von_mises(self) -> np.ndarray:
        sxx, syy, szz, sxy, syz, szx = self.S.T
        return np.sqrt(0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2)
                       + 3.0 * (sxy ** 2 + syz ** 2 + szx ** 2))

    def principal(self) -> np.ndarray:
        """(n,3) principal stresses sorted ascending."""
        T = np.zeros((len(self.S), 3, 3))
        sxx, syy, szz, sxy, syz, szx = self.S.T
        T[:, 0, 0], T[:, 1, 1], T[:, 2, 2] = sxx, syy, szz
        T[:, 0, 1] = T[:, 1, 0] = sxy
        T[:, 1, 2] = T[:, 2, 1] = syz
        T[:, 0, 2] = T[:, 2, 0] = szx
        return np.linalg.eigvalsh(T)

    def displacement_magnitude(self) -> np.ndarray:
        return np.linalg.norm(self.U, axis=1)


@dataclass
class HeatModel:
    mesh: Mesh
    conductivity: float           # mW/(mm K)
    density: float                # t/mm3
    specific_heat: float          # mJ/(t K)
    fixed_temps: list[tuple[str, np.ndarray, float]] = field(default_factory=list)   # (name, nodes, degC)
    films: list[tuple[str, np.ndarray, float, float]] = field(default_factory=list)  # (name, tri mask, h, T_inf)
    fluxes: list[tuple[str, np.ndarray, float]] = field(default_factory=list)        # (name, tri mask, q mW/mm2)
    radiation: list[tuple[str, np.ndarray, float, float]] = field(default_factory=list)  # (name, mask, emissivity, T_env)
    volumetric: float = 0.0       # heat generation, mW/mm3, over the whole body
    initial_temp: float = 20.0
    transient: bool = False
    duration: float = 0.0         # s
    time_step: float = 0.0        # initial step, s
    output_times: list[float] = field(default_factory=list)
    ramp_time: float = 0.0        # s, a held temperature away from initial_temp rises to it linearly over this time


@dataclass
class HeatResult:
    T: np.ndarray                 # (n,) final temperature, degC
    history_t: np.ndarray = field(default_factory=lambda: np.zeros(0))      # output times, s
    history_T: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))  # (steps, n)
    flux: np.ndarray | None = None
    t_solve: float = 0.0
    log: str = ""
    solver: str = ""
    reaction_flux: np.ndarray | None = None   # (n,) heat flow at fixed temperature nodes, mW
    reaction_flux_history: np.ndarray | None = None   # (steps, n) the same at each output time of a transient


def mass_properties(mesh: Mesh, density: float) -> tuple[float, np.ndarray]:
    c = mesh.X[mesh.tets[:, :4]]
    v = np.abs(np.einsum("ij,ij->i", c[:, 1] - c[:, 0], np.cross(c[:, 2] - c[:, 0], c[:, 3] - c[:, 0]))) / 6.0
    cen = c.mean(1)
    vol = v.sum()
    return float(vol * density), (cen * v[:, None]).sum(0) / max(vol, 1e-300)
