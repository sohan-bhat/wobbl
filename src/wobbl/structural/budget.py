"""A strength study's time budget, spent by a cost model of mesh sizes instead of read off the clock.

Refinement stops when the next mesh would not fit the budget. Judged by the clock, that decision moved with how
busy or how fast the machine was, so the level reached, and with it the answer, could change from one run to
the next. Here every mesh and solve is charged a modeled cost from its node count, so the same study stops at
the same level every time, on any machine. The model is calibrated on a 2018 six core laptop (Intel i7-8750H,
CalculiX on four threads); a faster machine simply finishes sooner.

The clock is still a backstop at three times the budget, for a machine far slower than the model or a solver
that stalls. A study stopped that way says so, and only then can the level reached depend on the machine.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

BACKSTOP = 3.0   # the wall clock stops a study at this multiple of its budget


def solve_cost(n_nodes: int) -> float:
    """Modeled seconds for one linear static solve of a second order tet mesh, reading the results included."""
    return 1.0 + 1.6 * (n_nodes / 1e4) ** 1.4


def mesh_cost(n_nodes: int) -> float:
    """Modeled seconds to mesh: serial gmsh makes about 15,000 nodes a second."""
    return 0.5 + n_nodes / 15000.0


@dataclass
class Budget:
    seconds: float
    start: float = field(default_factory=time.time)
    spent: float = 0.0           # modeled seconds charged so far

    def charge(self, seconds: float):
        self.spent += seconds

    def charge_level(self, n_nodes: int, solves: int = 1, meshed: bool = True):
        """One refinement level: a mesh (unless it reuses one) and its solves (several with wall contact)."""
        self.spent += (mesh_cost(n_nodes) if meshed else 0.0) + max(1, solves) * solve_cost(n_nodes)

    def next_level_fits(self, n_nodes: int) -> bool:
        """Whether one more level after a mesh of ``n_nodes`` fits: its solve can cost 2.6 times this one (the
        mesh grows, and a direct solve grows faster than the mesh) and its meshing about twice as long."""
        return self.spent + 2.6 * solve_cost(n_nodes) + 2.0 * mesh_cost(n_nodes) <= self.seconds

    def clock(self) -> float:
        return time.time() - self.start

    def clock_ran_out(self) -> bool:
        return self.clock() > BACKSTOP * self.seconds

    def timeout(self, floor: float = 120.0) -> float:
        """Wall clock limit for one solver run: only a stalled solver or a very slow machine reaches it."""
        return max(floor, BACKSTOP * self.seconds - self.clock())
