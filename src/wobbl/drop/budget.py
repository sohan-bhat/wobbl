"""How long a drop will take, worked out before it is solved, and how to keep it inside a time budget.

An explicit march costs the number of steps times the work of one step. A step multiplies the stiffness matrix
by a vector, so it costs about the matrix's nonzeros (about 220 per quadratic tet node). The number of steps is the
duration of the impact over the stable time step, and the time step is the shortest element over the wave speed,
so halving the element size multiplies the cost by about sixteen: eight times the nodes and twice the steps.
Stress recovery on the kept frames is paid once per node as well.

The constants below are fit on the enclosure example (a 3 mm walled shell, meshes of 12 to 45 thousand nodes) on
two threads, and the study reports the estimate beside the time it measured. Thin walls hold the node count up
whatever element size is asked for, so the levers that really cut the cost are the curvature setting (fewer
elements round every fillet and hole, which cuts nodes and lengthens the step), and a little more mass scaling.
"""
from __future__ import annotations

from dataclasses import dataclass, field

DEFAULT_BUDGET = 600.0       # s, the time a drop study may be predicted to take before it coarsens its mesh
STEP_NNZ = 8.0e-9            # s per stiffness nonzero per step, two threads, with the contact and the logs
FRAME_NODE = 1.0e-5          # s per node per kept frame, stress recovery and the checks that read it
SETUP_NODE = 1.0e-3          # s per node, meshing and building the matrices
DT_WMAX = 1.15               # the step is about this over the mesh's top frequency, contact springs included
SCREEN_SHARE = 0.05          # a linear tet landing, stopped as the part leaves the floor, against a full one
REFINE_SHARE = 2.5           # the refined level: a few more nodes and a time step half as long
FRAMES = 150
# What can be given up, finest first: (element size factor, elements per circle round curves, mass scaling cap).
RUNGS = [(1.0, 12, 0.02), (1.0, 8, 0.02), (1.0, 6, 0.02), (1.0, 6, 0.05), (1.4, 4, 0.05), (2.0, 4, 0.05)]


@dataclass
class DropCost:
    """Predicted seconds of one drop study and what it rests on."""
    seconds: float
    nodes: int
    steps: int
    dt: float
    t_end: float
    runs: dict = field(default_factory=dict)       # name -> seconds
    fit: str = ""                                  # what was given up to fit the budget, in words
    fit: str = ""                                  # what was given up to fit the budget, in words

    def to_dict(self) -> dict:
        return {"seconds": self.seconds, "nodes": self.nodes, "steps": self.steps, "dt": self.dt,
                "t_end": self.t_end, "runs": self.runs, "fit": self.fit}


def stable_dt(wmax: float) -> float:
    """The explicit step of a mesh whose top frequency is wmax (rad/s): shortest element over the wave speed,
    in the form the integrator uses it, 80% of the stable limit."""
    return DT_WMAX / max(wmax, 1e-30)


def run_seconds(nodes: float, nnz: float, dt: float, t_end: float, frames: int = FRAMES) -> float:
    """One quadratic tet landing: its steps plus the stress recovery on its frames."""
    return (t_end / max(dt, 1e-30)) * STEP_NNZ * nnz + FRAME_NODE * nodes * frames


def study_cost(nodes: float, nnz: float, wmax: float, t_end: float, landings: int, top: int,
               refine: bool = True) -> DropCost:
    """The whole study: set up the mesh, screen the landings on linear tets when there are several, run the worst
    few on quadratic tets and refine the worst. t_end is the longest a landing is marched, so this is a ceiling."""
    dt = stable_dt(wmax)
    one = run_seconds(nodes, nnz, dt, t_end)
    runs = {"setup": SETUP_NODE * nodes}
    if landings > 1:
        runs["screen"] = landings * SCREEN_SHARE * one
    runs["quadratic"] = max(1, top) * one
    if refine:
        runs["refined"] = REFINE_SHARE * one
    return DropCost(sum(runs.values()), int(nodes), int(t_end / dt), dt, t_end, runs)


def hhmm(s: float) -> str:
    return f"{s:.0f} s" if s < 90 else f"{s / 60:.0f} min"
