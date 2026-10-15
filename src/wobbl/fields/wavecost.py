"""How long a 3D antenna study will take, worked out from its mesh before any frequency is solved.

The cost of a run is its frequency solves. Each is a complex edge element system whose size is set by the mesh: about
seven unknowns a node (measured on the patch, dipole and loop meshes of the tests), and a direct solve of a 3D
system grows a little faster than the unknowns raised to the power 1.4. The checks at the end of a normal run solve
the working frequency twice more, on a mesh a quarter finer (2.4 times the unknowns) and with the absorbing sphere
1.4 times farther out (2.7 times the volume). The constants are fit to those measured runs, and the study reports its
prediction beside the time it took.

When the prediction is far over the budget the study gives up what it can without changing the answer's meaning, in
this order: fewer frequencies, a coarser mesh, no verification solves. If even that is far over, it refuses and says
what a cheaper question would be, as the drop study does.
"""
from __future__ import annotations

from dataclasses import dataclass, field

DEFAULT_BUDGET = 900.0            # s of wall time a study may be predicted to take before it gives something up
REFUSE_OVER = 2.5                 # a study predicted this many times over budget at its coarsest is refused
UNKNOWNS_PER_NODE = 7.1           # edge elements on tetrahedra, plus the absorbing sphere's surface unknowns
AIR_CELLS = 0.72                  # nodes per far sized cell of air
SKIN_CELLS = 1.09                 # nodes per cell of surface, with the graded mesh around it
EDGE_CELLS = 12.4                 # nodes per element length of a copper sheet's edge, which the mesh refines around
WIRE_CELLS = 24.4                 # nodes per element length of a thin wire, for the fine skirt that grows around it
CPU_AT_100K = 175.0               # cpu seconds for one frequency (solve and field sampling) at 100 thousand unknowns
EXPONENT = 1.4
MESH_CPU_PER_NODE = 4.0e-3        # cpu seconds a node costs to mesh
SPEEDUP = 2.0                     # wall time shrinks by this much on the threads a solve gets
BYTES_PER_UNKNOWN = 18_000        # peak memory of a direct solve, per unknown at 100 thousand (about 0.8 GB at 49 thousand)
FINE_FACTOR = (1 / 0.75) ** 3     # unknowns of the verification mesh against the base mesh
FAR_FACTOR = 1.4 ** 3
# element size factors of the qualities, the coarsest first (wave3d.QUALITY)
K = {"fast": 1.3, "normal": 1.0, "fine": 0.8}


class CostRefusal(ValueError):
    """The study would take far longer than its budget even at its coarsest."""

    def __init__(self, message: str, hint: str = "", plan: "Plan | None" = None):
        super().__init__(message)
        self.hint = hint
        self.plan = plan


@dataclass
class Plan:
    quality: str
    points: int                 # frequencies in the first sweep, with the working frequency and any hints
    refine: int                 # extra frequencies it may add where the response bends
    nodes: int
    unknowns: int
    solves: float               # frequency solves expected, verification included
    cpu: float
    seconds: float              # predicted wall time
    memory: float               # bytes at the peak
    verify: bool = True
    gave_up: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"quality": self.quality, "points": self.points, "refine": self.refine, "nodes": self.nodes,
                "unknowns": self.unknowns, "solves": self.solves, "cpu_s": self.cpu, "seconds": self.seconds,
                "memory_bytes": self.memory, "verify": self.verify, "gave_up": list(self.gave_up)}


def nodes_from(inp: dict) -> float:
    """The base mesh's node count from what sets its element sizes (wave3d.mesh_inputs): the air sphere's volume in
    cells of the far size, and each surface's area in cells of its own size, with the refinement that grows around
    it, a thin wire's fine skirt by its length in cells and a copper sheet's edge by its perimeter in cells. Fit to the
    node counts of eleven meshes (six patches, two dipoles, two loops and a trace antenna on a board), to 12% or
    better on all of them."""
    import math

    air = 4 / 3 * math.pi * inp["R"] ** 3 / inp["far"] ** 3
    skin = sum(a / (h * h) for a, h in inp["surfaces"])
    wires = sum(length / h for length, h in inp.get("wires", ()))
    edges = sum(length / h for length, h in inp.get("edges", ()))
    return AIR_CELLS * air + SKIN_CELLS * skin + WIRE_CELLS * wires + EDGE_CELLS * edges


def solve_cpu(unknowns: float) -> float:
    return CPU_AT_100K * (unknowns / 1e5) ** EXPONENT


def mesh_cpu(nodes: float) -> float:
    return MESH_CPU_PER_NODE * nodes


def cost(nodes: float, points: int, refine: int, verify: bool, speedup: float = SPEEDUP) -> tuple[float, float, float]:
    """(cpu seconds, wall seconds, solves) of a study on a base mesh of this many nodes. The extra frequencies the
    sweep adds are counted at three quarters of the most it may add."""
    n = nodes * UNKNOWNS_PER_NODE
    solves = points + 0.75 * refine
    cpu = solves * solve_cpu(n) + mesh_cpu(nodes)
    if verify:
        cpu += solve_cpu(n * FINE_FACTOR) + solve_cpu(n * FAR_FACTOR) + mesh_cpu(nodes * (FINE_FACTOR + FAR_FACTOR))
        solves += 2
    return cpu, cpu / speedup, solves


def memory(unknowns: float, verify: bool) -> float:
    big = unknowns * (max(FINE_FACTOR, FAR_FACTOR) if verify else 1.0)
    return BYTES_PER_UNKNOWN * big * (big / 1e5) ** 0.3


def assess(nodes: int, quality: str, points: int, refine: int, budget: float, speedup: float = SPEEDUP,
           hints: int = 0) -> Plan:
    """The study as asked if it fits the budget, else the richest version that does, else a refusal. nodes is the
    base mesh's node count at the quality asked for."""
    if quality not in K:
        raise ValueError(f"unknown quality '{quality}'. Use fast, normal or fine")
    nodes0, k0 = float(nodes), K[quality]
    give: list[str] = []
    rungs = [(quality, points, refine, quality != "fast", [])]
    p3 = max(min(points, 5), 3)
    if points > p3:
        rungs.append((quality, p3, min(refine, 3), quality != "fast", [f"{points - p3} fewer frequencies in the sweep"]))
    if quality != "fast":
        rungs.append(("fast", p3, min(refine, 3), False,
                      [f"{points - p3} fewer frequencies in the sweep"] * (points > p3) +
                      ["a coarser mesh (the 'fast' quality)", "no verification solves on finer or larger meshes"]))
    rungs.append(("fast", 3, min(refine, 2), False, ["the sweep cut to three frequencies and what the response asks for"]))
    plan = None
    for q, pts, ref, ver, why in rungs:
        n = nodes0 * (k0 / K[q]) ** 3
        cpu, wall, solves = cost(n, pts, ref, ver, speedup)
        plan = Plan(q, pts, ref, int(n), int(n * UNKNOWNS_PER_NODE), solves, cpu, wall, memory(n * UNKNOWNS_PER_NODE, ver),
                    ver, list(why))
        if budget <= 0 or wall <= budget:
            return plan
    assert plan is not None
    if plan.seconds > REFUSE_OVER * budget:
        raise CostRefusal(
            f"this study is predicted to take {_t(plan.seconds)} even with a coarse mesh and a short sweep "
            f"({plan.unknowns:,} unknowns a frequency), against a budget of {_t(budget)}",
            "Make the problem smaller or cheaper. Solve at a lower frequency (the mesh grows with the cube of the "
            "frequency), shrink the ground plane or the air around it, or raise the budget with --set budget=2h",
            plan)
    plan.gave_up.append("and it still runs over the budget")
    return plan


def _t(s: float) -> str:
    return f"{s:.0f} s" if s < 90 else (f"{s / 60:.0f} min" if s < 5400 else f"{s / 3600:.1f} h")
