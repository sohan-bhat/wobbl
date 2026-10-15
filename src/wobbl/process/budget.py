"""How long a manufacturing study will take, worked out before it is solved, and how to give up mesh fineness
rung by rung until two mesh levels fit a time budget.

A mesh convergence check needs at least two meshes, so a part that is slow to solve cannot just be run once. The
shape of the fix is the one the drop studies use (wobbl.drop.budget). Predict the cost from the mesh, and when it is
over budget give up fineness in a fixed order, fewer elements round curves first and larger elements after. The
numbers are not the drop's, since a print warp is a static CalculiX solve, a casting an implicit enthalpy march and
a mold fill a front by front march, so each process has its own cost law, and the levels are meshed coarse to fine.

Meshing the finest rung of a slender truss costs minutes by itself, so the node counts of the rungs are not found by
meshing each one. Two pilot meshes are made at the coarsest rungs, which are cheap by construction, and every finer
rung is priced from them. The count grows as the element size to the minus a, where the pilots measure a (about
1.7 for a thin walled bracket and under one for a truss whose curved tubes, not its element size, set the count), and
as the elements per circle to the power 1.5 on curved faces, and the count is taken to grow by the larger of the
two. The study then meshes the rung it chose, and prices any level it adds on top from the mesh it really has.

The constants are fit on this machine under a load of about thirty on twelve cores, so they run on the slow side,
and the study reports the prediction beside what it measured.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

DEFAULT_BUDGET = 600.0     # s, the time the mesh levels of one study may be predicted to take
MIN_LEVELS = 2
# What can be given up, finest first: (element size factor, elements per circle round curves).
RUNGS = [(1.0, 16), (1.0, 12), (1.0, 8), (1.4, 8), (1.4, 6), (2.0, 6), (2.0, 4), (3.0, 4)]


@dataclass(frozen=True)
class CostLaw:
    """Seconds of one mesh level of n nodes, meshing it and then solving it. A solve costs coeff n^power."""
    mesh_node: float
    coeff: float
    power: float
    fixed: float = 0.0

    def mesh(self, n: float) -> float:
        return self.mesh_node * n

    def solve(self, n: float) -> float:
        return self.coeff * max(n, 1.0) ** self.power

    def level(self, n: float) -> float:
        return self.fixed + self.mesh(n) + self.solve(n)


# print warp, CalculiX on quadratic tets: 5.4 thousand nodes in 8 s and 174 thousand in 1125 s, meshing 1 ms a node
PRINT = CostLaw(1.0e-3, 3.6e-5, 1.43, 2.0)
# casting, one implicit march of the metal and the sand: 13.9 thousand nodes in about 80 s
CASTING = CostLaw(1.0e-3, 1.4e-3, 1.15, 2.0)
# mold filling, the fill of one level, a march of steps that each solve a pressure on the filled nodes: 1.2 thousand
# nodes in 14 s and 4.2 thousand in 67 s
MOLDING = CostLaw(1.0e-3, 1.8e-3, 1.26, 8.0)


@dataclass
class Fit:
    """What was chosen and what it was predicted to cost."""
    rung: int
    factor: float
    curvature: int
    nodes: list[float]                 # predicted nodes of each required level
    seconds: float
    budget: float
    over: bool = False                 # even the coarsest rung is over the budget
    kept: bool = False                 # a mesh size was given, so nothing was given up
    power: float = 1.5                 # nodes grow as the element size to the minus this, measured on the pilots
    measured: dict = field(default_factory=dict)

    @property
    def given_up(self) -> bool:
        return self.rung > 0

    def to_dict(self) -> dict:
        return {"rung": self.rung, "size_factor": self.factor, "elements_per_circle": self.curvature,
                "predicted_s": round(self.seconds, 1), "budget_s": self.budget, "over_budget": self.over,
                "predicted_nodes": [int(n) for n in self.nodes], "size_power": round(self.power, 2), **self.measured}


def hhmm(s: float) -> str:
    return f"{s:.0f} s" if s < 90 else f"{s / 60:.0f} min"


CURVE_POWER = 1.5          # nodes against elements per circle, from a truss meshed at 4, 8 and 16 (1.4 to 1.8)
PILOTS = [(3.0, 4), (2.0, 4), (1.4, 4), (1.0, 6)]    # tried in this order until two of them mesh


@dataclass
class Pilot:
    factor: float
    curvature: int
    nodes: float
    mesh: object = None


def pilots(mesh_at) -> list[Pilot]:
    """Two coarse meshes of the part, the first rungs of PILOTS that mesh at all. A mesher can fail on a very coarse
    size next to a thin wall, so a rung that fails is skipped and the next finer one is tried."""
    got, err = [], None
    for f, c in PILOTS:
        try:
            m = mesh_at(f, c)
        except Exception as e:  # noqa: BLE001  the mesher says why in its own words
            err = e
            continue
        got.append(Pilot(f, c, float(len(getattr(m, "X", m))), m))
        if len(got) == 2:
            break
    if not got:
        raise err
    return got


def size_power(n_a: float, n_b: float, h_a: float, h_b: float) -> float:
    """How the node count grows with the element size, a in n ~ h^-a, from two meshes of one part at sizes h_a > h_b."""
    return float(min(max(math.log(max(n_b, 1.0) / max(n_a, 1.0)) / math.log(h_a / h_b), 0.3), 3.0))


def grow(n_ref: float, h_ref: float, curv_ref: int, h: float, curv: int, power: float = 1.5) -> float:
    """The nodes of a mesh of element size h with curv elements per circle, from a mesh of size h_ref and curv_ref
    that has n_ref. The bulk grows as (h_ref / h)^power and the curved faces as (curv / curv_ref)^1.5, and the
    count is taken to grow by the larger of the two, since a part is mostly one or the other."""
    return n_ref * max((h_ref / h) ** power, (curv / curv_ref) ** CURVE_POWER, 1.0)


def fit(law: CostLaw, found: list[Pilot], h_base: float, scales: list[float], budget: float,
        fixed_size: bool = False, extra=None) -> Fit:
    """The finest rung whose first MIN_LEVELS levels are predicted to fit the budget. The levels run at h_base times
    each of scales times the rung's size factor, and the pilots were meshed at their own rungs on the first scale.
    extra(nodes of the first level) gives the seconds the study spends beyond its levels, such as the check that
    reruns the first level with a finer step."""
    ref = found[-1]
    h_b = h_base * scales[0] * ref.factor
    a = 1.5
    if len(found) == 2:
        a = size_power(found[0].nodes, ref.nodes, h_base * scales[0] * found[0].factor, h_b)
    rungs = RUNGS[:1] if fixed_size else RUNGS
    best = None
    for k, (f, c) in enumerate(rungs):
        nodes = [grow(ref.nodes, h_b, ref.curvature, h_base * s * f, c, a) for s in scales[:MIN_LEVELS]]
        sec = sum(law.level(n) for n in nodes) + (extra(nodes[0]) if extra else 0.0)
        best = Fit(k, f, c, nodes, sec, budget, kept=fixed_size, power=a)
        if sec <= budget:
            return best
    best.over = not fixed_size
    return best


def room_for(law: CostLaw, n_last: float, h_last: float, curv: int, h_next: float, spent: float,
             budget: float, power: float = 1.5, last_seconds: float = 0.0) -> bool:
    """Whether one more level, priced from the mesh just solved, fits in what is left of the budget. When the level
    just solved took last_seconds, the law is scaled by how far that was from what it predicted, so a part that is
    slower than the machines the law was fit on is priced at its own pace."""
    pace = last_seconds / max(law.level(n_last), 1e-9) if last_seconds > 0 else 1.0
    return spent + pace * law.level(grow(n_last, h_last, curv, h_next, curv, power)) <= budget


def note(f: Fit) -> str:
    """The sentence for the assumptions, what was predicted, what was given up and what it came to."""
    base = f"Predicted {hhmm(f.seconds)} for the first {MIN_LEVELS} mesh levels against a budget of {hhmm(f.budget)}"
    if f.kept:
        text = base + (", and a mesh size was set, so it was not coarsened" if f.seconds > f.budget
                       else ", so the meshes were left as asked")
    elif not f.given_up:
        text = base + ", so the meshes were left as they were"
    else:
        gave = []
        if f.curvature != RUNGS[0][1]:
            gave.append(f"{f.curvature} elements round each curve instead of {RUNGS[0][1]}")
        if f.factor != 1.0:
            gave.append(f"elements {f.factor:g} times larger")
        text = base + ", so the meshes were coarsened to " + " and ".join(gave) + \
            (", and the pair is still predicted over the budget at the coarsest mesh allowed" if f.over else "")
    done = f.measured
    if done.get("levels"):
        text += (f". It solved {done['levels']} levels of " + ", ".join(f"{n:,}" for n in done["nodes"]) +
                 f" nodes in {hhmm(done['seconds'])}")
    return text + ". The budget can be raised with --budget."


HARD_SHARE = 4.0           # a study is stopped when it has run this many times its budget


class OverBudget(RuntimeError):
    """A solve ran past the hard limit, which is a few times the budget, since the cost laws are estimates. The
    study stops with this instead of running on for hours."""

    def __init__(self, what: str, limit: float):
        super().__init__(f"{what} was still running after {hhmm(limit)}, over four times the budget, so it was stopped")
        self.hint = "Raise --budget, or pass a larger --mesh-size, so the mesh is coarser"


def deadline(budget: float) -> float:
    """The clock time past which a study with this budget is stopped."""
    return time.time() + HARD_SHARE * budget


def check(limit: float | None, what: str, budget: float) -> None:
    if limit is not None and time.time() > limit:
        raise OverBudget(what, HARD_SHARE * budget)
