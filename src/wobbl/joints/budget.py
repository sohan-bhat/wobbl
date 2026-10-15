"""What each mesh level of a contact or bolted study will cost, worked out before it is solved, and how the mesh is
coarsened until two levels fit the time budget.

A contact solve is a Newton iteration. Every iteration builds and factors a stiffness matrix whose cost grows a
little faster than the node count (about the 1.3 power on these tet meshes) and contact makes it more expensive
because the pressed faces couple their nodes. The model below is seconds per iteration times the iterations a study
needs, fit on 40 levels of contact, bolt and plasticity solves of 1,400 to 17,000 nodes (two threads, a machine busy
with other solvers, so the constants lean slow). Meshing is charged by the node as in structural/budget.py.

The first level is meshed for real at each element size and coarsened rung by rung until both levels are predicted
to fit. The second level is then sized from what the first one really took, so a slow machine or a hard
contact makes the study coarser instead of letting the solver run into its time limit. A mesh size the user set is
kept, and the prediction is only reported.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from ..structural.budget import BACKSTOP, mesh_cost

DEFAULT_BUDGET = 1500.0      # s, the time the two levels together may be predicted to take
ITER_S = 1.0e-4              # s per Newton iteration per node^1.3
POWER = 1.3
CONTACT_ITERS = 40           # iterations of a pressed assembly loaded in one step
# What can be given up, finest first: (factor on the element size, elements per circle round curves). Round holes
# and bores hold the node count up whatever size is asked for, so the curvature setting is the lever that matters.
RUNGS = ((1.0, 16), (1.0, 12), (1.25, 9), (1.6, 7), (2.0, 6), (2.5, 5), (3.2, 4))
FINEST = 0.87                # the second level is at least this fraction of the first element size
SLACK = 0.85                 # the second level may use this share of what the budget has left


def solve_seconds(nodes: float, iterations: float, pairs: int = 1) -> float:
    """Modeled seconds of one nonlinear solve with its result reading."""
    return 2.0 + ITER_S * iterations * nodes ** POWER * (1.0 + 0.04 * max(0, pairs - 1))


@dataclass
class Plan:
    """The two levels and what was predicted for them."""
    sizings: list
    h: list
    mesh0: object | None = None                 # the first level's mesh, built while fitting
    nodes: list = field(default_factory=lambda: [0, 0])
    seconds: list = field(default_factory=lambda: [0.0, 0.0])
    ratio: float = 1.0                          # real nodes over estimated nodes
    rung: float = 1.0
    curv: int = 16
    rung_index: int = 0
    user_size: bool = False
    iterations: float = CONTACT_ITERS
    pairs: int = 1
    spent: float = 0.0
    budget: float = DEFAULT_BUDGET
    notes: list = field(default_factory=list)

    @property
    def predicted(self) -> float:
        return float(sum(self.seconds))

    def to_dict(self) -> dict:
        return {"budget_s": self.budget, "predicted_s": round(self.predicted, 1),
                "predicted_nodes": [int(n) for n in self.nodes], "element_mm": [round(float(h), 3) for h in self.h],
                "coarsened_by": self.rung, "elements_per_circle": self.curv, "user_mesh_size": self.user_size}

    def timeout(self, floor: float = 300.0) -> float:
        """Wall clock limit for one solver run: only a stalled solver or a very slow machine reaches it."""
        return max(floor, BACKSTOP * self.budget - self.spent)


def _predict(plan: Plan, sizings, est, ratio: float) -> tuple[list, list]:
    nodes = [e * ratio for e in est]
    sec = [mesh_cost(n) + solve_seconds(n, plan.iterations, plan.pairs) for n in nodes]
    return nodes, sec


def fit(asm, make, h_auto: float, h_user: float | None, budget: float, iterations: float, pairs: int, say=None,
        mesh_fn=None) -> Plan:
    """Choose the element size. ``make(h)`` gives the two sizings for a first level of size h. The first level is
    meshed for real, the second is estimated from it, and the size is coarsened until both levels are predicted to
    fit the budget."""
    from ..mesh.mesher import estimate_nodes
    from .assembly import build_assembly_mesh

    say = say or (lambda a, b: None)
    mesh_fn = mesh_fn or build_assembly_mesh
    plan = Plan([], [], user_size=bool(h_user), iterations=iterations, pairs=pairs, budget=budget)
    base = float(h_user or h_auto)
    rungs = RUNGS[:1] if h_user else RUNGS
    k = 0
    while True:
        f, curv = rungs[k]
        sizings = make(base * f, curv=curv)
        # the estimate misses on coarse meshes, where curves and holes set the size, so each rung is meshed for real
        mesh0 = mesh_fn(asm, sizings[0])
        ratio = mesh0.n_nodes / max(estimate_nodes(asm.prep, sizings[0]), 1.0)
        est = [estimate_nodes(asm.prep, s) for s in sizings]
        nodes, sec = _predict(plan, sizings, est, ratio)
        nodes[0] = mesh0.n_nodes
        sec[0] = mesh_cost(nodes[0]) + solve_seconds(nodes[0], iterations, pairs)
        plan.sizings, plan.h, plan.nodes, plan.seconds, plan.ratio, plan.rung = sizings, [s.h for s in sizings], \
            nodes, sec, ratio, f
        plan.curv, plan.rung_index = curv, k
        plan.mesh0 = mesh0
        if sum(sec) <= budget or k == len(rungs) - 1:
            break
        say("budget", f"predicted {sum(sec):.0f} s for two levels of {nodes[0]:,.0f} and {nodes[1]:,.0f} nodes "
                      f"against a budget of {budget:.0f} s, coarsening")
        k = min(k + (2 if sum(sec) > 4 * budget else 1), len(rungs) - 1)
    if h_user and plan.predicted > budget:
        plan.notes.append(f"The mesh size you set is predicted to take {plan.predicted:.0f} s over its two meshes, "
                          f"more than the {budget:.0f} s budget, so it was kept. Pass a larger --budget to silence "
                          "this or a larger --mesh-size to run faster.")
    elif plan.rung_index > 0:
        plan.notes.append(f"To fit the {budget:.0f} s budget the mesh was coarsened (element size {plan.h[0]:.3g} mm "
                          f"and {plan.curv} elements round each circle), with about {plan.nodes[0]:,.0f} nodes on the first mesh and "
                          f"{plan.nodes[1]:,.0f} on the second. A larger --budget keeps a finer mesh.")
    return plan


def second_level(plan: Plan, asm, make, t_first: float, n_first: int, iters_first: int, mesh_fn=None):
    """The second mesh, with its cost predicted from what the first level really took. The cost model is scaled by
    measured over modeled, and the mesh is made less refined than the usual 0.75 of the first element size, step by
    step, until its real node count fits what the budget has left. Returns the mesh."""
    from .assembly import build_assembly_mesh

    mesh_fn = mesh_fn or build_assembly_mesh
    plan.spent += t_first + mesh_cost(n_first)
    modeled = solve_seconds(n_first, max(iters_first, 1), plan.pairs)
    scale = min(max(t_first / max(modeled, 1e-9), 0.3), 6.0) if t_first > 0 else 1.0
    left = SLACK * (plan.budget - plan.spent)
    h0 = plan.h[0]
    for q in (0.75, 0.82, FINEST):          # the ladder from the usual refinement to the least refinement
        sizing = make(h0, q, plan.curv)[1]
        mesh = mesh_fn(asm, sizing)
        cost = mesh_cost(mesh.n_nodes) + scale * solve_seconds(mesh.n_nodes, max(iters_first, 1), plan.pairs)
        if cost <= left:
            break
    plan.sizings[1], plan.h[1], plan.nodes[1], plan.seconds[1] = sizing, sizing.h, mesh.n_nodes, cost
    if cost > left and not plan.user_size:
        plan.notes.append(f"The second mesh is predicted to take {cost:.0f} s against {left:.0f} s left in the "
                          f"budget even at its least refinement ({q:g} of the first element size). It runs anyway, "
                          "so the two meshes differ little and the convergence check is weak.")
    return mesh
