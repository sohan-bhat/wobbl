"""The crack path: solve, read K along the front, turn and advance every front point, remesh, repeat.

Each step solves the cracked part at the loads given and reads K_I, K_II and K_III along each front. Every front
point turns by the maximum tangential stress angle from its own K_I and K_II and advances in proportion to its
growth rate under the load spectrum, so the fastest point moves a set step and a point below threshold stays put.
The new front, reaching a little past the part's surfaces, becomes the next section of the crack, which is cut
into the part and meshed again. The path is predicted from the geometry and loads, never assumed.
"""
from __future__ import annotations

import math
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..solvers.base import SolverError
from .crackmesh import Crack, CrackError, PartSeparated, Sizing, build_cracked_mesh
from .growth import GrowthLaw, Spectrum
from .life import LifeInputs, block_rate, cycle_types
from .sif import FrontK, extract
from .solve import solve_displacements


@dataclass
class Step:
    index: int
    size: float                   # crack size measure, mm
    k_lead: float                 # largest equivalent K along the fronts, MPa sqrt(m)
    area: float                   # crack face area, mm2
    nodes: int
    h_front: float
    fronts: list[dict] = field(default_factory=list)   # points, K and kink angles along each front
    advance: float = 0.0          # how far the fastest point moved after this step, mm
    kink_lead: float = 0.0        # turn at the fastest point, degrees
    t: float = 0.0

    def to_dict(self) -> dict:
        return {"step": self.index, "size_mm": self.size, "k_lead": self.k_lead, "area_mm2": self.area,
                "nodes": self.nodes, "h_front_mm": self.h_front, "advance_mm": self.advance,
                "kink_lead_deg": self.kink_lead, "time_s": round(self.t, 1), "fronts": self.fronts}


def resample(fk: FrontK, n: int) -> dict:
    """Points, frames and smoothed K at n points evenly along a front."""
    t = np.linspace(0.0, fk.length, n)
    xyz = np.column_stack([np.interp(t, fk.s, fk.xyz[:, i]) for i in range(3)])

    def vec(a):
        v = np.column_stack([np.interp(t, fk.s, a[:, i]) for i in range(3)])
        return v / np.maximum(np.linalg.norm(v, axis=1), 1e-300)[:, None]

    k = np.column_stack([np.interp(t, fk.s, fk.k[:, i]) for i in range(3)])
    return {"xyz": xyz, "forward": vec(fk.forward), "normal": vec(fk.normal), "tangent": vec(fk.tangent), "k": k}


def advance(front: dict, law: GrowthLaw, spectrum: Spectrum, nu: float, size: float, da_max: float,
            reach: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """New front points, each turned by its maximum tangential stress angle and moved in proportion to its growth
    rate, and the front extended past the part's surfaces by ``reach`` at both ends. Returns the new points, the
    advance and the turn (radians) at each old point."""
    from .handbook import k_equivalent, mts_angle

    k = front["k"]
    keq = k_equivalent(k[:, 0], k[:, 1], k[:, 2], nu)
    inp = LifeInputs(law, spectrum, 0.0)
    rate = block_rate(inp, keq, np.full_like(keq, size), 1.0, 1.0, 0.0, cycle_types(spectrum, 0.0))
    top = float(rate.max())
    da = np.zeros_like(keq) if top <= 0 else da_max * rate / top
    theta = mts_angle(k[:, 0], k[:, 1])
    step = np.cos(theta)[:, None] * front["forward"] + np.sin(theta)[:, None] * front["normal"]
    pts = front["xyz"] + da[:, None] * step
    t0 = pts[0] - pts[1]
    t1 = pts[-1] - pts[-2]
    t0 /= max(np.linalg.norm(t0), 1e-300)
    t1 /= max(np.linalg.norm(t1), 1e-300)
    pts = np.vstack([pts[0] + reach * t0, pts, pts[-1] + reach * t1])
    return pts, da, theta


def orient(fk: FrontK, section: np.ndarray) -> FrontK:
    """Turn a front's node order to run the same way as the crack section it grew from."""
    a, b = fk.xyz[0], fk.xyz[-1]
    if np.linalg.norm(a - section[0]) + np.linalg.norm(b - section[-1]) > \
            np.linalg.norm(b - section[0]) + np.linalg.norm(a - section[-1]):
        idx = slice(None, None, -1)
        fk = FrontK(fk.nodes[idx], fk.xyz[idx], fk.length - fk.s[idx], fk.forward[idx], fk.normal[idx],
                    -fk.tangent[idx], fk.raw[idx], fk.k[idx], fk.nu, fk.fit_points[idx])
    return fk


def match_fronts(fks: list[FrontK], crack: Crack) -> dict[int, FrontK]:
    """Which measured front belongs to which live section of the crack."""
    live = crack.front_sections()
    out = {}
    for i in live:
        mid = crack.sections[i][len(crack.sections[i]) // 2]
        near = [f for f in fks if f.length > 0]
        if not near:
            continue
        f = min(near, key=lambda f: float(np.min(np.linalg.norm(f.xyz - mid, axis=1))))
        out[i] = orient(f, crack.sections[i])
    return out


@dataclass
class PathResult:
    steps: list[Step]
    crack: Crack                  # including the last front, which was not solved
    end: str
    first: list[FrontK]           # the first step's fronts, for the checks
    convergence: list[dict] = field(default_factory=list)
    balance: tuple | None = None
    notes: list[str] = field(default_factory=list)
    end_fails: bool = False       # the crack has cut through, so reaching the end of the path is failure


def front_size(size: float, crack: Crack, h: float, wall: float, quality: str) -> float:
    """Element size along the front: about a fourteenth of the crack, at least six elements across a wall it runs
    through, and never above a quarter of the far field size."""
    frac = {"fast": 1 / 10, "fine": 1 / 20}.get(quality, 1 / 14)
    hf = frac * size
    if wall > 0:
        hf = min(hf, wall / 6.0)
    return float(min(hf, h / 4.0))


def grow(prep, crack: Crack, build_model, material, law: GrowthLaw, spectrum: Spectrum, adapter, workdir: Path,
         h: float, wall: float, quality: str, max_steps: int, growth_frac: float, stop_k: float,
         stop_net, budget: float, progress=None, keep: bool = False, part=None) -> PathResult:
    """Grow the crack step by step until it fails, cuts through the part, stops growing, or the steps or the time
    budget run out. ``build_model(cm)`` gives the static model on a cracked mesh, ``stop_net(size, area)`` says
    whether the ligament has yielded and ``part`` (a place.PartShape) tells when the next front would leave the
    uncracked part."""
    say = progress or (lambda a, b: None)
    t_start = time.time()
    steps: list[Step] = []
    first: list[FrontK] = []
    convergence: list[dict] = []
    notes: list[str] = []
    balance = None
    size = crack.size
    E, nu = material.E, material.nu
    end = f"the step limit ({max_steps})"
    end_fails = False
    for j in range(max_steps + 1):
        t0 = time.time()
        hf = front_size(size, crack, h, wall, quality)
        sz = Sizing(h=h, h_front=hf, h_crack=float(np.clip(0.3 * size, hf, h)))
        say("mesh", f"step {j}: crack {size:.3g} mm, front elements {hf:.3g} mm")
        try:
            if j == 0:
                cm, fks, convergence, balance = _first_step(prep, crack, sz, build_model, adapter, workdir, E, nu,
                                                            quality, say)
            else:
                cm = build_cracked_mesh(prep, crack, sz, dump=workdir / f"step{j}" / "crack.pkl" if keep else None)
                say("solve", f"step {j}: {cm.mesh.n_nodes:,} nodes")
                U, _ = solve_displacements(adapter, build_model(cm).model, workdir / f"step{j}")
                fks = extract(cm, U, E, nu)
        except PartSeparated:
            end = "the crack cutting through the part"
            end_fails = True
            break
        except (CrackError, SolverError) as e:
            if j == 0:
                raise
            notes.append(f"Step {j} could not be solved ({str(e)[:160]}), so the path stops at step {j - 1}.")
            end = f"a failed step ({str(e)[:80]})"
            break
        if j == 0:
            first = fks
        fronts = match_fronts(fks, crack)
        if len(fronts) < len(crack.front_sections()):
            notes.append(f"At step {j} a front was lost, so the path stops there.")
            end = "a lost front"
            break
        n_pts = {i: int(np.clip(round(f.length / (1.5 * hf)), 7, 31)) for i, f in fronts.items()}
        sampled = {i: resample(f, n_pts[i]) for i, f in fronts.items()}
        keq = max(float(_keq(s, nu).max()) for s in sampled.values())
        step = Step(j, size, keq, cm.area, cm.mesh.n_nodes, hf, [], t=time.time() - t0)
        for i, s in sampled.items():
            step.fronts.append({"section": i, "xyz": np.round(s["xyz"], 4).tolist(),
                                "k": np.round(s["k"], 4).tolist()})
        steps.append(step)
        say("check", f"step {j}: largest K {keq:.3g} MPa sqrt(m)")
        if keq >= stop_k:
            end = "fracture (K reached the toughness of the toughest part sampled)"
            break
        if stop_net(size, cm.area):
            end = "net section yield of the weakest part sampled"
            break
        if j == max_steps:
            break
        if time.time() - t_start > budget:
            end = f"the time budget ({budget:.0f} s)"
            notes.append(f"The path stopped at step {j} when the time budget ran out; a longer --budget follows it "
                         "further.")
            break
        da_max = growth_frac * size
        reach = max(1.5 * da_max, 4.0 * hf)
        moved = {}
        for i, s in sampled.items():
            pts, da, theta = advance(s, law, spectrum, nu, size, da_max, reach)
            moved[i] = (pts, da, theta)
        if all(m[1].max() <= 0 for m in moved.values()):
            end = "the crack stopping below threshold"
            notes.append("Every front point is below the growth threshold at this load, so the crack stops "
                         "growing.")
            break
        if part is not None and any(_leaves(part, sampled[i]["xyz"], m[0], m[1]) for i, m in moved.items()):
            # the next front would leave the part: the crack has reached the far side of the wall it runs through
            end = "the crack cutting through the wall it grows across"
            end_fails = True
            notes.append(f"After step {j} the crack reaches the far side of the wall it grows across. The few cycles "
                         "it takes to cut the last of that wall are not counted, which errs short.")
            break
        lead_i = max(moved, key=lambda i: moved[i][1].max())
        lead = int(np.argmax(moved[lead_i][1]))
        step.advance = da_max
        step.kink_lead = math.degrees(float(moved[lead_i][2][lead]))
        for i, (pts, _, _) in moved.items():
            if i == 0:
                crack.sections.insert(0, pts)
            else:
                crack.sections.append(pts)
        size += da_max
        if not keep:
            shutil.rmtree(workdir / f"step{j}", ignore_errors=True)
    return PathResult(steps, crack, end, first, convergence, balance, notes, end_fails)


def _leaves(part, old: np.ndarray, new: np.ndarray, da: np.ndarray) -> bool:
    """Whether a front moving from ``old`` to ``new[1:-1]`` (``new`` carries one extra point past each end) leaves
    the part: a point past the far side of the wall, or a point whose way ahead meets the far side within its step.
    The second catches a front whose fastest point is at a surface, where the first cannot tell."""
    if (~part.inside(new[2:-2])).any():
        return True
    for p0, p1, d in zip(old, new[1:-1], da):
        if d > 0 and part.ahead(p0, (p1 - p0) / d, 0.02 * d) < d:
            return True
    return False


def _keq(s: dict, nu: float) -> np.ndarray:
    from .handbook import k_equivalent
    return k_equivalent(s["k"][:, 0], s["k"][:, 1], s["k"][:, 2], nu)


def _first_step(prep, crack: Crack, sz: Sizing, build_model, adapter, workdir: Path, E, nu, quality, say):
    """The starting crack on a coarser front mesh and then the regular one (and a finer one at --quality fine),
    to show K has settled with the mesh. Also returns the force balance of the regular solve."""
    from dataclasses import replace
    from types import SimpleNamespace

    from ..structural.run import support_reactions

    factors = [1.5, 1.0] + ([0.7] if quality == "fine" else [])
    conv, last = [], None
    for f in factors:
        s = replace(sz, h_front=sz.h_front * f, h_crack=max(sz.h_crack, sz.h_front * f))
        cm = build_cracked_mesh(prep, crack, s)
        say("solve", f"step 0: front elements {s.h_front:.3g} mm, {cm.mesh.n_nodes:,} nodes")
        built = build_model(cm)
        U, RF = solve_displacements(adapter, built.model, workdir / f"start{f:g}")
        fks = extract(cm, U, E, nu)
        conv.append({"h_front": s.h_front, "nodes": cm.mesh.n_nodes,
                     "k_lead": max(float(f_.k_eq().max()) for f_ in fks) if fks else float("nan")})
        last = (cm, fks, built, RF)
    cm, fks, built, RF = last
    applied = built.model.applied_total()
    reaction = support_reactions(built, SimpleNamespace(RF=RF), cm.mesh)
    return cm, fks, conv, (applied, reaction, float(np.linalg.norm(np.ptp(cm.mesh.X, axis=0))),
                           bool(built.model.normal_constraints))
