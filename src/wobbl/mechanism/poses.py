"""Poses of any mechanism: sample its joint space, close its loops, and search it for the worst case.

A serial robot's poses are just its joint angles within their ranges. A mechanism with loops (a slider crank, a
four bar, a parallel gripper written with equality constraints) has fewer free coordinates than joints, so
Wobbl picks independent joints, samples those, and solves the rest by Newton steps on the loop constraints.
Every search for a worst pose (the most torque, the farthest reach, the smallest tipping margin) starts from the
best of many random poses and then refines it a joint at a time with shrinking steps.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .install import require as _mj
from .model import Info, start_qpos


@dataclass
class Space:
    """The coordinates a pose search moves: independent joints with their ranges, and how to close loops."""
    model: object
    info: Info
    free: list[dict]                      # joints the search sets
    dep: list[dict]                       # joints the loop constraints set
    lo: np.ndarray
    hi: np.ndarray
    q0: np.ndarray                        # the model's starting qpos

    @property
    def n(self) -> int:
        return len(self.free)


def space(model, info: Info, joints: list[str] | None = None) -> Space:
    """The searchable space of ``model``: by default every hinge and slide joint, with independent ones picked
    when loops tie some together. An unlimited hinge spans a full turn; an unlimited slide stays where it is."""
    cand = [j for j in info.joints if joints is None or j["name"] in joints]
    cand = [j for j in cand if j["limited"] or j["kind"] == "hinge"]
    free, dep = cand, []
    if _active_eq(model):
        model.opt.jacobian = int(_mj().mjtJacobian.mjJAC_DENSE)
        free, dep = _split(model, info, cand)
    lo = np.array([j["range"][0] if j["limited"] else -math.pi for j in free])
    hi = np.array([j["range"][1] if j["limited"] else math.pi for j in free])
    return Space(model, info, free, dep, lo, hi, start_qpos(model))


def _active_eq(model) -> bool:
    return bool(model.neq) and bool(np.any(model.eq_active0))


def _eq_rows(model, d):
    """The loop constraints' Jacobian and violation at ``d``'s pose (the model's Jacobian is kept dense)."""
    mujoco = _mj()

    mujoco.mj_fwdPosition(model, d)
    rows = d.efc_type[:d.nefc] == int(mujoco.mjtConstraint.mjCNSTR_EQUALITY)
    J = np.asarray(d.efc_J[:d.nefc * model.nv]).reshape(d.nefc, model.nv)
    return J[rows].copy(), d.efc_pos[:d.nefc][rows].copy()


def _split(model, info: Info, cand: list[dict]) -> tuple[list[dict], list[dict]]:
    """Independent joints for a mechanism with loops: keep a joint free while the remaining ones can still close
    every loop, preferring joints that have an actuator, since those are the ones a user drives."""
    mujoco = _mj()

    d = mujoco.MjData(model)
    J, _ = _eq_rows(model, d)
    if not len(J):
        return cand, []
    driven = {int(model.actuator_trnid[a, 0]) for a in range(model.nu)
              if model.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT}
    order = sorted(cand, key=lambda j: (j["id"] not in driven, j["id"]))
    r = np.linalg.matrix_rank(J, 1e-9)
    free, dep = [], list(info.joints)
    for j in order:
        rest = [x for x in dep if x is not j]
        cols = [x["dof"] for x in rest]
        if cols and np.linalg.matrix_rank(J[:, cols], 1e-9) == r:
            free.append(j)
            dep = rest
    return free, dep


def set_pose(sp: Space, d, x: np.ndarray, iters: int = 30, tol: float = 1e-9) -> bool:
    """Put ``d`` at the pose with free coordinates ``x``, closing loops by Newton steps. Returns whether every
    loop closed."""
    mujoco = _mj()

    d.qpos[:] = sp.q0
    d.qvel[:] = 0
    for j, v in zip(sp.free, x):
        d.qpos[j["qadr"]] = v
    if not sp.dep:
        mujoco.mj_kinematics(sp.model, d)
        return True
    cols = [j["dof"] for j in sp.dep]
    adr = [j["qadr"] for j in sp.dep]
    for _ in range(iters):
        J, r = _eq_rows(sp.model, d)
        if not len(r) or float(np.max(np.abs(r))) < tol:
            return True
        step = np.linalg.lstsq(J[:, cols], r, rcond=None)[0]
        d.qpos[adr] -= np.clip(step, -0.5, 0.5)
    J, r = _eq_rows(sp.model, d)
    return not len(r) or float(np.max(np.abs(r))) < 1e-6


def sample(sp: Space, n: int, seed: int = 0) -> np.ndarray:
    """``n`` poses spread evenly in the free coordinates' ranges, the starting pose first."""
    rng = np.random.default_rng(seed)
    x0 = np.clip(np.array([sp.q0[j["qadr"]] for j in sp.free]), sp.lo, sp.hi)
    if not sp.n:
        return x0[None, :]
    return np.vstack([x0, sp.lo + (sp.hi - sp.lo) * rng.random((n - 1, sp.n))])


def worst(sp: Space, score, n: int = 300, seed: int = 0, levels: int = 7) -> tuple[float, np.ndarray, int]:
    """The pose that makes ``score(d)`` largest: the best of ``n`` samples, then refined a coordinate at a time
    with steps halving from a quarter of each range. Returns the score, the free coordinates and how many poses
    were tried. Poses whose loops cannot close are skipped."""
    mujoco = _mj()

    d = mujoco.MjData(sp.model)
    best, bx, tried = -math.inf, None, 0
    for x in sample(sp, n, seed):
        tried += 1
        if set_pose(sp, d, x):
            s = score(d)
            if s > best:
                best, bx = s, x.copy()
    if bx is None:
        return -math.inf, sample(sp, 1)[0], tried
    step = (sp.hi - sp.lo) / 4
    for _ in range(levels):
        improved = True
        while improved:
            improved = False
            for k in range(sp.n):
                for sgn in (1, -1):
                    x = bx.copy()
                    x[k] = min(max(x[k] + sgn * step[k], sp.lo[k]), sp.hi[k])
                    if x[k] == bx[k]:
                        continue
                    tried += 1
                    if set_pose(sp, d, x):
                        s = score(d)
                        if s > best + 1e-12 * max(1.0, abs(best)):
                            best, bx, improved = s, x, True
        step = step / 2
    set_pose(sp, d, bx)
    return best, bx, tried


def describe_pose(sp: Space, x: np.ndarray) -> str:
    """A pose in words: each free joint's angle in degrees or slide in mm."""
    parts = []
    for j, v in zip(sp.free, x):
        parts.append(f"{j['name']} {math.degrees(v):.0f} deg" if j["kind"] == "hinge" else
                     f"{j['name']} {v * 1000:.0f} mm")
    return ", ".join(parts) if parts else "its starting pose"


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% interval on a share k of n, which stays sensible at 0 and n."""
    if n == 0:
        return 0.0, 1.0
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, mid - half), min(1.0, mid + half)
