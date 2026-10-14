"""How far a mechanism reaches and whether its parts run into each other, for any model.

Reach is searched like the worst torque: many random poses, then the farthest one refined a joint at a time. The
straight line through every joint along the chain bounds it, which is the hand check. Interference is counted
over random poses with every collision shape switched on, skipping pairs joined directly by a joint, which touch
at the joint by design, and pairs that already overlap as the model is drawn, which are reported on their own.
"""
from __future__ import annotations

import math
import re

import numpy as np

from ..units import fmt
from ..verify.checks import Check
from . import plots
from .findings import Finding, Setup
from .holding import quiet
from .install import require as _mj
from .model import end_effector
from .poses import describe_pose, sample, set_pose, space, wilson, worst

POSES = 600              # random poses for interference, twice over with different seeds


def _tip(su: Setup, model):
    if su.payload:
        names = su.info.names["body"]
        return names.index(su.payload[1]), np.asarray(su.payload[2], float)
    return end_effector(model, su.info)


def reach(su: Setup) -> Finding:
    mujoco = _mj()

    model = quiet(su.ld.compile())
    sp = space(model, su.info)
    if not sp.n:
        return Finding("kinematics", "info", "The mechanism has no joint that moves it.")
    body, pt = _tip(su, model)
    bname = su.info.names["body"][body]
    root = _root_of(model, body)

    def where(d):
        return d.xpos[body] + d.xmat[body].reshape(3, 3) @ pt

    def base(d):
        return d.xpos[root]

    su.log("solve", f"sweeping the range of {sp.n} joints")
    d = mujoco.MjData(model)
    pts = []
    for x in sample(sp, su.n(2000), seed=11):
        if set_pose(sp, d, x):
            pts.append(where(d) - base(d))
    pts = np.array(pts)
    far, xf, _ = worst(sp, lambda dd: float(np.linalg.norm(where(dd) - base(dd))), n=300, seed=1)
    far2, _, _ = worst(sp, lambda dd: float(np.linalg.norm(where(dd) - base(dd))), n=300, seed=5)
    bound = _chain_length(model, sp, body, pt, root)
    radial = np.hypot(pts[:, 0], pts[:, 1])
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    gap = abs(far - far2) / max(far, far2, 1e-12)
    checks = [Check("farthest pose search", "pass" if gap < 0.005 else ("warn" if gap < 0.03 else "fail"),
                    f"searching again from another start reaches {fmt(max(far, far2) * 1000, 4)} mm against "
                    f"{fmt(min(far, far2) * 1000, 4)} mm ({100 * gap:.2g}% apart)", {"first": far, "second": far2})]
    if math.isfinite(bound):
        ok = far <= bound * (1 + 1e-6) + 1e-9
        checks.append(Check("chain length", "pass" if ok else "fail",
                            f"the straight line from the base through every joint to the {bname} is "
                            f"{fmt(bound * 1000, 4)} mm, and the farthest reach found, {fmt(far * 1000, 4)} mm, "
                            + ("stays inside it" if ok else "goes past it, which no linkage can"),
                            {"chain_m": bound, "reach_m": far}))
    rows = []
    for j in su.info.joints:
        if j["limited"]:
            a, b = j["range"]
            rows.append(f"{j['name']} {math.degrees(a):.0f} to {math.degrees(b):.0f} deg" if j["kind"] == "hinge"
                        else f"{j['name']} {a * 1000:.0f} to {b * 1000:.0f} mm")
        else:
            rows.append(f"{j['name']} turns freely" if j["kind"] == "hinge" else f"{j['name']} slides freely")
    sentence = (f"The {bname} reaches up to {fmt(far * 1000)} mm from the base, {fmt(radial.max() * 1000)} mm out "
                f"sideways and from {fmt(lo[2] * 1000)} to {fmt(hi[2] * 1000)} mm in height.")
    verdict = "info"
    target, want = _asked(su.question)
    values = {"reach_m": far, "radial_m": float(radial.max()), "z_range_m": [float(lo[2]), float(hi[2])],
              "box_m": [lo.tolist(), hi.tolist()], "farthest_pose": describe_pose(sp, xf), "tip": bname,
              "joint_ranges": rows, "_poses": {"farthest": {j["name"]: float(v) for j, v in zip(sp.free, xf)}}}
    if target is not None:
        miss, xt, _ = worst(sp, lambda dd: -float(np.linalg.norm(where(dd) - base(dd) - target)), n=400, seed=2)
        miss = -miss
        tol = max(0.002, 0.005 * far)
        values["target_m"], values["target_miss_m"] = target.tolist(), miss
        if miss <= tol:
            sentence = (f"The {bname} reaches the point at {', '.join(fmt(v * 1000) for v in target)} mm, with "
                        f"{describe_pose(sp, xt)}. " + sentence)
        else:
            verdict = "fails"
            sentence = (f"The {bname} cannot reach the point at {', '.join(fmt(v * 1000) for v in target)} mm, "
                        f"coming no closer than {fmt(miss * 1000)} mm. " + sentence)
    elif want is not None:
        if far + 1e-9 < want:
            verdict = "fails"
            sentence = f"The {bname} falls short of {fmt(want * 1000)} mm, reaching only {fmt(far * 1000)} mm. " + \
                       sentence
        else:
            sentence = f"The {bname} reaches past {fmt(want * 1000)} mm. " + sentence
    su.img.mkdir(parents=True, exist_ok=True)
    images = {"workspace": plots.cloud(su.img / "workspace.png", pts, f"where the {bname} can go")}
    return Finding("kinematics", verdict, sentence,
                   [("reach", f"{fmt(far * 1000)} mm"), ("height", f"{fmt(lo[2] * 1000)} to {fmt(hi[2] * 1000)} mm")],
                   checks, values, [], [], [["workspace", f"Where the {bname} can go, from above and from the side"]],
                   images, [f"Reach is measured from the base's origin to the {bname}, over 2000 random poses and a "
                            "search for the farthest one. Joint ranges: " + "; ".join(rows) + "."])


def _root_of(model, body: int) -> int:
    while model.body_parentid[body] != 0:
        body = int(model.body_parentid[body])
    return body


def _chain_length(model, sp, body: int, pt, root: int) -> float:
    """Length of the straight segments from the base through each joint anchor up to the tip, plus each slide's
    full travel, at the starting pose: no pose of the chain can reach farther than that."""
    mujoco = _mj()

    d = mujoco.MjData(model)
    d.qpos[:] = sp.q0
    mujoco.mj_kinematics(model, d)
    chain = []
    b = body
    while b != 0:
        for k in range(model.body_jntnum[b]):
            chain.append(model.body_jntadr[b] + k)
        b = int(model.body_parentid[b])
    pts = [d.xpos[root]] + [d.xanchor[j] for j in reversed(chain)] + \
        [d.xpos[body] + d.xmat[body].reshape(3, 3) @ pt]
    extra = sum(abs(model.jnt_range[j]).max() if model.jnt_limited[j] else math.inf for j in chain
                if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_SLIDE)
    return float(sum(np.linalg.norm(b - a) for a, b in zip(pts, pts[1:])) + extra)


def _asked(q: str):
    """A point the question asks about, as three numbers in mm, or a reach distance it asks for."""
    m = re.search(r"\(?\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)?\s*mm", q)
    if m:
        return np.array([float(v) / 1000 for v in m.groups()]), None
    m = re.search(r"reach\w*\s+(?:out\s+)?(?:to\s+)?(\d+(?:\.\d+)?)\s*(mm|cm|m)\b", q.lower())
    if m:
        return None, float(m.group(1)) * {"mm": 1e-3, "cm": 1e-2, "m": 1.0}[m.group(2)]
    return None, None


def interference(su: Setup) -> Finding:
    mujoco = _mj()

    model = su.ld.compile()
    model.opt.disableflags &= ~int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
    solid = (model.geom_contype != 0) | (model.geom_conaffinity != 0)
    world = model.geom_bodyid == 0
    moving = solid & ~world
    if not moving.any():
        return Finding("collision", "info", "The model has no collision shapes on its moving parts, so Wobbl "
                                               "cannot check whether they run into each other.",
                       extra=[("low", "the moving parts have no collision shapes")])
    groups = su.info.groups
    # the model's own collision filter says which of its parts are meant to touch; a model that lets none of its
    # parts touch each other (common in models built for speed) is checked with every shape on instead
    gid = [g for g in np.nonzero(moving)[0]]
    declared = any((model.geom_contype[a] & model.geom_conaffinity[b]) or (model.geom_contype[b] &
                                                                           model.geom_conaffinity[a])
                   for i, a in enumerate(gid) for b in gid[i + 1:]
                   if groups[model.geom_bodyid[a]] != groups[model.geom_bodyid[b]] and
                   frozenset((int(groups[model.geom_bodyid[a]]), int(groups[model.geom_bodyid[b]])))
                   not in su.info.adjacent)
    note = []
    if not declared:
        model.geom_contype[:] = np.where(solid, 1, 0)
        model.geom_conaffinity[:] = np.where(solid, 1, 0)
        note = ["The model turns off collisions between its own parts, so every pair of its collision shapes was "
                "checked."]
    else:
        note = ["Only the pairs of parts the model's own collision filter lets touch were checked."]
    plane = model.geom_type == 0
    names = su.info.names["body"]
    tol = max(2e-4, 1e-3 * float(model.stat.extent))
    sp = space(model, su.info)
    d = mujoco.MjData(model)

    def touching(x):
        if not set_pose(sp, d, x):
            return None
        mujoco.mj_collision(model, d)
        selfs, floor = {}, {}
        for i in range(d.ncon):
            c = d.contact[i]
            if c.dist > -tol:
                continue
            g1, g2 = int(model.geom_bodyid[c.geom1]), int(model.geom_bodyid[c.geom2])
            a, b = int(groups[g1]), int(groups[g2])
            if a == b or frozenset((a, b)) in su.info.adjacent:
                continue
            key = tuple(sorted((a, b)))
            ground = g1 == 0 or g2 == 0 or plane[c.geom1] or plane[c.geom2]
            bucket = floor if ground else selfs
            bucket[key] = max(bucket.get(key, 0.0), -float(c.dist))
            who.setdefault(key, (g1, g2))
        return selfs, floor

    who = {}                               # a pair of groups, and the two bodies first seen touching for it
    x0 = sample(sp, 1)[0]
    drawn = touching(x0) or ({}, {})
    skip = set(drawn[0])

    def sweep(seed, count):
        hits, floor_hits, pairs, example, n = 0, 0, {}, None, 0
        for x in sample(sp, count, seed)[1:]:
            r = touching(x)
            if r is None:
                continue
            n += 1
            s = {k: v for k, v in r[0].items() if k not in skip}
            if s:
                hits += 1
                example = example or (x, max(s, key=s.get))
                for k, v in s.items():
                    e = pairs.setdefault(k, [0, 0.0])
                    e[0] += 1
                    e[1] = max(e[1], v)
            if r[1] and not su.info.floating:
                floor_hits += 1
        return hits, floor_hits, n, pairs, example

    su.log("solve", f"checking {POSES} poses for parts running into each other")
    h1, f1, n1, pairs, example = sweep(21, su.n(POSES))
    h2, f2, n2, _, _ = sweep(22, su.n(POSES))

    def label(k):
        return " and ".join("the floor" if g == 0 else names[g] for g in who.get(k, k))

    p1, p2 = h1 / max(n1, 1), h2 / max(n2, 1)
    lo1, hi1 = wilson(h1, n1)
    lo2, hi2 = wilson(h2, n2)
    agree = not (hi1 < lo2 or hi2 < lo1)
    checks = [Check("pose sampling", "pass" if agree else "warn",
                    f"a second set of {n2} random poses finds {100 * p2:.3g}% interfering against {100 * p1:.3g}%, "
                    f"inside each other's 95% intervals" if agree else
                    f"a second set of {n2} random poses finds {100 * p2:.3g}% interfering against {100 * p1:.3g}%, "
                    "outside each other's 95% intervals, so the share is rough", {"first": p1, "second": p2})]
    top = sorted(pairs.items(), key=lambda kv: -kv[1][0])
    if h1:
        k, (cnt, depth) = top[0]
        verdict = "marginal"
        sentence = (f"Parts run into each other in about {100 * p1:.2g}% of the joint range (95% interval "
                    f"{100 * lo1:.2g} to {100 * hi1:.2g}%), most often {label(k)}, by up to {fmt(depth * 1000)} mm, "
                    f"for example at {describe_pose(sp, example[0])}.")
    else:
        verdict = "ok"
        sentence = (f"No two parts run into each other anywhere in the {n1 + n2} poses checked across the range "
                    f"(under {100 * max(hi1, hi2):.2g}% of it at 95% confidence).")
    if f1:
        verdict = "marginal"
        sentence += f" It strikes the floor or the table in about {100 * f1 / max(n1, 1):.2g}% of its range."
    if drawn[0]:
        sentence += (" As drawn, " + "; ".join(f"{label(k)} already overlap by {fmt(v * 1000)} mm"
                                               for k, v in list(drawn[0].items())[:3]) +
                     ", which is left out of the count as part of the model.")
    values = {"share": p1, "interval": [lo1, hi1], "pairs": {label(k): {"poses": v[0], "depth_m": v[1]}
                                                             for k, v in top[:8]},
              "floor_share": f1 / max(n1, 1), "as_drawn": {label(k): v for k, v in drawn[0].items()},
              "poses": n1 + n2}
    if example:
        values["example_pose"] = describe_pose(sp, example[0])
    assumptions = [f"Parts count as running into each other when their collision shapes overlap by more than "
                   f"{fmt(tol * 1000)} mm. Parts joined directly by a joint touch there by design and are left out."
                   ] + note
    return Finding("collision", verdict, sentence, [("interfering", f"{100 * p1:.2g}% of range")], checks,
                   values, [], [], [], {}, assumptions)
