"""Where a mechanism's joints are at every recorded instant, from its own model.xml and the geom poses MuJoCo
recorded, so no solver is needed to draw them.

A joint is fixed in its body's frame. Each geom of that body was recorded with its world pose, and model.xml says
where the geom sits in the body, so the body's world pose is the geom's pose with that placement undone. The joint's
anchor and axis follow from it. This holds for any MJCF, and for a URDF once MuJoCo has turned it into one.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


def qmul(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    w1, x1, y1, z1 = np.moveaxis(a, -1, 0)
    w2, x2, y2, z2 = np.moveaxis(b, -1, 0)
    return np.stack([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], -1)


def conj(q):
    q = np.asarray(q, float)
    return q * np.array([1, -1, -1, -1])


def rotate(q, v):
    """Vector v turned by quaternion q (w, x, y, z); either may carry a leading frame axis."""
    q = np.asarray(q, float)
    v = np.broadcast_to(np.asarray(v, float), q.shape[:-1] + (3,))
    p = np.concatenate([np.zeros(v.shape[:-1] + (1,)), v], -1)
    return qmul(qmul(q, p), conj(q))[..., 1:]


def z2vec(d) -> np.ndarray:
    """The shortest turn of +z onto d."""
    d = np.asarray(d, float)
    d = d / max(np.linalg.norm(d), 1e-30)
    if d[2] < -0.999999:
        return np.array([0.0, 1.0, 0.0, 0.0])
    q = np.array([1 + d[2], -d[1], d[0], 0.0])
    return q / np.linalg.norm(q)


def _vec(el, key, default):
    s = el.get(key)
    return np.array([float(x) for x in s.split()]) if s else np.array(default, float)


def _local(g) -> tuple[np.ndarray, np.ndarray]:
    """A geom's position and orientation in its body's frame."""
    if g.get("fromto"):
        f = _vec(g, "fromto", [0] * 6)
        return (f[:3] + f[3:]) / 2, z2vec(f[:3] - f[3:])          # MuJoCo turns z from the end back to the start
    q = _vec(g, "quat", [1, 0, 0, 0])
    if g.get("axisangle"):
        aa = _vec(g, "axisangle", [0, 0, 1, 0])
        a = aa[:3] / max(np.linalg.norm(aa[:3]), 1e-30)
        q = np.r_[np.cos(aa[3] / 2), a * np.sin(aa[3] / 2)]
    return _vec(g, "pos", [0, 0, 0]), q / np.linalg.norm(q)


def joints(out: Path, pb: dict) -> dict[str, dict]:
    """{joint: {"type", "anchor" (frames x 3), "axis" (frames x 3), "body", "grounded"}} for every hinge and slide
    joint whose body has a recorded geom. Empty when the run kept no model.xml."""
    p = Path(out) / "model.xml"
    if not p.exists():
        return {}
    try:
        root = ET.parse(p).getroot()
    except ET.ParseError:
        return {}
    rec = {b["name"]: b for b in pb.get("bodies", []) if b.get("name")}
    n = len(pb.get("t", [])) or 1
    found = {}
    world = root.find("worldbody")
    if world is None:
        return {}

    def walk(el, grounded):
        for body in el.findall("body"):
            pose = None
            for g in body.findall("geom"):
                r = rec.get(g.get("name"))
                if r is None:
                    continue
                gp, gq = np.asarray(r["pos"], float), np.asarray(r["quat"], float)
                gp, gq = np.broadcast_to(gp, (n, 3)) if gp.ndim == 1 else gp, \
                    np.broadcast_to(gq, (n, 4)) if gq.ndim == 1 else gq
                lp, lq = _local(g)
                bq = qmul(gq, conj(lq))
                pose = (gp - rotate(bq, lp), bq)
                break
            for j in body.findall("joint"):
                kind = j.get("type", "hinge")
                if pose is None or kind not in ("hinge", "slide"):
                    continue
                bp, bq = pose
                ax = _vec(j, "axis", [0, 0, 1])
                found[j.get("name") or body.get("name")] = {
                    "type": kind, "body": body.get("name"), "grounded": grounded,
                    "anchor": bp + rotate(bq, _vec(j, "pos", [0, 0, 0])),
                    "axis": rotate(bq, ax / max(np.linalg.norm(ax), 1e-30))}
            walk(body, False)

    walk(world, True)
    return found


def masses(out: Path) -> dict[str, float]:
    """Each named geom's mass in kg, where model.xml gives one."""
    p = Path(out) / "model.xml"
    try:
        root = ET.parse(p).getroot()
    except (ET.ParseError, OSError):
        return {}
    return {g.get("name"): float(g.get("mass")) for g in root.iter("geom") if g.get("name") and g.get("mass")}
