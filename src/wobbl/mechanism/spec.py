"""A mechanism described in plain terms: bodies, joints, drives, couplings, loops, loads and a move.

Templates build one of these, and so does a TOML or JSON spec file. Everything inside is SI (m, kg, s, rad, N,
N m). In a spec file a bare number is mm, g, degrees, N m, rpm or s, as ``quantities`` explains. A spec file can
also point at an MJCF or URDF model with ``model = "robot.urdf"`` and add only the drives and the move.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

from . import quantities as Q
from .motors import Motor, motor_from


class SpecError(ValueError):
    pass


@dataclass
class Body:
    name: str
    parent: str = "world"
    pos: tuple = (0.0, 0.0, 0.0)          # body frame origin in the parent's frame
    euler: tuple = (0.0, 0.0, 0.0)        # rad, rotations about x, y, z of the parent frame
    shape: str = "capsule"                # capsule, cylinder, box, sphere, none
    fromto: tuple | None = None           # capsule or cylinder axis, two points in this body's frame
    radius: float = 0.01
    size: tuple = ()                      # box full lengths
    center: tuple = (0.0, 0.0, 0.0)       # box or sphere center in this body's frame
    mass: float = 0.0
    collide: bool = True
    support: bool = False                 # meant to rest on the floor, so touching it is not interference
    color: str = ""


@dataclass
class Joint:
    name: str
    body: str
    kind: str = "hinge"                   # hinge, slide, free
    axis: tuple = (0.0, 0.0, 1.0)
    pos: tuple = (0.0, 0.0, 0.0)          # in the body's frame
    range: tuple | None = None
    damping: float = 0.0                  # N m s/rad
    friction: float = 0.0                 # dry friction torque, N m
    armature: float = 0.0                 # reflected rotor inertia, kg m2
    start: float = 0.0
    speed: float | tuple = 0.0            # starting speed, or a free body's velocity (x, y, z)


@dataclass
class PointMass:
    name: str
    body: str
    mass: float
    pos: tuple = (0.0, 0.0, 0.0)
    radius: float = 0.0                   # 0 picks the size of a ball of that mass at 2000 kg/m3
    collide: bool = False


@dataclass
class Drive:
    joint: str
    motor: Motor | None = None            # None is an ideal drive with no limits
    mode: str = "servo"                   # servo follows the move, voltage runs the motor open loop
    ratio: float = 1.0                    # extra reduction between the motor and the joint
    efficiency: float = 1.0
    voltage: float = 1.0                  # share of full voltage in voltage mode, sign sets the direction

    @property
    def at_joint(self) -> Motor | None:
        if self.motor is None:
            return None
        return self.motor if self.ratio == 1 and self.efficiency == 1 else self.motor.geared(self.ratio,
                                                                                          self.efficiency)


@dataclass
class Coupling:
    driver: str
    driven: str
    ratio: float                          # driver turns per driven turn
    efficiency: float = 1.0
    kind: str = "gear"                    # gear reverses the direction, belt keeps it


@dataclass
class Loop:
    body1: str
    body2: str
    point: tuple                          # in body1's frame


@dataclass
class Load:
    joint: str = ""
    torque: float = 0.0                   # N m on a joint (N on a slide)
    kind: str = "drag"                    # drag opposes the motion, constant pushes one way
    body: str = ""                        # or a force on a body
    force: tuple = (0.0, 0.0, 0.0)        # N, world frame
    point: tuple = (0.0, 0.0, 0.0)        # where on the body, in its frame
    ramp: float = 0.0                     # s to grow from zero, 0 applies it at once


@dataclass
class Mechanism:
    name: str = "mechanism"
    bodies: list[Body] = field(default_factory=list)
    joints: list[Joint] = field(default_factory=list)
    masses: list[PointMass] = field(default_factory=list)
    drives: list[Drive] = field(default_factory=list)
    couplings: list[Coupling] = field(default_factory=list)
    loops: list[Loop] = field(default_factory=list)
    loads: list[Load] = field(default_factory=list)
    move: list[tuple[float, dict[str, float]]] = field(default_factory=list)
    spin: dict[str, tuple[float, float]] = field(default_factory=dict)   # joint: (rad/s, s to reach it)
    hold: float = 0.0                     # s to stay at the last waypoint
    duration: float = 0.0                 # s for a run with no move
    gravity: tuple = (0.0, 0.0, -9.80665)
    floor: bool = False
    floor_friction: float = 0.8
    timestep: float = 1e-3
    contact: dict = field(default_factory=dict)   # timeconst, dampratio, or stiffness and damping (direct)
    model_file: str = ""                  # an MJCF or URDF file used in place of bodies and joints
    question: str = ""

    def body(self, name: str) -> Body:
        for b in self.bodies:
            if b.name == name:
                return b
        raise SpecError(f"no body named '{name}'")

    def joint(self, name: str) -> Joint:
        for j in self.joints:
            if j.name == name:
                return j
        raise SpecError(f"no joint named '{name}'")

    def drive(self, joint: str) -> Drive | None:
        return next((d for d in self.drives if d.joint == joint), None)

    @property
    def end_time(self) -> float:
        t_move = self.move[-1][0] if self.move else max([r for _, r in self.spin.values()] + [0.0])
        return max(t_move + self.hold, self.duration)

    def check(self):
        names = {b.name for b in self.bodies}
        if len(names) != len(self.bodies):
            raise SpecError("two bodies share a name")
        for b in self.bodies:
            if b.parent != "world" and b.parent not in names:
                raise SpecError(f"body '{b.name}' hangs from '{b.parent}', which is not a body")
        jn = {j.name for j in self.joints}
        if not self.model_file:
            for j in self.joints:
                if j.body not in names:
                    raise SpecError(f"joint '{j.name}' moves '{j.body}', which is not a body")
            for d in self.drives:
                if d.joint not in jn:
                    raise SpecError(f"a drive names joint '{d.joint}', which does not exist")
        for t, q in self.move:
            for k in q:
                if not self.model_file and k not in jn:
                    raise SpecError(f"the move names joint '{k}', which does not exist")
        ts = [t for t, _ in self.move]
        if ts != sorted(ts) or len(set(ts)) != len(ts):
            raise SpecError("move waypoints need increasing times")


# --- reading spec files ------------------------------------------------------------------------------------
def _vec(v, read=Q.length, n=3) -> tuple:
    if v is None:
        return (0.0,) * n
    vals = Q.listed(v, read)
    if len(vals) != n:
        raise SpecError(f"expected {n} values, got '{v}'")
    return tuple(vals)


def _axis(v) -> tuple:
    if isinstance(v, str) and v.strip().lstrip("+-") in ("x", "y", "z"):
        s = -1.0 if v.strip().startswith("-") else 1.0
        return tuple(s * float(c == v.strip()[-1]) for c in "xyz")
    a = _vec(v, Q.number)
    n = math.sqrt(sum(x * x for x in a))
    if n == 0:
        raise SpecError("a joint axis cannot be zero")
    return tuple(x / n for x in a)


def _take(d: dict, where: str, allowed: set):
    extra = set(d) - allowed
    if extra:
        raise SpecError(f"unknown settings in {where}: {', '.join(sorted(extra))}")


def read_spec(source: str | Path | dict) -> Mechanism:
    """A Mechanism from a TOML or JSON file, a TOML or JSON string, or an already parsed dict."""
    base = Path(".")
    if isinstance(source, dict):
        d = source
    else:
        text, path = str(source), Path(str(source))
        if len(text) < 1000 and "\n" not in text and path.exists():
            base = path.resolve().parent
            text = path.read_text(encoding="utf-8")
        d = _parse_text(text, path.suffix.lower())
    return from_dict(d, base)


def _parse_text(text: str, suffix: str = "") -> dict:
    t = text.lstrip()
    if suffix == ".json" or t.startswith("{"):
        return json.loads(text)
    try:
        import tomllib
    except ImportError:  # Python 3.10
        import tomli as tomllib
    return tomllib.loads(text)


def from_dict(d: dict, base: Path = Path(".")) -> Mechanism:
    d = dict(d)
    top = {"name", "question", "gravity", "floor", "floor_friction", "duration", "timestep", "hold", "model",
           "body", "joint", "mass", "drive", "coupling", "loop", "load", "move", "spin", "contact"}
    _take(d, "the spec", top)
    m = Mechanism(name=str(d.get("name", "mechanism")), question=str(d.get("question", "")))
    if "gravity" in d:
        g = d["gravity"]
        m.gravity = (0.0, 0.0, -Q.accel(g)) if not isinstance(g, (list, tuple)) else _vec(g, Q.accel)
    m.floor = bool(d.get("floor", False))
    m.floor_friction = float(d.get("floor_friction", 0.8))
    m.duration = Q.time(d.get("duration", 0))
    m.timestep = Q.time(d.get("timestep", "1 ms"), "ms")
    m.hold = Q.time(d.get("hold", 0))
    if "model" in d:
        p = Path(str(d["model"]))
        m.model_file = str(p if p.is_absolute() else (base / p).resolve())
    for b in d.get("body", []):
        _take(b, f"body {b.get('name', '')}", {"name", "parent", "pos", "euler", "shape", "from", "to", "radius",
                                                "size", "center", "mass", "collide", "support", "color", "joint"})
        shape = str(b.get("shape", "capsule"))
        body = Body(str(b["name"]), str(b.get("parent", "world")), _vec(b.get("pos")),
                    _vec(b.get("euler"), Q.angle), shape, radius=Q.length(b.get("radius", 10)),
                    center=_vec(b.get("center")), mass=Q.mass(b.get("mass", 0)),
                    collide=bool(b.get("collide", True)), support=bool(b.get("support", False)),
                    color=str(b.get("color", "")))
        if shape in ("capsule", "cylinder"):
            body.fromto = _vec(b.get("from")) + _vec(b.get("to", [100, 0, 0]))
        if shape == "box":
            body.size = _vec(b.get("size", [20, 20, 20]))
        m.bodies.append(body)
        if "joint" in b:   # a joint written inline on its body
            j = b["joint"] if isinstance(b["joint"], dict) else {"name": b["joint"]}
            m.joints.append(_joint({**j, "body": body.name}))
    m.joints += [_joint(j) for j in d.get("joint", [])]
    for pm in d.get("mass", []):
        _take(pm, "a mass", {"name", "body", "mass", "pos", "radius", "collide"})
        m.masses.append(PointMass(str(pm.get("name", f"mass{len(m.masses) + 1}")), str(pm["body"]),
                                  Q.mass(pm["mass"]), _vec(pm.get("pos")), Q.length(pm.get("radius", 0)),
                                  bool(pm.get("collide", False))))
    for dr in d.get("drive", []):
        dr = dict(dr)
        joint = str(dr.pop("joint"))
        mode = str(dr.pop("mode", "servo"))
        ratio = Q.number(dr.pop("ratio", 1))
        eff = Q.number(dr.pop("efficiency", 1))
        volt = Q.number(dr.pop("voltage", 1))
        motor = motor_from(dr) if dr else None
        m.drives.append(Drive(joint, motor, mode, ratio, eff, volt))
    for c in d.get("coupling", []):
        _take(c, "a coupling", {"driver", "driven", "ratio", "efficiency", "kind"})
        m.couplings.append(Coupling(str(c["driver"]), str(c["driven"]), Q.number(c["ratio"]),
                                    Q.number(c.get("efficiency", 1)), str(c.get("kind", "gear"))))
    for lp in d.get("loop", []):
        _take(lp, "a loop", {"body1", "body2", "point"})
        m.loops.append(Loop(str(lp["body1"]), str(lp["body2"]), _vec(lp.get("point"))))
    for ld in d.get("load", []):
        _take(ld, "a load", {"joint", "torque", "kind", "body", "force", "point", "ramp"})
        m.loads.append(Load(str(ld.get("joint", "")), Q.torque(ld.get("torque", 0)), str(ld.get("kind", "drag")),
                            str(ld.get("body", "")), _vec(ld.get("force"), Q.force), _vec(ld.get("point")),
                            Q.time(ld.get("ramp", 0))))
    for k, v in dict(d.get("spin", {})).items():   # crank = "45 rpm", or {speed = "45 rpm", ramp = "0.3 s"}
        v = v if isinstance(v, dict) else {"speed": v}
        m.spin[str(k)] = (Q.speed(v["speed"]), Q.time(v.get("ramp", 0.3)))
    for wp in d.get("move", []):
        wp = dict(wp)
        t = Q.time(wp.pop("t"))
        m.move.append((t, {k: _joint_value(m, k, v) for k, v in wp.items()}))
    if "contact" in d:
        m.contact = {k: float(v) for k, v in dict(d["contact"]).items()}
    m.check()
    return m


def _joint(j: dict) -> Joint:
    _take(j, f"joint {j.get('name', '')}", {"name", "body", "type", "axis", "pos", "range", "damping", "friction",
                                            "armature", "start", "speed"})
    kind = str(j.get("type", "hinge"))
    lin = kind == "slide"
    ang = Q.length if lin else Q.angle
    rng = tuple(Q.listed(j["range"], ang)) if "range" in j else None
    return Joint(str(j["name"]), str(j["body"]), kind, _axis(j.get("axis", "z")), _vec(j.get("pos")), rng,
                 Q.number(j.get("damping", 0)), (Q.force if lin else Q.torque)(j.get("friction", 0)),
                 Q.number(j.get("armature", 0)), ang(j.get("start", 0)),
                 (Q.velocity if lin else Q.speed)(j.get("speed", 0)))


def _joint_value(m: Mechanism, name: str, v) -> float:
    try:
        lin = m.joint(name).kind == "slide"
    except SpecError:
        lin = False
    return Q.length(v) if lin else Q.angle(v)
