"""Write a Mechanism as MJCF, MuJoCo's model format, or load an MJCF or URDF file the user already has.

Wobbl applies every drive and load itself as joint forces, so a model never needs MuJoCo actuators, and the
actuators inside a loaded MJCF file are switched off. Contact and constraint softness are set here from the
mechanism, not from the time step, so halving the time step changes only the integration.
"""
from __future__ import annotations

import math
import xml.etree.ElementTree as ET

from .install import require as _mj
from .spec import Mechanism, SpecError

PALETTE = ["0.98 0.83 0.0 1", "0.29 0.56 0.85 1", "0.24 0.81 0.48 1", "1.0 0.42 0.42 1", "0.75 0.6 0.95 1",
           "0.95 0.6 0.25 1"]
POINT_DENSITY = 7800.0          # kg/m3, steel, sets the size of a point mass drawn as a ball
# MuJoCo scales a soft constraint's stiffness with the inertia it holds, so a tiny gear carrying real torque would
# slip by radians at the default impedance; 0.9999 makes loops and meshes as near to rigid as MuJoCo allows
HARD = "0.9999 0.9999 0.001 0.5 2"


def _f(v) -> str:
    return " ".join(f"{x + 0.0:.9g}" for x in (v if isinstance(v, (list, tuple)) else [v]))   # no -0


def contact_params(m: Mechanism) -> tuple[str, str]:
    """solref and solimp for every contact. A direct stiffness and damping (an impact tuned to its materials)
    gives a linear spring and dashpot; otherwise a stiff contact settles in 5 ms, critically damped."""
    c = m.contact
    if "stiffness" in c:
        return f"{-c['stiffness']:.9g} {-c.get('damping', 0.0):.9g}", "0.99 0.99 0.001 0.5 2"
    return f"{c.get('timeconst', 0.005):.9g} {c.get('dampratio', 1.0):.9g}", "0.95 0.99 0.001 0.5 2"


def constraint_timeconst(m: Mechanism) -> float:
    """Loops and gear meshes are soft constraints in MuJoCo; this keeps them stiff but stable at the base step."""
    return max(0.002, 4 * m.timestep)


def to_mjcf(m: Mechanism) -> str:
    m.check()
    root = ET.Element("mujoco", model=m.name.replace(" ", "_"))
    ET.SubElement(root, "compiler", angle="radian", eulerseq="XYZ", autolimits="true", boundmass="1e-6",
                  boundinertia="1e-10")
    opt = ET.SubElement(root, "option", timestep=f"{m.timestep:.9g}", gravity=_f(m.gravity), integrator="Euler")
    ET.SubElement(opt, "flag", energy="enable")
    solref, solimp = contact_params(m)
    dflt = ET.SubElement(root, "default")
    ET.SubElement(dflt, "geom", solref=solref, solimp=solimp, friction=f"{m.floor_friction:g} 0.005 0.0001",
                  condim=str(int(m.contact.get("condim", 3))))
    world = ET.SubElement(root, "worldbody")
    if m.floor:
        ET.SubElement(world, "geom", name="floor", type="plane", size="0 0 0.05", rgba="0.2 0.21 0.23 1")
    elems = {"world": world}
    pending = list(m.bodies)
    color = 0
    while pending:
        progressed = False
        for b in list(pending):
            if b.parent not in elems:
                continue
            e = ET.SubElement(elems[b.parent], "body", name=b.name, pos=_f(b.pos))
            if any(b.euler):
                e.set("euler", _f(b.euler))
            for j in (j for j in m.joints if j.body == b.name):
                if j.kind == "free":
                    ET.SubElement(e, "freejoint", name=j.name)
                    continue
                je = ET.SubElement(e, "joint", name=j.name, type=j.kind, axis=_f(j.axis), pos=_f(j.pos))
                if j.range:
                    je.set("range", _f(j.range))
                for attr, v in (("damping", j.damping), ("frictionloss", j.friction), ("armature", j.armature)):
                    if v:
                        je.set(attr, f"{v:.9g}")
            if b.shape != "none":
                g = ET.SubElement(e, "geom", name=b.name, type=b.shape, mass=f"{b.mass:.9g}",
                                  rgba=b.color or PALETTE[color % len(PALETTE)])
                color += 1
                if b.shape in ("capsule", "cylinder"):
                    g.set("fromto", _f(b.fromto))
                    g.set("size", f"{b.radius:.9g}")
                elif b.shape == "box":
                    g.set("size", _f([s / 2 for s in b.size]))
                    g.set("pos", _f(b.center))
                elif b.shape == "sphere":
                    g.set("size", f"{b.radius:.9g}")
                    g.set("pos", _f(b.center))
                else:
                    raise SpecError(f"unknown shape '{b.shape}' (use capsule, cylinder, box, sphere or none)")
                if not b.collide:
                    g.set("contype", "0")
                    g.set("conaffinity", "0")
            for pm in (p for p in m.masses if p.body == b.name and p.mass > 0):
                r = pm.radius or (3 * pm.mass / (4 * math.pi * POINT_DENSITY)) ** (1 / 3)
                g = ET.SubElement(e, "geom", name=pm.name, type="sphere", size=f"{r:.9g}", pos=_f(pm.pos),
                                  mass=f"{pm.mass:.9g}", rgba="0.85 0.86 0.88 1")
                if not pm.collide:
                    g.set("contype", "0")
                    g.set("conaffinity", "0")
            elems[b.name] = e
            pending.remove(b)
            progressed = True
        if not progressed:
            raise SpecError("bodies " + ", ".join(b.name for b in pending) + " do not connect back to the world")
    tc = constraint_timeconst(m)
    if m.loops or m.couplings:
        eq = ET.SubElement(root, "equality")
        for lp in m.loops:
            ET.SubElement(eq, "connect", body1=lp.body1, body2=lp.body2, anchor=_f(lp.point), solref=f"{tc:.9g} 1",
                          solimp=HARD)
        for c in m.couplings:
            k = (-1.0 if c.kind == "gear" else 1.0) / c.ratio
            ET.SubElement(eq, "joint", joint1=c.driven, joint2=c.driver, polycoef=f"0 {k:.12g} 0 0 0",
                          solref=f"{tc:.9g} 1", solimp=HARD)
    if m.loops:
        ex = ET.SubElement(root, "contact")
        for lp in m.loops:
            ET.SubElement(ex, "exclude", body1=lp.body1, body2=lp.body2)
    ET.indent(root, space=" ")
    return ET.tostring(root, encoding="unicode")


FILE_STEP = 0.002        # a model file's own time step is kept up to this, so contacts and the checks stay sharp


def build(m: Mechanism, timestep: float | None = None):
    """The compiled MjModel of a mechanism, at its own time step or the one given. A model file keeps its own
    gravity, contacts and floor, and its own time step up to FILE_STEP unless the mechanism sets one."""
    mujoco = _mj()

    if m.model_file or m.edits:
        from .model import loaded

        model = loaded(m).compile()
        model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_ACTUATION)
        own = min(float(model.opt.timestep), FILE_STEP) if m.model_file else m.timestep
        model.opt.timestep = timestep or m.timestep or own
        return model
    try:
        model = mujoco.MjModel.from_xml_string(to_mjcf(m))
    except ValueError as e:
        raise SpecError(f"MuJoCo could not build the model: {e}") from e
    model.opt.timestep = timestep or m.timestep
    return model
