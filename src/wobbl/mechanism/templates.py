"""Ready made mechanisms: a pendulum, a robot arm doing a pick and place, a four bar linkage, a gear train, two
parts colliding and a stand that may tip. Each builds a Mechanism from plain values and keeps the closed form
numbers its checks need.

Values come in as text or numbers (``--set payload=500g``). A bare number is mm, g, degrees, N m, rpm or s.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from . import quantities as Q
from . import theory as T
from .motors import Motor, get_motor, motor_from
from .spec import Body, Coupling, Drive, Joint, Load, Loop, Mechanism, PointMass, SpecError


@dataclass
class Built:
    mech: Mechanism
    kind: str
    params: dict                       # the values used, readable, for the outcome
    info: dict = field(default_factory=dict)


class Params:
    """Reads template values and remembers which ones the user gave, so unknown keys are caught."""

    def __init__(self, raw: dict, defaults: dict):
        self.raw = {str(k).strip().lower(): v for k, v in (raw or {}).items()}
        self.defaults = defaults
        self.used = {}

    def get(self, key: str, read=None):
        v = self.raw.get(key, self.defaults.get(key))
        self.used[key] = v
        if v is None or read is None:
            return v
        try:
            return read(v)
        except (ValueError, TypeError) as e:
            raise SpecError(f"could not read {key}={v}: {e}") from None

    def given(self, key: str) -> bool:
        return key in self.raw

    def done(self, extra: set = frozenset()):
        unknown = set(self.raw) - set(self.defaults) - set(self.used) - set(extra)
        if unknown:
            raise SpecError(f"unknown values {', '.join(sorted(unknown))}; this template takes "
                            f"{', '.join(sorted(self.defaults))}")


def _motor(p: Params, key: str, fallback: str | None):
    """A drive's motor: a catalog name, or stall_torque and no_load_speed given under the template."""
    name = p.get(key)
    if p.given("stall_torque") and key in ("motor", "servo"):
        d = {"stall_torque": p.get("stall_torque"), "no_load_speed": p.get("no_load_speed") or "inf"}
        for k in ("continuous_torque", "no_load_current", "stall_current"):
            if p.given(k):
                d[k] = p.get(k)
        return motor_from(d)
    if name in (None, "", "none", "ideal"):
        return None if fallback is None else get_motor(fallback)
    return motor_from(str(name))


# --- pendulum -----------------------------------------------------------------------------------------------
def pendulum(raw: dict) -> Built:
    p = Params(raw, {"length": "1000 mm", "angle": "120 deg", "mass": "1000 g", "periods": 3, "timestep": "1 ms"})
    L, th0, mass = p.get("length", Q.length), p.get("angle", Q.angle), p.get("mass", Q.mass)
    if not 0 < th0 < math.pi:
        raise SpecError("the release angle must be between 0 and 180 degrees")
    period = T.pendulum_period(L, th0)
    m = Mechanism("pendulum", timestep=p.get("timestep", Q.time))
    m.bodies.append(Body("rod", fromto=(0, 0, 0, 0, 0, -L), radius=max(L / 500, 1e-3), mass=0.0, collide=False))
    m.joints.append(Joint("pivot", "rod", axis=(0.0, 1.0, 0.0), start=th0))
    m.masses.append(PointMass("bob", "rod", mass, (0.0, 0.0, -L), radius=max(L / 1000, 5e-4)))
    m.duration = p.get("periods", Q.number) * period
    p.done()
    return Built(m, "pendulum", p.used, {"period_s": period, "small_angle_s": 2 * math.pi * math.sqrt(L / T.G),
                                         "length_m": L, "angle_rad": th0})


# --- robot arm ----------------------------------------------------------------------------------------------
ARM = {"links": "120,120,60", "link_mass": "60,50,30", "servo": "mg996r", "base_servo": None, "shoulder_servo": None,
       "elbow_servo": None, "wrist_servo": None, "payload": "500 g", "gripper": "40 g", "base_height": "60 mm",
       "pick": "200,0,40", "place": "0,200,40", "lift": "80 mm", "move_time": "1.2 s", "dwell": "0.5 s",
       "mount": "clamped", "base_mass": "400 g", "base_size": "140,140", "ratio": 1, "ratio_joint": "shoulder",
       "counterweight": "0 g", "counterweight_arm": "60 mm", "base_ratio": None, "shoulder_ratio": None,
       "elbow_ratio": None, "wrist_ratio": None, "timestep": "1 ms"}


def arm_poses(links: list[float], base_h: float, targets: list[tuple[float, float, float]]) -> list[dict]:
    """Joint angles that put the gripper tip on each target, elbow up, with a three link arm's gripper pointing
    straight down."""
    out = []
    l1, l2 = links[0], links[1]
    for x, y, z in targets:
        r, zr = math.hypot(x, y), z - base_h
        if len(links) == 3:
            zr += links[2]
        sol = T.arm_ik(l1, l2, r, zr)
        if sol is None:
            raise SpecError(f"the point ({x * 1000:.0f}, {y * 1000:.0f}, {z * 1000:.0f}) mm is out of the arm's reach")
        pose = {"base": math.atan2(y, x), "shoulder": sol[0], "elbow": sol[1]}
        if len(links) == 3:
            pose["wrist"] = -math.pi / 2 - sol[0] - sol[1]
        out.append(pose)
    return out


def arm(raw: dict) -> Built:
    p = Params(raw, ARM)
    links = p.get("links", lambda v: Q.listed(v, Q.length))
    lmass = p.get("link_mass", lambda v: Q.listed(v, Q.mass))
    if len(links) not in (2, 3) or len(lmass) != len(links):
        raise SpecError("links and link_mass need two or three values each (upper arm, forearm, wrist to tip)")
    pitch = ["shoulder", "elbow", "wrist"][:len(links)]
    servos = {j: _motor(p, f"{j}_servo", None) or _motor(p, "servo", "mg996r") for j in ["base", *pitch]}
    payload, gripper = p.get("payload", Q.mass), p.get("gripper", Q.mass)
    base_h = p.get("base_height", Q.length)
    pick, place = (p.get(k, lambda v: tuple(Q.listed(v, Q.length))) for k in ("pick", "place"))
    lift = p.get("lift", Q.length)
    tseg, dwell = p.get("move_time", Q.time), p.get("dwell", Q.time)
    mount = str(p.get("mount")).lower()
    ratio, rjoint = p.get("ratio", Q.number), str(p.get("ratio_joint"))
    cw, cw_arm = p.get("counterweight", Q.mass), p.get("counterweight_arm", Q.length)
    if mount not in ("clamped", "free"):
        raise SpecError("mount is clamped (screwed to the table) or free (standing on it)")
    targets = [pick, (pick[0], pick[1], pick[2] + lift), (place[0], place[1], place[2] + lift), place]
    poses = arm_poses(links, base_h, targets)

    m = Mechanism("robot arm", timestep=p.get("timestep", Q.time), floor=True)
    bw, bd = p.get("base_size", lambda v: Q.listed(v, Q.length))
    box_h = min(0.03, 0.5 * base_h)
    base_mass = p.get("base_mass", Q.mass)
    m.bodies.append(Body("base", shape="box", size=(bw, bd, box_h), center=(0, 0, box_h / 2), mass=base_mass,
                         support=True, color="0.35 0.37 0.4 1"))
    if mount == "free":
        m.joints.append(Joint("base_free", "base", kind="free"))
    m.bodies.append(Body("turret", "base", (0, 0, box_h), fromto=(0, 0, 0, 0, 0, base_h - box_h), radius=0.022,
                         mass=servos["shoulder"].mass + 0.02, shape="cylinder", collide=False))
    m.joints.append(Joint("base", "turret"))
    parent, ppos = "turret", (0.0, 0.0, base_h - box_h)
    names = {"shoulder": "upper arm", "elbow": "forearm", "wrist": "hand"}
    for k, (jn, L, lm) in enumerate(zip(pitch, links, lmass)):
        bname = names[jn]
        m.bodies.append(Body(bname, parent, ppos, fromto=(0, 0, 0, L, 0, 0), radius=0.012 - 0.002 * k, mass=lm))
        m.joints.append(Joint(jn, bname, axis=(0.0, -1.0, 0.0), range=(-math.pi, math.pi)))
        nxt = pitch[k + 1] if k + 1 < len(pitch) else None
        if nxt:
            m.masses.append(PointMass(f"{nxt} servo", bname, servos[nxt].mass, (L, 0.0, 0.0)))
        parent, ppos = bname, (L, 0.0, 0.0)
    tip_body, tip = names[pitch[-1]], (links[-1], 0.0, 0.0)
    m.masses.append(PointMass("gripper", tip_body, gripper, tip))
    m.masses.append(PointMass("payload", tip_body, payload, tip))
    if cw > 0:
        m.masses.append(PointMass("counterweight", "upper arm", cw, (-cw_arm, 0.0, 0.0)))
    for j in ["base", *pitch]:
        mot = servos[j]
        r = p.get(f"{j}_ratio", Q.number) or (ratio if j == rjoint else 1.0)
        m.drives.append(Drive(j, mot, "servo", r, 0.9 if r != 1 else 1.0))   # a belt or gear stage loses 10%
    m.move = [(i * tseg, pose) for i, pose in enumerate(poses)]
    m.hold = dwell
    p.done()
    static_masses = []   # (link index, distance along it, kg) for the hand statics check
    for k, (L, lm) in enumerate(zip(links, lmass)):
        static_masses.append((k, L / 2, lm))
        if k + 1 < len(links):
            static_masses.append((k, L, servos[pitch[k + 1]].mass))
    static_masses += [(len(links) - 1, links[-1], gripper), (len(links) - 1, links[-1], payload)]
    if cw > 0:
        static_masses.append((0, -cw_arm, cw))
    return Built(m, "arm", p.used, {"links": links, "pitch": pitch, "servos": servos, "poses": poses,
                                    "static_masses": static_masses, "mount": mount, "base_mass": base_mass,
                                    "base_size": (bw, bd), "payload": payload, "segment_s": tseg, "dwell_s": dwell,
                                    "ratio": ratio, "ratio_joint": rjoint, "targets": targets})


# --- four bar linkage ---------------------------------------------------------------------------------------
FOURBAR = {"ground": "120 mm", "crank": "30 mm", "coupler": "125 mm", "rocker": "45 mm", "motor": None,
           "stall_torque": None, "no_load_speed": None, "continuous_torque": None, "speed": "45 rpm",
           "load": "0 N*m", "link_mass": "auto", "start": "0 deg", "turns": 2, "sweep": None, "timestep": "0.5 ms"}


def fourbar(raw: dict) -> Built:
    p = Params(raw, FOURBAR)
    d, a, b, c = (p.get(k, Q.length) for k in ("ground", "crank", "coupler", "rocker"))
    motor = _motor(p, "motor", None)
    speed = p.get("speed", Q.speed)
    load = p.get("load", Q.torque)
    t2_0 = p.get("start", Q.angle)
    turns = p.get("turns", Q.number)
    lm = p.get("link_mass")
    masses = ([L * 0.4 for L in (a, b, c)] if str(lm).strip().lower() == "auto" else  # 0.4 g/mm, a 10x3 mm strip
              Q.listed(lm, Q.mass) if "," in str(lm) else [Q.mass(lm)] * 3)
    cls, full = T.grashof(d, a, b, c)
    t4, t3 = T.freudenstein(d, a, b, c, t2_0)
    if not math.isfinite(float(t4)):
        raise SpecError(f"these links cannot be assembled with the crank at {math.degrees(t2_0):.0f} degrees; "
                        "pick another start")
    t4, t3 = float(t4), float(t3)
    m = Mechanism("four bar linkage", timestep=p.get("timestep", Q.time), gravity=(0.0, 0.0, -T.G))
    # drawn assembled at the start, since MuJoCo pins the loop where the model sits
    m.bodies.append(Body("crank", euler=(0, 0, t2_0), fromto=(0, 0, 0, a, 0, 0), radius=0.004, mass=masses[0],
                         collide=False))
    m.bodies.append(Body("coupler", "crank", (a, 0, 0), (0, 0, t3 - t2_0), fromto=(0, 0, 0, b, 0, 0), radius=0.003,
                         mass=masses[1], collide=False))
    m.bodies.append(Body("rocker", "world", (d, 0, 0), (0, 0, t4), fromto=(0, 0, 0, c, 0, 0), radius=0.004,
                         mass=masses[2], collide=False))
    for j in ("crank", "coupler", "rocker"):
        m.joints.append(Joint(j, j))
    m.loops.append(Loop("coupler", "rocker", (b, 0, 0)))
    if load:
        m.loads.append(Load("rocker", load, "drag"))
    w_est = motor.no_load_speed if motor is not None and math.isfinite(motor.no_load_speed) else speed
    if motor is not None:
        m.drives.append(Drive("crank", motor, "voltage"))
        m.duration = turns * 2 * math.pi / w_est * 1.3 + 0.3
    else:
        m.drives.append(Drive("crank", None, "servo"))
        m.spin = {"crank": (speed, 0.3)}
        m.duration = turns * 2 * math.pi / speed + 0.3
    sweep = p.get("sweep", Q.angle)
    p.done()
    return Built(m, "fourbar", p.used, {"lengths": (d, a, b, c), "grashof": cls, "full_turn": full,
                                        "start": (t2_0, t3, t4), "motor": motor, "speed": speed, "load": load,
                                        "sweep": sweep, "turns": turns})


# --- gear train ---------------------------------------------------------------------------------------------
GEARS = {"teeth": "12:36,10:40", "efficiency": 0.97, "motor": "fa-130", "stall_torque": None, "no_load_speed": None,
         "continuous_torque": None, "no_load_current": None, "stall_current": None, "load": "0.01 N*m",
         "load_inertia": "0 kg*m2", "module": "0.5 mm", "face": "4 mm", "density": "1410 kg/m3", "speed": None,
         "rotor_inertia": "auto", "time": None, "timestep": "0.1 ms"}


ROTOR_TAU = 0.02


def _stages(v) -> list[tuple[int, int]]:
    out = []
    for s in str(v).replace(" ", "").split(","):
        if ":" not in s:
            raise SpecError(f"each stage is driver:driven teeth, like 12:36, got '{s}'")
        a, b = s.split(":")
        out.append((int(a), int(b)))
    if not out or any(a < 6 or b < 6 for a, b in out):
        raise SpecError("gears need at least 6 teeth")
    return out


def gears(raw: dict) -> Built:
    p = Params(raw, GEARS)
    stages = p.get("teeth", _stages)
    eff = p.get("efficiency", Q.number)
    motor = _motor(p, "motor", "fa-130")
    load = p.get("load", Q.torque)
    module, face = p.get("module", Q.length), p.get("face", Q.length)
    rho = p.get("density", lambda v: Q.number(str(v).lower().replace("kg/m3", "")))
    j_load = p.get("load_inertia", lambda v: Q.number(str(v).lower().replace("kg*m2", "").replace("kgm2", "")))
    rotor = str(p.get("rotor_inertia")).strip().lower()
    # without a datasheet value, a rotor that gives the bare motor a 20 ms mechanical time constant, typical of small
    # brushed motors: tau = J w0 / Ts
    rotor_auto = rotor == "auto"
    j_rotor = (ROTOR_TAU * motor.stall_torque / motor.no_load_speed if rotor_auto and math.isfinite(motor.no_load_speed)
               else 0.0 if rotor_auto else Q.number(rotor.replace("kg*m2", "").replace("kgm2", "")))
    ratio, eff_total = T.gear_train(stages, eff)
    m = Mechanism("gear train", timestep=p.get("timestep", Q.time), gravity=(0.0, 0.0, -T.G))
    x = 0.0
    j_shafts = []
    for i in range(len(stages) + 1):
        m.bodies.append(Body(f"shaft{i}", pos=(x, 0, 0), shape="none"))
        m.joints.append(Joint(f"shaft{i}", f"shaft{i}", armature=j_rotor if i == 0 else 0.0))
        js = 0.0
        for side, k in ((1, i - 1), (0, i)):       # the driven gear of the stage before, the driver of this one
            if 0 <= k < len(stages):
                z = stages[k][side]
                r = module * z / 2
                mass = rho * math.pi * r * r * face
                js += 0.5 * mass * r * r
                m.bodies.append(Body(f"gear{k + 1}{'ab'[side]}", f"shaft{i}", (0, 0, k * (face + 0.002)),
                                     shape="cylinder", fromto=(0, 0, 0, 0, 0, face), radius=r, mass=mass,
                                     collide=False))
        j_shafts.append(js)
        if i < len(stages):
            a, b = stages[i]
            x += module * (a + b) / 2
    if j_load > 0:
        r = 0.05
        m.bodies.append(Body("load", f"shaft{len(stages)}", (0, 0, -0.01), shape="cylinder",
                             fromto=(0, 0, 0, 0, 0, -0.004), radius=r, mass=2 * j_load / (r * r), collide=False,
                             color="0.6 0.62 0.66 1"))
    for i, (a, b) in enumerate(stages):
        m.couplings.append(Coupling(f"shaft{i}", f"shaft{i + 1}", b / a, eff))
    m.drives.append(Drive("shaft0", motor, "voltage"))
    if load:
        m.loads.append(Load(f"shaft{len(stages)}", load, "drag"))
    # how long the train takes to come up to speed: inertia seen at the motor over the motor's line
    j_eq = j_rotor + sum(js / math.prod(b / a for a, b in stages[:i]) ** 2 for i, js in enumerate(j_shafts))
    j_eq += j_load / ratio ** 2
    tau_m = j_eq * motor.no_load_speed / motor.stall_torque if math.isfinite(motor.no_load_speed) else 0.05
    m.duration = p.get("time", Q.time) or max(0.2, min(10.0, 8 * tau_m))
    want = p.get("speed", Q.speed)
    p.done()
    w_m, t_m = T.motor_steady(motor.stall_torque, motor.no_load_speed, ratio, eff_total, load)
    return Built(m, "gears", p.used, {"stages": stages, "ratio": ratio, "efficiency": eff_total, "mesh_eff": eff,
                                      "motor": motor, "load": load, "hand_motor_speed": w_m, "hand_motor_torque": t_m,
                                      "j_eq": j_eq, "tau_mech_s": tau_m, "want_speed": want, "j_rotor": j_rotor,
                                      "rotor_auto": rotor_auto})


# --- impact -------------------------------------------------------------------------------------------------
IMPACT = {"mass": "200 g", "speed": "2 m/s", "drop": None, "target_mass": None, "target_speed": "0 m/s",
          "restitution": None, "radius": "10 mm", "target_radius": None, "material": "mild-steel",
          "target_material": None, "limit": None, "stiffness": None}


def impact(raw: dict) -> Built:
    from ..materials import resolve_material

    p = Params(raw, IMPACT)
    m1 = p.get("mass", Q.mass)
    drop = p.get("drop", Q.length)
    v = math.sqrt(2 * T.G * drop) if drop else p.get("speed", Q.velocity)
    tm = p.get("target_mass")
    fixed = drop is not None or tm in (None, "", "fixed", "ground", "inf")
    m2 = math.inf if fixed else Q.mass(tm)
    v2 = 0.0 if fixed else p.get("target_speed", Q.velocity)
    r1 = p.get("radius", Q.length)
    r2 = p.get("target_radius", Q.length) or (math.inf if fixed else r1)
    mat1 = resolve_material(str(p.get("material")))
    mat2 = resolve_material(str(p.get("target_material") or p.get("material")))
    e = p.get("restitution", Q.number)
    e_given = e is not None
    if e is None:   # metals bounce, plastics and wood less; a typical value until the user knows better
        e = 0.6 if mat1.category == "metal" and mat2.category == "metal" else 0.4
    if not 0 <= e <= 1:
        raise SpecError("restitution is between 0 (no bounce) and 1 (perfectly elastic)")
    m_eff = m1 if fixed else m1 * m2 / (m1 + m2)
    r_eff = r1 if math.isinf(r2) else r1 * r2 / (r1 + r2)
    e_eff = T.effective_modulus(mat1.E * 1e6, mat1.nu, mat2.E * 1e6, mat2.nu)
    vrel = v - v2
    if vrel <= 0:
        raise SpecError("the parts are not moving toward each other")
    hz = T.hertz_impact(m_eff, vrel, e_eff, r_eff)
    # a linear contact that stores the impact's energy at Hertz's peak force: k = F^2 / (m v^2), per unit mass
    k_given = p.get("stiffness", lambda s: Q.number(str(s).lower().replace("n/m", "")))
    k_lin = k_given or hz["force_N"] ** 2 / (m_eff * vrel * vrel)
    w0 = math.sqrt(k_lin / m_eff)
    zeta = T.zeta_for_restitution(e)
    t_c = math.pi / w0
    m = Mechanism("impact", timestep=t_c / 400, gravity=(0.0, 0.0, -T.G) if drop else (0.0, 0.0, 0.0),
                  floor=fixed, contact={"stiffness": w0 * w0, "damping": 2 * zeta * w0, "condim": 1})
    gap = max(1e-6, 0.05 * vrel * t_c)
    if fixed:
        m.bodies.append(Body("part", pos=(0, 0, r1 + gap), shape="sphere", radius=r1, mass=m1))
        m.joints.append(Joint("part_free", "part", kind="free", speed=(0.0, 0.0, -v)))
    else:
        m.bodies.append(Body("part", pos=(-(r1 + gap), 0, 0), shape="sphere", radius=r1, mass=m1))
        m.bodies.append(Body("target", pos=(r2, 0, 0), shape="sphere", radius=r2, mass=m2,
                             color="0.29 0.56 0.85 1"))
        m.joints += [Joint("part_free", "part", kind="free", speed=(v, 0.0, 0.0)),
                     Joint("target_free", "target", kind="free", speed=(v2, 0.0, 0.0))]
    m.duration = gap / vrel + 3 * t_c
    limit = p.get("limit", Q.force)
    p.done()
    after = T.collision(m1, v, m2, v2, e) if not fixed else (-e * v, 0.0)
    return Built(m, "impact", p.used, {"m1": m1, "m2": m2, "v1": v, "v2": v2, "fixed": fixed, "e": e,
                                       "e_given": e_given, "hertz": hz, "k_lin": k_lin, "w0": w0, "zeta": zeta,
                                       "m_eff": m_eff, "contact_time_s": t_c, "after": after,
                                       "materials": (mat1, mat2), "radii": (r1, r2), "limit": limit, "drop": drop,
                                       "k_given": bool(k_given),
                                       "linear_peak_N": T.dashpot_peak(m_eff, vrel, w0, zeta)})


# --- tipping ------------------------------------------------------------------------------------------------
TIP = {"base": "160,160", "base_mass": "1200 g", "base_height": "20 mm", "mass": "300 g", "height": "250 mm",
       "payload": "500 g", "reach": "120 mm", "payload_height": "300 mm", "force": "0 N", "force_height": None,
       "accel": "0 m/s2", "friction": 0.6, "counterweight": "0 g", "time": "2 s"}


def tip(raw: dict) -> Built:
    p = Params(raw, TIP)
    bw, bd = p.get("base", lambda v: Q.listed(v, Q.length))
    bm, bh = p.get("base_mass", Q.mass), p.get("base_height", Q.length)
    um, uh = p.get("mass", Q.mass), p.get("height", Q.length)
    pm, reach, ph = p.get("payload", Q.mass), p.get("reach", Q.length), p.get("payload_height", Q.length)
    force = p.get("force", Q.force)
    fh = p.get("force_height", Q.length) or ph
    acc = p.get("accel", lambda v: Q.accel(v) if not str(v).strip().endswith("g") else
                Q.number(str(v).strip()[:-1]) * T.G)
    mu = p.get("friction", Q.number)
    cw = p.get("counterweight", Q.mass)
    m = Mechanism("stand", timestep=1e-3, floor=True, floor_friction=mu,
                  gravity=(acc, 0.0, -T.G))
    m.bodies.append(Body("stand", pos=(0, 0, 0), shape="box", size=(bw, bd, bh), center=(0, 0, bh / 2), mass=bm,
                         support=True, color="0.35 0.37 0.4 1"))
    m.joints.append(Joint("stand_free", "stand", kind="free"))
    m.bodies.append(Body("mast", "stand", (0, 0, bh), fromto=(0, 0, 0, 0, 0, max(ph, uh) - bh), radius=0.01,
                         mass=0.0, collide=False))
    m.masses.append(PointMass("upper", "stand", um, (0.0, 0.0, uh)))
    if pm > 0:
        m.bodies.append(Body("boom", "stand", (0, 0, ph), fromto=(0, 0, 0, reach, 0, 0), radius=0.008, mass=0.0,
                             collide=False))
        m.masses.append(PointMass("payload", "stand", pm, (reach, 0.0, ph)))
    if cw > 0:   # ballast low at the back edge
        m.masses.append(PointMass("counterweight", "stand", cw, (-bw / 2 + 0.01, 0.0, bh / 2)))
    ramp = 0.5
    if force:
        m.loads.append(Load(body="stand", force=(force, 0.0, 0.0), point=(0.0, 0.0, fh), ramp=ramp))
    m.duration = p.get("time", Q.time)
    p.done()
    masses = [(bm, 0.0, bh / 2), (um, 0.0, uh)] + ([(pm, reach, ph)] if pm > 0 else [])
    masses += [(cw, -bw / 2 + 0.01, bh / 2)] if cw > 0 else []
    hand = T.tipping(masses, bw / 2, force, fh, acc)
    return Built(m, "tip", p.used, {"masses": masses, "edge": bw / 2, "force": force, "force_height": fh,
                                    "accel": acc, "friction": mu, "hand": hand, "base": (bw, bd)})


TEMPLATES = {"pendulum": pendulum, "arm": arm, "fourbar": fourbar, "gears": gears, "impact": impact, "tip": tip}
_ALIASES = {"four-bar": "fourbar", "linkage": "fourbar", "wiper": "fourbar", "gear": "gears", "gearbox": "gears",
            "robot-arm": "arm", "collision": "impact", "drop": "impact", "tipping": "tip", "stability": "tip"}


def template(name: str, raw: dict) -> Built:
    key = _ALIASES.get(name.lower(), name.lower())
    if key not in TEMPLATES:
        raise SpecError(f"unknown mechanism '{name}' (use {', '.join(TEMPLATES)}, a spec file or an MJCF or URDF "
                        "model)")
    return TEMPLATES[key](raw)


def is_template(name: str) -> bool:
    return _ALIASES.get(name.lower(), name.lower()) in TEMPLATES


__all__ = ["Built", "template", "is_template", "TEMPLATES", "Motor"]
