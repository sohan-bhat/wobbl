"""Run a Mechanism in MuJoCo and keep what the checks need.

Each step Wobbl works out every drive torque itself and hands MuJoCo the sum as joint forces. A servo follows
the move by computed torque (MuJoCo's inverse dynamics at the current state, with the move's acceleration plus a
stiff correction), clipped to its motor's torque speed line. A motor in voltage mode runs open loop on that
line. Gear meshes lose a share of the torque they carry. Along the way the run adds up the work of every force,
so the energy account can be closed, and records contacts, penetration and constraint drift.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

from .install import require as _mj
from .mjcf import build
from .spec import Mechanism, SpecError

SERVO_HZ = 8.0            # bandwidth of the servo's correction loop
DRAG_SPEED = 0.05         # rad/s over which a drag load builds to its full value


@dataclass
class Run:
    t: np.ndarray
    joints: list[str]
    q: np.ndarray
    v: np.ndarray
    q_des: np.ndarray
    v_des: np.ndarray
    tau: np.ndarray                   # drive torque applied at each joint
    tau_need: np.ndarray              # what the drive asked for before its motor's limit
    kinetic: np.ndarray
    potential: np.ndarray
    work: dict[str, np.ndarray]       # running work of each kind of force since t = 0, J
    contacts: dict[str, dict]
    peak_force: np.ndarray            # largest contact normal force at each sample, N (support contacts left out)
    depth: np.ndarray                 # deepest penetration at each sample, m
    eq_err: np.ndarray                # largest loop or mesh constraint violation at each sample (m or rad)
    momentum: np.ndarray              # total linear momentum, kg m/s
    cop: np.ndarray                   # center of pressure of the support contacts on the floor (x, y), m
    free: dict[str, dict]             # free bodies: tilt (deg), height, position and velocity over time
    points: dict[str, np.ndarray]     # where each point mass (payload, gripper) is over time, m
    frames: list[tuple[float, np.ndarray, np.ndarray]]
    dt: float
    steps: int
    wall_s: float
    model_info: dict = field(default_factory=dict)

    def col(self, joint: str) -> int:
        return self.joints.index(joint)

    def energy_residual(self, floor: float = 0.0) -> tuple[float, float, dict]:
        """Change in kinetic plus potential energy against the work of every force, at the end of the run and
        at its worst moment, relative to the largest energy flow in the account. ``floor`` keeps a run where almost
        nothing moves from judging micro joules of settling against themselves."""
        mech = self.kinetic + self.potential
        dE = mech - mech[0]
        w = sum(self.work.values())
        res = dE - w
        scale = max([float(np.max(np.abs(x))) for x in [dE, self.kinetic, *self.work.values()]] + [floor, 1e-12])
        totals = {k: float(v[-1]) for k, v in self.work.items()}
        return float(abs(res[-1]) / scale), float(np.max(np.abs(res)) / scale), {"scale_J": scale, **totals,
                                                                                  "delta_E_J": float(dE[-1])}


class Trajectory:
    """Minimum jerk between waypoints, joint by joint; a joint left out of a waypoint holds where it was. A
    spinning joint ramps up to its speed on a minimum jerk profile and then turns steadily."""

    def __init__(self, move: list[tuple[float, dict[str, float]]], start: dict[str, float],
                 spin: dict[str, tuple[float, float]] | None = None):
        self.spin = dict(spin or {})
        self.start = start
        self.joints = sorted({k for _, q in move for k in q} | set(self.spin))
        self.knots = {}
        for j in self.joints:
            if j in self.spin:
                continue
            pts, last = [], start.get(j, 0.0)
            for t, q in move:
                last = q.get(j, last)
                pts.append((t, last))
            self.knots[j] = pts

    def at(self, j: str, t: float) -> tuple[float, float, float]:
        if j in self.spin:
            w, ramp = self.spin[j]
            q0 = self.start.get(j, 0.0)
            if ramp <= 0 or t >= ramp:
                return q0 + w * (t - ramp / 2), w, 0.0
            s = max(t, 0.0) / ramp
            return (q0 + w * ramp * (2.5 * s ** 4 - 3 * s ** 5 + s ** 6), w * (10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5),
                    w / ramp * (30 * s ** 2 - 60 * s ** 3 + 30 * s ** 4))
        pts = self.knots[j]
        if t <= pts[0][0]:
            return pts[0][1], 0.0, 0.0
        for (ta, qa), (tb, qb) in zip(pts, pts[1:]):
            if t <= tb:
                T = tb - ta
                s = (t - ta) / T
                dq = qb - qa
                return (qa + dq * (10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5),
                        dq * (30 * s ** 2 - 60 * s ** 3 + 30 * s ** 4) / T,
                        dq * (60 * s - 180 * s ** 2 + 120 * s ** 3) / T ** 2)
        return pts[-1][1], 0.0, 0.0


def _names(model, kind) -> list[str]:
    mujoco = _mj()

    n = {"joint": model.njnt, "body": model.nbody, "geom": model.ngeom}[kind]
    obj = {"joint": mujoco.mjtObj.mjOBJ_JOINT, "body": mujoco.mjtObj.mjOBJ_BODY, "geom": mujoco.mjtObj.mjOBJ_GEOM}
    return [mujoco.mj_id2name(model, obj[kind], i) or f"{kind}{i}" for i in range(n)]


def simulate(m: Mechanism, timestep: float | None = None, end: float | None = None, samples: int = 800,
             frames: int = 9, ideal: bool = False) -> Run:
    """Simulate ``m`` to ``end`` (default its move plus hold, or its duration). ``ideal`` drops every motor limit,
    which shows what a move needs."""
    mujoco = _mj()

    t_wall = time.time()
    model = build(m, timestep)
    dt = float(model.opt.timestep)
    end = float(end if end is not None else m.end_time)
    if end <= 0:
        raise SpecError("nothing to simulate: give a move, a duration or a hold time")
    d = mujoco.MjData(model)
    di = mujoco.MjData(model)
    jnames = _names(model, "joint")
    bnames = _names(model, "body")
    jid = {n: i for i, n in enumerate(jnames)}
    hinge = [i for i in range(model.njnt) if model.jnt_type[i] in (mujoco.mjtJoint.mjJNT_HINGE,
                                                                   mujoco.mjtJoint.mjJNT_SLIDE)]
    rec = [jnames[i] for i in hinge]
    qadr = np.array([model.jnt_qposadr[i] for i in hinge], int)
    dadr = np.array([model.jnt_dofadr[i] for i in hinge], int)
    free = [i for i in range(model.njnt) if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE]

    # starting state: joint start values, then the move's first waypoint
    start = {j.name: j.start for j in m.joints}
    if m.move:
        start.update(m.move[0][1])
    for name, val in start.items():
        if name in jid and jid[name] in hinge:
            d.qpos[model.jnt_qposadr[jid[name]]] = val
    for j in m.joints:
        if j.name in jid and isinstance(j.speed, tuple):   # a free body's starting velocity
            a = model.jnt_dofadr[jid[j.name]]
            d.qvel[a:a + 3] = j.speed
        elif j.speed and j.name in jid:
            d.qvel[model.jnt_dofadr[jid[j.name]]] = j.speed
    traj = Trajectory(m.move, start, m.spin) if (m.move or m.spin) else None
    for name in (traj.joints if traj else []):
        if name not in jid:
            raise SpecError(f"the move names joint '{name}', which the model does not have")

    drives = []
    for dr in m.drives:
        if dr.joint not in jid:
            raise SpecError(f"a drive names joint '{dr.joint}', which the model does not have")
        follows = dr.mode == "servo" and traj is not None and dr.joint in traj.joints
        if dr.mode == "servo" and not follows:
            raise SpecError(f"servo '{dr.joint}' has nothing to follow; add it to the move or use mode = voltage")
        drives.append((dr, int(model.jnt_dofadr[jid[dr.joint]]), None if ideal else dr.at_joint, follows))
    servo_dofs = np.array([dof for dr, dof, _, f in drives if f], int)
    loads = []
    for ld in m.loads:
        if ld.joint:
            if ld.joint not in jid:
                raise SpecError(f"a load names joint '{ld.joint}', which the model does not have")
            loads.append((ld, int(model.jnt_dofadr[jid[ld.joint]]), -1))
        else:
            loads.append((ld, -1, bnames.index(ld.body)))
    meshes = []
    n_loops = len(m.loops)
    for k, c in enumerate(m.couplings):
        meshes.append((n_loops + k, int(model.jnt_dofadr[jid[c.driven]]), c.efficiency))

    if model.neq:
        _assemble(model, d, [int(model.jnt_qposadr[jid[n]]) for n in start if n in jid and jid[n] in hinge])
    mujoco.mj_forward(model, d)

    steps = max(1, int(round(end / dt)))
    every = max(1, steps // samples)
    nrec = steps // every + 1
    nj = len(rec)
    R = {k: np.full((nrec, nj), np.nan) for k in ("q", "v", "qd", "vd", "tau", "need")}
    kin, pot = np.zeros(nrec), np.zeros(nrec)
    work_keys = ("drives", "loads", "gear losses", "damping", "constraints")
    Wrec = {k: np.zeros(nrec) for k in work_keys}
    t_rec = np.zeros(nrec)
    peak, depth, eqerr = np.zeros(nrec), np.zeros(nrec), np.zeros(nrec)
    cop = np.full((nrec, 2), np.nan)
    mom = np.zeros((nrec, 3))
    fr = {jnames[i]: {"body": bnames[model.jnt_bodyid[i]], "tilt": np.zeros(nrec), "z": np.zeros(nrec),
                      "xy": np.zeros((nrec, 2)), "vel": np.zeros((nrec, 3))} for i in free}
    frame_at = set(np.linspace(0, steps, frames).round().astype(int).tolist()) if frames else set()
    frame_list = []
    contacts: dict[str, dict] = {}
    gname = _names(model, "geom")
    tracked = {pm.name: gname.index(pm.name) for pm in m.masses if pm.name in gname}
    pts = {k: np.zeros((nrec, 3)) for k in tracked}
    support_bodies = {b.name for b in m.bodies if b.support}
    f6 = np.zeros(6)
    ftmp = np.zeros(model.nv)
    loss = np.zeros(model.nv)
    f_load, f_drive, v0, vm = (np.zeros(model.nv) for _ in range(4))
    tau_cur, need_cur = np.zeros(model.nv), np.zeros(model.nv)
    kp, kd = (2 * math.pi * SERVO_HZ) ** 2, 2 * (2 * math.pi * SERVO_HZ)
    total_mass = float(model.body_mass.sum())
    # MuJoCo's arrays keep their memory for the whole run, so their views are taken once
    qpos, qvel, qacc, applied = d.qpos, d.qvel, d.qacc, d.qfrc_applied
    passive, spring, constraint, energy = d.qfrc_passive, d.qfrc_spring, d.qfrc_constraint, d.energy
    di_qacc, di_applied = di.qacc, di.qfrc_applied
    plan = [(dr, dof, int(model.jnt_qposadr[jid[dr.joint]]), motor, follows) for dr, dof, motor, follows in drives]
    servo_plan = [x for x in plan if x[4]]
    probe = [x[3].stall_torque if x[3] is not None else 1.0 for x in servo_plan]
    A = np.zeros((len(servo_plan), len(servo_plan)))
    a_cmd = np.zeros(len(servo_plan))
    W_d = W_l = W_g = W_p = W_c = 0.0
    r = 0

    for step in range(steps + 1):
        t = float(d.time)
        f_load[:] = 0
        for ld, dof, body in loads:
            ramp = min(1.0, t / ld.ramp) if ld.ramp > 0 else 1.0
            if dof >= 0:
                f_load[dof] += (ld.torque * ramp if ld.kind == "constant" else
                                -ld.torque * ramp * math.tanh(qvel[dof] / DRAG_SPEED))
            else:
                ftmp[:] = 0
                p = d.xpos[body] + d.xmat[body].reshape(3, 3) @ np.asarray(ld.point)
                mujoco.mj_applyFT(model, d, np.asarray(ld.force) * ramp, np.zeros(3), p, body, ftmp)
                f_load += ftmp
        f_drive[:] = 0
        des = {j: traj.at(j, t) for j in traj.joints} if traj is not None else {}
        for dr, dof, qa, motor, follows in plan:     # motors running open loop on their line
            if follows:
                continue
            w = float(qvel[dof])
            need = motor.stall_torque * dr.voltage if motor is not None else 0.0
            if motor is not None and math.isfinite(motor.no_load_speed):
                need -= motor.stall_torque * w / motor.no_load_speed
            tau = min(max(need, motor.limit(w)[0]), motor.limit(w)[1]) if motor is not None else need
            f_drive[dof] += tau
            tau_cur[dof], need_cur[dof] = tau, need
        if len(servo_dofs):
            # computed torque that also holds for closed loops and gear meshes: the acceleration of the servo
            # joints is linear in their torques, so probe each torque once and solve for the commanded motion
            np.add(f_load, loss, out=di_applied)
            di_applied += f_drive
            di.qpos[:] = qpos
            di.qvel[:] = qvel
            mujoco.mj_forward(model, di)
            a0 = di_qacc[servo_dofs].copy()
            for k, dof in enumerate(servo_dofs):
                di_applied[dof] += probe[k]
                mujoco.mj_forwardSkip(model, di, mujoco.mjtStage.mjSTAGE_VEL, 1)
                A[:, k] = (di_qacc[servo_dofs] - a0) / probe[k]
                di_applied[dof] -= probe[k]
            for k, (dr, dof, qa, motor, follows) in enumerate(servo_plan):
                qd, vd, ad = des[dr.joint]
                a_cmd[k] = ad + kp * (qd - qpos[qa]) + kd * (vd - qvel[dof]) - a0[k]
            needs = a_cmd / A[0, 0] if len(a_cmd) == 1 else np.linalg.solve(A, a_cmd)
            for k, (dr, dof, qa, motor, follows) in enumerate(servo_plan):
                need = float(needs[k])
                tau = min(max(need, motor.limit(qvel[dof])[0]), motor.limit(qvel[dof])[1]) if motor else need
                f_drive[dof] += tau
                tau_cur[dof], need_cur[dof] = tau, need
        np.add(f_drive, f_load, out=applied)
        applied += loss
        v0[:] = qvel
        sample = step % every == 0 and r < nrec
        if sample:
            q_k, v_k = qpos[qadr].copy(), qvel[dadr].copy()
        if step == steps:   # the final state: forces and energy without stepping past the end
            mujoco.mj_forward(model, d)
        else:
            mujoco.mj_step(model, d)
        if sample:
            t_rec[r] = t
            R["q"][r], R["v"][r] = q_k, v_k
            R["tau"][r], R["need"][r] = tau_cur[dadr], need_cur[dadr]
            for c, name in enumerate(rec):
                if name in des:
                    R["qd"][r, c], R["vd"][r, c] = des[name][0], des[name][1]
            pot[r], kin[r] = energy[0], energy[1]
            for k, v in zip(work_keys, (W_d, W_l, W_g, W_p, W_c)):
                Wrec[k][r] = v
            mujoco.mj_subtreeVel(model, d)
            mom[r] = d.subtree_linvel[0] * total_mass
            for i in free:
                b = model.jnt_bodyid[i]
                z = d.xmat[b].reshape(3, 3)[:, 2]
                e = fr[jnames[i]]
                e["tilt"][r] = math.degrees(math.acos(max(-1.0, min(1.0, float(z[2])))))
                e["z"][r], e["xy"][r] = d.xipos[b][2], d.xipos[b][:2]
                e["vel"][r] = v0[model.jnt_dofadr[i]:model.jnt_dofadr[i] + 3]
            for k, g in tracked.items():
                pts[k][r] = d.geom_xpos[g]
        if step in frame_at:
            frame_list.append((t, d.geom_xpos.copy(), d.geom_xmat.copy()))
        # contacts and constraints belong to the state at the start of this step
        pk = dp = fsum = 0.0
        fxy = [0.0, 0.0]
        for i in range(d.ncon):
            c = d.contact[i]
            b1, b2 = bnames[model.geom_bodyid[c.geom1]], bnames[model.geom_bodyid[c.geom2]]
            mujoco.mj_contactForce(model, d, i, f6)
            fn, pen = float(f6[0]), float(max(0.0, -c.dist))
            key = " / ".join(sorted((gname[c.geom1], gname[c.geom2])))
            sup = (b1 == "world" and b2 in support_bodies) or (b2 == "world" and b1 in support_bodies)
            e = contacts.setdefault(key, {"bodies": sorted((b1, b2)), "first_s": t, "last_s": t, "peak_N": 0.0,
                                          "peak_s": t, "depth_m": 0.0, "support": sup})
            e["last_s"] = t
            if fn > e["peak_N"]:
                e["peak_N"], e["peak_s"] = fn, t
            e["depth_m"] = max(e["depth_m"], pen)
            dp = max(dp, pen)
            if not sup:
                pk = max(pk, fn)
            else:
                fsum += fn
                fxy[0] += fn * c.pos[0]
                fxy[1] += fn * c.pos[1]
        if sample:
            peak[r], depth[r] = pk, dp
            if fsum > 0:
                cop[r] = fxy[0] / fsum, fxy[1] / fsum
            if model.neq:
                eqm = d.efc_type[:d.nefc] == 0
                eqerr[r] = float(np.max(np.abs(d.efc_pos[:d.nefc][eqm]))) if eqm.any() else 0.0
            r += 1
        if step == steps:
            break
        # gear mesh losses for the next step, from the torque each mesh carried in this one
        loss[:] = 0
        if meshes:
            for eq_id, dof, eff in meshes:
                rows = np.nonzero((d.efc_type[:d.nefc] == 0) & (d.efc_id[:d.nefc] == eq_id))[0]
                if len(rows):
                    carried = abs(float(d.efc_force[rows[0]]))
                    loss[dof] -= (1 - eff) * carried * math.tanh(d.qvel[dof] / DRAG_SPEED)
        # work over the step: forces held through it, against the mean of the speeds before and after
        np.add(v0, qvel, out=vm)
        vm *= 0.5 * dt
        W_d += float(f_drive @ vm)
        W_l += float(f_load @ vm)
        W_g += float((applied - f_drive - f_load) @ vm) if meshes else 0.0
        W_p += float(passive @ vm) - float(spring @ vm)
        W_c += float(constraint @ vm)

    n = r
    # a blown up step (MuJoCo resets the state and warns) or a full constraint buffer make the run worthless
    warned = {k.replace("mjWARN_", "").lower(): int(d.warning[int(w)].number)
              for k, w in mujoco.mjtWarning.__members__.items()
              if k.startswith("mjWARN_") and int(w) < len(d.warning) and d.warning[int(w)].number}
    return Run(t_rec[:n], rec, R["q"][:n], R["v"][:n], R["qd"][:n], R["vd"][:n], R["tau"][:n], R["need"][:n],
               kin[:n], pot[:n], {k: v[:n] for k, v in Wrec.items()}, contacts, peak[:n], depth[:n], eqerr[:n],
               mom[:n], cop[:n],
               {k: {**v, **{f: v[f][:n] for f in ("tilt", "z", "xy", "vel")}} for k, v in fr.items()},
               {k: v[:n] for k, v in pts.items()},
               frame_list, dt, steps, time.time() - t_wall,
               {"nbody": model.nbody, "njnt": model.njnt, "nv": model.nv, "neq": model.neq, "warnings": warned,
                "mass_kg": total_mass,
                "geoms": _geom_table(model, gname)})


def _assemble(model, d, held: list[int], tol: float = 1e-9, max_steps: int = 20000):
    """Pull a closed loop together before the run: MuJoCo has no assembly solver, so let its constraint forces
    close the loop with gravity off, the driven joints held and the velocity cleared every step."""
    mujoco = _mj()

    g = model.opt.gravity.copy()
    model.opt.gravity[:] = 0
    hold = {a: float(d.qpos[a]) for a in held}
    try:
        for _ in range(max_steps):
            mujoco.mj_step(model, d)
            for a, v in hold.items():
                d.qpos[a] = v
            d.qvel[:] = 0
            eqm = d.efc_type[:d.nefc] == 0
            if not eqm.any() or float(np.max(np.abs(d.efc_pos[:d.nefc][eqm]))) < tol:
                break
    finally:
        model.opt.gravity[:] = g
    d.time = 0.0
    d.qvel[:] = 0
    d.qacc[:] = 0


def _geom_table(model, gname) -> list[dict]:
    """Shape and size of every geom, so a run's frames can be drawn without MuJoCo's renderer."""
    kinds = {0: "plane", 2: "sphere", 3: "capsule", 5: "cylinder", 6: "box"}
    out = []
    for i in range(model.ngeom):
        out.append({"name": gname[i], "type": kinds.get(int(model.geom_type[i]), "other"),
                    "size": model.geom_size[i].tolist(), "rgba": model.geom_rgba[i].tolist(),
                    "body": int(model.geom_bodyid[i])})
    return out


def inverse_along(m: Mechanism, times: np.ndarray, timestep: float | None = None):
    """Joint torques an open chain needs to follow its move exactly, from MuJoCo's inverse dynamics on the
    move's positions, speeds and accelerations, with the center of mass of the whole model at each time. Every
    joint must be in the move or stay still."""
    mujoco = _mj()

    model = build(m, timestep)
    d = mujoco.MjData(model)
    jnames = _names(model, "joint")
    start = {j.name: j.start for j in m.joints}
    if m.move:
        start.update(m.move[0][1])
    traj = Trajectory(m.move, start, m.spin)
    out = np.zeros((len(times), len(traj.joints)))
    com = np.zeros((len(times), 3))
    loads = [(ld, int(model.jnt_dofadr[jnames.index(ld.joint)])) for ld in m.loads if ld.joint]
    for k, t in enumerate(times):
        d.qpos[:] = 0
        d.qvel[:] = 0
        d.qacc[:] = 0
        for name, val in start.items():
            if name in jnames:
                d.qpos[model.jnt_qposadr[jnames.index(name)]] = val
        for name in traj.joints:
            i = jnames.index(name)
            q, v, a = traj.at(name, float(t))
            d.qpos[model.jnt_qposadr[i]], d.qvel[model.jnt_dofadr[i]], d.qacc[model.jnt_dofadr[i]] = q, v, a
        mujoco.mj_inverse(model, d)
        f = d.qfrc_inverse.copy()
        com[k] = d.subtree_com[0]
        for ld, dof in loads:
            v = d.qvel[dof]
            f[dof] -= ld.torque if ld.kind == "constant" else -ld.torque * math.tanh(v / DRAG_SPEED)
        for c, name in enumerate(traj.joints):
            out[k, c] = f[model.jnt_dofadr[jnames.index(name)]]
    return traj.joints, out, com
