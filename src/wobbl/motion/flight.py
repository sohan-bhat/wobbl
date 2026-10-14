"""Fly a body as a point mass in a vertical plane under gravity, drag, wind, thrust and parachutes.

The state is the distance x along the launch direction, the altitude z above the ground, their speeds, the
mass and, for a rocket, the angle its nose points at. Next to it the integrator carries seven running sums
the checks read: the work done by thrust, drag and gravity, the kinetic energy carried off by burned
propellant, and the speed lost to steering, drag and gravity. The flight is integrated piecewise with
scipy's solve_ivp (DOP853). Each phase (on the rail, under thrust, coasting, under the drogue, under the
main, falling) runs until one of its events (rail exit, apogee, the main's deployment altitude, the ground)
or a breakpoint where a force has a kink (a point of the thrust curve, burnout, a parachute finishing its
opening), so every stretch it sees is smooth.

Off the rail a stable rocket turns into the wind it feels, its air relative velocity, and the thrust turns
with it. It turns at about its weathercock frequency, sqrt(CN q A (xcp - xcg) / I), from Barrowman's normal
force slope and margin and the pitch inertia of a rod of its length and mass. A rocket without fins in its
spec, or any unpowered body, points along its air relative velocity at once. Wind blows back toward the start
along the launch direction, so a rail tilted into the wind leans toward positive x and the descent drifts
toward negative x.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np
from scipy.integrate import solve_ivp

from .atmosphere import G0, R_AIR, air, gravity
from .spec import MotionStudy, SpecError, barrowman

X, Z, VX, VZ, M, TH = range(6)
W_THRUST, W_DRAG, W_GRAV, E_PROP, L_STEER, L_DRAG, L_GRAV = range(6, 13)
N = 13
RTOL = 1e-6
ATOL = np.array([1e-4, 1e-4, 1e-5, 1e-5, 1e-7, 1e-7, 1e-2, 1e-2, 1e-2, 1e-2, 1e-6, 1e-6, 1e-6])
FILL_DIAMETERS = 8.0       # a canopy opens over about eight of its diameters of travel
# Under a parachute the sink rate settles within a second or two, and steps much longer than that let DOP853's
# error estimate miss a small lag behind the terminal speed as the air thickens, so they are capped.
CHUTE_STEP = 1.0
MAX_TIME = 7200.0


@dataclass(frozen=True)
class Phase:
    name: str                       # rail, boost, coast, drogue, main, fall
    held: bool = False              # on the pad before thrust passes the weight
    rail: bool = False
    body_drag: bool = True          # off once a parachute is out
    burning: bool = False           # the motor is on the curve, from ignition to burnout
    chutes: tuple = ()              # (name, drag area m2, deploy time s, fill time s) of each open parachute


@dataclass
class Segment:
    phase: Phase
    sol: object                     # scipy OdeSolution, dense over [t0, t1]
    t0: float
    t1: float


@dataclass
class Event:
    name: str
    t: float
    y: np.ndarray
    phase: Phase


class Model:
    """Forces on the body at a time and state, for one study."""

    def __init__(self, st: MotionStudy):
        self.st = st
        self.cd = st.drag_coefficient()
        a = math.radians(st.angle)
        self.u = (math.cos(a), math.sin(a))          # launch direction, along the rail
        self.motor = st.motor
        self.g_ground = gravity(st.site_altitude)
        self.cn = self.cp = None
        af = st.airframe
        if self.motor and af is not None and af.cg is not None:
            bm = barrowman(af, st.diameter)
            self.cn, self.cp = bm["cn_nose"] + bm["cn_fins"], bm["cp"]
            self.x_prop = af.length - self.motor.length / 2 if self.motor.length > 0 else af.cg

    def air(self, z: float) -> tuple[float, float]:
        """Density and speed of sound at z above the ground."""
        if self.st.air == "standard":
            rho, _, _, a = air(self.st.site_altitude + z)
            return rho, a
        rho = 0.0 if self.st.air == "vacuum" else float(self.st.air)
        return rho, math.sqrt(1.4 * R_AIR * 288.15)

    def g(self, z: float) -> float:
        return self.g_ground if self.st.uniform_gravity else gravity(self.st.site_altitude + z)

    def thrust(self, t: float, ph: Phase) -> float:
        return self.motor.thrust(t) if self.motor and ph.burning else 0.0

    def drag_area(self, t: float, mach: float, ph: Phase) -> float:
        cda = self.cd(mach) * self.st.area if ph.body_drag else 0.0
        for _, area, t_open, fill in ph.chutes:
            cda += area * min(1.0, max(t - t_open, 0.0) / fill) ** 2
        return cda

    def turn_rate(self, m: float, q: float) -> float:
        """Weathercock frequency, rad/s: restoring moment slope over pitch inertia."""
        burned = self.st.liftoff_mass - m
        cg = (self.st.liftoff_mass * self.st.airframe.cg - burned * self.x_prop) / m
        lever = self.cp - cg
        inertia = m * self.st.airframe.length ** 2 / 12.0
        return math.sqrt(max(self.cn * q * self.st.area * lever / inertia, 0.0))

    def forces(self, t: float, y, ph: Phase) -> dict:
        """Acceleration, turn rate and everything the running sums and the trajectory table need."""
        z, vx, vz, m = y[Z], y[VX], y[VZ], y[M]
        rho, sound = self.air(z)
        g = self.g(z)
        T = self.thrust(t, ph)
        wx, wz = vx + self.st.wind, vz              # velocity through the air
        vair = math.hypot(wx, wz)
        mach = vair / sound
        q = 0.5 * rho * vair * vair
        k = 0.5 * rho * self.drag_area(t, mach, ph) * vair      # drag force is -k times the air velocity
        speed = math.hypot(vx, vz)
        turn = 0.0
        if ph.rail or ph.held:
            dx, dz = self.u
        elif self.cn is not None:                   # turns into the air relative velocity at its own rate
            dx, dz = math.cos(y[TH]), math.sin(y[TH])
            if T > 0.0 and vair > 1e-9:
                off = math.atan2(wz, wx) - y[TH]
                turn = self.turn_rate(m, q) * math.atan2(math.sin(off), math.cos(off))
        elif speed < 1e-9:
            dx, dz = self.u
        elif self.st.air == "vacuum" or vair < 1e-9:
            dx, dz = vx / speed, vz / speed
        else:
            dx, dz = wx / vair, wz / vair           # weathercocked at once
        fx, fz = T * dx - k * wx, T * dz - k * wz - m * g
        if ph.held:
            ax = az = 0.0
        elif ph.rail:
            along = (fx * self.u[0] + fz * self.u[1]) / m
            if along < 0.0 and speed < 1e-6:        # sitting on the pad
                along = 0.0
            ax, az = along * self.u[0], along * self.u[1]
        else:
            ax, az = fx / m, fz / m
        if ph.rail or ph.held or speed < 1e-9:
            ux, uz = self.u
        else:
            ux, uz = vx / speed, vz / speed          # unit vector along the flight path
        return {"ax": ax, "az": az, "T": T, "dir": (dx, dz), "k": k, "air": (wx, wz), "g": g, "mach": mach,
                "q": q, "rho": rho, "path": (ux, uz), "speed": speed, "turn": turn}

    def rhs(self, t: float, y, ph: Phase):
        f = self.forces(t, y, ph)
        m, T = y[M], f["T"]
        mdot = T / self.motor.exhaust_speed if self.motor else 0.0
        vx, vz = y[VX], y[VZ]
        ux, uz = f["path"]
        dx, dz = f["dir"]
        wx, wz = f["air"]
        dydt = np.zeros(N)
        if ph.held:
            dydt[M] = -mdot
            dydt[L_GRAV] = T / m                    # thrust spent holding the rocket on the pad
            return dydt
        dydt[X], dydt[Z], dydt[VX], dydt[VZ], dydt[M], dydt[TH] = vx, vz, f["ax"], f["az"], -mdot, f["turn"]
        dydt[W_THRUST] = T * (dx * vx + dz * vz)
        dydt[W_DRAG] = -f["k"] * (wx * vx + wz * vz)
        dydt[W_GRAV] = -m * f["g"] * vz
        dydt[E_PROP] = 0.5 * mdot * (vx * vx + vz * vz)
        dydt[L_STEER] = T / m * (1.0 - (dx * ux + dz * uz))
        dydt[L_DRAG] = f["k"] * (wx * ux + wz * uz) / m
        dydt[L_GRAV] = f["g"] * uz
        return dydt

    def liftoff_time(self) -> float:
        """When the thrust first beats the weight along the rail and the wind pushing down it, s."""
        st, mo = self.st, self.motor
        rho, _ = self.air(0.0)
        push = 0.5 * rho * self.cd(0.0) * st.area * st.wind * abs(st.wind) * self.u[0]

        def excess(t):
            return mo.thrust(t) - (st.liftoff_mass - mo.burned(t)) * self.g(0.0) * self.u[1] - push

        ts = mo.times
        if excess(0.0) > 0:
            return 0.0
        for a, b in zip(ts, ts[1:]):
            if excess(b) > 0:
                from scipy.optimize import brentq
                return float(brentq(excess, a, b, xtol=1e-12))
        peak = float(mo.forces.max())
        raise SpecError(f"the motor never lifts the rocket, because its peak thrust of {peak:.4g} N is under the "
                        f"{st.liftoff_mass * G0:.4g} N the rocket weighs", "Use a bigger motor or a lighter rocket.")


def _event(fn, name: str, terminal: bool, direction: float):
    fn.terminal, fn.direction, fn.label = terminal, direction, name
    return fn


@dataclass
class Flight:
    study: MotionStudy
    model: Model
    rtol: float
    segments: list[Segment] = field(default_factory=list)
    events: list[Event] = field(default_factory=list)

    def event(self, name: str) -> Event | None:
        return next((e for e in self.events if e.name == name), None)

    @property
    def end(self) -> Event:
        return self.events[-1]

    def samples(self, per_segment: int = 60) -> list[tuple[float, np.ndarray, Phase]]:
        """Points along the whole flight: every integrator step, an even spread over each segment and every
        event. A point where one phase ends and the next begins belongs to the phase that ends there."""
        out = []
        marks = {e.t for e in self.events}
        for k, s in enumerate(self.segments):
            ts = np.union1d(np.asarray(s.sol.ts), np.linspace(s.t0, s.t1, per_segment))
            ts = np.union1d(ts, [t for t in marks if s.t0 < t < s.t1])
            ts = ts[(ts >= s.t0) & (ts <= s.t1)]
            if k and len(out) and abs(ts[0] - out[-1][0]) < 1e-12:
                ts = ts[1:]
            ys = s.sol(ts)
            out += [(float(t), ys[:, i], s.phase) for i, t in enumerate(ts)]
        return out


def fly(st: MotionStudy, rtol: float = RTOL) -> Flight:
    """Integrate the flight from release or ignition to the ground."""
    st.validate()
    mdl = Model(st)
    fl = Flight(st, mdl, rtol)
    a = math.radians(st.angle)
    y = np.zeros(N)
    y[Z], y[VX], y[VZ], y[M], y[TH] = st.height, st.speed * math.cos(a), st.speed * math.sin(a), st.liftoff_mass, a
    atol = ATOL * rtol / RTOL
    t = 0.0
    mo = st.motor

    def run(ph: Phase, t0: float, t1: float, y0, events=()):
        sol = solve_ivp(mdl.rhs, (t0, t1), y0, method="DOP853", rtol=rtol, atol=atol, events=list(events) or None,
                        dense_output=True, args=(ph,), max_step=CHUTE_STEP if ph.chutes else np.inf)
        if sol.status < 0:
            raise RuntimeError(f"the integrator stopped in the {ph.name} phase: {sol.message}")
        fl.segments.append(Segment(ph, sol.sol, t0, float(sol.t[-1])))
        for ev, te, ye in zip(events, sol.t_events or [], sol.y_events or []):
            if not ev.terminal:
                fl.events += [Event(ev.label, float(tt), yy, ph) for tt, yy in zip(te, ye)]
        fired = next((ev.label for ev, te in zip(events, sol.t_events or []) if ev.terminal and len(te)), None)
        return float(sol.t[-1]), sol.y[:, -1].copy(), fired

    if mo:
        t_lift = mdl.liftoff_time()
        held = Phase("rail", held=True, burning=True)
        for b in [b for b in mo.times if 0 < b < t_lift] + ([t_lift] if t_lift > 0 else []):
            t, y, _ = run(held, t, b, y)
        fl.events.append(Event("liftoff", t, y.copy(), held))
        phase = Phase("rail", rail=True, burning=True)
    else:
        fl.events.append(Event("release", 0.0, y.copy(), Phase("fall")))
        phase = Phase("fall")
        now = [c for c in st.chutes if c.altitude is None]
        if y[VZ] <= 0 and now:
            phase = _open(mdl, phase, now, t, y)
            fl.events += [Event(f"{c.name} deploy", t, y.copy(), phase) for c in now]
    rising = y[VZ] > 0

    rail_exit = _event(lambda t, y, ph: (y[X] * mdl.u[0] + (y[Z] - st.height) * mdl.u[1]) - st.rail_length,
                       "rail exit", True, 1)
    ground = _event(lambda t, y, ph: y[Z], "ground", True, -1)
    while True:
        stops = [b for b in (mo.times if mo and phase.burning else []) if b > t + 1e-12]
        stops += [t_open + fill for _, _, t_open, fill in phase.chutes if t_open + fill > t + 1e-12]
        t_stop = min(stops, default=t + MAX_TIME)
        if phase.rail:
            events = [rail_exit]
        else:
            events = [ground]
            if phase.name in ("boost", "coast"):
                events.append(_event(lambda t, y, ph: y[VZ], "apogee", True, -1))
            elif not mo and rising and not fl.event("apogee"):
                events.append(_event(lambda t, y, ph: y[VZ], "apogee", False, -1))
            if phase.name not in ("boost", "coast"):
                for c in st.chutes:
                    if c.altitude is not None and c.name not in _open_names(phase):
                        events.append(_event(lambda t, y, ph, h=c.altitude: y[Z] - h, f"{c.name} deploy", True,
                                             -1))
        t, y, fired = run(phase, t, t_stop, y, events)
        if fired is None:
            if t >= MAX_TIME:
                raise RuntimeError(f"the body has not landed after {MAX_TIME:.0f} s")
            if mo and phase.burning and abs(t - mo.burn_time) < 1e-12:
                if phase.rail:
                    raise SpecError("the motor burns out before the rocket leaves the rail",
                                    "Use a shorter rail or a motor that burns longer.")
                fl.events.append(Event("burnout", t, y.copy(), phase))
                phase = Phase("coast") if phase.name == "boost" else replace(phase, burning=False)
            continue
        if fired == "ground":
            fl.events.append(Event("landing" if phase.chutes else "impact", t, y.copy(), phase))
            return fl
        fl.events.append(Event(fired, t, y.copy(), phase))
        if fired == "rail exit":
            phase = Phase("boost", burning=True) if mo and t < mo.burn_time else Phase("coast")
        elif fired == "apogee":
            now = [c for c in st.chutes if c.altitude is None or y[Z] <= c.altitude]
            fall = Phase("fall", burning=phase.burning)
            phase = _open(mdl, fall, now, t, y) if now else fall
            fl.events += [Event(f"{c.name} deploy", t, y.copy(), phase) for c in now]
        else:
            c = next(c for c in st.chutes if f"{c.name} deploy" == fired)
            phase = _open(mdl, phase, [c], t, y)


def _open_names(ph: Phase) -> set[str]:
    return {name for name, _, _, _ in ph.chutes}


def _open(mdl: Model, ph: Phase, chutes, t: float, y) -> Phase:
    """The phase after these parachutes come out. Each opens over the time it takes to travel FILL_DIAMETERS of
    its diameters through the air, its drag area growing with the square of time, between 0.1 and 3 s."""
    vair = math.hypot(y[VX] + mdl.st.wind, y[VZ])
    opened = tuple((c.name, c.drag_area, t, min(3.0, max(0.1, FILL_DIAMETERS * c.diameter / max(vair, 1e-9))))
                   for c in chutes)
    names = {c.name for c in chutes} | _open_names(ph)
    return Phase("main" if "main" in names else "drogue", body_drag=False, chutes=ph.chutes + opened,
                 burning=ph.burning)
