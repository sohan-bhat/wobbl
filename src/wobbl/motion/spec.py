"""What flies and where: a body or a rocket, its motor and parachutes, the launch and the air.

A falling rock or a thrown ball is a mass and a diameter with a drag coefficient. A rocket adds a motor with a
thrust curve, a nose, fins and a balance point for Barrowman's stability margin, a launch rail, and a drogue
and a main parachute. Everything here is stored in SI (m, kg, s, N).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

SPHERE_CD = 0.47           # a sphere below the drag crisis
LANDING_LIMIT = 6.1        # m/s, about 20 ft/s, the usual ceiling for landing under a main
RAIL_EXIT_MIN = 15.0       # m/s, about 50 ft/s, the usual recommendation off the rail
RAIL_EXIT_FLOOR = 9.0      # m/s, about 30 ft/s, below which fins cannot steer at all
STABILITY_MIN = 1.0        # calibers
STABILITY_FIX = 1.5        # calibers a stability fix aims for

# A slender finned rocket's drag rises through Mach 1 as shock waves form on the nose and fins. A scalar drag
# coefficient is the subsonic value, and these factors raise it through the transonic range, peaking at 1.7
# times near Mach 1.1 and easing off as the flow goes fully supersonic. A table against Mach replaces them.
TRANSONIC = [(0.0, 1.0), (0.6, 1.0), (0.8, 1.03), (0.9, 1.12), (0.95, 1.28), (1.0, 1.52), (1.05, 1.66),
             (1.1, 1.70), (1.2, 1.64), (1.5, 1.42), (2.0, 1.20), (3.0, 0.98)]
NOSE_SHAPES = ("cone", "ogive", "elliptical", "parabolic", "haack")


class SpecError(ValueError):
    """A spec that cannot be flown. The message says which key, and the hint how to fix it."""

    def __init__(self, msg: str, hint: str = ""):
        super().__init__(msg)
        self.hint = hint


@dataclass
class Motor:
    name: str
    curve: list[tuple[float, float]]   # (s, N), straight lines between points
    propellant: float                  # kg
    loaded_mass: float                 # kg, case and propellant
    length: float = 0.0                # m, for the balance point at burnout

    def __post_init__(self):
        pts = [(float(t), float(f)) for t, f in self.curve]
        if not pts:
            raise SpecError(f"motor {self.name} has no thrust curve")
        if pts[0][0] > 0:              # RASP curves start after ignition and ramp from zero
            pts.insert(0, (0.0, 0.0))
        if any(b[0] <= a[0] for a, b in zip(pts, pts[1:])) or any(f < 0 for _, f in pts):
            raise SpecError(f"the thrust curve of motor {self.name} needs rising times and no negative thrust")
        self.curve = pts
        if not 0 < self.propellant < self.loaded_mass:
            raise SpecError(f"motor {self.name} needs a propellant mass above zero and under its loaded mass",
                            "Give loaded_mass (case and propellant) and propellant, for example 3.4 kg and 1.9 kg.")
        self.times = np.array([t for t, _ in pts])
        self.forces = np.array([f for _, f in pts])
        self.cumulative = np.concatenate([[0.0], np.cumsum(np.diff(self.times) * 0.5 *
                                                           (self.forces[1:] + self.forces[:-1]))])

    @property
    def impulse(self) -> float:
        """Total impulse, N s, exact for a curve of straight lines."""
        return float(self.cumulative[-1])

    @property
    def burn_time(self) -> float:
        return float(self.times[-1])

    @property
    def exhaust_speed(self) -> float:
        """Effective exhaust speed, m/s. Propellant burns in proportion to thrust, so all of it is gone at the
        end of the curve."""
        return self.impulse / self.propellant

    def thrust(self, t: float) -> float:
        """Thrust on the curve, N, including its end points, so a stretch that ends at burnout sees the curve's
        last value there. Phases after burnout do not ask."""
        if t < 0.0 or t > self.times[-1]:
            return 0.0
        return float(np.interp(t, self.times, self.forces))

    def burned(self, t: float) -> float:
        """Propellant burned by time t, kg."""
        if t <= 0.0:
            return 0.0
        if t >= self.times[-1]:
            return self.propellant
        k = int(np.searchsorted(self.times, t, side="right")) - 1
        dt = t - self.times[k]
        return self.propellant * (self.cumulative[k] + 0.5 * dt * (self.forces[k] + self.thrust(t))) / self.impulse


@dataclass
class Chute:
    name: str                       # drogue or main
    diameter: float                 # m
    cd: float
    altitude: float | None = None   # m above the ground on the way down, None to open at apogee

    @property
    def drag_area(self) -> float:
        return self.cd * math.pi * self.diameter ** 2 / 4.0


@dataclass
class Fins:
    count: int
    root: float
    tip: float
    span: float
    sweep: float                    # how far the tip's leading edge sits behind the root's, m
    position: float                 # root leading edge from the nose tip, m

    @property
    def area(self) -> float:
        return self.count * 0.5 * (self.root + self.tip) * self.span


@dataclass
class Airframe:
    length: float                   # nose tip to aft end, m
    nose_length: float
    nose_shape: str
    fins: Fins
    cg: float | None = None         # balance point from the nose tip with the motor loaded, m


@dataclass
class MotionStudy:
    name: str = "body"
    mass: float = 0.0                   # kg, everything but the motor
    diameter: float = 0.0               # m, sets the frontal area for drag and the caliber for stability
    cd: float | list = SPHERE_CD        # a number, or [[mach, cd], ...]
    transonic: bool = False             # raise a numeric cd through Mach 1, for rockets
    height: float = 0.0                 # m above the ground at the start
    speed: float = 0.0                  # m/s at the start
    angle: float = 90.0                 # deg above horizontal, the launch direction (along the rail for rockets)
    wind: float = 0.0                   # m/s, blowing back toward the start along the launch direction
    site_altitude: float = 0.0          # m above sea level
    rail_length: float = 0.0            # m of travel on the rail
    motor: Motor | None = None
    chutes: list[Chute] = field(default_factory=list)
    airframe: Airframe | None = None
    target_apogee: float | None = None  # m above the ground
    landing_limit: float = LANDING_LIMIT
    rail_exit_min: float = RAIL_EXIT_MIN
    stability_min: float = STABILITY_MIN
    air: str | float = "standard"       # standard, vacuum, or a constant density in kg/m3
    uniform_gravity: bool = False       # gravity held at its ground value
    question: str = ""
    source: str = ""                    # the spec file, empty for flags
    notes: list[str] = field(default_factory=list)

    @property
    def area(self) -> float:
        return math.pi * self.diameter ** 2 / 4.0

    @property
    def rocket(self) -> bool:
        return self.motor is not None

    @property
    def liftoff_mass(self) -> float:
        return self.mass + (self.motor.loaded_mass if self.motor else 0.0)

    @property
    def burnout_mass(self) -> float:
        return self.liftoff_mass - (self.motor.propellant if self.motor else 0.0)

    def chute(self, name: str) -> Chute | None:
        return next((c for c in self.chutes if c.name == name), None)

    def drag_coefficient(self):
        """A smooth function of Mach: the table through a monotone cubic, a numeric cd raised by TRANSONIC for a
        rocket, or the number itself. Beyond the table's ends it holds the end values."""
        if isinstance(self.cd, (list, tuple)):
            return _monotone_cubic([(float(m), float(c)) for m, c in self.cd])
        cd = float(self.cd)
        if not self.transonic:
            return lambda mach: cd
        return _monotone_cubic([(m, f * cd) for m, f in TRANSONIC])

    def validate(self):
        if self.mass <= 0:
            raise SpecError("the body needs a positive mass", "Give a mass like '1 kg', or a density with a "
                                                              "diameter for a round body.")
        if self.diameter <= 0:
            raise SpecError("the body needs a diameter for its drag", "Give the diameter that meets the air, "
                                                                     "for example '9 cm'.")
        if self.rocket and self.rail_length <= 0:
            raise SpecError("a rocket needs a launch rail", "Set rail_length under [launch], for example "
                                                            "'2.44 m', so the rocket points straight until its "
                                                            "fins can steer it.")
        if self.height < 0 or not 0 <= self.angle <= 90:
            raise SpecError("the start height cannot be negative and the angle runs from 0 (level) to 90 deg "
                            "(straight up)")


def _monotone_cubic(pts: list[tuple[float, float]]):
    """A PCHIP curve through the points, evaluated in plain Python. The integrator asks for the drag coefficient
    at every step, where scipy's interpolator call costs more than the whole rest of the force model."""
    from bisect import bisect_right

    from scipy.interpolate import PchipInterpolator

    if len(pts) == 1:
        return lambda mach: pts[0][1]
    xs = [p[0] for p in pts]
    pp = PchipInterpolator(xs, [p[1] for p in pts])
    coef = pp.c.T.tolist()          # one row of cubic coefficients per interval, highest power first
    lo, hi = pts[0], pts[-1]

    def cd(mach: float) -> float:
        if mach <= lo[0]:
            return lo[1]
        if mach >= hi[0]:
            return hi[1]
        i = bisect_right(xs, mach) - 1
        a, b, c, d = coef[i]
        h = mach - xs[i]
        return ((a * h + b) * h + c) * h + d

    return cd


# --- Barrowman stability -------------------------------------------------------------
def nose_radius(shape: str, x: np.ndarray, length: float, radius: float) -> np.ndarray:
    """Radius of the nose at distance x from its tip."""
    s = np.clip(x / length, 0.0, 1.0)
    if shape == "cone":
        return radius * s
    if shape == "ogive":            # tangent ogive
        rho = (radius ** 2 + length ** 2) / (2 * radius)
        return np.sqrt(np.maximum(rho ** 2 - (length - x) ** 2, 0.0)) + radius - rho
    if shape == "elliptical":
        return radius * np.sqrt(np.maximum(1 - (1 - s) ** 2, 0.0))
    if shape == "parabolic":
        return radius * (2 * s - s ** 2)
    if shape == "haack":            # von Karman, the Haack series with C = 0
        th = np.arccos(1 - 2 * s)
        return radius / math.sqrt(math.pi) * np.sqrt(th - np.sin(2 * th) / 2)
    raise SpecError(f"unknown nose shape '{shape}'", f"Use one of {', '.join(NOSE_SHAPES)}.")


def barrowman(af: Airframe, diameter: float, fins: Fins | None = None) -> dict:
    """Normal force slopes and centers of pressure of the nose and fins by Barrowman's method, and the rocket's
    center of pressure from the nose tip, m. The nose's slope is 2 for any shape and its center sits at
    L - V / A, so its volume decides it. The body tube adds no lift in this method."""
    f = fins or af.fins
    R = diameter / 2.0
    x = np.linspace(0.0, af.nose_length, 2001)
    r = nose_radius(af.nose_shape, x, af.nose_length, R)
    vol = float(np.sum(np.diff(x) * math.pi * 0.5 * (r[1:] ** 2 + r[:-1] ** 2)))
    x_nose = af.nose_length - vol / (math.pi * R ** 2)
    cn_nose = 2.0
    mid = math.hypot(f.span, f.sweep + f.tip / 2 - f.root / 2)        # mid chord line
    interference = 1 + R / (f.span + R)
    cn_fins = (interference * 4 * f.count * (f.span / diameter) ** 2 /
               (1 + math.sqrt(1 + (2 * mid / (f.root + f.tip)) ** 2)))
    x_fins = (f.position + f.sweep * (f.root + 2 * f.tip) / (3 * (f.root + f.tip)) +
              ((f.root + f.tip) - f.root * f.tip / (f.root + f.tip)) / 6)
    cp = (cn_nose * x_nose + cn_fins * x_fins) / (cn_nose + cn_fins)
    return {"cp": cp, "cn_nose": cn_nose, "x_nose": x_nose, "cn_fins": cn_fins, "x_fins": x_fins}


def margin(study: MotionStudy, cg: float | None = None, fins: Fins | None = None) -> float:
    """Static margin in calibers, the center of pressure behind the balance point over the diameter."""
    af = study.airframe
    return (barrowman(af, study.diameter, fins)["cp"] - (af.cg if cg is None else cg)) / study.diameter


def burnout_cg(study: MotionStudy) -> float | None:
    """Balance point once the propellant is gone, with the motor flush with the aft end."""
    af, mo = study.airframe, study.motor
    if af is None or af.cg is None or mo is None or mo.length <= 0:
        return None
    x_prop = af.length - mo.length / 2
    return (study.liftoff_mass * af.cg - mo.propellant * x_prop) / study.burnout_mass
