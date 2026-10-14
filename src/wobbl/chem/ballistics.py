"""Internal ballistics: chamber pressure, mass flow and thrust over a burn, from the grains, propellant and nozzle.

The gas in the chamber gains what the burning surface makes and loses what the throat passes,

    V / (R T) dP/dt = (rho_p - rho_g) A_b r(P) - P A_t / c*,        dx/dt = r(P)

where V is the free volume, which grows as the propellant burns, rho_g the gas density and x the regression every
surface shares. A pressure that holds still gives back the closed form P = (a (rho_p - rho_g) c* Kn)^(1/(1-n))
with Kn = A_b / A_t, and the same equation fills the chamber at ignition and empties it after burnout. The burning
itself is the burn rate law, not resolved flame chemistry. Pressure settles in milliseconds while the grains take
seconds, so a fourth order Runge-Kutta step is set by the faster of the two, and the run is repeated at half the
step until the impulse and peak pressure stop moving.

Thrust is C_F P A_t, with the ideal thrust coefficient for the frozen ratio of specific heats and the nozzle's area
ratio, times the divergence factor (1 + cos a) / 2 and a nozzle efficiency for two phase flow, friction and heat
loss. When the exit pressure would fall below 40% of ambient (Summerfield's criterion) the flow is taken to
separate there and only the attached part of the nozzle counts. SI units throughout.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .grains import Grain
from .propellants import Propellant
from .thermo import G0, P_REF, _ideal_exit_ratio

SEPARATION = 0.4          # Summerfield: separation when the wall pressure drops under 0.4 of ambient
CLASSES = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


@dataclass
class Nozzle:
    throat: float                       # m, diameter
    exit: float                         # m, diameter
    efficiency: float = 0.90
    half_angle: float = 15.0            # degrees, divergent cone

    @property
    def At(self) -> float:
        return math.pi / 4 * self.throat ** 2

    @property
    def area_ratio(self) -> float:
        return (self.exit / self.throat) ** 2

    @property
    def divergence(self) -> float:
        return (1 + math.cos(math.radians(self.half_angle))) / 2


@dataclass
class Motor:
    grains: list[Grain]                 # forward end first
    propellant: Propellant
    nozzle: Nozzle
    chamber_volume: float = 0.0         # m3 inside the casing over the grains, 0 for the grains' own envelope
    ambient: float = P_REF
    cstar_efficiency: float = 1.0

    @property
    def cstar(self) -> float:
        return self.propellant.cstar * self.cstar_efficiency

    @property
    def web(self) -> float:
        return max(g.web for g in self.grains)

    def __post_init__(self):
        # identical segments burn identically, so each distinct grain is evaluated once and counted
        groups: dict[int, list] = {}
        for g in self.grains:
            key = next((k for k, (h, _) in groups.items() if h == g), id(g))
            groups.setdefault(key, [g, 0])[1] += 1
        self._groups = [(g, n) for g, n in groups.values()]
        self._envelope = max(self.chamber_volume, sum(math.pi / 4 * g.D ** 2 * g.L for g in self.grains))

    def burn_area(self, x: float) -> float:
        return sum(n * g.burn_area(x) for g, n in self._groups)

    def propellant_volume(self, x: float) -> float:
        return sum(n * g.volume(x) for g, n in self._groups)

    @property
    def envelope(self) -> float:
        return self._envelope

    def kn(self, x: float) -> float:
        return self.burn_area(x) / self.nozzle.At

    @property
    def mass(self) -> float:
        return self.propellant.density * self.propellant_volume(0.0)


@dataclass
class Burn:
    t: np.ndarray
    P: np.ndarray                       # Pa
    F: np.ndarray                       # N
    mdot: np.ndarray                    # kg/s through the throat
    x: np.ndarray                       # m regression
    dt: float
    steps: int
    mass: float                         # kg of propellant
    gas_start: float                    # kg of gas in the chamber at the start and end
    gas_end: float
    extra: dict = field(default_factory=dict)

    @property
    def impulse(self) -> float:
        return float(np.sum(0.5 * (self.F[1:] + self.F[:-1]) * np.diff(self.t)))

    @property
    def peak_pressure(self) -> float:
        return float(self.P.max())

    @property
    def peak_thrust(self) -> float:
        return float(self.F.max())

    @property
    def burn_time(self) -> float:
        """Time the thrust stays above 5% of its peak, the way certification bodies time a burn."""
        on = np.nonzero(self.F >= 0.05 * self.F.max())[0]
        return float(self.t[on[-1]] - self.t[on[0]]) if len(on) > 1 else 0.0

    @property
    def average_thrust(self) -> float:
        bt = self.burn_time
        return self.impulse / bt if bt > 0 else 0.0

    @property
    def isp(self) -> float:
        return self.impulse / (self.mass * G0) if self.mass > 0 else 0.0

    @property
    def expelled(self) -> float:
        return float(np.sum(0.5 * (self.mdot[1:] + self.mdot[:-1]) * np.diff(self.t)))

    def designation(self) -> str:
        return motor_class(self.impulse) + f"{self.average_thrust:.0f}"


def motor_class(impulse: float) -> str:
    """The letter class: A up to 2.5 N s, each letter after doubling the one before."""
    if impulse <= 2.5:
        return "A" if impulse > 1.25 else "1/2A" if impulse > 0.625 else "1/4A"
    k = int(math.ceil(math.log2(impulse / 2.5)))
    return CLASSES[min(k, len(CLASSES) - 1)]


def thrust_coefficient(k: float, eps: float, P: float, Pa: float, pe_ratio: float) -> float:
    """Ideal thrust coefficient, with separation past Summerfield's 0.4 Pa wall pressure."""
    if P <= Pa:
        return 0.0
    gam = math.sqrt(k) * (2 / (k + 1)) ** ((k + 1) / (2 * (k - 1)))
    crit = (2 / (k + 1)) ** (k / (k - 1))
    if Pa / P > crit:     # the throat no longer chokes: a plain subsonic jet
        pr = Pa / P
        return gam * math.sqrt(2 * k / (k - 1) * max(1 - pr ** ((k - 1) / k), 0.0)) * _subsonic_flow(k, pr)
    pr, e = pe_ratio, eps
    if pr * P < SEPARATION * Pa:            # the flow leaves the wall where the pressure reaches 0.4 Pa
        pr = SEPARATION * Pa / P
        e = _area_ratio(pr, k)
    mom = gam * math.sqrt(2 * k / (k - 1) * (1 - pr ** ((k - 1) / k)))
    return mom + (pr - Pa / P) * e


def _area_ratio(pr: float, k: float) -> float:
    t = pr ** ((k - 1) / k)
    m2 = 2 / (k - 1) * (1 / t - 1)
    return math.sqrt(((2 / (k + 1)) * (1 + (k - 1) / 2 * m2)) ** ((k + 1) / (k - 1)) / m2)


def _subsonic_flow(k: float, pr: float) -> float:
    """Mass flow of an unchoked throat as a fraction of the choked flow at the same chamber state."""
    gam = math.sqrt(k) * (2 / (k + 1)) ** ((k + 1) / (2 * (k - 1)))
    if pr >= 1.0:
        return 0.0
    crit = (2 / (k + 1)) ** (k / (k - 1))
    if pr <= crit:
        return 1.0
    return math.sqrt(2 * k / (k - 1) * (pr ** (2 / k) - pr ** ((k + 1) / k))) / gam


def simulate(m: Motor, dt: float | None = None, max_steps: int = 400_000) -> Burn:
    """One burn with a fixed Runge-Kutta step, from ambient pressure at ignition to the end of the tail off."""
    prop, noz = m.propellant, m.nozzle
    k, At, cs, rho_p, Pa = prop.k, noz.At, m.cstar, prop.density, m.ambient
    gam = math.sqrt(k) * (2 / (k + 1)) ** ((k + 1) / (2 * (k - 1)))
    RT = (cs * gam) ** 2                     # R T of the gas, from c* = sqrt(R T) / Gamma
    web = m.web
    v0 = m.envelope - m.propellant_volume(0.0)
    if v0 <= 0:
        raise ValueError("the grains fill the chamber completely, so there is no room for gas")
    pe_ratio = _ideal_exit_ratio(noz.area_ratio, k) if noz.area_ratio > 1 else (2 / (k + 1)) ** (k / (k - 1))
    eta = noz.efficiency * noz.divergence

    def rates(P, x):
        xb = min(x, web)
        Ab = m.burn_area(xb) if x < web else 0.0
        r = prop.burn_rate.rate(P) if x < web else 0.0
        V = m.envelope - m.propellant_volume(xb)
        flow = At * P / cs * _subsonic_flow(k, Pa / P) if P > Pa else 0.0
        dP = RT / V * ((rho_p - P / RT) * Ab * r - flow)
        return dP, r, flow

    if dt is None:
        P_eq = _steady_pressure(m, 0.0)
        r0 = prop.burn_rate.rate(max(P_eq, Pa))
        tau = v0 * cs / (RT * At)            # how fast the chamber fills or empties, s
        dt = min(tau, web / max(r0, 1e-6) / 1000)
    P, x, t = Pa, 0.0, 0.0
    ts, Ps, Fs, ms, xs = [0.0], [P], [0.0], [0.0], [0.0]
    peak_F = 0.0
    for step in range(max_steps):
        k1, r1, _ = rates(P, x)
        k2, r2, _ = rates(max(P + 0.5 * dt * k1, Pa * 0.5), x + 0.5 * dt * r1)
        k3, r3, _ = rates(max(P + 0.5 * dt * k2, Pa * 0.5), x + 0.5 * dt * r2)
        k4, r4, _ = rates(max(P + dt * k3, Pa * 0.5), x + dt * r3)
        P = max(P + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4), Pa * 0.5)
        x = x + dt / 6 * (r1 + 2 * r2 + 2 * r3 + r4)
        t += dt
        _, _, flow = rates(P, x)
        F = thrust_coefficient(k, noz.area_ratio, P, Pa, pe_ratio) * eta * P * At if P > Pa else 0.0
        ts.append(t)
        Ps.append(P)
        Fs.append(F)
        ms.append(flow)
        xs.append(x)
        peak_F = max(peak_F, F)
        if x >= web and (P < Pa * 1.02 or F < 0.002 * peak_F):
            break
    else:
        raise RuntimeError(f"the burn did not finish within {max_steps} steps of {dt:.3g} s")
    t_arr = np.array(ts)
    P_arr = np.array(Ps)
    return Burn(t_arr, P_arr, np.array(Fs), np.array(ms), np.array(xs), dt, step + 1, m.mass,
                Pa * v0 / RT, P * m.envelope / RT)


def _steady_pressure(m: Motor, x: float, P_guess: float | None = None) -> float:
    """The closed form chamber pressure at regression x: P = (a (rho_p - rho_g) c* Kn)^(1/(1-n)), solved in the
    burn rate band it lands in, with the gas density taken at that pressure."""
    prop = m.propellant
    kn = m.kn(x)
    if kn <= 0:
        return m.ambient
    k = prop.k
    gam = math.sqrt(k) * (2 / (k + 1)) ** ((k + 1) / (2 * (k - 1)))
    RT = (m.cstar * gam) ** 2
    P = P_guess or 3e6
    for i in range(100):
        a, n = prop.burn_rate.band(P)
        a_si = a / 1000.0 * (1e-6) ** n                # m/s at P in Pa
        if n >= 1:
            raise ValueError("a burn rate exponent of 1 or more has no steady chamber pressure")
        P_new = (a_si * (prop.density - P / RT) * m.cstar * kn) ** (1 / (1 - n))
        if abs(P_new - P) < 1e-9 * P:
            return P_new
        P = 0.5 * (P + P_new) if i > 50 else P_new     # damped when two bands keep handing it back and forth
    return P


def kn_for_pressure(m: Motor, P: float) -> float:
    """The Kn whose closed form chamber pressure is P, in the burn rate band P falls in."""
    prop = m.propellant
    a, n = prop.burn_rate.band(P)
    k = prop.k
    gam = math.sqrt(k) * (2 / (k + 1)) ** ((k + 1) / (2 * (k - 1)))
    rho_g = P / (m.cstar * gam) ** 2
    return P ** (1 - n) / (a / 1000.0 * (1e-6) ** n * (prop.density - rho_g) * m.cstar)


def steady_pressure(m: Motor, x: float, classic: bool = False) -> float:
    """Closed form pressure at regression x, or the textbook form without the gas density when classic."""
    if not classic:
        return _steady_pressure(m, x)
    kn = m.kn(x)
    prop = m.propellant
    P = 3e6
    for _ in range(100):
        a, n = prop.burn_rate.band(P)
        P_new = (a / 1000.0 * (1e-6) ** n * prop.density * m.cstar * kn) ** (1 / (1 - n))
        if abs(P_new - P) < 1e-9 * P:
            break
        P = P_new
    return P


def converge(m: Motor, tol: float = 0.003, max_halvings: int = 4) -> tuple[Burn, list[dict]]:
    """Run with the automatic step, then halve it until impulse, peak pressure and burn time each move less
    than tol. Returns the finest burn and the history of every run."""
    runs, burns = [], []
    dt = None
    for _ in range(max_halvings + 1):
        b = simulate(m, dt)
        burns.append(b)
        runs.append({"dt_s": b.dt, "steps": b.steps, "impulse_Ns": b.impulse, "peak_Pa": b.peak_pressure,
                     "burn_time_s": b.burn_time})
        if len(burns) >= 2:
            a, c = burns[-2], burns[-1]
            change = max(abs(c.impulse - a.impulse) / max(c.impulse, 1e-12),
                         abs(c.peak_pressure - a.peak_pressure) / c.peak_pressure,
                         abs(c.burn_time - a.burn_time) / max(c.burn_time, 1e-12))
            runs[-1]["change"] = change
            if change < tol:
                break
        dt = b.dt / 2
    return burns[-1], runs
