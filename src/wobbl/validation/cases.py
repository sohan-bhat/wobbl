"""Validation cases with known answers, run through the same pipeline users get.

Each case builds its geometry, runs Wobbl end to end (meshing, convergence loop,
solver, checks) and compares one number with the textbook or benchmark value.
Run them with 'wobbl validate'.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from . import geometry as G

E_STEEL, NU = 200000.0, 0.3
STEEL = f"custom:E={E_STEEL},nu={NU},yield=250,density=7850"


@dataclass
class Case:
    id: str
    title: str
    domain: str
    source: str
    quantity: str
    unit: str
    tol: float
    exact: bool                       # textbook value is exact for this setup
    run: Callable[[Path], tuple[float, float, dict]]   # returns (measured, expected, info)


def _strength(part, material, loads, fixtures, out, quality="normal"):
    from ..structural.pipeline import analyze_strength
    from ..structural.spec import StructuralStudy, parse_fixture, parse_load

    st = StructuralStudy(str(part), material, [parse_load(x) for x in loads], [parse_fixture(x) for x in fixtures],
                         quality=quality, budget=600)
    o = analyze_strength(st, out, render=False, fixes=False)
    f = np.load(Path(out) / "work" / "fields.npz")
    return o, f


def _near(X, p):
    return int(np.argmin(np.linalg.norm(X - np.asarray(p, float), axis=1)))


def _info(o):
    return {"confidence": o.confidence, "levels": len(o.levels), "nodes": o.levels[-1]["n_nodes"],
            "time_s": round(o.t_total, 1)}


# --- structural -------------------------------------------------------------------
def cantilever_tip(out: Path):
    L, b, h, F = 200.0, 20.0, 10.0, 100.0
    part = G.box(out / "bar.step", L, b, h)
    o, f = _strength(part, STEEL, [f"{F} N -z @ x=max"], ["x=min"], out / "run")
    X, U = f["X"], f["U"]
    end = np.abs(X[:, 0] - L) < 1e-6
    I = b * h ** 3 / 12
    Gm = E_STEEL / (2 * (1 + NU))
    expected = F * L ** 3 / (3 * E_STEEL * I) + F * L / (5 / 6 * Gm * b * h)
    return float(-U[end, 2].mean()), expected, _info(o)


def cantilever_stress(out: Path):
    L, b, h, F = 200.0, 20.0, 10.0, 100.0
    part = G.box(out / "bar.step", L, b, h)
    o, f = _strength(part, STEEL, [f"{F} N -z @ x=max"], ["x=min"], out / "run")
    i = _near(f["X"], (L / 2, b / 2, h))
    expected = F * (L / 2) * (h / 2) / (b * h ** 3 / 12)
    return float(f["S"][i, 0]), expected, _info(o)


def plate_hole(out: Path):
    Lp, W, t, d, s = 400.0, 200.0, 4.0, 20.0, 10.0
    part = G.plate_with_hole(out / "plate.step", Lp, W, t, d)
    o, f = _strength(part, STEEL, [f"-{s} MPa @ x=max"], ["roller x=min", "roller y=min", "roller z=min"],
                     out / "run")
    r = d / W
    kt_net = 2 + 0.284 * (1 - r) - 0.600 * (1 - r) ** 2 + 1.32 * (1 - r) ** 3
    expected = kt_net / (1 - r) * s
    return o.governing_MPa, expected, _info(o)


def lame(out: Path):
    ri, ro, p = 50.0, 100.0, 100.0
    part = G.quarter_tube(out / "tube.step", ri, ro, 20.0)
    o, f = _strength(part, STEEL, [f"{p} MPa @ cylinders r={ri:g}"],
                     ["roller x=0", "roller y=0", "roller z=min", "roller z=max"], out / "run")
    st = p * (ro ** 2 + ri ** 2) / (ro ** 2 - ri ** 2)
    sr = -p
    sz = NU * (st + sr)
    expected = math.sqrt(0.5 * ((st - sr) ** 2 + (sr - sz) ** 2 + (sz - st) ** 2))
    return o.governing_MPa, expected, _info(o)


def _faces_where(part, out, pred):
    from ..structural.run import describe_part

    _, feats = describe_part(str(part), out / "probe")
    return ",".join(str(f.id) for f in feats.faces.values() if pred(f))


def le1(out: Path):
    part = G.nafems_le1(out / "le1.step")
    outer = _faces_where(part, out, lambda f: f.kind != "plane" and (f.centroid[0] / 3250) ** 2 +
                         (f.centroid[1] / 2750) ** 2 > 0.6)
    o, f = _strength(part, "custom:E=210000,nu=0.3,yield=500", [f"-10 MPa @ faces {outer}"],
                     ["roller x=0", "roller y=0", "roller z=min"], out / "run")
    i = _near(f["X"], (2000, 0, 50))
    return float(f["S"][i, 1]), 92.7, _info(o)


def le10(out: Path):
    part = G.nafems_le10(out / "le10.step")
    ell = lambda f: (f.centroid[0] / 3250) ** 2 + (f.centroid[1] / 2750) ** 2 > 0.6  # noqa: E731
    outer = _faces_where(part, out, lambda f: f.kind != "plane" and ell(f))
    band = _faces_where(part, out, lambda f: f.kind != "plane" and ell(f) and abs(f.centroid[2]) < 2)
    o, f = _strength(part, "custom:E=210000,nu=0.3,yield=500", ["1 MPa @ z=max"],
                     ["roller x=0", "roller y=0", f"fixed-xy faces {outer}", f"fixed-z faces {band}"], out / "run")
    i = _near(f["X"], (2000, 0, 300))
    return float(f["S"][i, 1]), -5.38, _info(o)


def torsion(out: Path):
    d, L, T = 20.0, 100.0, 50.0
    part = G.shaft(out / "shaft.step", d, L)
    o, f = _strength(part, STEEL, [f"{T} N*m +z @ z=max"], ["z=min"], out / "run")
    tau = 16 * T * 1000 / (math.pi * d ** 3)
    return o.governing_MPa, math.sqrt(3) * tau, _info(o)


def self_weight(out: Path):
    L, b = 1000.0, 20.0
    part = G.box(out / "beam.step", L, b, b)
    o, f = _strength(part, STEEL, ["1 g down"], ["x=min"], out / "run")
    q = 7850e-12 * 9806.65 * b * b
    I = b ** 4 / 12
    end = np.abs(f["X"][:, 0] - L) < 1e-6
    return float(-f["U"][end, 2].mean()), q * L ** 4 / (8 * E_STEEL * I), _info(o)


# --- real parts from the bake off ------------------------------------------------
# References: hand built gmsh + CalculiX models converged over four to six meshes (up to 400k nodes).
PARTS = Path(__file__).parent / "parts"
PLA_SPEC = "pla"


def cast_motor_mount(out: Path):
    """Sand cast A356-T6 motor mount, 2 kN belt pull at a pulley off the part, shared by four flange bolts."""
    part = PARTS / "cast_motor_mount.step"
    o, f = _strength(part, "custom:E=72400,nu=0.33,yield=165,process=cast",
                     ["2000 N -0.8660254,0,-0.5 at 0,-30,66 @ holes d=6.6"], ["z=min"], out / "run")
    return o.governing_MPa, 124.3, {**_info(o), "max_disp_mm": round(o.max_disp_mm, 4), "reference_disp_mm": 0.266}


def enclosure_shell(out: Path):
    """Molded ABS shell with bosses, a snap tongue and a USB opening, 250 N pressed near one corner."""
    part = PARTS / "enclosure_shell.step"
    rim = _faces_where(part, out, lambda f: f.kind == "plane" and abs(f.bbox[0][2]) < 1e-3 and
                       abs(f.bbox[1][2]) < 1e-3 and not (32 < f.centroid[0] < 44 and f.centroid[1] < -42))
    o, f = _strength(part, "custom:E=2200,nu=0.35,yield=40", ["250 N -z @ top and within 15 of 50,-25,35"],
                     [f"faces {rim}"], out / "run")
    down = float(-f["U"][:, 2].min())
    return o.governing_MPa, 93.0, {**_info(o), "top_deflection_mm": round(down, 4), "reference_deflection_mm": 1.93}


def drone_arm_truss(out: Path):
    """PA12 drone arm with Warren truss windows, 40 N thrust and 0.8 N m torque on a rigid motor."""
    part = PARTS / "drone_arm_truss.step"
    pad = "top and within 14 of 210,0,25"
    o, f = _strength(part, "custom:E=1800,nu=0.39,yield=48",
                     [f"40 N +z at 210,0,25 @ {pad} rigid", f"0.8 N*m +z at 210,0,25 @ {pad} rigid"], ["x=min"],
                     out / "run")
    X, U = f["X"], f["U"]
    m = (np.linalg.norm(X[:, :2] - [210, 0], axis=1) < 14) & (np.abs(X[:, 2] - 25) < 0.01)
    rise = float(U[m, 2].mean()) if m.any() else float("nan")
    return o.governing_MPa, 11.5, {**_info(o), "motor_rise_mm": round(rise, 4), "reference_rise_mm": 4.09}


def _bracket(name: str, out: Path):
    part = PARTS / name
    return _strength(part, PLA_SPEC, ["60 N -y @ normal +y and box 45,5,-1,55,7,21"],
                     ["hole d=4.5", "wall x=min"], out / "run")


def bracket_sharp(out: Path):
    """PLA L bracket with a 0.5 mm inside corner, screwed to a wall, 60 N shelf load. Judged in tension."""
    o, f = _bracket("shelf_bracket_sharp.step", out)
    return o.governing_MPa, 78.0, _info(o)


def bracket_fillet(out: Path):
    """The same bracket with a 6 mm fillet."""
    o, f = _bracket("shelf_bracket_fillet.step", out)
    return o.governing_MPa, 25.3, _info(o)


# --- heat -------------------------------------------------------------------------
def slab(out: Path):
    from ..thermal.study import run_thermal
    from ..thermal.tools import study_from

    L, k, h, t1, tinf = 100.0, 167.0, 50.0, 100.0, 20.0
    part = G.box(out / "slab.step", L, 20, 20)
    st = study_from(part, "6061-t6", films=[f"{h} W/m2K at {tinf} C @ x=max"], fixed=[f"{t1} C @ x=min"],
                    ambient=f"{tinf} C")
    o = run_thermal(st, out / "run")
    f = np.load(out / "run" / "fields.npz")
    end = np.abs(f["X"][:, 0] - L) < 1e-6
    q = (t1 - tinf) / (L / 1000 / k + 1 / h)
    return float(f["T"][end].mean()), tinf + q / h, {"confidence": o.confidence, "time_s": round(o.t_total, 1)}


def lumped(out: Path):
    from ..materials import get_material
    from ..thermal.study import run_thermal
    from ..thermal.tools import study_from

    a, h, t0, tinf, dur = 20.0, 10.0, 100.0, 20.0, 600.0
    part = G.box(out / "cube.step", a, a, a)
    st = study_from(part, "6061-t6", films=[f"{h} W/m2K at {tinf} C @ all"], ambient=f"{tinf} C",
                    duration=f"{dur} s")
    st.initial = t0
    o = run_thermal(st, out / "run")
    f = np.load(out / "run" / "fields.npz")
    m = get_material("6061-t6")
    tau = m.density * m.specific_heat * (a / 1000) ** 3 / (h * 6 * (a / 1000) ** 2)
    expected = tinf + (t0 - tinf) * math.exp(-dur / tau)
    return float(f["T"].mean()), expected, {"confidence": o.confidence, "tau_s": round(tau, 1)}


def sudden_face(out: Path):
    """A 100 mm aluminum bar at 20 C whose end is held at 120 C from t = 0, read 10 mm in after 2 s. Heat has
    not reached the far end yet, so the bar is a semi infinite solid and the rise is 100 erfc(x / (2 sqrt(a t)))."""
    from ..materials import get_material
    from ..thermal.study import run_thermal
    from ..thermal.tools import study_from

    L, t0, t1, depth, when = 100.0, 20.0, 120.0, 10.0, 2.0
    part = G.box(out / "bar.step", L, 20, 20)
    st = study_from(part, "6061-t6", films=[f"10 W/m2K at {t0} C @ x=max"], fixed=[f"{t1} C @ x=min"],
                    ambient=f"{t0} C", duration="4 s")
    o = run_thermal(st, out / "run")
    f = np.load(out / "run" / "fields.npz")
    i = int(np.argmin(np.abs(f["hist_t"] - when)))
    j = _near(f["X"], (depth, 10, 10))
    m = get_material("6061-t6")
    alpha = m.conductivity / (m.density * m.specific_heat) * 1e6      # mm2/s
    x = float(f["X"][j, 0])
    expected = (t1 - t0) * math.erfc(x / (2 * math.sqrt(alpha * float(f["hist_t"][i]))))
    return float(f["hist_T"][i, j]) - t0, expected, {"confidence": o.confidence, "depth_mm": round(x, 3),
                                                     "time_s": round(o.t_total, 1)}


# --- circuits ---------------------------------------------------------------------
def led(out: Path):
    from ..circuits.run import run_circuit
    from ..circuits.templates import led as led_tpl

    p = {"supply": "5", "resistor": "150", "led": "red"}
    o = run_circuit("led", p, out)
    return o.values["i_led_A"] * 1000, led_tpl(p).expect["i_led"] * 1000, {"confidence": o.confidence}


def rc(out: Path):
    from ..circuits.run import run_circuit

    o = run_circuit("rc", {"r": "10k", "c": "10u", "v": "5"}, out)
    return o.values["v_at_tau"], 5 * (1 - math.exp(-1)), {"confidence": o.confidence}


def buck(out: Path):
    from ..circuits.run import run_circuit
    from ..circuits.templates import buck as buck_tpl

    p = {"vin": "12", "vout": "5", "current": "2", "frequency": "500k", "inductor": "10u", "capacitor": "22u"}
    o = run_circuit("buck", p, out)
    return o.values["ripple_V"] * 1000, buck_tpl(p).expect["ripple_v"] * 1000, {"confidence": o.confidence}


# --- motion -----------------------------------------------------------------------
# Each flight goes through the motion domain's own integrator, events, phases and checks, reduced to a setup with
# an exact answer: gravity held at 9.80665 m/s2 and either no air or air of one density.
def _flight(**kw):
    from ..motion.spec import MotionStudy
    from ..motion.study import analyze

    a = analyze(MotionStudy(uniform_gravity=True, **kw))
    return a.flight, {"confidence": a.confidence, "events": len(a.flight.events)}


def free_fall(out: Path):
    """A ball dropped from 300 m with no air, t = sqrt(2h/g)."""
    from ..motion.atmosphere import G0

    fl, info = _flight(name="ball", mass=1.0, diameter=0.1, height=300.0, air="vacuum")
    return fl.end.t, math.sqrt(2 * 300.0 / G0), info


def drag_fall(out: Path):
    """The falling rock example's 9 cm granite sphere, Cd 0.6, dropped 300 m through air of constant density.
    The fall time comes from h = vt^2/g ln cosh(g t / vt), and the impact speed is vt tanh(g t / vt)."""
    from ..motion.atmosphere import G0

    m, d, cd, rho, h = 2700 * math.pi / 6 * 0.09 ** 3, 0.09, 0.6, 1.225, 300.0
    fl, info = _flight(name="rock", mass=m, diameter=d, cd=cd, height=h, air=rho)
    vt = math.sqrt(2 * m * G0 / (rho * cd * math.pi * d * d / 4))
    t = vt / G0 * math.acosh(math.exp(G0 * h / vt ** 2))
    v = math.hypot(fl.end.y[2], fl.end.y[3])
    return v, vt * math.tanh(G0 * t / vt), {**info, "time_s": round(fl.end.t, 6), "closed_form_time_s": round(t, 6)}


def projectile_range(out: Path):
    """A ball thrown at 30 m/s, 40 deg above level, from the ground with no air, R = v^2 sin(2 theta) / g."""
    from ..motion.atmosphere import G0

    fl, info = _flight(name="ball", mass=0.2, diameter=0.07, speed=30.0, angle=40.0, air="vacuum")
    return float(fl.end.y[0]), 30.0 ** 2 * math.sin(math.radians(80.0)) / G0, info


def _vertical_rocket():
    """A rocket on a vertical rail in a vacuum, 1000 N for 4 s burning 4 kg of propellant (an exhaust speed of
    1000 m/s), 11 kg at ignition and 7 kg at burnout."""
    from ..motion.spec import Motor

    thrust, burn, dry, case, prop = 1000.0, 4.0, 5.0, 2.0, 4.0
    motor = Motor("constant", [(0.0, thrust), (burn, thrust)], prop, case + prop)
    fl, info = _flight(name="rocket", mass=dry, diameter=0.1, air="vacuum", rail_length=1.0, motor=motor)
    return fl, info, dry + case + prop, dry + case, thrust * burn / prop, burn


def rocket_speed(out: Path):
    """Burnout speed of the vertical rocket, ve ln(m0 / m1) - g tb."""
    from ..motion.atmosphere import G0

    fl, info, m0, m1, ve, tb = _vertical_rocket()
    return float(fl.event("burnout").y[3]), ve * math.log(m0 / m1) - G0 * tb, info


def rocket_height(out: Path):
    """Burnout altitude of the vertical rocket, ve tb - ve (m1 / mdot) ln(m0 / m1) - g tb^2 / 2."""
    from ..motion.atmosphere import G0

    fl, info, m0, m1, ve, tb = _vertical_rocket()
    mdot = (m0 - m1) / tb
    exact = ve * tb - ve * m1 / mdot * math.log(m0 / m1) - G0 * tb ** 2 / 2
    return float(fl.event("burnout").y[1]), exact, info


def chute_descent(out: Path):
    """A 5 kg payload let go from 500 m under a 2 m parachute with Cd 1.5 in air of constant density. It lands at
    the terminal speed sqrt(2 m g / (rho Cd A))."""
    from ..motion.atmosphere import G0
    from ..motion.spec import Chute

    chute = Chute("main", 2.0, 1.5)
    fl, info = _flight(name="payload", mass=5.0, diameter=0.2, height=500.0, air=1.225, chutes=[chute])
    return float(-fl.end.y[3]), math.sqrt(2 * 5.0 * G0 / (1.225 * chute.drag_area)), info


# --- uncertainty and calibration ----------------------------------------------------
# The cantilever of cantilever-tip with a lognormal load (10%) and a lognormal modulus (5%). Tip deflection is
# F L^3 / (3 E I) + F L / (k G A), both terms F over E, so it is lognormal too and its mean and spread are exact.
_UQ_RUNS: dict[str, tuple] = {}


def _uq_cantilever(out: Path):
    from ..credibility.uncertainty import run_uncertainty

    key = str(out.parent)
    if key not in _UQ_RUNS:
        L, b, h, F = 200.0, 20.0, 10.0, 100.0
        part = G.box(out / "bar.step", L, b, h)
        o, _ = _strength(part, STEEL, [f"{F} N -z @ x=max"], ["x=min"], out / "run")
        r = run_uncertainty(out / "run", ["load lognormal 10%", "E lognormal 5%"], samples=200000, numerical=False)
        Gm = E_STEEL / (2 * (1 + NU))
        tip = F * L ** 3 / (3 * E_STEEL * b * h ** 3 / 12) + F * L / (5 / 6 * Gm * b * h)
        mean = tip * (1 + 0.05 ** 2)      # E[1/E] = (1 + cov^2) / mean for a lognormal modulus
        sd = mean * math.sqrt((1 + 0.1 ** 2) * (1 + 0.05 ** 2) - 1)
        _UQ_RUNS[key] = (r["outputs"]["max_deflection_mm"], mean, sd, _info(o), out / "run")
    return _UQ_RUNS[key]


def uq_cantilever_mean(out: Path):
    s, mean, sd, info, _ = _uq_cantilever(out)
    return s["mean"], mean, {**info, "sd_mm": round(s["sd"], 6), "closed_form_sd_mm": round(sd, 6)}


def uq_cantilever_sd(out: Path):
    s, mean, sd, info, _ = _uq_cantilever(out)
    return s["sd"], sd, {**info, "mean_mm": round(s["mean"], 6), "closed_form_mean_mm": round(mean, 6)}


def sobol_ishigami(out: Path):
    """Sobol indices of the Ishigami function, inputs uniform on [-pi, pi], against the analytic values. The number
    returned is whichever of the five nonzero indices lands farthest from its value, relatively."""
    from ..credibility import uq

    ins = [uq.Input(f"x{i + 1}", "uniform", 0.0, 2 * math.pi / math.sqrt(12), -math.pi, math.pi) for i in range(3)]
    got = {r["input"]: r for r in uq.sobol_indices(uq.ishigami, ins, n=2 ** 14, boot=50)}
    rows = []
    for name, (s1, st) in uq.ishigami_indices().items():
        rows += [(f"first order {name}", got[name]["first_order"], s1), (f"total {name}", got[name]["total"], st)]
    worst = max((r for r in rows if r[2] > 0), key=lambda r: abs(r[1] - r[2]) / r[2])
    info = {r[0]: round(r[1], 5) for r in rows}
    info["worst"] = worst[0]
    info["first_order_x3_abs_error"] = round(abs(got["x3"]["first_order"]), 5)
    if abs(got["x3"]["first_order"]) > 0.01:
        raise ValueError(f"first order index of x3 is {got['x3']['first_order']:.4f}, analytic 0")
    return worst[1], worst[2], info


def calibration_twin(out: Path):
    """A twin experiment on the cantilever: strain gauge and dial gauge readings made from Wobbl's own solution at
    E = 194 GPa, with 5 microstrain and 0.005 mm of noise, then calibrated from a prior of 200 GPa +- 5% through
    'wobbl validate-against'. The case fails if the true modulus is outside the stated 95% interval."""
    from ..credibility.against import run_validation
    from ..credibility.probes import gauge, load_fields, surface

    run = _uq_cantilever(out)[4]
    f = load_fields(run)
    surf = surface(f)
    true = 194000.0
    rng = np.random.default_rng(2026)
    rows = ["# synthetic twin experiment for the calibration case, not a real test",
            "name,quantity,x,y,z,direction,load,value,uncertainty,unit"]
    for name, x in (("SG1", 40.0), ("SG2", 100.0), ("SG3", 160.0)):
        g = gauge(f, (x, 10.0, 10.0), E_STEEL, NU, (1, 0, 0), surf=surf)
        for load in (50.0, 100.0):
            eps = g["strain"] * load / 100.0 * E_STEEL / true * 1e6
            rows.append(f"{name},strain,{x},10,10,x,{load},{eps + 5.0 * rng.standard_normal():.4f},5,microstrain")
    i = _near(f["X"], (200.0, 10.0, 0.0))
    for load in (50.0, 100.0):
        w = -f["U"][i, 2] * load / 100.0 * E_STEEL / true
        rows.append(f"D1,displacement,200,10,0,-z,{load},{w + 0.005 * rng.standard_normal():.5f},0.005,mm")
    csv = out / "readings.csv"
    csv.write_text("\n".join(rows) + "\n")
    r = run_validation(run, csv, ["E 5%"])
    c = r["calibration"]["inputs"][0]
    if not c["ci95"][0] <= true <= c["ci95"][1]:
        raise ValueError(f"true modulus {true:g} MPa is outside the stated 95% interval {c['ci95']}")
    return c["mean"], true, {"posterior_sd_MPa": round(c["sd"], 1), "ci95_MPa": [round(v, 1) for v in c["ci95"]],
                             "narrowed": round(c["narrowed"], 2), "readings": len(rows) - 2}


def calibration_cooling(out: Path):
    """Two parameters at once: the film coefficient and the air temperature of a 20 mm aluminum cube cooling from
    100 C, T = Ta + (T0 - Ta) exp(-h A t / (rho c V)), recovered from twelve readings with 0.3 C of noise. The case
    fails if either true value is outside its stated 95% interval."""
    from ..credibility import uq
    from ..credibility.calibrate import calibrate
    from ..materials import get_material

    m = get_material("6061-t6")
    a = 0.02
    k = 6 * a * a / (m.density * m.specific_heat * a ** 3)        # h A / (rho c V) per unit h, 1/s
    t = np.linspace(60.0, 1800.0, 12)

    def model(X):
        return X[:, 1:2] + (100.0 - X[:, 1:2]) * np.exp(-np.outer(X[:, 0], t) * k)
    true = np.array([[12.0, 22.0]])
    data = model(true)[0] + 0.3 * np.random.default_rng(7).standard_normal(len(t))
    prior = [uq.Input("h", "lognormal", 10.0, 3.0, unit="W/m2K"), uq.Input("ambient", "normal", 25.0, 3.0, unit="C")]
    r = calibrate(model, prior, data, np.full(len(t), 0.3))
    h, ta = r["inputs"]
    for c, v in ((h, 12.0), (ta, 22.0)):
        if not c["ci95"][0] <= v <= c["ci95"][1]:
            raise ValueError(f"true {c['input']} {v:g} is outside the stated 95% interval {c['ci95']}")
    return h["mean"], 12.0, {"h_ci95": [round(x, 3) for x in h["ci95"]], "ambient_mean_C": round(ta["mean"], 3),
                             "ambient_ci95_C": [round(x, 3) for x in ta["ci95"]],
                             "acceptance": round(r["acceptance"], 3)}


CASES = [
    Case("cantilever-tip", "Cantilever tip deflection", "strength", "Timoshenko beam theory", "tip deflection",
         "mm", 0.03, True, cantilever_tip),
    Case("cantilever-stress", "Cantilever bending stress at mid span", "strength", "Euler-Bernoulli, M c / I",
         "sigma_xx at mid span", "MPa", 0.03, True, cantilever_stress),
    Case("plate-hole", "Plate with a hole in tension", "strength", "Kirsch with Peterson finite width Kt",
         "peak von Mises at the hole", "MPa", 0.04, True, plate_hole),
    Case("lame", "Thick walled pressure vessel", "strength", "Lame, plane strain", "von Mises at the bore",
         "MPa", 0.03, True, lame),
    Case("nafems-le1", "NAFEMS LE1 elliptic membrane", "strength", "NAFEMS LE1 target 92.7 MPa",
         "sigma_yy at point D", "MPa", 0.04, True, le1),
    Case("nafems-le10", "NAFEMS LE10 thick plate under pressure", "strength", "NAFEMS LE10 target -5.38 MPa",
         "sigma_yy at point D", "MPa", 0.05, True, le10),
    Case("torsion", "Round shaft in torsion", "strength", "tau = 16 T / (pi d^3)", "von Mises at the surface",
         "MPa", 0.03, True, torsion),
    Case("self-weight", "Cantilever under its own weight", "strength", "q L^4 / (8 E I)", "tip deflection", "mm",
         0.03, True, self_weight),
    Case("cast-motor-mount", "Cast motor mount with a belt pull through four bolts", "strength",
         "converged reference FEA, 278k nodes", "peak von Mises at the rib tip", "MPa", 0.05, False, cast_motor_mount),
    Case("enclosure-shell", "ABS enclosure shell pressed near a corner", "strength",
         "converged reference FEA, 281k nodes", "peak von Mises at the corner boss", "MPa", 0.10, False,
         enclosure_shell),
    Case("drone-arm-truss", "PA12 truss drone arm at full thrust", "strength", "converged reference FEA, 403k nodes",
         "peak von Mises at the first window corner", "MPa", 0.08, False, drone_arm_truss),
    Case("bracket-sharp", "PLA shelf bracket, 0.5 mm inside corner", "strength",
         "converged reference FEA with wall contact", "max principal at the corner", "MPa", 0.08, False,
         bracket_sharp),
    Case("bracket-fillet", "PLA shelf bracket, 6 mm fillet", "strength", "converged reference FEA with wall contact",
         "max principal at the fillet", "MPa", 0.05, False, bracket_fillet),
    Case("slab", "Conduction through a slab into convection", "heat", "1D series resistance",
         "cooled face temperature", "C", 0.01, True, slab),
    Case("lumped", "Aluminum cube cooling in air", "heat", "lumped capacitance (Biot 2e-4)",
         "mean temperature after 10 min", "C", 0.01, True, lumped),
    Case("sudden-face", "Aluminum bar whose end jumps to 120 C", "heat", "semi infinite solid, erfc",
         "temperature rise 10 mm in after 2 s", "C", 0.02, True, sudden_face),
    Case("led", "Red LED on 5 V through 150 ohm", "circuit", "diode equation solved by Newton", "LED current",
         "mA", 0.005, True, led),
    Case("rc", "RC charging", "circuit", "V (1 - e^-1) at t = RC", "voltage at one time constant", "V", 0.002,
         True, rc),
    Case("buck", "Buck converter output ripple", "circuit", "ideal triangular current through C and ESR",
         "peak to peak ripple", "mV", 0.15, False, buck),
    Case("free-fall", "Ball dropped 300 m without air", "motion", "t = sqrt(2 h / g)", "fall time", "s", 0.001,
         True, free_fall),
    Case("drag-fall", "Granite sphere dropped 300 m through air", "motion",
         "v = vt tanh(g t / vt), h = vt^2 / g ln cosh(g t / vt)", "impact speed", "m/s", 0.001, True, drag_fall),
    Case("projectile", "Ball thrown at 30 m/s and 40 deg without air", "motion", "R = v^2 sin(2 theta) / g",
         "range", "m", 0.001, True, projectile_range),
    Case("rocket-speed", "Vertical rocket at constant thrust in a vacuum", "motion", "ve ln(m0 / m1) - g tb",
         "burnout speed", "m/s", 0.001, True, rocket_speed),
    Case("rocket-height", "Vertical rocket at constant thrust in a vacuum", "motion",
         "ve tb - ve (m1 / mdot) ln(m0 / m1) - g tb^2 / 2", "burnout altitude", "m", 0.001, True, rocket_height),
    Case("chute-descent", "Payload under a 2 m parachute", "motion", "v = sqrt(2 m g / (rho Cd A))",
         "descent rate at the ground", "m/s", 0.001, True, chute_descent),
    Case("uq-cantilever-mean", "Cantilever with load and stiffness scatter, mean deflection", "uncertainty",
         "lognormal propagation of Timoshenko beam theory", "mean tip deflection", "mm", 0.03, True,
         uq_cantilever_mean),
    Case("uq-cantilever-sd", "Cantilever with load and stiffness scatter, spread of deflection", "uncertainty",
         "lognormal propagation of Timoshenko beam theory", "standard deviation of tip deflection", "mm", 0.03, True,
         uq_cantilever_sd),
    Case("sobol-ishigami", "Sobol indices of the Ishigami function", "uncertainty",
         "analytic first order and total indices", "index farthest from its value", "", 0.02, True, sobol_ishigami),
    Case("calibration-twin", "Modulus calibrated from synthetic cantilever gauges", "uncertainty",
         "the modulus the readings were made with, 194 GPa", "posterior mean modulus", "MPa", 0.01, True,
         calibration_twin),
    Case("calibration-cooling", "Film coefficient and air temperature from a synthetic cooling curve",
         "uncertainty", "the values the readings were made with, 12 W/m2K and 22 C", "posterior mean film coefficient",
         "W/m2K", 0.05, True, calibration_cooling),
]


def run_cases(out_dir: str | Path, ids: list[str] | None = None, log=print) -> list[dict]:
    out = Path(out_dir)
    results = []
    for c in CASES:
        if ids and c.id not in ids:
            continue
        work = out / c.id
        work.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        try:
            got, want, info = c.run(work)
            err = abs(got - want) / abs(want)
            ok = err <= c.tol
            row = {"id": c.id, "title": c.title, "domain": c.domain, "source": c.source, "quantity": c.quantity,
                   "unit": c.unit, "expected": want, "wobbl": got, "error": err, "tolerance": c.tol, "pass": ok,
                   "exact_theory": c.exact, **info, "wall_s": round(time.time() - t0, 1)}
        except Exception as e:  # noqa: BLE001
            row = {"id": c.id, "title": c.title, "domain": c.domain, "error_text": str(e), "pass": False}
        results.append(row)
        log(f"{c.id:<18} " + (f"expected {row['expected']:.4g} got {row['wobbl']:.4g} {c.unit} "
                              f"({100 * row['error']:.2f}%) {'pass' if row['pass'] else 'FAIL'}" if "wobbl" in row
                              else f"ERROR {row['error_text']}"))
    (out / "results.json").write_text(json.dumps(results, indent=1))
    return results


def markdown_table(results: list[dict]) -> str:
    lines = ["| Case | Quantity | Reference | Expected | Wobbl | Error | Tolerance | Wobbl's confidence | Result |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        if "wobbl" not in r:
            lines.append(f"| {r['title']} | | | | error | | | | FAIL |")
            continue
        lines.append(f"| {r['title']} | {r['quantity']} | {r['source']} | {r['expected']:.4g} {r['unit']} | "
                     f"{r['wobbl']:.4g} {r['unit']} | {100 * r['error']:.2f}% | {100 * r['tolerance']:.1f}% | "
                     f"{r.get('confidence', '')} | {'pass' if r['pass'] else 'FAIL'} |")
    return "\n".join(lines) + "\n"
