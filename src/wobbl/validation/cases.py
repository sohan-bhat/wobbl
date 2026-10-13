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
    Case("slab", "Conduction through a slab into convection", "heat", "1D series resistance",
         "cooled face temperature", "C", 0.01, True, slab),
    Case("lumped", "Aluminum cube cooling in air", "heat", "lumped capacitance (Biot 2e-4)",
         "mean temperature after 10 min", "C", 0.01, True, lumped),
    Case("led", "Red LED on 5 V through 150 ohm", "circuit", "diode equation solved by Newton", "LED current",
         "mA", 0.005, True, led),
    Case("rc", "RC charging", "circuit", "V (1 - e^-1) at t = RC", "voltage at one time constant", "V", 0.002,
         True, rc),
    Case("buck", "Buck converter output ripple", "circuit", "ideal triangular current through C and ESR",
         "peak to peak ripple", "mV", 0.15, False, buck),
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
    lines = ["| Case | Quantity | Reference | Expected | Wobbl | Error | Tolerance | Result |",
             "|---|---|---|---|---|---|---|---|"]
    for r in results:
        if "wobbl" not in r:
            lines.append(f"| {r['title']} | | | | error | | | FAIL |")
            continue
        lines.append(f"| {r['title']} | {r['quantity']} | {r['source']} | {r['expected']:.4g} {r['unit']} | "
                     f"{r['wobbl']:.4g} {r['unit']} | {100 * r['error']:.2f}% | {100 * r['tolerance']:.1f}% | "
                     f"{'pass' if r['pass'] else 'FAIL'} |")
    return "\n".join(lines) + "\n"
