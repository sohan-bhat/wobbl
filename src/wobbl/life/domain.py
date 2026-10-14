"""Vibration, buckling and fatigue of any CAD part, for wobbl simulate.

Each analysis takes the part, its material, supports and loads the way the strength analysis does
(wobbl.structural.domain.study), so whatever the question or the options say about them reaches these too.
"""
from __future__ import annotations

import re

from ..simulate import Analysis, Option, register
from ..structural.domain import PART_KINDS, study


def _rpm(q: str) -> list[str]:
    return [f"{m.group(1)} rpm" for m in re.finditer(r"(\d+(?:\.\d+)?(?:\s*-\s*\d+(?:\.\d+)?)?)\s*rpm\b", q.lower())]


def cycles_in(q: str) -> str | None:
    """'a million cycles', '2e6 cycles', '500k cycles' or '10^6 times' as a number of cycles."""
    low = q.lower()
    m = re.search(r"(\d+(?:\.\d+)?(?:e\d+)?|a|one)?\s*(million|thousand|billion|k|m)?\s*(?:load )?(?:cycles|times)",
                  low)
    if not m or not (m.group(1) or m.group(2)):
        m2 = re.search(r"10\^(\d+)\s*(?:cycles|times)", low)
        return f"1e{m2.group(1)}" if m2 else None
    n = 1.0 if m.group(1) in (None, "a", "one") else float(m.group(1))
    n *= {"million": 1e6, "m": 1e6, "thousand": 1e3, "k": 1e3, "billion": 1e9}.get(m.group(2) or "", 1)
    return f"{n:g}"


def sweep_in(q: str, loads: bool, axis: str = "z") -> str | None:
    """A sine sweep the question describes: a band of frequencies said with 'sweep', 'sine' or 'shake'. With loads it
    sweeps them; without, a level like '2 g' makes it a sweep of the base."""
    low = q.lower()
    band = re.search(r"(\d+(?:\.\d+)?)\s*(?:-|to)\s*(\d+(?:\.\d+)?)\s*(k?hz)\b", low)
    if not band or not re.search(r"\bswe(?:ep|pt)|\bsine\b|\bshak", low):
        return None
    text = f"{band.group(1)}-{band.group(2)} {band.group(3)}"
    if loads:
        return text
    # a level said with the sweep ('2 g sine', '2 g z sweep') or after at or of ('swept at 2 g'), never a mass
    # like '45 g motor'
    g = re.search(r"\b(\d+(?:\.\d+)?)\s*g\s+(?:[xyz]\s+)?(?:sine|swe|shak|base|input|level|peak)", low) or \
        re.search(r"\b(?:at|of)\s+(\d+(?:\.\d+)?)\s*g\b(?!\s*[a-z])", low)
    return f"{g.group(1)} g {axis} {text}" if g else None


def _supports(base) -> list[str]:
    return [f.text for f in base.fixtures]


def vibration(job):
    from ..structural.spec import parse_load
    from .vibration import run_vibration
    from .vibspec import VibrationStudy, parse_sine

    base, pre = study(job, need_load=False)
    rpm = job.values("rpm") or _rpm(job.focus)
    if rpm:
        job.use("rpm", rpm)
    loads = job.values("load")
    sine = job.value("sine") or sweep_in(job.focus, bool(loads), job.value("axis") or "z") or ""
    if sine and not job.value("sine"):
        job.use("sine", sine)
    # loads shake the part only in a sweep of the loads (a band alone); with a base sweep or none they belong to
    # another analysis of the same question, such as its strength
    if sine and loads and parse_sine(sine).accel == 0:
        base.loads = [parse_load(x) for x in loads]
        job.use("load", loads)
    num = job.number
    st = VibrationStudy(base, masses=job.values("mass"), modes=int(num("modes", 15)), rpm=rpm,
                        blades=int(num("blades", 2)), frequencies=job.values("frequency"), margin=num("margin", 0.2),
                        sine=sine, psd=job.value("psd") or "", axis=job.value("axis") or "z",
                        duration=job.value("duration") or "", damping=num("damping"), target_sf=base.target_sf,
                        keep=job.flag("keep"))
    st.excitations()
    job.situate(supports=_supports(base), loads=[x.text for x in base.loads], objects=job.values("mass"),
                environment={k: v for k, v in (("rpm", rpm), ("sine", sine), ("psd", st.psd)) if v})
    job.ready()
    o = run_vibration(st, job.out, job.log, pre=pre, render=not job.quick)
    if o.values.get("sf") is not None:
        job.criterion("vibration", "sf", o.values["target_sf"], "min")
    return o


def buckling(job):
    from .buckling import BucklingStudy, run_buckling

    base, pre = study(job)
    st = BucklingStudy(base, base.target_sf, job.value("shape") or "auto", int(job.number("modes", 4)),
                       job.flag("keep"))
    job.situate(supports=_supports(base), loads=[x.text for x in base.loads])
    job.ready()
    o = run_buckling(st, job.out, job.log, pre=pre, render=not job.quick)
    if o.values.get("design_factor") is not None:
        job.criterion("buckling", "design_factor", o.values["target_sf"], "min")
    return o


def fatigue(job):
    from .fatigue_study import FatigueStudy, run_fatigue

    source = ""
    if job.kind == "result" and job.path is not None:
        source = str(job.path)
    elif job.earlier.get("stress") is not None or job.earlier.get("vibration") is not None:
        prior = job.earlier.get("stress") or job.earlier.get("vibration")
        source = str(getattr(prior, "workdir", "") or prior.get("workdir", ""))
    cycles = job.value("cycles") or cycles_in(job.focus)
    if cycles:
        job.use("cycles", cycles)
    else:
        cycles = job.value("cycles", "1e7", "required life not given")
    num = job.number
    st = FatigueStudy(source=source, material=job.value("material") or "", amplitude=job.value("amplitude") or "",
                      mean=job.value("mean") or "", kt=num("kt"), radius=job.value("radius") or "",
                      ratio=num("ratio", 0.0) or 0.0, cycles=cycles, blocks=job.values("block"),
                      repeats=num("repeats", 1.0) or 1.0, duration=job.value("duration") or "",
                      surface=job.value("surface") or "auto", size=job.value("size") or "",
                      loading=job.value("loading") or "auto", reliability=num("reliability", 0.9) or 0.9,
                      mean_model=job.value("mean-stress") or "goodman", target_sf=num("target-sf"),
                      question=job.question, history=job.value("history") or "")
    job.ready()
    o = run_fatigue(st, job.out, job.log, render=not job.quick)
    job.criterion("fatigue", "sf", o.values["target_sf"], "min")
    return o


PART_OPTS = (Option("keep", "Keep the solver's work files", flag=True),)

register(Analysis(
    "part", "vibration", "Natural frequencies against rotors and tones, and the stress of a sine sweep of the base "
    "or of the loads, or of random input", vibration, dry=True, kinds=PART_KINDS, legacy="wobbl vibration",
    keywords=(r"vibrat", r"resona", r"natural frequenc", r"\bmodes?\b", r"\bmodal\b", r"\brpm\b", r"\bshaker\b",
              r"sine sweep", r"random vibration", r"\bpsd\b", r"gevs", r"blade pass", r"\bhz\b", r"\bharmonic\b",
              r"forced response", r"\b(?:un|im)balance"),
    aliases=("modal", "resonance", "frequencies", "harmonic"),
    options=PART_OPTS + (
        Option("mass", "A mass the part carries, e.g. '45 g at 150,0,25 @ top'", multiple=True),
        Option("rpm", "Rotor speed or range, e.g. '4000-9000 rpm'", multiple=True),
        Option("blades", "Blades on the rotor, for blade pass", "2"),
        Option("frequency", "Other tone to avoid, e.g. '120 Hz'", multiple=True),
        Option("margin", "Keep natural frequencies this share away from each excitation", "0.2"),
        Option("sine", "Sine sweep of the base, e.g. '2 g z 10-500 Hz', or a band alone, '10-500 Hz', that sweeps "
                       "the loads as harmonic forces"),
        Option("psd", "Random input, e.g. gevs-qual"),
        Option("axis", "Axis of the random input", "z"), Option("duration", "Time at the input, e.g. '2 min'"),
        Option("damping", "Damping ratio, e.g. 0.02"), Option("modes", "Natural frequencies to find", "15")),
    example='wobbl simulate arm.step "does it resonate between 4000 and 9000 rpm?"'))

register(Analysis(
    "part", "buckling", "Buckling load factor with an imperfection knockdown for the shape", buckling,
    dry=True, kinds=PART_KINDS, covers=("stress",), legacy="wobbl buckling",
    keywords=(r"buckl", r"crippl", r"euler", r"\bcollapse under\b"),
    options=PART_OPTS + (Option("shape", "auto, column, plate, cylinder, dome or general", "auto"),
                         Option("modes", "Buckling modes to find", "4")),
    example='wobbl simulate strut.step "will it buckle under 2 kN?"'))

register(Analysis(
    "part", "fatigue", "Fatigue life: cycles before a crack starts and the safety factor on stress, under one load, "
    "load blocks or a load history", fatigue,
    dry=True, kinds=PART_KINDS + ("result",), needs=("stress",), legacy="wobbl fatigue",
    keywords=(r"fatigue", r"\bcycles?\b", r"how long (?:will|does) it last", r"\blifetime\b|\blife\b",
              r"\b(?:million|thousand) times\b", r"endurance", r"duty cycle", r"load history", r"rainflow"),
    aliases=("life", "cycles"),
    options=(Option("cycles", "Life the part needs, in cycles", "1e7"),
             Option("ratio", "Load ratio min over max: 0 on and off, -1 fully reversed", "0"),
             Option("block", "Duty cycle block, e.g. '100% x 2e4'", multiple=True),
             Option("history", "A load history to rainflow count: a file or numbers, multiples of the studied load"),
             Option("repeats", "Times the duty cycle or history runs", "1"),
             Option("amplitude", "Stress amplitude, e.g. '80 MPa'"),
             Option("mean", "Mean stress"), Option("kt", "Stress concentration at the notch"),
             Option("radius", "Notch radius"), Option("surface", "polished, machined, as-cast and so on", "auto"),
             Option("size", "Equivalent diameter of the section"), Option("loading", "bending, axial or torsion"),
             Option("reliability", "Share of parts that must survive", "0.9"),
             Option("mean-stress", "goodman, gerber or soderberg", "goodman"), Option("duration", "Time at the input")),
    example='wobbl simulate mount.step "will it last a million cycles at 150 N?"'))
