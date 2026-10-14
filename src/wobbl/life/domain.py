"""Vibration, buckling and fatigue of a CAD part, for wobbl simulate."""
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


def vibration(job):
    from .vibration import run_vibration
    from .vibspec import VibrationStudy

    base, _ = study(job, need_load=False)
    rpm = job.values("rpm") or _rpm(job.question)
    if rpm:
        job.use("rpm", rpm)
    num = job.number
    st = VibrationStudy(base, job.values("mass"), int(num("modes", 15)), rpm, int(num("blades", 2)),
                        job.values("frequency"), num("margin", 0.2), job.value("sine") or "", job.value("psd") or "",
                        job.value("axis") or "z", job.value("duration") or "", num("damping"), base.target_sf,
                        job.flag("keep"))
    st.excitations()
    job.situate(supports=[f.text for f in base.fixtures], loads=rpm + job.values("mass"))
    return run_vibration(st, job.out, job.log)


def buckling(job):
    from .buckling import BucklingStudy, run_buckling

    base, _ = study(job)
    st = BucklingStudy(base, base.target_sf, job.value("shape") or "auto", int(job.number("modes", 4)),
                       job.flag("keep"))
    job.situate(supports=[f.text for f in base.fixtures], loads=[x.text for x in base.loads])
    return run_buckling(st, job.out, job.log)


def fatigue(job):
    from .fatigue_study import FatigueStudy, run_fatigue

    source = ""
    if job.kind == "result" and job.path is not None:
        source = str(job.path)
    elif job.earlier.get("stress") is not None or job.earlier.get("vibration") is not None:
        prior = job.earlier.get("stress") or job.earlier.get("vibration")
        source = str(getattr(prior, "workdir", "") or prior.get("workdir", ""))
    cycles = job.value("cycles") or cycles_in(job.question)
    if cycles:
        job.use("cycles", cycles)
    else:
        cycles = job.value("cycles", "1e7", "required life not given")
    num = job.number
    st = FatigueStudy(source, job.value("material") or "", job.value("amplitude") or "", job.value("mean") or "",
                      num("kt"), job.value("radius") or "", num("ratio", 0.0) or 0.0, cycles, job.values("block"),
                      num("repeats", 1.0) or 1.0, job.value("duration") or "", job.value("surface") or "auto",
                      job.value("size") or "", job.value("loading") or "auto", num("reliability", 0.9) or 0.9,
                      job.value("mean-stress") or "goodman", num("target-sf"), job.question)
    return run_fatigue(st, job.out, job.log)


PART_OPTS = (Option("keep", "Keep the solver's work files", flag=True),)

register(Analysis(
    "part", "vibration", "Natural frequencies against rotors and tones, and the stress of a sine sweep or random "
    "input", vibration, kinds=PART_KINDS, legacy="wobbl vibration",
    keywords=(r"vibrat", r"resonan", r"natural frequenc", r"\bmodes?\b", r"\bmodal\b", r"\brpm\b", r"\bshaker\b",
              r"sine sweep", r"random vibration", r"\bpsd\b", r"gevs", r"blade pass", r"\bhz\b"),
    aliases=("modal", "resonance", "frequencies"),
    options=PART_OPTS + (
        Option("mass", "A mass the part carries, e.g. '45 g at 150,0,25 @ top'", multiple=True),
        Option("rpm", "Rotor speed or range, e.g. '4000-9000 rpm'", multiple=True),
        Option("blades", "Blades on the rotor, for blade pass", "2"),
        Option("frequency", "Other tone to avoid, e.g. '120 Hz'", multiple=True),
        Option("margin", "Keep natural frequencies this share away from each excitation", "0.2"),
        Option("sine", "Shaker sine sweep, e.g. '2 g z 10-500 Hz'"), Option("psd", "Random input, e.g. gevs-qual"),
        Option("axis", "Axis of the random input", "z"), Option("duration", "Time at the input, e.g. '2 min'"),
        Option("damping", "Damping ratio, e.g. 0.02"), Option("modes", "Natural frequencies to find", "15")),
    example='wobbl simulate arm.step "does it resonate between 4000 and 9000 rpm?"'))

register(Analysis(
    "part", "buckling", "Buckling load factor with an imperfection knockdown for the shape", buckling,
    kinds=PART_KINDS, covers=("stress",), legacy="wobbl buckling",
    keywords=(r"buckl", r"crippl", r"euler"),
    options=PART_OPTS + (Option("shape", "auto, column, plate, cylinder, dome or general", "auto"),
                         Option("modes", "Buckling modes to find", "4")),
    example='wobbl simulate strut.step "will it buckle under 2 kN?"'))

register(Analysis(
    "part", "fatigue", "Fatigue life: cycles before a crack starts and the safety factor on stress", fatigue,
    kinds=PART_KINDS + ("result",), needs=("stress",), legacy="wobbl fatigue",
    keywords=(r"fatigue", r"\bcycles?\b", r"how long (?:will|does) it last", r"\blifetime\b|\blife\b",
              r"\b(?:million|thousand) times\b", r"endurance", r"duty cycle"),
    aliases=("life", "cycles"),
    options=(Option("cycles", "Life the part needs, in cycles", "1e7"),
             Option("ratio", "Load ratio min over max: 0 on and off, -1 fully reversed", "0"),
             Option("block", "Duty cycle block, e.g. '100% x 2e4'", multiple=True),
             Option("repeats", "Times the duty cycle runs", "1"),
             Option("amplitude", "Stress amplitude, e.g. '80 MPa'"),
             Option("mean", "Mean stress"), Option("kt", "Stress concentration at the notch"),
             Option("radius", "Notch radius"), Option("surface", "polished, machined, as-cast and so on", "auto"),
             Option("size", "Equivalent diameter of the section"), Option("loading", "bending, axial or torsion"),
             Option("reliability", "Share of parts that must survive", "0.9"),
             Option("mean-stress", "goodman, gerber or soderberg", "goodman"), Option("duration", "Time at the input")),
    example='wobbl simulate mount.step "will it last a million cycles at 150 N?"'))

