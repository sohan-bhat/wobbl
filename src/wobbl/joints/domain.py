"""Assemblies whose parts press on each other, slide, and are bolted together, for wobbl simulate."""
from __future__ import annotations

import shlex

from ..simulate import Analysis, Option, register
from ..structural.domain import PART_KINDS


def _bodies(job, name: str) -> dict:
    """'2=pc' or '2 pc' options as {body: value}."""
    out = {}
    for v in job.values(name):
        b, _, m = v.replace("=", " ").partition(" ")
        if b.strip().isdigit() and m.strip():
            out[int(b)] = m.strip()
    return out


def _common(job):
    from ..structural.domain import study

    base, _ = study(job)
    quality = "fast" if job.quick else base.quality
    job.situate(supports=[f.text for f in base.fixtures], loads=[x.text for x in base.loads])
    return base, quality


def contact(job):
    from .contact import ContactStudy, run_contact

    base, quality = _common(job)
    friction = float(job.value("friction", "0.2", "friction coefficient between the parts not given"))
    bonded = [tuple(int(x) for x in v.replace("-", ",").split(",")[:2]) for v in job.values("bond")]
    st = ContactStudy(base.part, base.material, [x.text for x in base.loads], [f.text for f in base.fixtures],
                      job.values("move"), friction, _bodies(job, "body-material"), bonded,
                      job.number("contact-size"), job.number("target-sf"), quality, job.question, base.scale,
                      base.params, job.flag("keep"), job.flag("playback"))
    o = run_contact(st, job.out, job.log, render_images=not job.quick)
    job.criterion("stress", "governing_MPa", o.values["governing_MPa"] * o.values["safety_factor"], "max")
    return o


def bolts(job):
    from .bolts import BoltStudy, run_bolts

    base, quality = _common(job)
    given = job.values("bolt")
    if not given:
        job.assume("bolts not given: assumed a class 8.8 bolt sized to its hole in every hole that runs through two "
                   "parts, tightened to 70% of its proof load", "--bolt " + shlex.quote("M8 8.8 torque 25 N*m"))
    friction = float(job.value("friction", "0.15", "friction coefficient between the clamped parts not given"))
    nut = float(job.value("nut-factor", "0.2", "nut factor K in T = K F d not given"))
    st = BoltStudy(base.part, base.material, given, [x.text for x in base.loads], [f.text for f in base.fixtures],
                   friction, nut, _bodies(job, "body-material"), [], quality, job.question, base.scale, base.params,
                   job.flag("keep"), job.flag("playback"))
    o = run_bolts(st, job.out, job.log, render_images=not job.quick)
    job.criterion("bolt proof use", "proof_use", 1.0, "max")
    job.criterion("clamp left", "least_clamp_N", 0.0, "min")
    return o


_SHARED = (Option("friction", "Friction coefficient between touching parts"),
           Option("body-material", "Material of one body of the assembly, as '2=pc'", multiple=True),
           Option("playback", "Also write playback.json with every increment, for a player", flag=True),
           Option("keep", "Keep the solver's work files", flag=True))

register(Analysis(
    "part", "contact", "Parts of an assembly that press on each other, slide and open, with friction",
    contact, kinds=PART_KINDS, covers=("stress",), legacy="",
    keywords=(r"\bcontact", r"\bpress(?:es|ed|ing)?\s+(?:on|against|into)\b", r"\bslid(?:e|es|ing)\b",
              r"\bslip(?:s|ping)?\b", r"\bfriction", r"\bclamp(?:s|ed|ing)?\b", r"\bpin\b", r"\binterference",
              r"press[- ]fit", r"\btouch(?:es|ing)?\b", r"\bhertz", r"\bbearing pressure"),
    aliases=("touching", "friction", "assembly"),
    options=_SHARED + (Option("move", "Prescribed movement of some faces after the loads, as '0.1 mm +x @ body 2 "
                                      "and x=min'", multiple=True),
                       Option("bond", "Two bodies glued together, as '1,2'", multiple=True),
                       Option("contact-size", "Element size on the contact faces, mm")),
    example='wobbl simulate clamp.step "does the jaw hold the rod with 2 kN, and how hard does it press?"'))

register(Analysis(
    "part", "bolts", "Bolted joints: preload from torque, then the load, checked for separation, slip and strength",
    bolts, kinds=PART_KINDS, covers=("stress", "contact"), legacy="",
    keywords=(r"\bbolt(?:s|ed)?\b", r"\bscrews?\b", r"\bfasten", r"\bpreload", r"\btighten", r"\bnuts?\b",
              r"\bflange", r"\bseparat", r"\bjoint\s+open", r"\bvdi\s*2230"),
    aliases=("bolt", "bolted", "fasteners", "joint"),
    options=_SHARED + (Option("bolt", "Bolts, as 'M8 8.8 @ holes d=9 torque 25 N*m'; sized from the holes when left "
                                      "out", multiple=True),
                       Option("nut-factor", "Nut factor K in T = K F d", "0.2")),
    example='wobbl simulate flange.step "do four M8 bolts at 25 N m keep it closed under 10 kN?"'))
