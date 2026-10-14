"""Composite layups on a CAD part's walls, for wobbl simulate."""
from __future__ import annotations

import re

from ..simulate import Analysis, Option, register
from ..structural.domain import PART_KINDS, study

LAYUP = r"\[[0-9\s±+\-/_:,.a-z]+\]\s*\d*s?"


def layup_in(question: str) -> str | None:
    m = re.search(LAYUP, question.lower())
    return m.group(0).replace(" ", "") if m else None


def ply_in(question: str) -> str | None:
    q = question.lower()
    for word, key in (("im7", "im7-8552"), ("as4", "as4-3501-6"), ("t300", "t300-5208"), ("glass", "e-glass-epoxy"),
                      ("fiberglass", "e-glass-epoxy"), ("gfrp", "e-glass-epoxy")):
        if word in q:
            return key
    return None


def composite(job):
    from .study import run_laminate
    from .tools import study_from

    # a layup is what the walls are made of, so it holds for every load the question gives, not one clause
    job.part = ""
    q = job.focus
    lay = job.value("layup") or layup_in(q)
    if lay:
        job.use("layup", lay)
    else:
        lay = job.value("layup", "[0/±45/90]s", "layup not given, so a quasi isotropic stack is assumed")
    ply = job.value("ply") or ply_in(q)
    if ply:
        job.use("ply", ply)
    else:
        ply = job.value("ply", "t300-5208", "ply material not given")
    # the plies are the material, so the strength study's material guess is skipped
    given = bool(job.options.get("material"))
    job.options.setdefault("material", [ply])
    st, pre = study(job)
    if not given:
        job.used.pop("material", None)
    lam = study_from(st.part, lay, ply, [x.text for x in st.loads], [f.text for f in st.fixtures],
                     job.number("ply-thickness"), st.target_sf, job.value("reference") or "",
                     "fast" if job.quick else "normal", q, st.params)
    lam.scale = st.scale
    job.situate(supports=[f.text for f in st.fixtures], loads=[x.text for x in st.loads])
    job.ready()
    o = run_laminate(lam, job.out, job.log, pre=pre, render=not job.quick)
    job.criterion("first ply failure", "strength_ratio", 1.0, "min")
    return o


register(Analysis(
    "part", "composite", "Composite layup on the walls: ply stresses, first ply failure by Tsai-Wu and Hashin, "
    "and layup changes that reach the target", composite, dry=True, kinds=PART_KINDS, default=False,
    covers=("stress",), covers_all=True,
    keywords=(r"\bcomposite\b", r"\bcarbon fib(?:er|re)\b", r"\bcfrp\b", r"\bgfrp\b", r"\bfiberglass\b",
              r"\bglass fib(?:er|re)\b", r"\blay ?up\b", r"\bplies\b", r"\bply\b", r"\blaminate\b", LAYUP),
    aliases=("laminate", "layup", "composites"), legacy="wobbl run --layup",
    options=(Option("layup", "Ply angles from the inner face out, e.g. '[0/±45/90]s'", "[0/±45/90]s"),
             Option("ply", "Ply material: t300-5208, im7-8552, as4-3501-6 or e-glass-epoxy", "t300-5208"),
             Option("ply-thickness", "Cured ply thickness in mm (default the material's)"),
             Option("reference", "Direction the 0 degree plies run along, e.g. x (default the long axis)")),
    example='wobbl simulate arm.step "carbon fiber [0/±45/0]s, 40 N at the tip, is it strong enough?"'))
