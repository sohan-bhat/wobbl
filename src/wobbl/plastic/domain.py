"""A metal part loaded past first yield, for wobbl simulate."""
from __future__ import annotations

from ..simulate import Analysis, Option, register
from ..structural.domain import PART_KINDS, study


def plastic(job):
    from .study import PlasticStudy, run_plastic

    base, _ = study(job)
    st = PlasticStudy(base.part, base.material, [x.text for x in base.loads], [f.text for f in base.fixtures],
                      job.number("target"), job.value("curve") or "auto", base.quality, job.question, base.scale,
                      base.body, base.params, job.flag("keep"))
    job.situate(supports=[f.text for f in base.fixtures], loads=[x.text for x in base.loads])
    o = run_plastic(st, job.out, job.log)
    if isinstance(o.values.get("collapse_factor"), (int, float)):
        job.criterion("collapse", "collapse_factor", 1.0, "min")
    return o


register(Analysis(
    "part", "plastic", "Past first yield: permanent set after unloading and the load that collapses a metal part",
    plastic, kinds=PART_KINDS, covers=("stress",), legacy="wobbl plastic",
    keywords=(r"\bplastic(?:ity|ally)?\b(?! part)", r"permanent(?:ly)?\s+(?:bend|set|deform)",
              r"bend\w*(?:\s+\w+)?\s+for good",
              r"past yield", r"\bcollapse", r"limit load"),
    aliases=("plasticity", "collapse", "yield"),
    options=(Option("target", "Margin wanted on the collapse load", "1.5"),
             Option("curve", "auto (true stress strain) or perfect (flat at yield)", "auto"),
             Option("keep", "Keep the solver's work files", flag=True)),
    example='wobbl simulate hook.step "will 3 kN bend it for good?"'))
