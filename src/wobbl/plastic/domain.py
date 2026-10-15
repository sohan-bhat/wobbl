"""A part loaded past first yield, for wobbl simulate."""
from __future__ import annotations

from ..simulate import Analysis, Option, register
from ..structural.domain import PART_KINDS, study


def plastic(job):
    from .study import PlasticStudy, run_plastic

    base, pre = study(job)
    st = PlasticStudy(base.part, base.material, [x.text for x in base.loads], [f.text for f in base.fixtures],
                      job.number("target"), job.value("curve") or "auto", "fast" if job.quick else base.quality,
                      job.question, base.scale, base.body, base.params, job.flag("keep"), job.flag("playback"))
    job.situate(supports=[f.text for f in base.fixtures], loads=[x.text for x in base.loads])
    job.ready()
    o = run_plastic(st, job.out, job.log, pre=pre, render_images=not job.quick)
    if isinstance(o.values.get("collapse_factor"), (int, float)):
        job.criterion("collapse", "collapse_factor", 1.0, "min")
    return o


register(Analysis(
    "part", "plastic", "Past first yield: permanent set after unloading and the load that collapses the part",
    plastic, dry=True, kinds=PART_KINDS, covers=("stress",), legacy="wobbl plastic",
    keywords=(r"\bplastic(?:ity|ally)?\b(?! part)", r"permanent(?:ly)?\s+(?:bend|set|deform)",
              r"bend\w*(?:\s+\w+){0,3}\s+for good",
              r"past yield", r"\bcollapse", r"limit load",
              r"spr(?:ing|ung)\s*-?back"),
    aliases=("plasticity", "collapse", "yield"),
    options=(Option("target", "Margin wanted on the collapse load", "1.5"),
             Option("curve", "Past yield: auto (the material's true curve), perfect (flat at yield), 'true 250@0, "
                             "300@0.02' (true stress MPa at plastic strain), 'engineering 250@0.00125, 400@0.2' "
                             "(a tensile test) or a CSV file of either", "auto"),
             Option("playback", "Also write playback.json with the load going on and off, for a player", flag=True),
             Option("keep", "Keep the solver's work files", flag=True)),
    example='wobbl simulate hook.step "will 3 kN bend it for good?"'))
