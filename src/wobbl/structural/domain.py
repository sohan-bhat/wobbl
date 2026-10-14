"""Static strength of a CAD part for wobbl simulate, and the part setup the other part analyses share."""
from __future__ import annotations

import re
import shlex
from pathlib import Path

from ..simulate import Analysis, Option, register

PART_KINDS = ("step", "stl", "brep", "iges", "parametric")
FORCE = r"\b\d+(?:\.\d+)?\s*(?:k?n|newtons?|lbf|lbs?|pounds?|kgf|kips?)\b"


def material(job, default: str = "6061-t6") -> str:
    """The material as given, read from the question, or the default with the guess recorded."""
    from ..ask import find_material

    m = job.value("material") or find_material(job.question)
    if m:
        job.use("material", m)
        return m
    return job.value("material", default, "material not mentioned")


def study(job, need_load: bool = True, need_fix: bool = True):
    """The structural study for a part analysis from the job's options, with what the question leaves out filled
    in the way 'wobbl simulate part.step "question"' does, each guess recorded with its exact override."""
    from ..ask import find_material, read_question
    from ..structural.run import describe_part
    from ..structural.spec import Load, StructuralStudy, parse_fixture, parse_load

    if job.path is None:
        raise ValueError(f"file not found: {job.thing}")
    num = job.number
    st = StructuralStudy(part=str(job.path), material=job.value("material") or find_material(job.question) or "auto",
                         target_sf=num("target-sf"), quality=job.value("quality") or "normal",
                         question=job.question, scale=num("scale", 1.0) or 1.0,
                         body=int(num("body")) if job.value("body") else None, mesh_size=num("mesh-size"))
    st.params = dict(p.split("=", 1) for p in job.values("param") if "=" in p)
    st.loads = [parse_load(x) for x in job.values("load")] if need_load else []
    st.fixtures = [parse_fixture(x) for x in job.values("fix")]
    pre = None
    gaps = st.material == "auto" or (need_load and not st.loads) or (need_fix and not st.fixtures)
    if gaps:
        pre = describe_part(st.part, job.out / "work", st.scale, st.body, st.params)
        placeholder = None
        if not need_load and not st.loads:
            placeholder = Load("gravity", magnitude=1.0)
            st.loads = [placeholder]
        if not need_fix and not st.fixtures:
            st.fixtures = [parse_fixture("fixed bottom")]
        guesses = read_question(job.focus, pre[1], st).guesses
        st.loads = [x for x in st.loads if x is not placeholder]
        for g in guesses:
            text = g.split(" (override with")[0]
            if g.startswith("material"):
                job.assume(text, f"--material {st.material}")
            elif g.startswith("supports"):
                job.assume(text, " ".join(f"--fix {shlex.quote(f.text)}" for f in st.fixtures))
            elif g.startswith("load"):
                newton = [f"{x.magnitude:.4g} N" for x in st.loads if x.kind in ("force", "bolt", "remote")
                          and not re.search(r"\d\s*n\b", x.text.lower())]
                job.assume(text + (f" ({', '.join(newton)})" if newton else ""),
                           " ".join(f"--load {shlex.quote(x.text)}" for x in st.loads))
            else:
                job.assume(text)
    job.use("material", st.material)
    if st.loads:
        job.use("load", [x.text for x in st.loads])
    if st.fixtures and need_fix:
        job.use("fix", [f.text for f in st.fixtures])
    job.situate(supports=[f.text for f in st.fixtures] if need_fix or job.values("fix") else [],
                loads=[x.text for x in st.loads], material=st.material)
    return st, pre


def stress(job):
    from ..materials import resolve_material
    from ..structural.pipeline import analyze_strength

    st, pre = study(job)
    resolve_material(st.material)
    st.budget = job.number("budget", 300.0)
    st.print_direction = job.value("print-direction")
    job.ready()
    if job.flag("shell"):
        from ..structural.shell_path import analyze_shell
        return analyze_shell(st, job.out, job.log, render=not job.quick, pre=pre)
    if job.flag("solid"):
        o = analyze_strength(st, job.out, job.log, render=not job.quick, fixes=not job.quick, pre=pre)
    else:
        # thin walled parts go to the shell model, checked against a coarse solid, and back to the solid if not
        from ..structural.shell_path import analyze_auto
        o = analyze_auto(st, job.out, job.log, render=not job.quick, fixes=not job.quick, pre=pre)
    if o.allowable_MPa:
        job.criterion("stress", "governing_MPa", o.allowable_MPa, "max")
    return o


register(Analysis(
    "part", "stress", "Static strength: peak stress, safety factor and deflection, with mesh convergence",
    stress, dry=True, kinds=PART_KINDS, default=True, legacy="wobbl run",
    keywords=(r"\bhold", r"\bcrack", r"\bbreak", r"\bsurviv", r"\bfail", r"\bbend", r"\bstress", r"safety factor",
              r"\bstrong", r"\bstrength", r"\bdeflect", r"\bsnap", FORCE, r"\b\d+(?:\.\d+)?\s*(?:mpa|kpa|psi|bar)\b",
              r"\b\d+(?:\.\d+)?\s*g'?s\b|\bg load"),
    aliases=("strength", "static", "structural"),
    options=(Option("shell", "Solve thin walls as shells on their midsurface, checked against a coarse solid (thin "
                             "walled parts get this on their own)", flag=True),
             Option("solid", "Solve the solid even when the walls are thin", flag=True),
             Option("budget", "Seconds for the convergence study", "300"),
             Option("print-direction", "Axis the print layers stack along, e.g. z")),
    example='wobbl simulate bracket.step "will it hold 200 N?"'))


def review(job):
    """A finished result asked a yes or no question: its verdict as it stands, or the same study run again with
    the options given, as a new load or material."""
    import json

    src = job.path
    if src is None or not (src / "outcome.json").exists():
        raise ValueError(f"no finished result in {job.thing}")
    if job.options and (src / "request.json").exists():
        from ..simulate import rerun
        job.ready()
        job.log("rerun", f"{src.name} with {', '.join(job.options)} changed")
        return rerun(src, job.out, job.options, job.quick)
    job.ready()
    d = json.loads((src / "outcome.json").read_text())
    job.out.mkdir(parents=True, exist_ok=True)
    d.setdefault("assumptions", []).insert(0, f"Wobbl read this verdict from the finished result in {src}.")
    d["workdir"] = str(job.out)
    for k in ("report",):
        if d.get(k) and not Path(d[k]).is_absolute():
            d[k] = str(src / d[k])
    (job.out / "outcome.json").write_text(json.dumps(d, indent=1))
    return d


register(Analysis(
    "part", "review", "A finished result's verdict, or the same study run again with what the question changes",
    review, dry=True, kinds=("result",), default=True,
    keywords=(r"\bhold", r"\bsurviv", r"\bfail", r"\bbreak", r"\bpass", r"\bsafe", r"\bok\b", r"\bverdict",
              r"safety factor", r"\bstrong", r"what if", r"\binstead\b"),
    aliases=("verdict", "result", "again"),
    example='wobbl simulate wobbl-out/bracket "does it still hold with 250 N?" --load "250 N down @ top"'))


register(Analysis(
    "part", "submodel", "Submodel: a fine local model of the peak, driven by the coarse result's displacements",
    "wobbl.structural.submodel:simulate_entry", kinds=PART_KINDS + ("result",), needs=("stress",),
    keywords=(r"\bsubmodel", r"\brefine\b.*\b(?:peak|locally|around|hot ?spot|stress)", r"\bzoom\b.*\bpeak",
              r"local(?:ly)?\s+(?:fine|refine)", r"\bfiner\b.*\b(?:peak|hole|fillet|corner)"),
    aliases=("submodeling", "local-refinement", "zoom"),
    options=(Option("at", "Center of the local model as x,y,z in mm", "the global peak"),
             Option("radius", "Radius of the local model in mm", "2.5 feature sizes"),
             Option("size", "Element size in the local model in mm", "a sixth of the feature size")),
    example='wobbl simulate output "refine the stress around the peak"'))
