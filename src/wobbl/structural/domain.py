"""Static strength of a CAD part for wobbl simulate, and the part setup the other part analyses share."""
from __future__ import annotations

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
        guesses = read_question(job.question, pre[1], st).guesses
        st.loads = [x for x in st.loads if x is not placeholder]
        for g in guesses:
            text = g.split(" (override with")[0]
            if g.startswith("material"):
                job.assume(text, f"--material {st.material}")
            elif g.startswith("supports"):
                job.assume(text, " ".join(f"--fix {shlex.quote(f.text)}" for f in st.fixtures))
            elif g.startswith("load"):
                job.assume(text, " ".join(f"--load {shlex.quote(x.text)}" for x in st.loads))
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
    if job.flag("shell"):
        from ..structural.shell_path import analyze_shell
        return analyze_shell(st, job.out, job.log, render=not job.quick, pre=pre)
    o = analyze_strength(st, job.out, job.log, render=not job.quick, fixes=not job.quick, pre=pre)
    if o.allowable_MPa:
        job.criterion("stress", "governing_MPa", o.allowable_MPa, "max")
    return o


register(Analysis(
    "part", "stress", "Static strength: peak stress, safety factor and deflection, with mesh convergence",
    stress, kinds=PART_KINDS, default=True, legacy="wobbl run",
    keywords=(r"\bhold", r"\bcrack", r"\bbreak", r"\bsurviv", r"\bfail", r"\bbend", r"\bstress", r"safety factor",
              r"\bstrong", r"\bstrength", r"\bdeflect", r"\bsnap", FORCE, r"\b\d+(?:\.\d+)?\s*(?:mpa|kpa|psi|bar)\b",
              r"\b\d+(?:\.\d+)?\s*g'?s\b|\bg load"),
    aliases=("strength", "static", "structural"),
    options=(Option("shell", "Solve thin walls as shells on their midsurface, checked against a coarse solid",
                    flag=True),
             Option("budget", "Seconds for the convergence study", "300"),
             Option("print-direction", "Axis the print layers stack along, e.g. z")),
    example='wobbl simulate bracket.step "will it hold 200 N?"'))
