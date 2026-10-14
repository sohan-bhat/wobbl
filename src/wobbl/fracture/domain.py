"""Fatigue crack growth on a CAD part, for wobbl simulate.

The stress analysis runs first and its result supplies the part, material, loads, supports and the governing
point, where an automatic crack starts. A strength or fatigue result folder works the same way, and a fatigue
result also brings its duty cycle and required life.
"""
from __future__ import annotations

from pathlib import Path

from ..life.domain import cycles_in
from ..simulate import Analysis, Option, register
from ..structural.domain import PART_KINDS


def _source(job) -> str | None:
    """The result folder the crack study builds on: the one given, or the stress result run just before."""
    if job.kind == "result" and job.path is not None:
        return str(job.path)
    prior = job.earlier.get("stress")
    if prior is None:
        return None
    work = getattr(prior, "workdir", "") if not isinstance(prior, dict) else prior.get("workdir", "")
    return str(work) if work and (Path(work) / "outcome.json").exists() else None


def crack(job):
    from . import api

    num, val = job.number, job.value
    blocks = job.values("block")
    spectrum = "; ".join(blocks) if blocks else (val("spectrum") or "")
    target = num("target-life")
    if target is None and cycles_in(job.question):
        target = float(cycles_in(job.question))
        job.use("target-life", f"{target:g}")
    opts = {"r_ratio": num("r-ratio"), "spectrum": spectrum, "target_life": target, "max_pof": num("max-pof"),
            "law": val("law"), "growth": val("growth") or "", "kc": num("kc"), "wheeler": num("wheeler"),
            "detectable": val("detectable"), "samples": int(num("samples", 4000 if job.quick else 20000)),
            "steps": int(num("steps", 5 if job.quick else 12)), "budget": num("budget", 1200.0),
            "quality": "fast" if job.quick else (val("quality") or "normal"), "question": job.question,
            "keep": job.flag("keep")}
    opts = {k: v for k, v in opts.items() if v not in (None, "")}
    where = val("crack") or "auto"
    src = _source(job)
    if src is not None:
        st = api.from_result(src, where, val("material") or "", job.values("load") or None,
                             job.values("fix") or None, **opts)
    else:
        from ..structural.domain import study

        base, _ = study(job)
        params = dict(p.split("=", 1) for p in job.values("param") if "=" in p)
        st = api.study(base.part, base.material, [x.text for x in base.loads], [f.text for f in base.fixtures],
                       where, scale=base.scale, params=params, **opts)
    if where == "auto":
        job.assume(f"crack not given: the damage tolerance flaw at {st.crack.where}", "--crack 'through 1.27 mm at "
                   "hole d=8'")
    if not st.spectrum and num("r-ratio") is None:
        job.assume(f"load cycles not given: constant amplitude at R = {st.r_ratio:g}", f"--r-ratio {st.r_ratio:g}")
    job.use("crack", st.crack.text or where)
    job.use("material", st.material)
    job.use("load", [x.text for x in st.loads])
    job.use("fix", [f.text for f in st.fixtures])
    job.situate(supports=[f.text for f in st.fixtures], loads=[x.text for x in st.loads],
                crack={"start": st.crack.text or where, "spectrum": st.spectrum or f"R = {st.r_ratio:g}"})
    o = api.run(st, job.out, job.log, job.quick)
    if st.target_life:
        job.criterion("crack life", "nominal_cycles", st.target_life, "min")
    return o


register(Analysis(
    "part", "crack", "Fatigue crack growth: where a crack goes, cycles until it fails the part and how lives spread",
    crack, kinds=PART_KINDS + ("result",), needs=("stress",), legacy="wobbl crack",
    keywords=(r"crack\w*\s+(?:grow|growth|propagat|from|at|in)\b", r"\bgrows?\s+critical", r"critical crack",
              r"\bflaws?\b", r"\binspect", r"damage toleran", r"fracture mechanic", r"stress intensity",
              r"\bpropagat", r"\bparis law\b", r"\bnasgro\b"),
    aliases=("crack-growth", "fracture", "damage-tolerance"),
    options=(
        Option("crack", "Starting crack: auto, 'auto at 30,0,5', 'through 1.27 mm at hole d=8', 'corner 1 mm at "
                        "hole d=8' or 'surface a=1 c=2 at hottest'", "auto"),
        Option("r-ratio", "Valley over peak load for constant amplitude cycles", "0.1"),
        Option("spectrum", "Load cycles: a file of 'max,min[,count]' factors, or '100% x 3650 R=0; 70% x 1e8'"),
        Option("block", "A load block, e.g. '100% x 2e4 R=0' (repeatable)", multiple=True),
        Option("target-life", "Cycles every part should last"),
        Option("max-pof", "Share of parts allowed to fail before the target", "0.01"),
        Option("law", "nasgro or paris", "nasgro"),
        Option("growth", "Measured constants 'C=2.7e-8,n=3.7,dkth=1.8,kic=29' (mm/cycle, MPa sqrt(m))"),
        Option("kc", "Fracture toughness, MPa sqrt(m)"),
        Option("wheeler", "Wheeler retardation exponent from tests", "0"),
        Option("detectable", "Crack size an inspection finds reliably", "1.27 mm"),
        Option("samples", "Parts in the Monte Carlo", "20000"),
        Option("steps", "Most crack growth steps to solve", "12"),
        Option("budget", "Seconds for the crack path", "1200"),
        Option("keep", "Keep the solver's work files", flag=True)),
    example='wobbl simulate bracket.step "how long until a crack from the bolt hole grows critical?"'))

