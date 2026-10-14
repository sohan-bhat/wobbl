"""Steady and transient heat in a CAD part, for wobbl simulate."""
from __future__ import annotations

from ..simulate import Analysis, Option, register
from ..structural.domain import PART_KINDS, material


def heat(job):
    from .study import run_thermal
    from .tools import read_thermal_question, study_from

    r = read_thermal_question(job.focus)
    power = job.value("power") or r.get("power")
    if not power:
        raise ValueError("the question has no heat load I can read. Say it like '12 W', or pass --power '12 W'")
    job.use("power", power)
    films = job.values("film")
    if not films:
        films = [r["film"]]
        job.assume(r["guesses"][0].split(" (override with")[0], f"--film '{r['film']}'")
        job.use("film", films)
    ambient = job.value("ambient") or r.get("ambient")
    if ambient:
        job.use("ambient", ambient)
    else:
        ambient = job.value("ambient", "25 C", "air temperature not given")
    limit = job.value("limit") or r.get("limit", "")
    if limit:
        job.use("limit", limit)
    duration = job.value("duration") or r.get("duration", "")
    if duration:
        job.use("duration", duration)
    mat = material(job)
    st = study_from(str(job.path), mat, power, job.value("heat-faces") or "", films, ambient,
                    job.values("fixed-temp"), duration, limit, job.number("radiation", 0.0) or 0.0, job.question,
                    job.value("quality") or "normal")
    job.situate(environment={"ambient": ambient, "cooling": films}, loads=[f"{power} of heat"])
    job.ready()
    o = run_thermal(st, job.out, job.log)
    if o.values.get("limit_C") is not None:
        job.criterion("temperature", "t_max_C", o.values["limit_C"], "max")
    return o


register(Analysis(
    "part", "heat", "Steady or transient heat: the hottest point against a temperature limit", heat,
    dry=True, kinds=PART_KINDS, legacy="wobbl thermal",
    keywords=(r"overheat", r"\bheat\b(?!\s*sink)", r"\bhot\b", r"temperature", r"thermal", r"\bcool", r"\bwarm",
              r"\b\d+(?:\.\d+)?\s*(?:k?w|watts?)\b", r"\b-?\d+(?:\.\d+)?\s*(?:°\s*)?[cf]\b(?![\w/])",
              r"celsius|fahrenheit|°"),
    aliases=("thermal", "temperature"),
    options=(Option("power", "Heat made inside the part, e.g. '12 W'"),
             Option("heat-faces", "Put the power on these faces instead of through the volume"),
             Option("film", "Convection, e.g. '5 W/m2K @ all' or '25 W/m2K at 30 C @ top'", multiple=True),
             Option("ambient", "Air temperature", "25 C"),
             Option("fixed-temp", "Held temperature, e.g. '40 C @ bottom'", multiple=True),
             Option("duration", "Transient run length, e.g. '30 min'. Left out, the run is steady"),
             Option("limit", "Temperature limit, e.g. '60 C' (default the material's)"),
             Option("radiation", "Surface emissivity, 0 turns radiation off", "0")),
    example='wobbl simulate pack.step "does it stay under 60 C at 12 W in a sealed box?"'))
