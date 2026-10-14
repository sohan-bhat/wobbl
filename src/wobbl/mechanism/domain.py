"""Mechanisms for wobbl simulate: any URDF, MJCF or spec, and the ready made arm, linkage, gear train, impact,
tipping stand and pendulum described in words, each analysis named by its physics.

A model file runs the general analyses the question asks for (general.py). A mechanism described in words runs
the template that fits it, whose own analysis covers that physics in depth.
"""
from __future__ import annotations

import dataclasses

from ..simulate import Analysis, Option, register
from ..simulate.registry import domain
from .general import KEYWORDS

FILES = ("urdf", "mjcf", "spec")
GENERAL = (Option("payload", "A load it carries, e.g. '500 g', held at its end effector"),
           Option("payload-at", "The body that carries the payload, when not the end effector"),
           Option("servo", "A servo or motor from the catalog for every driven joint, e.g. 'mg996r' or 'xm430'"),
           Option("base", "'free' to stand a fixed base model loose on a floor"),
           Option("passive", "Nothing drives the joints", flag=True),
           Option("duration", "How long a run lasts, e.g. '2 s'"),
           Option("timestep", "Time step, e.g. '1 ms' (default the model's own, at most 2 ms)"),
           Option("set", "A value of a ready made mechanism, e.g. 'payload=500g' or 'crank=30'", multiple=True))
OBJECTS = {"arm": ["table", "payload"], "tip": ["floor", "payload"], "impact": ["floor"], "fourbar": [],
           "gears": [], "pendulum": []}


def _sniff(path):
    """The router's own recognizer, plus a spec that assembles its parts from CAD with [[part]] tables."""
    from ..simulate.things import _tables, sniff_mechanism

    kind = sniff_mechanism(path)
    if kind is None and path.suffix.lower() in (".toml", ".json") and _tables(path) & {"part", "cad"}:
        return "spec"
    return kind


register(dataclasses.replace(domain("mechanism"), sniff=_sniff))


def _params(job, path) -> dict:
    """The general values from the job's options and its words, each guess recorded with its override so a
    rerun repeats it and --plan shows it."""
    from .general import _payload_in, _servo_in, asked

    p = {}
    for opt, key in (("payload", "payload"), ("payload-at", "payload_at"), ("servo", "servo"), ("base", "base"),
                     ("duration", "duration"), ("timestep", "timestep")):
        v = job.value(opt)
        if v:
            p[key] = v
    words = job.focus
    servo = p.get("servo") or _servo_in(words)
    if servo and "servo" not in p:
        p["servo"] = servo
        job.use("servo", servo)
    if "payload" not in p and set(asked(words)) & {"torque", "stability", "kinematics"}:
        kg = _payload_in(words)
        if kg:
            p["payload"] = f"{kg * 1000:g} g"
            job.use("payload", p["payload"])
    if p.get("payload") and not p.get("payload_at"):
        from .model import describe, end_effector, load

        model = load(path).compile()
        body = describe(model).names["body"][end_effector(model, describe(model))[0]]
        p["payload_at"] = body
        job.assume(f"the payload is held at the end of the {body}, the end of the arm", f"--payload-at {body}")
    if job.flag("passive"):
        p["passive"] = True
    if job.quick:
        p["quick"] = True
    return p


def _template(job):
    """A ready made mechanism from words: the kind the router recognized, its values read from the description
    and this analysis's part of the question, each guess recorded with its override, and --set values on top."""
    from .study import run_mechanism
    from .tools import read_for

    text = " ".join(x for x in (job.thing if job.path is None and job.thing != "-" else "", job.focus) if x)
    params, guesses = read_for(job.kind, text)
    for g in guesses:
        what, _, override = g.partition(" (override with ")
        job.assume(what, override.rstrip(")") if override else "")
    for s in job.values("set"):
        if "=" in s:
            k, v = s.split("=", 1)
            params[k.strip()] = v.strip()
    job.use("set", [f"{k}={v}" for k, v in params.items()])
    job.situate(objects=OBJECTS.get(job.kind, []))
    job.ready()
    return run_mechanism(job.kind, params, job.out, job.focus, job.log)


def _general(name: str | None):
    """The entry for one physics on a model file, or for the usual set when ``name`` is None."""

    def run(job):
        from .study import run_mechanism

        if job.kind not in FILES:
            return _template(job)
        if job.path is None:
            raise ValueError(f"file not found: {job.thing}")
        params = _params(job, job.path)
        if name:
            params["analyses"] = name
        job.situate(objects=["payload"] if params.get("payload") else [])
        job.ready()
        o = run_mechanism(str(job.path), params, job.out, job.focus, job.log)
        v = o.values
        if set(v.get("analyses", [])) & {"contact", "stability"}:
            job.situate(objects=["floor"])
        tq = v.get("torque", v) if name in (None, "torque") else {}
        if isinstance(tq, dict) and tq.get("worst_usage") is not None:
            job.criterion("drive use", "worst_usage", 1.0, "max")
        return o

    return run


def _add(name: str, summary: str, template_kinds: tuple[str, ...], extra_keywords: tuple[str, ...] = (),
         aliases: tuple[str, ...] = (), example: str = "") -> None:
    register(Analysis("mechanism", name, summary, _general(name), keywords=KEYWORDS[name] + extra_keywords,
                      options=GENERAL, aliases=aliases, kinds=FILES + template_kinds, example=example, dry=True))


_add("torque", "Whether the motors hold and move it: each driven joint's worst torque against its torque speed "
               "limit, with fixes solved again", ("arm", "gears"),
     (r"output torque", r"\bgear ?(?:box|train|ratio)\b", r"\bkg\s*\*?\s*cm\b", r"\bstrong enough\b"),
     ("motors", "servos", "sizing"), 'wobbl simulate arm.urdf "can these servos lift 500 g?"')
_add("kinematics", "How far it reaches and through what range, and for a linkage its sweep, dead points and "
                   "transmission angle", ("fourbar",),
     (r"\bbind", r"\block(?:s|ing)? up\b", r"\bsweep", r"swing angle", r"\btoggle"),
     ("range", "reach", "workspace"), 'wobbl simulate arm.urdf "how far can it reach?"')
_add("collision", "Whether its parts run into each other or the floor anywhere in its range", (),
     (r"\binterference\b",), ("interference", "self-collision"),
     'wobbl simulate robot.urdf "does it collide with itself?"')
_add("stability", "Whether it tips over standing, and in which pose it comes closest", ("tip",),
     (), ("tipping", "tip"), 'wobbl simulate robot.urdf "will it tip over with 2 kg in the gripper?"')
_add("contact", "What each foot, wheel or contact carries once it settles, or the peak force of an impact",
     ("impact",), (r"\bimpact\b", r"peak (?:contact )?force", r"dropped onto|drop onto", r"\bbounce"),
     ("forces", "impact"), 'wobbl simulate go1.xml "what does each foot carry?"')
_add("dynamics", "The mechanism run as written or along its move: what each drive needs, what touches, whether a "
                 "balance holds", ("pendulum",),
     (r"\bperiod\b", r"\bin time\b", r"\bclose\b", r"spin.?up"), ("motion", "run"),
     'wobbl simulate pendulum.xml "does it fall over if nothing holds it?"')
register(Analysis("mechanism", "overview", "The usual analyses for the model: torque, reach and collisions for a "
                                           "fixed base, or contact, stability and stance torque for a robot that "
                                           "stands", _general(None), options=GENERAL, kinds=FILES, default=True,
                  legacy="wobbl mechanism", aliases=("survey", "all"), dry=True,
                  example="wobbl simulate robot.urdf"))
