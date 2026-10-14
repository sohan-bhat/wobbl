"""wobbl simulate: one entry point that takes any thing and a question and works out the physics.

Register physics from a package's domain.py:

    from ..simulate import Analysis, Option, register

    register(Analysis(domain="part", name="drop", summary="Drop onto a hard floor", entry="wobbl.crash.run:drop",
                      keywords=(r"\\bdrop(?:ped|s)?\\b",), options=(Option("height", "Drop height", "1 m"),)))

and route or run with route(thing, question) and run_plan(plan, options, out).
"""
from .registry import Analysis, Domain, Option, analyses, domain, domains, errors, find, get, load, register, rerun
from .runner import Job, Step, grade, overall, run_plan

__all__ = ["Analysis", "Domain", "Option", "Job", "Step", "register", "run_plan", "analyses", "domain", "domains", "find",
           "get", "load", "rerun", "errors", "grade", "overall"]
