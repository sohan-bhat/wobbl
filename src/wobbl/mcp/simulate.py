"""The MCP simulate tool: any thing and question in, the route, every guess with its override and each physics'
result out, with the 3D view first."""
from __future__ import annotations

import uuid
from pathlib import Path


def _err(e: Exception) -> dict:
    hint = getattr(e, "hint", "")
    return {"error": str(e).strip("'\""), **({"hint": hint} if hint and hint not in str(e) else {})}


def _get(o, k, default=None):
    return o.get(k, default) if isinstance(o, dict) else getattr(o, k, default)


def compact(o) -> dict:
    """What an agent reads first from any outcome: verdict, numbers, checks, fixes and file paths."""
    if _get(o, "domain") == "strength" and not isinstance(o, dict):
        from .tools import compact as strength
        return strength(o)
    values = _get(o, "values") or {}
    checks = _get(o, "checks") or []
    out = {"verdict": _get(o, "verdict", ""), "headline": _get(o, "headline", ""),
           "metrics": dict(_get(o, "metrics") or []),
           "values": {k: v for k, v in values.items() if not str(k).startswith("_") and not isinstance(v, (list, dict))
                      or k in ("events",)},
           "confidence": _get(o, "confidence", ""), "confidence_reasons": _get(o, "confidence_reasons") or [],
           "checks": [{"name": _get(c, "name"), "status": _get(c, "status"), "summary": _get(c, "summary")}
                      for c in checks],
           "assumptions": _get(o, "assumptions") or [],
           "fixes": [{k: f.get(k) for k in ("kind", "title", "summary") if k in f} for f in _get(o, "fixes") or []],
           "images": _get(o, "images") or {}, "report": _get(o, "report", ""), "result_dir": _get(o, "workdir", ""),
           **view_of(_get(o, "workdir", ""))}
    return out


def view_of(result_dir) -> dict:
    """Where a result's interactive 3D view is, for the agent to open for the user."""
    if not result_dir:
        return {}
    from ..viz import view_path
    p = view_path(result_dir)
    return {"view": str(p)} if p else {}


def simulate_tool(thing: str = "", question: str = "", physics: list[str] | None = None, domain: str = "",
                  kind: str = "", material: str = "", loads: list[str] | None = None,
                  fixtures: list[str] | None = None, options: dict | None = None, plan_only: bool = False,
                  quick: bool = False, estimate: bool = False, background: bool = False, remote: str = "") -> dict:
    from ..simulate import RouteError, overall, route
    from ..simulate.runner import run_plan, view
    from ..solvers.locate import wobbl_home

    try:
        plan = route(thing, question, list(physics or []), domain or None, kind or None)
    except (RouteError, KeyError) as e:
        return _err(e)
    opts: dict[str, list[str]] = {str(k): (list(map(str, v)) if isinstance(v, (list, tuple)) else [str(v)])
                                  for k, v in (options or {}).items()}
    if material:
        opts["material"] = [material]
    if loads:
        opts.setdefault("load", []).extend(loads)
    if fixtures:
        opts.setdefault("fix", []).extend(fixtures)
    from ..simulate.runner import settle_options
    takes = {o.name for a in plan.analyses for o in a.all_options()}
    bad = settle_options(plan, opts)
    if bad:
        return {"error": f"no analysis in this plan reads '{bad[0]}'", "plan": plan.to_dict(),
                "options_read": sorted(takes)}
    head = {"plan": plan.to_dict(),
            "decisions": [{"decided": r, "override": o} for r, o in plan.reasons]}
    if plan_only:
        from ..simulate.runner import preview
        steps = preview(plan, opts)
        return head | {"options_read": sorted(takes - {"*"}),
                       "assumptions": [{"physics": s.analysis.name, "assumed": t, "override": o}
                                       for s in steps for t, o in s.job.assumptions],
                       "read": [{"physics": s.analysis.name, "option": k, "values": v}
                                for s in steps for k, v in s.job.readings()],
                       "pending": {s.analysis.name: s.error or s.hint for s in steps if s.error or s.hint}}
    try:
        if estimate:
            from ..estimate import estimate_plan
            est = estimate_plan(plan, opts)
            return head | {"estimate": est if isinstance(est, dict) else est.to_dict()}
        if background or remote:
            from ..jobs.manage import start_plan
            return head | {"job": start_plan(plan, opts, on=remote)}
    except ImportError:
        return head | {"error": "this version of Wobbl cannot estimate or queue a simulate plan yet"}
    except Exception as e:  # noqa: BLE001
        return head | _err(e)
    name = Path(plan.thing).stem if plan.thing and Path(plan.thing).exists() else "simulation"
    out = wobbl_home() / "results" / f"{name}-{uuid.uuid4().hex[:6]}"
    try:
        steps = run_plan(plan, opts, out, None, quick)
    except Exception as e:  # noqa: BLE001
        return head | _err(e)
    page = view(out)
    used: dict[str, list[str]] = {}
    for s in steps:
        for k, v in s.job.used.items():
            used.setdefault(k, v)
    return {"view": str(page or ""), "verdict": overall(steps),
            **head,
            "assumptions": [{"physics": s.analysis.name, "assumed": t, "override": o}
                            for s in steps for t, o in s.job.assumptions],
            "explicit": {"thing": plan.thing, "physics": plan.names,
                         "options": {k: v if len(v) > 1 else v[0] for k, v in used.items()}},
            "steps": [{"physics": s.analysis.name, **({"error": s.error} if s.error else compact(s.outcome))}
                      for s in steps],
            "result_dir": str(out)}


def physics_tool(domain: str = "") -> dict:
    """Every domain, the kinds of thing it reads and its physics with their options."""
    from ..simulate import analyses, domains, errors
    return {"domains": [{"name": d.name, "reads": d.summary, "example": d.example,
                         "shared_options": [o.name for o in d.options],
                         "physics": [{"name": a.name, "summary": a.summary, "aliases": list(a.aliases),
                                      "kinds": list(a.kinds),
                                      "options": [{"name": o.name, "help": o.help, "default": o.default}
                                                  for o in a.options]} for a in analyses(d.name)]}
                        for d in domains() if not domain or d.name == domain],
            **({"load_errors": errors} if errors else {})}
