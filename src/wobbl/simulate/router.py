"""Work out what a thing is and which physics a question asks for, and say why, with the option that overrides it.

The thing can be a file (its suffix, and its content when several domains share a suffix), a folder (a finished
result or a solver case), or words: a description, a name, a formula, SMILES or an equation. Each domain scores
the thing with its own words and recognizer, the best score wins, and the analyses come from the keywords of the
question, from --physics when given, or from the domain's usual analysis for that kind of thing.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from . import registry
from .registry import Analysis, Domain


class RouteError(ValueError):
    def __init__(self, msg: str, hint: str = ""):
        super().__init__(msg)
        self.hint = hint


@dataclass
class Plan:
    thing: str
    question: str
    domain: str
    kind: str
    analyses: list[Analysis]
    reasons: list[tuple[str, str]] = field(default_factory=list)    # (what was decided and why, the override)
    candidates: list[tuple[str, int]] = field(default_factory=list)   # every domain that scored, best first

    @property
    def names(self) -> list[str]:
        return [a.name for a in self.analyses]

    def to_dict(self) -> dict:
        return {"thing": self.thing, "question": self.question, "domain": self.domain, "kind": self.kind,
                "analyses": self.names, "reasons": [{"decided": r, "override": o} for r, o in self.reasons],
                "candidates": [{"domain": d, "score": s} for d, s in self.candidates]}


def _search(rx: str, text: str):
    return re.search(rx, text, re.I) if text else None


def _as_path(thing: str) -> Path | None:
    if not thing or thing == "-" or "\n" in thing or len(thing) > 1024:
        return None
    try:
        p = Path(thing).expanduser()
        return p if p.exists() else None
    except OSError:
        return None


def _result_domain(p: Path) -> tuple[str, str] | None:
    """A finished Wobbl result: the domain it came from, from request.json or from outcome.json's domain."""
    if (p / "request.json").exists():
        try:
            return json.loads((p / "request.json").read_text())["domain"], "result"
        except (ValueError, KeyError):
            pass
    if (p / "outcome.json").exists():
        try:
            dom = json.loads((p / "outcome.json").read_text()).get("domain", "")
        except ValueError:
            return None
        for d in registry.domains():
            if dom in d.results or dom == d.name:
                return d.name, "result"
    return None


def _kind_from_words(d: Domain, text: str, question: str) -> str:
    for src in (text, question):
        for k, rx in d.kinds.items():
            if _search(rx, src):
                return k
    return d.default_kind


def score(d: Domain, thing: str, question: str, path: Path | None) -> tuple[int, str, str]:
    """How strongly the thing looks like this domain's, the kind of thing it is, and the reason in words."""
    if path is not None:
        if path.is_dir():
            kind = registry.call(d.sniff, path) if d.sniff else None
            return (100, kind, f"{path.name} is a {kind} folder, which {d.name} reads") if kind else (0, "", "")
        suf = path.suffix.lower()
        if suf not in d.extensions:
            return 0, "", ""
        kind = d.extensions[suf] or (registry.call(d.sniff, path) if d.sniff else None)
        return (100, kind, f"{path.name} is a {suf} file, read as a {d.name} ({kind})") if kind else (0, "", "")
    s, hits = 0, []
    for rx in d.words:
        m = _search(rx, thing)
        if m:
            s, hits = s + 3, hits + [m.group(0)]
        elif (m := _search(rx, question)):
            s, hits = s + 1, hits + [m.group(0)]
    kind = ""
    r = registry.call(d.recognize, thing, question) if d.recognize else None
    if r:
        s += r[0]
        kind = r[1]
        hits.insert(0, thing or question)
    if not s:
        return 0, "", ""
    kind = kind or _kind_from_words(d, thing, question)
    shown = ", ".join(f'"{h.strip()[:40]}"' for h in dict.fromkeys(hits[:3]))
    return s, kind, f"the words {shown} point to {d.name} ({kind})"


def recognize(thing: str, question: str = "", only: str | None = None) -> list[tuple[int, Domain, str, str]]:
    """Every domain that recognizes the thing, best first, as (score, domain, kind, why)."""
    path = _as_path(thing)
    if path is not None and path.is_dir() and not only:
        hit = _result_domain(path)
        if hit:
            return [(100, registry.domain(hit[0]), hit[1], f"{path.name} is a finished {hit[0]} result")]
    if path is None and thing and thing != "-":
        suffix = Path(thing).suffix.lower()
        if suffix and any(suffix in d.extensions for d in registry.domains()) and " " not in thing:
            raise RouteError(f"file not found: {thing}", "Check the path, or describe the thing in words instead.")
    text = "" if path is not None or thing == "-" else thing
    out = []
    for d in registry.domains():
        if only and d.name != only:
            continue
        s, kind, why = score(d, text, question, path)
        if s:
            out.append((s, d, kind, why))
    out.sort(key=lambda r: (r[0], r[1].priority), reverse=True)
    return out


def _pool(d: Domain, kind: str) -> list[Analysis]:
    every = registry.analyses(d.name)
    pool = [a for a in every if not a.kinds or kind in a.kinds]
    if not pool and every:
        kinds = sorted({k for a in every for k in a.kinds if k != "result"})
        raise RouteError(f"Wobbl has no {d.name} analysis for a {kind} yet",
                         f"Say which kind it is with --kind ({', '.join(kinds)}), or describe it more closely.")
    return pool


def choose(d: Domain, kind: str, text: str, physics: list[str] | None = None) -> tuple[list[Analysis], str]:
    """The analyses to run and the reason, from --physics, from the question's keywords or from the default."""
    pool = _pool(d, kind)
    if not pool:
        raise RouteError(f"Wobbl has no {d.name} analysis registered yet",
                         f"Analyses register in a domain.py; see 'wobbl simulate --list'.")
    if physics:
        chosen = []
        for name in physics:
            a = registry.find(name, pool=pool)
            if a is None:
                other = registry.find(name, d.name)
                why = f" for a {kind}" if other else ""
                raise RouteError(f"no physics '{name}' in {d.name}{why}",
                                 f"The {d.name} physics for a {kind} are {', '.join(a.name for a in pool)}.")
            chosen.append(a)
        reason = f"physics {', '.join(a.name for a in chosen)}, as asked"
    else:
        hits: dict[str, str] = {}
        for a in pool:
            for rx in a.keywords:
                m = _search(rx, text)
                if m:
                    hits[a.key] = m.group(0).strip()
                    break
        chosen = [a for a in pool if a.key in hits]
        covered = {c for a in chosen for c in a.covers}
        chosen = [a for a in chosen if a.name not in covered]
        if chosen:
            reason = "physics " + ", ".join(f'{a.name} from "{hits[a.key]}"' for a in chosen)
        else:
            chosen = [a for a in pool if a.default][:1] or pool[:1]
            reason = f"the question names no physics, so {chosen[0].name}, the usual one for a {kind}"
    ordered: list[Analysis] = []

    def add(a: Analysis, seen=()):
        for n in a.needs:
            dep = registry.find(n, pool=pool)
            if dep is not None and dep not in ordered and dep not in seen:
                add(dep, (*seen, a))
        if a not in ordered:
            ordered.append(a)

    for a in chosen:
        add(a)
    extra = [a.name for a in ordered if a not in chosen]
    if extra:
        reason += f", with {', '.join(extra)} first because the others read its result"
    return ordered, reason


def route(thing: str = "", question: str = "", physics: list[str] | None = None, domain: str | None = None,
          kind: str | None = None) -> Plan:
    """The plan for a thing and a question: its domain, its kind and the analyses to run, each with a reason."""
    thing, question = (thing or "").strip(), (question or "").strip()
    if not thing and not question:
        raise RouteError("nothing to simulate", "Pass a file, a name or formula, or describe the thing in words.")
    physics = [p for x in (physics or []) for p in re.split(r"[,+\s]+", x) if p]
    if domain:
        registry.domain(domain)
    found = recognize(thing, question, only=domain)
    if domain and not found:
        d = registry.domain(domain)
        found = [(0, d, d.default_kind, f"{domain} as asked")]
    if physics and not domain:   # physics that only one domain offers settles a close call
        offer = [r for r in found if all(registry.find(p, r[1].name) for p in physics)]
        found = offer or found
    if not found:
        names = ", ".join(d.name for d in registry.domains())
        raise RouteError(f"Wobbl could not tell what '{thing or question}' is",
                         f"Pass a CAD file, a netlist, a spec, a name, a formula or an equation, or describe it in "
                         f"words, or name the domain with --domain ({names}).")
    s, d, k, why = found[0]
    k = kind or k
    reasons = [(why if not domain else f"domain {d.name}, as asked", f"--domain {d.name}")]
    if kind:
        reasons.append((f"kind {kind}, as asked", f"--kind {kind}"))
    text = " ".join(x for x in (question, "" if _as_path(thing) else thing) if x)
    chosen, reason = choose(d, k, text, physics)
    reasons.append((reason, f"--physics {','.join(a.name for a in chosen)}"))
    return Plan(thing, question, d.name, k, chosen, reasons, [(r[1].name, r[0]) for r in found])
