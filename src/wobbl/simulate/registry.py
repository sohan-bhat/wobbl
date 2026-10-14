"""One registry for every kind of physics Wobbl runs.

A domain says what kind of thing it takes (a CAD file, a netlist, a spec, a formula or a few words). An analysis
says what physics it runs on that thing, which words in a question ask for it, which options it reads and which
function runs it. Packages register both in a module named domain.py inside the package, and the registry imports
every such module the first time anyone asks, so new physics plugs into 'wobbl simulate' and the MCP simulate tool
without a new command or tool. Packages outside Wobbl can do the same through the 'wobbl.domains' entry point.
"""
from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


@dataclass(frozen=True)
class Option:
    """A value an analysis reads. It is '--name value' on the command line and options[name] over MCP."""
    name: str
    help: str = ""
    default: str | None = None     # what the entry assumes when it is left out, shown in the catalog
    multiple: bool = False         # may be given more than once, like --load
    flag: bool = False             # on or off, like --shell


@dataclass
class Domain:
    """A kind of thing to simulate and how to recognize it."""
    name: str
    summary: str
    extensions: dict[str, str] = field(default_factory=dict)   # file suffix to the kind of thing it holds
    sniff: Callable | str | None = None    # fn(path) -> kind or None, for folders and suffixes several domains use
    words: tuple[str, ...] = ()            # regexes that point to this domain in a description or a question
    kinds: dict[str, str] = field(default_factory=dict)        # kind to the regex that picks it from words
    recognize: Callable | str | None = None   # fn(thing, question) -> (score, kind) or None, beyond regexes
    options: tuple[Option, ...] = ()       # options every analysis in the domain reads
    default_kind: str = "words"
    results: tuple[str, ...] = ()         # the domain names its finished outcome.json files carry
    priority: int = 0                      # breaks a tie in favour of the higher one
    example: str = ""


@dataclass
class Analysis:
    """One kind of physics on one domain's things."""
    domain: str
    name: str                  # what --physics takes
    summary: str
    entry: Callable | str      # fn(job) -> outcome, or "module:function" imported only when it runs
    keywords: tuple[str, ...] = ()     # regexes on the question that ask for this analysis
    options: tuple[Option, ...] = ()
    aliases: tuple[str, ...] = ()      # other names --physics accepts
    kinds: tuple[str, ...] = ()        # the kinds of thing it applies to, all of the domain's when empty
    default: bool = False      # run it when the question asks for nothing in particular
    needs: tuple[str, ...] = ()        # analyses whose results it reads, run first
    covers: tuple[str, ...] = ()       # analyses it makes unnecessary when both are asked for
    legacy: str = ""           # the old command that ran the same study, kept as a hidden alias
    example: str = ""

    @property
    def key(self) -> str:
        return f"{self.domain}.{self.name}"

    def function(self) -> Callable:
        return _resolve(self.entry)

    def run(self, spec: dict, out_dir: str | Path, quick: bool = False) -> dict:
        """Run again from a request.json spec with nothing guessed, and return the outcome as outcome.json has it.
        quick skips images, reports and fixes where the analysis supports it."""
        from .runner import run_spec
        return run_spec(self, spec, Path(out_dir), quick)

    def takes(self, name: str) -> bool:
        return any(o.name == name for o in self.all_options())

    def all_options(self) -> tuple[Option, ...]:
        dom = _DOMAINS.get(self.domain)
        return self.options + (dom.options if dom else ())


_DOMAINS: dict[str, Domain] = {}
_ANALYSES: dict[str, Analysis] = {}
errors: dict[str, str] = {}       # module that failed to register, and why
_loaded = False


def register(item: Domain | Analysis) -> Domain | Analysis:
    """Add a domain or an analysis. Registering the same name again replaces it."""
    if isinstance(item, Domain):
        _DOMAINS[item.name] = item
    elif isinstance(item, Analysis):
        _ANALYSES[item.key] = item
    else:
        raise TypeError(f"register takes a Domain or an Analysis, not {type(item).__name__}")
    return item


def _resolve(ref: Callable | str | None) -> Callable | None:
    if ref is None or callable(ref):
        return ref
    mod, _, fn = ref.partition(":")
    return getattr(importlib.import_module(mod), fn)


def load() -> None:
    """Import every package's domain.py once, and any plugin on the 'wobbl.domains' entry point."""
    global _loaded
    if _loaded:
        return
    _loaded = True
    importlib.import_module("wobbl.simulate.things")   # the kinds of thing Wobbl knows how to read
    root = Path(__file__).resolve().parents[1]
    for pkg in sorted(p.parent.name for p in root.glob("*/domain.py")):
        try:
            importlib.import_module(f"wobbl.{pkg}.domain")
        except Exception as e:  # noqa: BLE001  one broken package must not take the others down
            errors[f"wobbl.{pkg}.domain"] = f"{type(e).__name__}: {e}"
    try:
        from importlib.metadata import entry_points
        for ep in entry_points(group="wobbl.domains"):
            try:
                ep.load()
            except Exception as e:  # noqa: BLE001
                errors[ep.value] = f"{type(e).__name__}: {e}"
    except Exception:  # noqa: BLE001
        pass
    importlib.import_module("wobbl.simulate.legacy").fill()   # bridges for domains nobody has registered yet


def domains() -> list[Domain]:
    load()
    return list(_DOMAINS.values())


def domain(name: str) -> Domain:
    load()
    if name not in _DOMAINS:
        raise KeyError(f"no domain '{name}'. The domains are {', '.join(_DOMAINS)}")
    return _DOMAINS[name]


def analyses(domain_name: str | None = None) -> list[Analysis]:
    load()
    return [a for a in _ANALYSES.values() if domain_name is None or a.domain == domain_name]


def find(name: str, domain_name: str | None = None, pool: list[Analysis] | None = None) -> Analysis | None:
    """The analysis a name or alias means, within a domain when one is given."""
    key = name.strip().lower().replace("_", "-").replace(" ", "-")
    pool = pool if pool is not None else analyses(domain_name)
    for a in pool:
        if key in (a.name, a.key) or key in (x.replace(" ", "-") for x in a.aliases):
            return a
    return None


def get(domain_name: str, name: str | None = None) -> Analysis:
    """The analysis by its domain and name, as get("part", "stress"), get("part.stress") or get("stress") when
    only one domain has it. Its .run(spec, out_dir, quick=False) runs a request.json spec again."""
    load()
    if name is None and "." in domain_name:
        domain_name, name = domain_name.split(".", 1)
    if name is None:
        hits = [a for a in _ANALYSES.values() if domain_name in (a.name, *a.aliases)]
        if len(hits) == 1:
            return hits[0]
        raise KeyError(f"'{domain_name}' names {len(hits)} analyses; ask with its domain, as get('part', 'stress')")
    a = find(name, domain_name)
    if a is None:
        raise KeyError(f"no analysis '{name}' in domain '{domain_name}'")
    return a


def rerun(result_dir: str | Path, out_dir: str | Path, changes: dict | None = None, quick: bool = False) -> dict:
    """Run a finished result again from its request.json, with some options changed."""
    import json
    req = json.loads((Path(result_dir) / "request.json").read_text())
    spec = dict(req["spec"])
    spec["options"] = {**spec.get("options", {}), **{k: v if isinstance(v, list) else [v]
                                                     for k, v in (changes or {}).items()}}
    return get(req["domain"], req["analysis"]).run(spec, out_dir, quick)


def call(ref: Callable | str | None, *args: Any) -> Any:
    fn = _resolve(ref)
    return fn(*args) if fn else None
