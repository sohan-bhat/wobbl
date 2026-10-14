"""Entry points for crack growth that any front end can call: the command line, an agent tool or a general router.

    study(part, material, loads, fixtures, crack="auto", ...)   a study from plain strings
    from_result(result_dir, crack="auto", ...)                    a study that picks up a strength or fatigue result
    run(study, out_dir, progress=None)                            run it and get the outcome
    simulate(request, out_dir, progress=None)                     the same from one dict, for a router

A strength result gives the part, material, loads and supports, and its governing point becomes the crack's
origin. A fatigue result adds its duty cycle as the load spectrum, its point as the origin and its required life
as the target. The crack can still be given anywhere, and options given here win over what the result says.
"""
from __future__ import annotations

import json
from pathlib import Path

from .spec import FractureStudy, study_from

RESULT_DOMAINS = ("strength", "fatigue")


def study(part: str, material: str, loads, fixtures, crack: str = "auto", **options) -> FractureStudy:
    """A crack growth study from the strings a user writes. ``crack`` is 'auto' (the damage tolerance flaw at the
    most stressed point), 'auto at <point or faces>', or a crack such as 'through 1.27 mm at hole d=8'. Options are
    the FractureStudy fields: r_ratio, spectrum (a file or blocks like '100% x 3650 R=0; 70% x 1e8 R=0.7'),
    target_life, max_pof, law, growth, kc, wheeler, samples, steps, budget, quality, scale, params and the rest."""
    from ..units import parse_quantity

    if isinstance(options.get("detectable"), str):
        options["detectable"] = parse_quantity(options["detectable"], "length", "mm").value
    st = study_from(str(Path(part).resolve()), material, list(loads or []), list(fixtures or []), crack or "auto",
                    **{k: v for k, v in options.items() if v is not None})
    return st


def is_result(path) -> bool:
    p = Path(path)
    return p.is_dir() and (p / "outcome.json").exists()


def _resolve(base: Path, part: str) -> str:
    p = Path(part)
    if p.is_absolute():
        return str(p)
    for root in (base, Path.cwd()):
        q = (root / p).resolve()
        if q.exists():
            return str(q)
    return str((base / p).resolve())


def _strength_inputs(d: Path) -> dict:
    o = json.loads((d / "outcome.json").read_text())
    st = json.loads((d / "study.json").read_text()) if (d / "study.json").exists() else {}
    loads = [x["text"] if isinstance(x, dict) else x for x in (st.get("loads") or [])] or list(o.get("loads", []))
    fixtures = [x["text"] if isinstance(x, dict) else x for x in (st.get("fixtures") or [])] or \
        list(o.get("fixtures", []))
    xyz = o.get("governing_xyz")
    return {"part": _resolve(d, st.get("part") or o["part"]), "material": st.get("material") or o.get("material", ""),
            "loads": loads, "fixtures": fixtures, "scale": float(st.get("scale") or 1.0),
            "params": dict(st.get("params") or {}), "where": ",".join(f"{v:.6g}" for v in xyz) if xyz else "hottest",
            "question": o.get("question", "")}


def result_inputs(result_dir) -> dict:
    """What a strength or fatigue result says about the part, its loads, the origin and the load cycles."""
    d = Path(result_dir)
    if not is_result(d):
        raise ValueError(f"no Wobbl result in '{result_dir}'; pass the folder a strength or fatigue run wrote")
    o = json.loads((d / "outcome.json").read_text())
    dom = o.get("domain")
    if dom == "strength":
        return _strength_inputs(d)
    if dom == "fatigue":
        st = json.loads((d / "study.json").read_text()) if (d / "study.json").exists() else {}
        src = st.get("source")
        # the source folder is written as given on the command line, so it can be relative to the run's folder
        found = [q.resolve() for q in (d / str(src), d.parent / str(src), Path(str(src))) if src and is_result(q)]
        if not found:
            raise ValueError(f"the fatigue result in '{result_dir}' does not lead back to a strength result, so it "
                             "has no part to grow a crack in")
        out = _strength_inputs(found[0])
        v = o.get("values", {})
        if v.get("xyz"):
            out["where"] = ",".join(f"{x:.6g}" for x in v["xyz"])
        blocks = st.get("blocks") or []
        ratio = float(st.get("ratio") or 0.0)
        if blocks:
            from .growth import parse_blocks

            sp = parse_blocks(blocks, ratio)
            out["spectrum"] = "; ".join(blocks)
            out["target_life"] = float(st.get("repeats") or 1.0) * sp.cycles
        else:
            out["r_ratio"] = ratio
            cycles = st.get("cycles")
            if cycles:
                out["target_life"] = float(cycles)
        out["question"] = o.get("question", "") or out["question"]
        return out
    raise ValueError(f"crack growth starts from a part, a strength result or a fatigue result, and '{result_dir}' is "
                     f"a {dom} result")


def from_result(result_dir, crack: str = "auto", material: str = "", loads=None, fixtures=None,
                **options) -> FractureStudy:
    """A study built on a finished strength or fatigue result. An automatic crack starts at the result's governing
    point; anything given here replaces what the result says."""
    got = result_inputs(result_dir)
    crack = (crack or "auto").strip()
    if crack.lower() == "auto":
        crack = f"auto at {got['where']}"
    for key in ("scale", "params", "spectrum", "target_life", "r_ratio", "question"):
        if key in got and options.get(key) in (None, "", {}, ()):
            options[key] = got[key]
    return study(got["part"], material or got["material"], loads or got["loads"], fixtures or got["fixtures"], crack,
                 **options)


def run(st: FractureStudy, out_dir, progress=None):
    """Run a crack growth study and return its outcome, which is also written to ``out_dir``."""
    from .study import run_fracture
    return run_fracture(st, out_dir, progress)


def simulate(request: dict, out_dir, progress=None):
    """Run crack growth from one dict: 'part' (a CAD file) or 'result' (a strength or fatigue result folder), and
    optionally 'material', 'loads', 'fixtures', 'crack' and any study option."""
    req = dict(request)
    src = req.pop("result", None) or req.pop("source", None)
    part = req.pop("part", None)
    if src is None and part and is_result(part):
        src, part = part, None
    crack = req.pop("crack", "auto")
    if src:
        st = from_result(src, crack, req.pop("material", ""), req.pop("loads", None), req.pop("fixtures", None), **req)
    else:
        if not part:
            raise ValueError("crack growth needs a part file or a strength or fatigue result folder")
        st = study(part, req.pop("material", ""), req.pop("loads", []), req.pop("fixtures", []), crack, **req)
    return run(st, out_dir, progress)
