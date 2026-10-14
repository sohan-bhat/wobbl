"""Agent facing operations behind the MCP server. Plain functions returning compact dicts.

Results stay small (numbers, statuses, one line summaries and file paths) so an
agent can read them in one go; images and the HTML report are files the agent
can open when it needs them.
"""
from __future__ import annotations

import json
import tempfile
import time
import uuid
from pathlib import Path

from ..solvers.locate import wobbl_home


def _studies() -> Path:
    p = wobbl_home() / "studies"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _results_root() -> Path:
    p = wobbl_home() / "results"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _err(e: Exception) -> dict:
    hint = getattr(e, "hint", "")
    return {"error": str(e), **({"hint": hint} if hint and hint not in str(e) else {})}


def inspect_geometry(path: str, scale: float = 1.0) -> dict:
    """Faces, holes, fillets, sharp corners, wall thickness and selector hints for a CAD file."""
    from ..structural.run import describe_part

    try:
        with tempfile.TemporaryDirectory() as tmp:
            prep, feat = describe_part(path, tmp, scale)
    except Exception as e:  # noqa: BLE001
        return _err(e)
    d = feat.to_dict(max_faces=40)
    d["notes"] = prep.notes
    d["units"] = "mm"
    d["selector_examples"] = _selector_examples(feat)
    return d


def _selector_examples(feat) -> list[str]:
    ex = ["top", "bottom", "x=min", "x=max", "normal -z"]
    holes = feat.holes()
    if holes:
        ds = sorted({round(2 * h.radius, 2) for h in holes})
        ex += [f"holes d={ds[0]:g}", f"bolt holes d={ds[-1]:g}"]
    if feat.fillets():
        ex.append("fillets")
    return ex


def mesh_part(path: str, size: float | None = None) -> dict:
    """Build the first level second order mesh and report its size and element counts."""
    from ..mesh.mesher import Sizing, build_mesh
    from ..structural.run import describe_part

    try:
        with tempfile.TemporaryDirectory() as tmp:
            prep, feat = describe_part(path, tmp)
            from ..mesh.mesher import auto_sizing
            s = auto_sizing(prep, thickness=feat.thickness_typical) if size is None else Sizing(h=float(size))
            m = build_mesh(prep, s)
    except Exception as e:  # noqa: BLE001
        return _err(e)
    return {"nodes": m.n_nodes, "elements": m.n_elems, "element_type": "10 node tetrahedron",
            "global_size_mm": round(s.h, 4), "volume_mm3": round(m.volume(), 3), "mesh_time_s": round(m.t_mesh, 2),
            "notes": m.notes + ["Studies refine this mesh automatically until the answer converges."]}


# --- stateful studies ---------------------------------------------------------
def new_study(path: str, material: str = "", question: str = "") -> dict:
    sid = uuid.uuid4().hex[:8]
    st = {"id": sid, "part": str(Path(path).expanduser().resolve()), "material": material or "auto",
          "loads": [], "fixtures": [], "question": question, "created": time.time()}
    (_studies() / f"{sid}.json").write_text(json.dumps(st))
    return {"study_id": sid, "part": st["part"],
            "next": "add supports with add_fixture, loads with apply_load, then solve_structural"}


def _get(sid: str) -> dict:
    p = _studies() / f"{sid}.json"
    if not p.exists():
        raise KeyError(f"no study '{sid}'; create one with new_study")
    return json.loads(p.read_text())


def _put(st: dict):
    (_studies() / f"{st['id']}.json").write_text(json.dumps(st))


def set_material(study_id: str, material: str) -> dict:
    from ..materials import resolve_material

    try:
        m = resolve_material(material)
        st = _get(study_id)
    except Exception as e:  # noqa: BLE001
        return _err(e)
    st["material"] = material
    _put(st)
    return {"study_id": study_id, "material": m.summary()}


def apply_load(study_id: str, load: str) -> dict:
    from ..structural.spec import parse_load

    try:
        ld = parse_load(load)
        st = _get(study_id)
    except Exception as e:  # noqa: BLE001
        return _err(e)
    st["loads"].append(load)
    _put(st)
    return {"study_id": study_id, "load": {"kind": ld.kind, "magnitude_internal": ld.magnitude,
                                           "direction": ld.direction, "faces": ld.faces}, "loads": st["loads"]}


def add_fixture(study_id: str, fixture: str) -> dict:
    from ..structural.spec import parse_fixture

    try:
        fx = parse_fixture(fixture)
        st = _get(study_id)
    except Exception as e:  # noqa: BLE001
        return _err(e)
    st["fixtures"].append(fixture)
    _put(st)
    return {"study_id": study_id, "fixture": {"kind": fx.kind, "faces": fx.faces}, "fixtures": st["fixtures"]}


def solve_structural(study_id: str = "", path: str = "", material: str = "", loads: list[str] | None = None,
                     fixtures: list[str] | None = None, target_sf: float | None = None, quality: str = "normal",
                     budget: float = 300.0, question: str = "", print_direction: str | None = None) -> dict:
    """Run the verified strength study. Either pass study_id or path + material + loads + fixtures."""
    from ..structural.pipeline import analyze_strength
    from ..structural.spec import StructuralStudy, parse_fixture, parse_load

    try:
        if study_id:
            st = _get(study_id)
            path = path or st["part"]
            material = material or st["material"]
            loads = (loads or []) + st["loads"]
            fixtures = (fixtures or []) + st["fixtures"]
            question = question or st.get("question", "")
        if not path:
            return {"error": "pass study_id, or path with loads and fixtures"}
        study = StructuralStudy(part=str(Path(path).expanduser().resolve()), material=material or "auto",
                                loads=[parse_load(x) for x in loads or []],
                                fixtures=[parse_fixture(x) for x in fixtures or []],
                                target_sf=target_sf, quality=quality, budget=budget, question=question,
                                print_direction=print_direction)
        pre = None
        guesses = []
        if study.material == "auto" or not study.loads or not study.fixtures:
            from ..ask import read_question
            from ..structural.run import describe_part
            out = _results_root() / f"{Path(path).stem}-{uuid.uuid4().hex[:6]}"
            pre = describe_part(study.part, out / "work")
            guesses = read_question(question, pre[1], study).guesses
        else:
            out = _results_root() / f"{Path(path).stem}-{uuid.uuid4().hex[:6]}"
        o = analyze_strength(study, out, pre=pre)
    except Exception as e:  # noqa: BLE001
        return _err(e)
    if guesses:
        o.assumptions = [f"Wobbl guessed: {g}." for g in guesses] + o.assumptions
        from ..structural.report import write_outputs
        write_outputs(o, out)
    return compact(o)


def compact(o) -> dict:
    return {
        "verdict": o.verdict, "headline": o.headline,
        "governing_stress_MPa": round(o.governing_MPa, 4), "where": o.governing_where,
        "location_mm": [round(v, 3) for v in o.governing_xyz], "measure": o.measure,
        "safety_factor": round(o.safety_factor, 3), "target_safety_factor": o.target_sf,
        "max_deflection_mm": round(o.max_disp_mm, 5),
        "confidence": o.confidence, "confidence_reasons": o.confidence_reasons,
        "checks": [{"name": c.name, "status": c.status, "summary": c.summary} for c in o.checks],
        "assumptions": o.assumptions,
        "fixes": [{"kind": f["kind"], "title": f["title"], "summary": f["summary"], "auto": f.get("auto", False)}
                  for f in o.fixes],
        "images": o.images, "report": o.report, "result_dir": o.workdir,
        "mesh_levels": [{"nodes": L["n_nodes"], "governing_MPa": round(L["governing"], 4)} for L in o.levels],
        "time_s": round(o.t_total, 1),
    }


def _outcome(result_dir: str):
    from ..structural.outcome import Outcome

    p = Path(result_dir) / "outcome.json"
    if not p.exists():
        raise FileNotFoundError(f"no outcome.json in {result_dir}")
    d = json.loads(p.read_text())
    if d.get("domain") == "strength":
        return Outcome.load(p)
    return d


def verify(result_dir: str) -> dict:
    """The verification checks and confidence of a finished study."""
    try:
        o = _outcome(result_dir)
    except Exception as e:  # noqa: BLE001
        return _err(e)
    if isinstance(o, dict):
        return {k: o.get(k) for k in ("confidence", "confidence_reasons", "checks", "assumptions")}
    return {"confidence": o.confidence, "confidence_reasons": o.confidence_reasons,
            "checks": [c.to_dict() for c in o.checks], "assumptions": o.assumptions,
            "excluded_peaks": [h for h in o.hotspots if h["excluded"]]}


def render(result_dir: str) -> dict:
    """Paths of the heatmaps and report of a finished study (stress, zoom, safety, deformed, convergence)."""
    try:
        o = _outcome(result_dir)
    except Exception as e:  # noqa: BLE001
        return _err(e)
    if isinstance(o, dict):
        return {"images": o.get("images", {}), "report": o.get("report", "")}
    return {"images": o.images, "report": o.report}


_RERUN_TOOL = {"thermal": "solve_thermal", "circuit": "solve_circuit", "motion": "solve_motion",
               "flow": "solve_flow", "motor": "solve_motor", "combustion": "solve_flame or solve_propellant"}
_DOMAIN_WORD = {"thermal": "heat", "circuit": "circuit", "motion": "motion", "flow": "flow", "motor": "motor",
                "combustion": "combustion"}
_RERUN_TOOL.update({"buckling": "solve_buckling", "vibration": "solve_vibration", "fatigue": "fatigue_life"})
_DOMAIN_WORD.update({"buckling": "buckling", "vibration": "vibration", "fatigue": "fatigue"})
_RERUN_TOOL.update({"plastic": "solve_plastic"})
_DOMAIN_WORD.update({"plastic": "plasticity"})
_RERUN_TOOL.update({"mechanism": "solve_mechanism"})
_DOMAIN_WORD.update({"mechanism": "mechanism"})
_RERUN_TOOL.update({d: f"solve_{d}" for d in ("magnetic", "electric", "board", "antenna", "sound")})
_DOMAIN_WORD.update({"magnetic": "magnetic", "electric": "electric field", "board": "board", "antenna": "antenna",
                     "sound": "sound"})


def suggest_fix(result_dir: str, apply: bool = False, which: list[str] | None = None) -> dict:
    """Fix suggestions with numbers for any finished study. For a strength result, apply=True builds the automatic
    fixes, solves each one again and compares. Heat, circuit and motion fixes change the inputs rather than the
    part, so they come back as suggestions with a note on which tool to call again."""
    try:
        o = _outcome(result_dir)
    except Exception as e:  # noqa: BLE001
        return _err(e)
    if isinstance(o, dict):
        return _input_fixes(o, apply)
    if apply:
        try:
            from ..fix.apply import apply_fixes
            return apply_fixes(result_dir, which)
        except Exception as e:  # noqa: BLE001
            return _err(e)
    return {"domain": "strength", "verdict": o.verdict, "safety_factor": round(o.safety_factor, 3),
            "fixes": o.fixes}


def _input_fixes(d: dict, apply: bool) -> dict:
    """The fixes of a heat, circuit or motion result. They change the cooling, the power, a part value or a
    rocket's nose weight, fins, rail or parachute, which only a new solve with those inputs can check."""
    dom = d.get("domain", "")
    tool = _RERUN_TOOL.get(dom, "the solve tool")
    fixes = [{"kind": f.get("kind", ""), "title": f.get("title", ""), "summary": f.get("summary", ""),
              "auto": False} for f in d.get("fixes") or []]
    out = {"domain": dom, "verdict": d.get("verdict", ""), "headline": d.get("headline", ""), "fixes": fixes}
    word = f"{_DOMAIN_WORD[dom]} " if dom in _DOMAIN_WORD else ""
    out["note"] = (f"Each {word}fix changes an input, so check it by calling {tool} again with the new value."
                   if fixes else "This result has no fixes to suggest.")
    if apply:
        out["applied"] = False
        out["note"] = "Wobbl builds and solves fixes by itself only for strength results. " + out["note"]
    return out


def quantify_uncertainty(result_dir: str, vary: list[str], samples: int = 200000, solves: int | None = None) -> dict:
    """Carry input scatter through a finished study: distributions, probability of failure and Sobol indices."""
    from ..credibility import uq
    from ..credibility.uncertainty import run_uncertainty

    try:
        r = run_uncertainty(result_dir, vary, samples, solves=solves)
    except Exception as e:  # noqa: BLE001
        return _err(e)
    return {"headline": r["headline"], "model": r["model"]["method"],
            "probability": {k: {"value": p["value"], "ci95": p["ci95"], "against": p["against"],
                                "in_words": uq.odds(p)} for k, p in r["probability"].items()},
            "outputs": {k: {x: round(v[x], 6) for x in ("mean", "sd", "p05", "p95")} | {"unit": v["unit"]}
                        for k, v in r["outputs"].items()},
            "sensitivity": [{"input": x["input"], "first_order": round(x["first_order"], 4),
                             "total": round(x["total"], 4)} for x in r["sensitivity"]["indices"]],
            "alternatives": [{"label": a["label"], "probability": a["probability"]["value"]}
                             for a in r.get("alternatives", [])],
            "files": {"uncertainty": str(Path(result_dir) / "uncertainty.json"),
                      "report": str(Path(result_dir) / "report.html")}}


def validate_against_tests(result_dir: str, tests: str, vary: list[str] | None = None,
                           load_uncertainty: float = 0.01) -> dict:
    """Compare a finished study with physical test readings (ASME V&V 20 and the area metric) and calibrate."""
    from ..credibility.against import run_validation

    try:
        r = run_validation(result_dir, tests, vary or [], load_uncertainty)
    except Exception as e:  # noqa: BLE001
        return _err(e)
    cal = r.get("calibration") or {}
    return {"headline": r["headline"], "synthetic": r["tests"]["synthetic"], "summary": r["summary"],
            "readings": [{k: x[k] for k in ("name", "quantity", "load", "value", "simulation", "error", "u_val",
                                            "within", "verdict")} for x in r["readings"]],
            "u_pooled_area": r["u_pooled"],
            "calibration": [{k: c[k] for k in ("input", "prior_mean", "mean", "sd", "ci95", "narrowed")}
                            for c in cal.get("inputs", [])],
            "notes": r["notes"], "files": {"validation": str(Path(result_dir) / "validation.json"),
                                           "report": str(Path(result_dir) / "report.html")}}


def assess_credibility(result_dir: str, influence: str = "medium", consequence: str = "medium",
                       question: str = "") -> dict:
    """A V&V 40 style credibility assessment of a finished study, with the test plan and credibility.html."""
    from ..credibility.assess import run_credibility

    try:
        r = run_credibility(result_dir, influence, consequence, question)
        plan = json.loads((Path(result_dir) / "test-plan.json").read_text())
    except Exception as e:  # noqa: BLE001
        return _err(e)
    return {"headline": r["headline"], "model_risk": r["risk"], "goal": r["goal"],
            "factors": [{k: f[k] for k in ("factor", "level", "goal", "meets", "evidence", "gap")}
                        for f in r["factors"]],
            "test_plan": _plan(plan), "disclaimer": r["disclaimer"],
            "files": {"credibility": str(Path(result_dir) / "credibility.html"),
                      "report": str(Path(result_dir) / "report.html")}}


def plan_tests(result_dir: str) -> dict:
    """The smallest set of physical tests that most reduce the risk, and which tests simulation can stand in for."""
    from ..credibility.report import refresh
    from ..credibility.testplan import plan
    from ..credibility.uncertainty import _clean

    try:
        r = _clean(plan(result_dir))
        (Path(result_dir) / "test-plan.json").write_text(json.dumps(r, indent=1))
        refresh(result_dir)
    except Exception as e:  # noqa: BLE001
        return _err(e)
    return _plan(r)


def _plan(r: dict) -> dict:
    return {"headline": r["headline"],
            "tests": [{k: t.get(k) for k in ("order", "kind", "what", "why", "how_many", "gauges")}
                      for t in r["tests"]],
            "replace": r["replace"]}


def list_materials() -> dict:
    from ..materials import all_materials

    return {"materials": [{"key": m.key, "name": m.name, "process": m.process, "E_MPa": m.E, "limit_MPa": m.allowable,
                           "criterion": m.criterion, "density_kg_m3": m.density,
                           "conductivity_W_mK": m.conductivity, "max_service_C": m.max_service_temp}
                          for m in all_materials()]}
