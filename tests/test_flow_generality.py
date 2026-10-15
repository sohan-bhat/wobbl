"""Generality proof for CFD on CAD parts: varied parts nobody tuned the flow study for, each asked a plain question
through wobbl simulate at fast quality. Every run has to finish with a finite, sensible answer and its own checks:
external runs a drag coefficient a real body can have, internal runs a balanced mass flow and a pressure drop no
lower than a plain duct's would allow for.

The parts are the repository's examples and validation parts, which were drawn for strength and heat studies,
and a handful of pipes, a cold plate, a fluid volume, a wing and an STL sphere drawn here. Run this file directly
for the table: python tests/test_flow_generality.py [out_dir]
"""
from __future__ import annotations

import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EX, PARTS = ROOT / "examples", ROOT / "src" / "wobbl" / "validation" / "parts"


@dataclass
class Part:
    name: str
    path: str                     # a file, or '{gen}/name' for a part drawn here
    question: str
    options: dict
    mode: str                     # what the study should decide: external, internal, volume or ducted


CASES = [
    Part("shelf bracket", str(EX / "shelf-bracket" / "bracket.py"), "how much drag does it feel at 10 m/s?", {},
         "external"),
    Part("drone arm", str(EX / "drone-arm" / "arm.py"), "what is the drag in a 15 m/s wind?", {}, "external"),
    Part("enclosure", str(EX / "enclosure" / "enclosure.py"), "what wind load does it take at 20 m/s?", {},
         "external"),
    Part("motor mount", str(EX / "motor-mount" / "mount.py"), "drag at 10 m/s?", {"direction": ["-y"]}, "external"),
    Part("strut", str(EX / "strut-buckling" / "strut.py"), "drag at 25 m/s across it?", {"direction": ["+y"]},
         "external"),
    Part("battery pack", str(EX / "battery-pack" / "pack.py"), "what is the drag at 8 m/s?", {}, "external"),
    Part("cast motor mount", str(PARTS / "cast_motor_mount.step"), "airflow drag at 12 m/s?", {}, "external"),
    Part("truss arm", str(PARTS / "drone_arm_truss.step"), "how much drag at 15 m/s?", {"direction": ["+z"]},
         "external"),
    Part("wing", "{gen}/wing.step", "lift and drag at 20 m/s?", {"up": ["+z"]}, "external"),
    Part("sphere STL", "{gen}/sphere.stl", "drag in water at 0.5 m/s?", {}, "external"),
    Part("tube", "{gen}/tube.step", "pressure lost at 10 L/min of water through it?", {}, "internal"),
    Part("elbow", "{gen}/elbow.step", "pressure lost with 6 L/min of engine oil at 60 C through it?", {},
         "internal"),
    Part("tee", "{gen}/tee.step", "how does 5 L/min of water split through it?", {}, "internal"),
    Part("cold plate", "{gen}/coldplate.step", "pressure lost with 2 L/min of glycol at 40 C through it?", {},
         "internal"),
    Part("fluid volume", "{gen}/stepvolume.step", "pressure lost with 8 L/min of water through it?",
         {"inlet": ["x=min"], "outlet": ["x=max"]}, "volume"),
    Part("heat sink in a duct", str(EX / "heat-sink" / "heatsink.py"), "pressure lost with 15 CFM of air through "
         "its fins?", {"mode": ["ducted"], "direction": ["+y"]}, "ducted"),
]


def draw(folder: Path) -> Path:
    """The parts drawn here, as STEP files (and one STL) with gmsh's OpenCASCADE kernel, in mm."""
    import gmsh

    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "wing.step").exists():
        return folder
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)

    def part(name, build, stl=False):
        gmsh.model.add(name)
        build(gmsh.model.occ)
        gmsh.model.occ.synchronize()
        gmsh.write(str(folder / f"{name}.step"))
        if stl:
            gmsh.option.setNumber("Mesh.MeshSizeMax", 3.0)
            gmsh.model.mesh.generate(2)
            gmsh.write(str(folder / f"{name}.stl"))
        gmsh.model.remove()

    def tube(o):
        o.cut([(3, o.addCylinder(0, 0, 0, 200, 0, 0, 12))], [(3, o.addCylinder(-1, 0, 0, 202, 0, 0, 10))])

    def bend(o, r):
        a = o.addCylinder(-60, 0, 0, 60, 0, 0, r)
        t = o.addTorus(0, 40, 0, 40, r, angle=math.pi / 2)
        o.rotate([(3, t)], 0, 40, 0, 0, 0, 1, -math.pi / 2)
        b = o.addCylinder(40, 40, 0, 0, 60, 0, r)
        return o.fuse([(3, a)], [(3, t), (3, b)])[0]

    def tee_solid(o, r, extra):
        a = o.addCylinder(-80 - extra, 0, 0, 160 + 2 * extra, 0, 0, r)
        return o.fuse([(3, a)], [(3, o.addCylinder(0, 0, 0, 0, 70 + extra, 0, r))])[0]

    def coldplate(o):
        b = o.addBox(0, 0, 0, 80, 50, 15)
        o.cut([(3, b)], [(3, o.addCylinder(-1, 12, 7.5, 71, 0, 0, 4)), (3, o.addCylinder(-1, 38, 7.5, 71, 0, 0, 4)),
                         (3, o.addCylinder(66, 12, 7.5, 0, 26, 0, 4))])

    def wing(o):
        o.fuse([(3, o.addBox(0, -60, -4, 60, 120, 8))], [(3, o.addCylinder(0, -60, 0, 0, 120, 0, 4))])

    part("tube", tube)
    part("elbow", lambda o: o.cut(bend(o, 10), bend(o, 8)))
    part("tee", lambda o: o.cut(tee_solid(o, 8, 0), tee_solid(o, 6, 1)))
    part("coldplate", coldplate)
    part("stepvolume", lambda o: o.fuse([(3, o.addCylinder(0, 0, 0, 60, 0, 0, 10))],
                                        [(3, o.addCylinder(60, 0, 0, 60, 0, 0, 6))]))
    part("wing", wing)
    part("sphere", lambda o: o.addSphere(0, 0, 0, 25), stl=True)
    gmsh.finalize()
    return folder


def run_case(case: Part, gen: Path, out: Path) -> dict:
    """Route and run one case at fast quality, and judge its answer."""
    from wobbl.simulate import route, run_plan

    path = case.path.replace("{gen}", str(gen))
    t0 = time.time()
    row = {"part": case.name, "file": Path(path).name, "question": case.question}
    try:
        plan = route(path, case.question, ["cfd"])
        steps = run_plan(plan, case.options, out, quick=True)
        o = steps[0].outcome
        if steps[0].error or o is None:
            raise RuntimeError(steps[0].error or "no outcome")
        v = o.values
        checks = {c.name: c.status for c in o.checks}
        row.update(mode=v["mode"], cells=v["cells"], model=v["model"], re=v["re"], confidence=o.confidence,
                   headline=o.headline, checks=checks)
        problems = []
        if v["mode"] != case.mode:
            problems.append(f"decided {v['mode']} flow, not {case.mode}")
        if v["mode"] == "external":
            row["answer"] = f"Cd {v['cd']:.3g}, drag {v['drag_N']:.3g} N"
            if not (math.isfinite(v["cd"]) and 0.02 <= v["cd"] <= 3.0):
                problems.append(f"drag coefficient {v['cd']:.3g} is outside what a real body has")
        else:
            row["answer"] = f"dp {v['dp_Pa']:.4g} Pa at {v['flow_m3_s'] * 6e4:.3g} L/min"
            if checks.get("mass conservation") == "fail":
                problems.append("mass is not conserved")
            if not (math.isfinite(v["dp_Pa"]) and v["dp_Pa"] > 0):
                problems.append(f"pressure drop {v['dp_Pa']:.3g} Pa is not positive")
            if checks.get("hand calculation") == "warn":
                problems.append("it loses less than a plain duct of its size")
        row["ok"] = not problems
        row["why"] = "; ".join(problems)
    except Exception as e:  # noqa: BLE001  a failure is a result here, with its reason
        row.update(ok=False, why=f"{type(e).__name__}: {e}"[:400])
    row["seconds"] = round(time.time() - t0)
    return row


@pytest.mark.slow
@pytest.mark.solver
@pytest.mark.parametrize("case", CASES[::4], ids=[c.name for c in CASES[::4]])
def test_flow_on_parts_nobody_tuned_for(case, tmp_path_factory):
    from conftest import _have
    if not _have("flow"):
        pytest.skip("OpenFOAM not installed")
    gen = draw(tmp_path_factory.getbasetemp() / "drawn")
    row = run_case(case, gen, tmp_path_factory.mktemp(case.name.replace(" ", "_")))
    assert row["ok"], row


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT / "tests"))
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("flow-generality")
    gen = draw(out / "drawn")
    rows = []
    for c in CASES:
        row = run_case(c, gen, out / c.name.replace(" ", "_"))
        rows.append(row)
        print(f"{'ok  ' if row['ok'] else 'FAIL'} {row['part']:22s} {row.get('mode', ''):9s} "
              f"{row.get('cells', 0):>7} cells {row['seconds']:>5}s  {row.get('answer', '')}  {row['why']}",
              flush=True)
        (out / "generality.json").write_text(json.dumps(rows, indent=1, default=str))
    good = sum(r["ok"] for r in rows)
    print(f"\n{good} of {len(rows)} parts gave a sensible answer")
