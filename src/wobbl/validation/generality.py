"""How general the nonlinear part studies are: plasticity on every example part and every validation part, and
contact and bolted joints on assemblies built here, none of which the code was tuned on.

    python -m wobbl.validation.generality OUT [--only id,id] [--workers 3]

writes OUT/generality.json and prints one line per case. A case succeeds when the study runs to an answer and none
of its self checks (convergence, balance, overlap, preload, load steps) fails; the verdict itself may be anything,
since a joint that opens is a correct answer. Each case runs in its own process with a time limit.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
EX = ROOT / "examples"
PARTS = Path(__file__).resolve().parent / "parts"
DESIGN = {"ductility", "hand calculation", "bolt strength", "tightening", "separation", "slip", "head pressure",
          "thread stripping", "material model", "clamped stiffness"}
LIMIT = 2400.0


def _step(path: Path, build) -> Path:
    from ..geometry.cad import gmsh_session

    with gmsh_session() as gmsh:
        gmsh.model.add("case")
        build(gmsh.model.occ)
        gmsh.model.occ.synchronize()
        gmsh.write(str(path))
    return path


# --------------------------------------------------------------------------------------------------- assemblies
def block_on_base(o):
    o.addBox(0, 0, 0, 80, 60, 12)
    o.addBox(25, 15, 12, 30, 30, 20)


def pin_in_hole(o):
    p = o.addBox(0, 0, 0, 60, 40, 10)
    h = o.addCylinder(30, 20, -1, 0, 0, 12, 5.0)
    o.cut([(3, p)], [(3, h)])
    o.addCylinder(30, 20, -8, 0, 0, 26, 4.99)


def cylinder_on_plate(o):
    o.addBox(0, 0, 0, 80, 40, 10)
    c = o.addCylinder(40, 5, 20, 0, 30, 0, 10)
    cut = o.addBox(20, 0, 27, 40, 40, 10)
    o.cut([(3, c)], [(3, cut)])


def lap_joint(o):
    a = o.addBox(0, 0, 0, 100, 40, 8)
    b = o.addBox(60, 0, 8, 100, 40, 8)
    for body, z in ((a, -1), (b, 7)):
        for x in (70, 90):
            o.cut([(3, body)], [(3, o.addCylinder(x, 20, z, 0, 0, 10, 4.5))])


def bracket_on_base(o):
    base = o.addBox(0, 0, 0, 100, 60, 10)
    flange = o.addBox(0, 10, 10, 100, 40, 8)
    web = o.addBox(0, 42, 18, 100, 8, 50)
    br, _ = o.fuse([(3, flange)], [(3, web)])
    for x in (25, 75):
        o.cut([(3, base)], [(3, o.addCylinder(x, 30, -1, 0, 0, 12, 5.5))])
        o.cut(br, [(3, o.addCylinder(x, 30, 9, 0, 0, 10, 5.5))])


def three_plate_stack(o):
    for k in range(3):
        p = o.addBox(0, 0, 8 * k, 50, 50, 8)
        o.cut([(3, p)], [(3, o.addCylinder(25, 25, 8 * k - 1, 0, 0, 10, 5.5))])


def flange_pair(o):
    import math

    for k in range(2):
        d = o.addCylinder(0, 0, 12 * k, 0, 0, 12, 60)
        o.cut([(3, d)], [(3, o.addCylinder(0, 0, 12 * k - 1, 0, 0, 14, 20))])
        for j in range(4):
            a = math.pi / 4 + j * math.pi / 2
            o.cut([(3, d)], [(3, o.addCylinder(45 * math.cos(a), 45 * math.sin(a), 12 * k - 1, 0, 0, 14, 4.5))])


def tapped_block(o):
    blk = o.addBox(0, 0, 0, 60, 40, 30)
    plate = o.addBox(0, 0, 30, 60, 40, 10)
    for x in (15, 45):
        o.cut([(3, blk)], [(3, o.addCylinder(x, 20, 10, 0, 0, 21, 3.4))])
        o.cut([(3, plate)], [(3, o.addCylinder(x, 20, 29, 0, 0, 12, 4.5))])


def sphere_on_plate(o):
    o.addBox(0, 0, 0, 60, 60, 10)
    s = o.addSphere(30, 30, 22, 12)
    o.cut([(3, s)], [(3, o.addBox(10, 10, 30, 40, 40, 20))])


# ------------------------------------------------------------------------------------------------------- cases
CASES = [
    # plasticity on the example parts, with their own loads and supports
    {"id": "cast-motor-mount", "kind": "plastic", "part": PARTS / "cast_motor_mount.step",
     "material": "custom:E=72400,nu=0.33,yield=165,uts=230", "fix": ["z=min"],
     "load": ["2000 N -0.8660254,0,-0.5 at 0,-30,66 @ holes d=6.6"]},
    {"id": "drone-arm", "kind": "plastic", "part": EX / "drone-arm" / "arm.py", "material": "pa12",
     "fix": ["bolt holes d=3.2 and x<0"],
     "load": ["25 N up at 150,0,15 @ top and within 15 of 150,0,15 rigid",
              "0.5 N*m +z at 150,0,15 @ top and within 15 of 150,0,15 rigid"]},
    {"id": "enclosure", "kind": "plastic", "part": EX / "enclosure" / "enclosure.py", "material": "pc",
     "fix": ["z=min"], "load": ["120 N down @ top and within 8 of 25,15,28"]},
    {"id": "heat-sink", "kind": "plastic", "part": EX / "heat-sink" / "heatsink.py", "material": "6063-t5",
     "fix": ["z=min"], "load": ["400 N -z @ top"], "note": "a thermal example; pressed on its fin tips here"},
    {"id": "motor-mount", "kind": "plastic", "part": EX / "motor-mount" / "mount.py", "material": "petg",
     "fix": ["bolt holes d=5.5"], "load": ["250 N down at 0,-15,31 @ normal +y and box -21,4,10,21,6,52 rigid"]},
    {"id": "shelf-bracket", "kind": "plastic", "part": EX / "shelf-bracket" / "bracket.py", "material": "6061-t6",
     "fix": ["bolt holes d=4.5", "wall x=min"], "load": ["40 N down @ top and within 12 of 65,12.5,6"],
     "note": "PLA in the example is brittle, which has no plastic reserve, so 6061-T6 here"},
    {"id": "strut", "kind": "plastic", "part": EX / "strut-buckling" / "strut.py", "material": "6061-t6",
     "fix": ["z=min", "fixed-xy z=max"], "load": ["7 kN -z @ z=max"]},
    {"id": "battery-pack", "kind": "plastic", "part": EX / "battery-pack" / "pack.py", "material": "6061-t6",
     "fix": ["z=min"], "load": ["5 kN -x @ x=max"],
     "note": "a thermal example of cells as one block; squeezed sideways here as an aluminum block"},
    # plasticity on the validation parts
    {"id": "enclosure-shell", "kind": "plastic", "part": PARTS / "enclosure_shell.step",
     "material": "custom:E=2200,nu=0.35,yield=40,uts=45", "fix": ["z=min"],
     "load": ["250 N -z @ top and within 15 of 50,-25,35"]},
    {"id": "drone-arm-truss", "kind": "plastic", "part": PARTS / "drone_arm_truss.step", "material": "pa12",
     "fix": ["x=min"], "load": ["40 N +z at 210,0,25 @ top and within 14 of 210,0,25 rigid"]},
    {"id": "bracket-fillet", "kind": "plastic", "part": PARTS / "shelf_bracket_fillet.step", "material": "6061-t6",
     "fix": ["hole d=4.5", "wall x=min"], "load": ["60 N -y @ normal +y and box 45,5,-1,55,7,21"]},
    {"id": "bracket-sharp", "kind": "plastic", "part": PARTS / "shelf_bracket_sharp.step", "material": "mild-steel",
     "fix": ["hole d=4.5", "wall x=min"], "load": ["60 N -y @ normal +y and box 45,5,-1,55,7,21"]},
    # contact on assemblies built here
    {"id": "block-on-base", "kind": "contact", "build": block_on_base, "material": "mild-steel",
     "fix": ["body 1 and bottom"], "load": ["5 MPa -z @ body 2 and top"]},
    {"id": "block-pushed", "kind": "contact", "build": block_on_base, "material": "mild-steel",
     "fix": ["body 1 and bottom"], "load": ["5 MPa -z @ body 2 and top"], "move": ["0.05 mm +x @ body 2 and x=min"],
     "friction": 0.3},
    {"id": "pin-in-hole", "kind": "contact", "build": pin_in_hole, "material": "mild-steel",
     "fix": ["body 1 and x=min"], "load": ["2 kN +x @ body 2 and top"]},
    {"id": "cylinder-on-plate", "kind": "contact", "build": cylinder_on_plate, "material": "mild-steel",
     "fix": ["body 1 and bottom"], "load": ["3 kN -z @ body 2 and top"]},
    {"id": "sphere-on-plate", "kind": "contact", "build": sphere_on_plate, "material": "mild-steel",
     "fix": ["body 1 and bottom"], "load": ["1 kN -z @ body 2 and top"]},
    # bolted joints, bolts found from the holes
    {"id": "lap-joint", "kind": "bolts", "build": lap_joint, "material": "mild-steel",
     "fix": ["body 1 and x=min"], "load": ["3 kN +x @ body 2 and x=max"]},
    {"id": "bracket-on-base", "kind": "bolts", "build": bracket_on_base, "material": "6061-t6",
     "fix": ["body 1 and bottom"], "load": ["2 kN +z @ body 2 and top"]},
    {"id": "three-plate-stack", "kind": "bolts", "build": three_plate_stack, "material": "mild-steel",
     "fix": ["body 1 and bottom"], "load": ["6 kN +z @ body 3 and top"]},
    {"id": "flange-pair", "kind": "bolts", "build": flange_pair, "material": "mild-steel",
     "fix": ["body 1 and bottom"], "load": ["10 kN +z @ body 2 and top"], "bolt": ["M8 8.8 torque 20 N*m"]},
    {"id": "tapped-block", "kind": "bolts", "build": tapped_block, "material": "mild-steel",
     "fix": ["body 1 and bottom"], "load": ["4 kN +z @ body 2 and top"]},
]


def run_case(case: dict, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    row = {"id": case["id"], "kind": case["kind"], "material": case["material"], "note": case.get("note", "")}
    try:
        part = case.get("part") or _step(out / f"{case['id']}.step", case["build"])
        if case["kind"] == "plastic":
            from ..plastic.study import PlasticStudy, run_plastic
            o = run_plastic(PlasticStudy(str(part), case["material"], case["load"], case["fix"], quality="fast"),
                            out / "run", render_images=False)
        elif case["kind"] == "contact":
            from ..joints.contact import ContactStudy, run_contact
            o = run_contact(ContactStudy(str(part), case["material"], case["load"], case["fix"], case.get("move", []),
                                         case.get("friction", 0.2), quality="fast"), out / "run", render_images=False)
        else:
            from ..joints.bolts import BoltStudy, run_bolts
            o = run_bolts(BoltStudy(str(part), case["material"], case.get("bolt", []), case["load"], case["fix"],
                                    quality="fast"), out / "run", render_images=False)
        own = [c for c in o.checks if c.name not in DESIGN]
        bad = [f"{c.name}: {c.summary}" for c in own if c.status == "fail"]
        row.update(ran=True, ok=not bad, verdict=o.verdict, confidence=o.confidence, headline=o.headline,
                   failed_checks=bad, warned=[c.name for c in own if c.status == "warn"],
                   nodes=(o.values.get("levels") or [{}])[-1].get("n_nodes"))
    except Exception as e:  # noqa: BLE001  a case that cannot run is a finding, not a crash of the harness
        row.update(ran=False, ok=False, error=f"{type(e).__name__}: {e}"[:600], hint=getattr(e, "hint", ""))
    row["seconds"] = round(time.time() - t0, 1)
    return row


def _worker(args):
    case, out = args
    return run_case(case, out)


def main(argv=None) -> int:
    import argparse
    import multiprocessing as mp

    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--only", default="")
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ids = [x for x in a.only.split(",") if x]
    cases = [c for c in CASES if not ids or c["id"] in ids]
    rows = []
    ctx = mp.get_context("spawn")
    with ctx.Pool(a.workers, maxtasksperchild=1) as pool:
        jobs = {c["id"]: pool.apply_async(_worker, ((c, out / c["id"]),)) for c in cases}
        for c in cases:
            try:
                row = jobs[c["id"]].get(timeout=LIMIT * (1 + len(cases) / max(a.workers, 1)))
            except Exception as e:  # noqa: BLE001
                row = {"id": c["id"], "kind": c["kind"], "ran": False, "ok": False, "error": f"no answer: {e}"}
            rows.append(row)
            mark = "ok  " if row["ok"] else ("ran " if row.get("ran") else "FAIL")
            print(f"{mark} {row['id']:<20} {row.get('verdict', ''):<10} {row.get('confidence', ''):<7} "
                  f"{row.get('seconds', '')}s  {row.get('error') or '; '.join(row.get('failed_checks', []))}",
                  flush=True)
            (out / "generality.json").write_text(json.dumps(rows, indent=1, default=str))
    n_ok = sum(r["ok"] for r in rows)
    print(f"{n_ok} of {len(rows)} cases ran and passed their own checks; "
          f"{sum(r.get('ran', False) for r in rows)} ran to an answer")
    return 0


if __name__ == "__main__":
    sys.exit(main())
