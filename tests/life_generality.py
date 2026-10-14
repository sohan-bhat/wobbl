"""Buckling, vibration and fatigue on every example part and on parts nobody tuned them for.

Each part comes with how it is held and loaded: the example parts with the supports and loads their own strength
runs used, read back from output/study.json, and the rest with a plain setup of the kind a user would write. Every
part then goes through the same five runs:

    modes        natural frequencies against a rotor band, a 1 g base sweep and a GEVS random input
    loads swept  its own loads swept from 10 to 1000 Hz as harmonic forces
    buckling     its own loads, as the reference for the buckling factor
    fatigue      the random input for two minutes, from the modes run
    history      a load history counted by rainflow, from the example's strength result or a fresh one

run_part returns one row per run with what came back, or the error and where it came from.
"""
from __future__ import annotations

import json
import math
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EX = ROOT / "examples"
HISTORY = "0, 1, 0, 0.6, -0.2, 1, 0, 0.4, 0, 1.1, 0"


@dataclass
class Part:
    name: str
    kind: str                       # example or varied
    material: str
    fixtures: list[str]
    loads: list[str]
    path: str = ""                  # filled in by make()
    params: dict = field(default_factory=dict)
    strength: str = ""              # a finished strength result of this setup, when one exists
    build: object = None            # for varied parts: writes the file into a folder and returns its path
    note: str = ""


def _example(name: str, folder: str, note: str = "") -> Part:
    d = EX / folder / "output"
    s = json.loads((d / "study.json").read_text())
    s = s.get("base", s)
    part = (d / s["part"]).resolve()
    return Part(name, "example", s["material"], [f["text"] for f in s["fixtures"]], [ld["text"] for ld in s["loads"]],
                str(part), dict(s.get("params") or {}), str(d) if (d / "outcome.json").exists() and
                json.loads((d / "outcome.json").read_text()).get("domain") == "strength" else "", note=note)


def _session(path: Path, build, stl: bool = False) -> str:
    """Write an OpenCASCADE part with gmsh, as STEP, or as an STL surface for a part that arrives as a mesh."""
    import gmsh

    from wobbl.validation.geometry import quiet_stdout

    if gmsh.isInitialized():
        gmsh.finalize()
    gmsh.initialize(readConfigFiles=False)
    gmsh.option.setNumber("General.Terminal", 0)
    try:
        gmsh.model.add("p")
        build(gmsh.model.occ)
        gmsh.model.occ.synchronize()
        with quiet_stdout():
            if stl:
                gmsh.option.setNumber("Mesh.MeshSizeMax", 2.5)
                gmsh.model.mesh.generate(2)
            gmsh.write(str(path))
    finally:
        gmsh.finalize()
    return str(path)


def _sheet(o):          # a 1.5 mm steel L bracket, the flange screwed to a wall and the web sticking out
    o.fuse([(3, o.addBox(0, 0, 0, 1.5, 40, 50))], [(3, o.addBox(0, 0, 0, 60, 40, 1.5))])


def _block(o):          # a chunky block with a cross hole
    o.cut([(3, o.addBox(0, 0, 0, 60, 40, 30))], [(3, o.addCylinder(30, -1, 15, 0, 42, 0, 6))])


def _hook(o):           # a C shaped hook, half a ring 10 mm thick
    outer = o.addCylinder(0, 0, 0, 0, 0, 10, 30, angle=math.pi)
    inner = o.addCylinder(0, 0, -1, 0, 0, 12, 20, angle=math.pi)
    o.cut([(3, outer)], [(3, inner)])


def _can(o):            # a thin walled can, 100 mm across with a 1 mm wall
    o.cut([(3, o.addCylinder(0, 0, 0, 0, 0, 150, 50))], [(3, o.addCylinder(0, 0, -1, 0, 0, 152, 49))])


def _dome(o):           # a 2 mm hemispherical dome on a short skirt
    shell = o.cut([(3, o.addSphere(0, 0, 20, 40))], [(3, o.addSphere(0, 0, 20, 38))])[0]
    shell = o.cut(shell, [(3, o.addBox(-50, -50, -30, 100, 100, 50))])[0]      # keep the half above z = 20
    skirt = o.cut([(3, o.addCylinder(0, 0, 0, 0, 0, 20, 40))], [(3, o.addCylinder(0, 0, -1, 0, 0, 22, 38))])[0]
    o.fuse(shell, skirt)


def _shaft(o):
    o.addCylinder(0, 0, 0, 0, 0, 200, 10)


def _plate(o):          # a plate with a hole, written as an STL mesh
    o.cut([(3, o.addBox(0, 0, 0, 80, 40, 5))], [(3, o.addCylinder(40, 20, -1, 0, 0, 7, 6))])


def _assembly(o):       # a post standing on a base plate, two bodies that touch
    o.addBox(0, 0, 0, 60, 60, 10)
    o.addBox(20, 20, 10, 20, 20, 80)


VARIED = [
    Part("sheet bracket", "varied", "mild-steel", ["fixed x=min"], ["20 N down @ x=max"], build=(_sheet, False),
         note="thin, 1.5 mm sheet"),
    Part("block with hole", "varied", "6061-t6", ["fixed z=min"], ["20 kN -z @ z=max"], build=(_block, False),
         note="chunky"),
    Part("hook", "varied", "6061-t6", ["fixed y=min and x>0"], ["500 N -y @ y=min and x<0"], build=(_hook, False),
         note="curved"),
    Part("thin can", "varied", "6061-t6", ["fixed z=min"], ["5 kN -z @ z=max"], build=(_can, False),
         note="thin shell, 1 mm wall on 100 mm"),
    Part("dome", "varied", "6061-t6", ["fixed z=min"], ["1 kN -z @ within 10 of 0,0,60"], build=(_dome, False),
         note="doubly curved shell, pressed on its crown"),
    Part("shaft", "varied", "4140", ["fixed z=min"], ["300 N +x @ z=max"], build=(_shaft, False), note="round"),
    Part("plate from STL", "varied", "6061-t6", ["fixed x=min"], ["500 N down @ x=max"], build=(_plate, True),
         note="mesh input, no CAD faces"),
    Part("post on plate", "varied", "6061-t6", ["fixed z=min"], ["200 N +x @ z=max"], build=(_assembly, False),
         note="two touching bodies"),
]


def parts() -> list[Part]:
    out = [_example("cast motor mount", "cast-motor-mount", "remote load through holes"),
           _example("shelf bracket", "shelf-bracket", "a wall the part pushes on"),
           _example("motor mount", "motor-mount", "rigid remote load"),
           _example("enclosure", "enclosure", "thin walls"),
           _example("drone arm", "drone-arm", "rigid force and moment"),
           _example("strut", "strut-buckling", "slender tube")]
    out += [Part("heat sink", "example", "6063-t5", ["fixed z=min"], ["50 N down @ z=max"],
                 str(EX / "heat-sink" / "heatsink.py"), note="thin fins, held by its base"),
            Part("battery pack", "example", "6061-t6", ["fixed z=min"], ["200 N down @ z=max"],
                 str(EX / "battery-pack" / "pack.py"), note="a block, held by its base")]
    return out + [Part(**{**p.__dict__}) for p in VARIED]


def make(p: Part, folder: Path) -> Part:
    if p.build is not None and not p.path:
        fn, stl = p.build
        folder.mkdir(parents=True, exist_ok=True)
        p.path = _session(folder / (p.name.replace(" ", "-") + (".stl" if stl else ".step")), fn, stl)
    return p


def _row(analysis: str, fn) -> dict:
    t0 = time.time()
    try:
        o = fn()
        v = o.values
        row = {"analysis": analysis, "ok": True, "verdict": o.verdict, "confidence": o.confidence,
               "reasons": list(o.confidence_reasons)[:4], "checks": {c.name: c.status for c in o.checks},
               "headline": o.headline, "failed_checks": [c.name for c in o.checks if c.status == "fail"],
               "playback_kB": round((Path(o.workdir) / "playback.json").stat().st_size / 1000, 1)
               if (Path(o.workdir) / "playback.json").exists() else None, "result_dir": o.workdir}
        for k in ("load_factor", "design_factor", "life", "sf", "frequencies"):
            if k in v:
                row[k] = v[k][:3] if isinstance(v[k], list) else v[k]
    except Exception as e:  # noqa: BLE001  the sweep records every failure with where it came from
        tb = traceback.extract_tb(e.__traceback__)
        where = next((f"{Path(f.filename).name}:{f.lineno}" for f in reversed(tb) if "wobbl" in f.filename), "")
        row = {"analysis": analysis, "ok": False, "error": f"{type(e).__name__}: {e}"[:300], "where": where}
    row["time_s"] = round(time.time() - t0, 1)
    return row


def run_part(p: Part, out: Path, quality: str = "fast") -> list[dict]:
    from wobbl.life import api

    p = make(p, out / "parts")
    kw = {"quality": quality, "params": p.params}
    rows = []
    d = out / p.name.replace(" ", "-")
    rows.append(_row("modes", lambda: api.vibration(
        p.path, p.material, p.fixtures, rpm=["3000-9000 rpm"], sine="1 g z 10-1000 Hz", psd="gevs-accept",
        modes=10, out=d / "modes", **kw)))
    rows.append(_row("loads swept", lambda: api.vibration(
        p.path, p.material, p.fixtures, loads=p.loads, sine="10-1000 Hz", modes=10, out=d / "swept", **kw)))
    rows.append(_row("buckling", lambda: api.buckling(p.path, p.material, p.fixtures, p.loads, out=d / "buckling",
                                                      **kw)))
    if rows[0]["ok"]:
        rows.append(_row("fatigue", lambda: api.fatigue(rows[0]["result_dir"], duration="2 min",
                                                        out=d / "fatigue")))
    else:
        rows.append({"analysis": "fatigue", "ok": False, "error": "skipped, the modes run failed", "time_s": 0})
    if p.strength:
        rows.append(_row("history", lambda: api.fatigue(p.strength, history=HISTORY, repeats=1e4,
                                                        out=d / "history")))
    else:
        rows.append(_row("history", lambda: api.fatigue(part=p.path, material=p.material, fixtures=p.fixtures,
                                                        loads=p.loads, history=HISTORY, repeats=1e4,
                                                        out=d / "history", **kw)))
    for r in rows:
        r["part"], r["kind"], r["note"] = p.name, p.kind, p.note
    return rows
