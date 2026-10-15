"""How general the manufacturing studies are: print warping, sand casting and mold filling on every example part,
every validation part and a set of parts built here, none of which the studies were tuned on.

    python -m wobbl.validation.process_generality OUT [--only id,id] [--process print,casting,molding] [--workers 2]
                                                      [--limit 1800]

writes OUT/process_generality.json and prints one line per case. A case succeeds when the study runs to an answer
and none of its own checks (mesh convergence, energy or volume balance, time step, hand calculation, units) fails.
The verdict may be anything, since a part that warps, a casting with porosity and a short shot are answers. Each
case runs in its own process with a time limit, at fast quality, with the defaults a user gets when they name only
the part and the process.
"""
from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

from .generality import EX, PARTS, _step

# checks that judge the part rather than the solve: failing one is the answer, not a failure of the study
DESIGN = {"flatness", "bed peel", "feeding", "porosity", "riser", "fill", "pressure", "thin walls"}
LIMIT = 1800.0     # s a case may run before it is killed and recorded as timed out
PROCESSES = ("print", "casting", "molding")


# ------------------------------------------------------------------------------------------------ parts built here
def housing(o):
    """A hollow box open at the top, with two ribs and four screw bosses inside."""
    outer = o.addBox(0, 0, 0, 80, 50, 30)
    inner = o.addBox(2.5, 2.5, 2.5, 75, 45, 30)
    o.cut([(3, outer)], [(3, inner)])
    parts = [(3, o.addBox(39, 2.5, 2.5, 2, 45, 15)), (3, o.addBox(2.5, 24, 2.5, 75, 2, 10))]
    for x, y in ((8, 8), (72, 8), (8, 42), (72, 42)):
        b = o.addCylinder(x, y, 2.5, 0, 0, 20, 3.5)
        o.cut([(3, b)], [(3, o.addCylinder(x, y, 5, 0, 0, 20, 1.3))])
        parts.append((3, b))
    o.fuse([(3, outer)], parts)


def l_bracket(o):
    """An angle bracket with a gusset and a hole in each leg."""
    a = o.addBox(0, 0, 0, 60, 40, 5)
    b = o.addBox(0, 0, 0, 5, 40, 50)
    g = o.addWedge(5, 17.5, 5, 30, 5, 30)
    fused, _ = o.fuse([(3, a)], [(3, b), (3, g)])
    o.cut(fused, [(3, o.addCylinder(40, 20, -1, 0, 0, 7, 3.3)), (3, o.addCylinder(-1, 20, 35, 7, 0, 0, 3.3))])


def gear_blank(o):
    """A disk with a raised hub and a bore."""
    d = o.addCylinder(0, 0, 0, 0, 0, 10, 40)
    h = o.addCylinder(0, 0, 10, 0, 0, 12, 15)
    f, _ = o.fuse([(3, d)], [(3, h)])
    o.cut(f, [(3, o.addCylinder(0, 0, -1, 0, 0, 24, 6))])


def tube(o):
    """A thick walled tube lying along x."""
    a = o.addCylinder(0, 0, 0, 120, 0, 0, 15)
    o.cut([(3, a)], [(3, o.addCylinder(-1, 0, 0, 122, 0, 0, 12))])


def knob(o):
    """A round knob: a sphere on a short shaft."""
    s = o.addSphere(0, 0, 30, 15)
    c = o.addCylinder(0, 0, 0, 0, 0, 20, 6)
    o.fuse([(3, s)], [(3, c)])


def thin_cover(o):
    """A thin cover 120 x 80 x 1.5 mm with a 6 mm rim."""
    p = o.addBox(0, 0, 0, 120, 80, 7.5)
    o.cut([(3, p)], [(3, o.addBox(1.5, 1.5, 1.5, 117, 77, 10))])


def tapered_block(o):
    """A wedge shaped block, thick at one end and thin at the other."""
    o.addWedge(0, 0, 0, 100, 40, 30, ltx=0)


def flanged_cup(o):
    """A cup with a flange around its rim."""
    c = o.addCylinder(0, 0, 0, 0, 0, 40, 25)
    o.cut([(3, c)], [(3, o.addCylinder(0, 0, 3, 0, 0, 40, 22))])
    o.fuse([(3, c)], [(3, o.addCylinder(0, 0, 37, 0, 0, 3, 35))])
    o.cut([(3, c)], [(3, o.addCylinder(0, 0, 3, 0, 0, 40, 22))])


# --------------------------------------------------------------------------------------------------------- parts
PARTS_LIST = [
    ("cast-motor-mount", PARTS / "cast_motor_mount.step", "validation part"),
    ("drone-arm-truss", PARTS / "drone_arm_truss.step", "validation part"),
    ("enclosure-shell", PARTS / "enclosure_shell.step", "validation part"),
    ("bracket-fillet", PARTS / "shelf_bracket_fillet.step", "validation part"),
    ("bracket-sharp", PARTS / "shelf_bracket_sharp.step", "validation part"),
    ("drone-arm", EX / "drone-arm" / "arm.py", "example"),
    ("carbon-arm", EX / "carbon-arm" / "arm.py", "example"),
    ("enclosure", EX / "enclosure" / "enclosure.py", "example"),
    ("heat-sink", EX / "heat-sink" / "heatsink.py", "example"),
    ("motor-mount", EX / "motor-mount" / "mount.py", "example"),
    ("shelf-bracket", EX / "shelf-bracket" / "bracket.py", "example"),
    ("bracket-crack", EX / "bracket-crack" / "bracket.py", "example"),
    ("steel-hook", EX / "steel-hook" / "hook.py", "example"),
    ("strut", EX / "strut-buckling" / "strut.py", "example"),
    ("battery-pack", EX / "battery-pack" / "pack.py", "example"),
    ("printed-lid", EX / "printed-lid" / "lid.py", "example"),
    ("cast-pulley", EX / "cast-pulley" / "pulley.py", "example"),
    ("cold-plate", EX / "cold-plate" / "plate.py", "example"),
    ("fan-heat-sink", EX / "fan-heat-sink" / "heatsink.py", "example"),
    ("bolted-flange", EX / "bolted-flange" / "flange.py", "example"),
    ("handheld", EX / "handheld-drop" / "handheld.py", "example"),
    ("housing", housing, "built here"),
    ("l-bracket", l_bracket, "built here"),
    ("gear-blank", gear_blank, "built here"),
    ("tube", tube, "built here"),
    ("knob", knob, "built here"),
    ("thin-cover", thin_cover, "built here"),
    ("tapered-block", tapered_block, "built here"),
    ("flanged-cup", flanged_cup, "built here"),
]

CASES = [{"id": f"{p}-{proc}", "part_id": p, "part": src, "source": where, "process": proc}
         for p, src, where in PARTS_LIST for proc in PROCESSES]


def run_case(case: dict, out: Path, keep: bool = False) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    row = {"id": case["id"], "part": case["part_id"], "source": case["source"], "process": case["process"]}
    try:
        part = case["part"] if isinstance(case["part"], Path) else _step(out / f"{case['part_id']}.step",
                                                                          case["part"])
        run = out / "run"
        if case["process"] == "print":
            from ..process.printing import run_print_warp
            from ..process.warp import WarpStudy
            o = run_print_warp(WarpStudy(str(part), "abs", quality="fast"), run)
        elif case["process"] == "casting":
            from ..process.casting import run_casting
            from ..process.tools import casting_study
            o = run_casting(casting_study(str(part), "a356", quality="fast"), run)
        else:
            from ..process.molding import MoldingStudy, run_molding
            o = run_molding(MoldingStudy(str(part), "gpps", quality="fast"), run)
        own = [c for c in o.checks if c.name not in DESIGN]
        bad = [f"{c.name}: {c.summary}" for c in own if c.status == "fail"]
        row.update(ran=True, ok=not bad, verdict=o.verdict, confidence=o.confidence, headline=o.headline,
                   failed_checks=bad, warned=[c.name for c in own if c.status == "warn"],
                   size_kb=round(sum(f.stat().st_size for f in run.rglob("*") if f.is_file()) / 1024))
    except Exception as e:  # noqa: BLE001  a part a study cannot take is a finding, not a crash of the harness
        row.update(ran=False, ok=False, error=f"{type(e).__name__}: {e}"[:600], hint=getattr(e, "hint", ""))
    row["seconds"] = round(time.time() - t0, 1)
    if not keep:
        shutil.rmtree(out, ignore_errors=True)
    return row


def run_limited(case: dict, out: Path, limit: float = LIMIT, keep: bool = False) -> dict:
    """Run one case in a child process of its own with a hard wall clock limit. The child leads a process group, so
    on timeout the study and the solvers and meshers it started are all stopped together."""
    out.mkdir(parents=True, exist_ok=True)
    row_file = out / "row.json"
    row_file.unlink(missing_ok=True)
    cmd = [sys.executable, "-m", "wobbl.validation.process_generality", "--one", case["id"], str(out)]
    if keep:
        cmd.append("--keep")
    t0 = time.time()
    log = open(out / "case.log", "wb")
    posix = os.name == "posix"
    proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                            start_new_session=posix)
    timed_out = False
    try:
        proc.wait(timeout=limit)
    except subprocess.TimeoutExpired:
        timed_out = True
        _stop_group(proc)
    finally:
        log.close()
    if timed_out:
        row = {"id": case["id"], "part": case["part_id"], "source": case["source"], "process": case["process"],
               "ran": False, "ok": False, "timed_out": True, "seconds": round(time.time() - t0, 1),
               "error": f"timed out after {limit:.0f} s and was stopped"}
    elif row_file.exists():
        row = json.loads(row_file.read_text())
    else:
        tail = (out / "case.log").read_text(errors="replace").strip().splitlines()[-3:]
        row = {"id": case["id"], "part": case["part_id"], "source": case["source"], "process": case["process"],
               "ran": False, "ok": False, "seconds": round(time.time() - t0, 1),
               "error": f"the case process ended with exit code {proc.returncode} and no answer: " + " ".join(tail)}
    if not keep:
        shutil.rmtree(out, ignore_errors=True)
    return row


def _stop_group(proc: subprocess.Popen) -> None:
    """Stop the child this harness started and everything in its process group, the group id being the child's own
    pid, never a process matched by name."""
    try:
        if os.name == "posix":
            os.killpg(proc.pid, signal.SIGKILL)
        else:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True)
    except (ProcessLookupError, PermissionError):
        pass
    proc.kill() if proc.poll() is None else None
    proc.wait()


def _one(case_id: str, out: Path, keep: bool) -> int:
    case = next(c for c in CASES if c["id"] == case_id)
    row = run_case(case, out / "work", keep=True)
    (out / "row.json").write_text(json.dumps(row, default=str))
    if not keep:
        shutil.rmtree(out / "work", ignore_errors=True)
    return 0


def main(argv=None) -> int:
    import argparse
    from concurrent.futures import ThreadPoolExecutor

    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--only", default="")
    ap.add_argument("--process", default="")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--limit", type=float, default=LIMIT, help="seconds a case may run before it is killed")
    ap.add_argument("--one", default="", help="run one case id in this process and write row.json (used internally)")
    ap.add_argument("--keep", action="store_true")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.one:
        return _one(a.one, out, a.keep)
    ids = [x for x in a.only.split(",") if x]
    procs = [x for x in a.process.split(",") if x]
    cases = [c for c in CASES if (not ids or c["id"] in ids or c["part_id"] in ids)
             and (not procs or c["process"] in procs)]
    rows: dict[str, dict] = {}
    lock = threading.Lock()

    def work(c):
        row = run_limited(c, out / c["id"], a.limit, a.keep)
        with lock:
            rows[c["id"]] = row
            done = [rows[x["id"]] for x in cases if x["id"] in rows]
            (out / "process_generality.json").write_text(json.dumps(done, indent=1, default=str))
            mark = "ok  " if row["ok"] else ("TIME" if row.get("timed_out") else ("ran " if row.get("ran") else "FAIL"))
            print(f"{mark} {row['id']:<26} {row.get('verdict', ''):<10} {row.get('confidence', ''):<7} "
                  f"{row.get('seconds', '')}s  {row.get('error') or '; '.join(row.get('failed_checks', []))}",
                  flush=True)

    with ThreadPoolExecutor(max(a.workers, 1)) as pool:
        list(pool.map(work, cases))
    allrows = [rows[c["id"]] for c in cases]
    for proc in PROCESSES:
        mine = [r for r in allrows if r.get("process") == proc]
        if mine:
            print(f"{proc}: {sum(r['ok'] for r in mine)} of {len(mine)} passed their own checks, "
                  f"{sum(r.get('ran', False) for r in mine)} ran to an answer, "
                  f"{sum(r.get('timed_out', False) for r in mine)} timed out")
    return 0


if __name__ == "__main__":
    sys.exit(main())
