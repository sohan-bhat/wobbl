"""Solve the website's motor mount for a belt pull in any direction in the pulley plane.

The hero lets visitors turn the belt pull around the pulley. The model is linear
elastic, so a pull of F newtons at angle t is exactly

    F * (cos t * case1 + sin t * case2)

where case1 and case2 are unit pulls along two perpendicular directions in the
plane the pulley turns in. This script runs the example's own study (the
arguments are read from its run.sh), takes the mesh Wobbl converged on for the
default direction, and solves both unit pulls on that same mesh, so the two
results line up node for node. Nothing else about the setup changes.

    python scripts/solve_site_directions.py --example examples/cast-motor-mount --out /tmp/dirs
    python scripts/solve_site_directions.py --example examples/cast-motor-mount --out /tmp/dirs \\
        --skip-cases --converged 0,60,90,120,150

The angle is measured in the pulley plane from the horizontal, positive below
it, with the default pull's horizontal side as zero, so the example's pull
"30 degrees below horizontal" is t = 30. Opposite pulls flip the sign of every
stress, so von Mises, the peak and the verdict repeat every 180 degrees; only
the way the part bends turns around.

``--converged`` then runs Wobbl's real study, with its own mesh refinement, for
each listed angle, and compares the converged peak with the superposed one on
the default mesh. That is the honest check of how well a mesh refined for one
direction serves the others. Each run's solver scratch is deleted when it
finishes; only its outcome and summary stay.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np


def example_args(example: Path) -> list[str]:
    """The arguments run.sh passes to 'wobbl run', with the part path made absolute."""
    text = (example / "run.sh").read_text().replace("\\\n", " ")
    env = {}
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("PART="):
            env["PART"] = shlex.split(line[5:])[0]
        if line.startswith("wobbl run"):
            args = shlex.split(line)[2:]
            break
    else:
        raise SystemExit(f"no 'wobbl run' line in {example / 'run.sh'}")
    out = []
    for a in args:
        if a == "$PART":
            a = str((example / env["PART"]).resolve())
        out.append(a)
    return out


def with_option(args: list[str], name: str, value: str | None) -> list[str]:
    """Replace (or with None, drop) an option and its value."""
    out, skip = [], False
    for i, a in enumerate(args):
        if skip:
            skip = False
            continue
        if a == name:
            skip = True
            continue
        out.append(a)
    return out + ([name, value] if value is not None else [])


def basis(load, features, sel) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Shaft axis, the two in plane unit directions and the default pull's angle in degrees."""
    axes = [np.asarray(features.faces[f].axis, float) for f in sel.faces if features.faces[f].axis is not None]
    if not axes:
        raise SystemExit("the load's faces have no axis, so the pulley plane is unknown")
    # hole axes are fitted to a surface mesh, so they are a fraction of a degree off; a shaft along a
    # coordinate axis snaps to it
    n = np.sum([a * np.sign(a @ axes[0]) for a in axes], axis=0)
    n /= np.linalg.norm(n)
    k = int(np.argmax(np.abs(n)))
    if abs(n[k]) > math.cos(math.radians(1.0)):
        n = np.eye(3)[k] * np.sign(n[k])
    d0 = np.asarray(load.direction, float)
    d0 /= np.linalg.norm(d0)
    if abs(d0 @ n) > 1e-6:
        raise SystemExit(f"the default pull {d0} is not in the pulley plane (axis {n})")
    down = np.array([0.0, 0.0, -1.0])
    e2 = down - (down @ n) * n
    e2 /= np.linalg.norm(e2)
    e1 = np.cross(e2, n)
    if e1 @ d0 < 0:
        e1 = -e1
    theta0 = math.degrees(math.atan2(d0 @ e2, d0 @ e1))
    return n, e1, e2, theta0


def direction(e1, e2, deg: float) -> np.ndarray:
    t = math.radians(deg)
    d = math.cos(t) * e1 + math.sin(t) * e2
    d[np.abs(d) < 1e-12] = 0.0
    return d


def von_mises(S: np.ndarray) -> np.ndarray:
    sxx, syy, szz, sxy, syz, szx = S.T
    return np.sqrt(0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2) + 3.0 * (sxy ** 2 + syz ** 2 + szx ** 2))


def solve_cases(example: Path, out: Path) -> dict:
    from wobbl.cli.main import _study_from, run as run_cmd
    from wobbl.structural.build import build_static
    from wobbl.structural.pipeline import _save_vtu
    from wobbl.structural.run import _solve, describe_part, resolve_selections, run_structural
    from wobbl.solvers import base as solvers

    args = example_args(example)
    kw = run_cmd.make_context("run", list(args)).params
    st = _study_from(kw.pop("part"), kw.get("question", ""), kw)
    work = out / "work"
    say = lambda stage, detail="": print(f"  {stage:<8} {detail}", flush=True)  # noqa: E731
    t0 = time.time()
    pre = describe_part(st.part, work, st.scale, st.body, st.params)
    run = run_structural(st, work, say, pre=pre)
    print(f"default study: {run.levels[-1].governing:.4f} MPa on {run.mesh.n_nodes:,} nodes "
          f"({time.time() - t0:.0f} s, stopped by {run.stopped_by})")
    _save_vtu(run, out / "default.vtu")

    features = pre[1]
    loads, fixtures, _ = resolve_selections(st, features)
    k = next(i for i, (ld, _) in enumerate(loads) if ld.kind == "remote")
    ld0, sel0 = loads[k]
    n, e1, e2, theta0 = basis(ld0, features, sel0)
    adapter = solvers.for_domain("structural")
    cases = {}
    for name, e in (("case1", e1), ("case2", e2)):
        unit = copy.copy(ld0)
        unit.direction, unit.magnitude = e.tolist(), 1.0
        alt = list(loads)
        alt[k] = (unit, sel0)
        built = build_static(run.mesh, alt, fixtures, run.material, features.faces)
        res, _, _ = _solve(adapter, built, work / name, None)
        fapp, _ = built.model.applied_total()
        print(f"{name}: unit pull along {np.round(e, 7).tolist()}, applied {np.round(fapp, 9).tolist()} N")
        _save_vtu(replace(run, result=res, measure=res.von_mises()), out / f"{name}.vtu")
        cases[name] = res

    # the default pull must come back from the two unit cases
    t = math.radians(theta0)
    S = ld0.magnitude * (math.cos(t) * cases["case1"].S + math.sin(t) * cases["case2"].S)
    U = ld0.magnitude * (math.cos(t) * cases["case1"].U + math.sin(t) * cases["case2"].U)
    ds = float(np.abs(S - run.result.S).max() / np.abs(run.result.S).max())
    du = float(np.abs(U - run.result.U).max() / np.abs(run.result.U).max())
    vm = von_mises(S)
    gov = float(vm[~run.excluded].max())
    print(f"superposed at {theta0:.4f} deg: governing {gov:.4f} MPa against {run.levels[-1].governing:.4f}; "
          f"largest stress difference {ds:.2e}, displacement {du:.2e} of the peak")

    info = {
        "args": args,
        "load_N": ld0.magnitude,
        "point": ld0.point,
        "axis": n.round(9).tolist(),
        "e1": e1.round(9).tolist(),
        "e2": e2.round(9).tolist(),
        "theta0_deg": round(theta0, 6),
        "nodes": run.mesh.n_nodes,
        "governing_MPa": run.levels[-1].governing,
        "superposed_governing_MPa": gov,
        "superposition_error": {"stress": ds, "displacement": du},
        "stopped_by": run.stopped_by,
    }
    (out / "cases.json").write_text(json.dumps(info, indent=1) + "\n")
    shutil.rmtree(work, ignore_errors=True)
    return info


def superposed_peak(out: Path, deg: float, load_n: float) -> tuple[float, list[float]]:
    import pyvista as pv

    c1, c2 = pv.read(out / "case1.vtu"), pv.read(out / "case2.vtu")
    t = math.radians(deg)
    S = load_n * (math.cos(t) * np.asarray(c1.point_data["stress_MPa"]) +
                  math.sin(t) * np.asarray(c2.point_data["stress_MPa"]))
    vm = von_mises(S)
    ok = np.asarray(c1.point_data["excluded_zone"]) < 0.5
    i = np.flatnonzero(ok)[np.argmax(vm[ok])]
    return float(vm[i]), np.asarray(c1.points[i]).round(3).tolist()


def converged(example: Path, out: Path, angles: list[float]) -> list[dict]:
    info = json.loads((out / "cases.json").read_text())
    e1, e2 = np.asarray(info["e1"]), np.asarray(info["e2"])
    rows = []
    for deg in angles:
        d = direction(e1, e2, deg)
        args = example_args(example)
        i = args.index("--load")
        text = args[i + 1].split()
        text[2] = ",".join(f"{v:.7f}".rstrip("0").rstrip(".") if v else "0" for v in d)
        args[i + 1] = " ".join(text)
        folder = out / f"converged-{deg:g}"
        args = with_option(args, "-o", str(folder)) + ["--no-images", "--json", "--quiet"]
        print(f"converged study at {deg:g} deg: {args[i + 1]}", flush=True)
        t0 = time.time()
        p = subprocess.run([sys.executable, "-m", "wobbl.cli.main", "run", *args], capture_output=True, text=True)
        if p.returncode:
            print(p.stdout[-2000:], p.stderr[-2000:])
            raise SystemExit(f"the study at {deg:g} deg failed")
        o = json.loads((folder / "outcome.json").read_text())
        shutil.rmtree(folder / "work", ignore_errors=True)
        for f in ("result.vtu", "report.html"):
            (folder / f).unlink(missing_ok=True)
        sup, sup_xyz = superposed_peak(out, deg, info["load_N"])
        conv = float(o["governing_MPa"])
        row = {
            "angle_deg": deg,
            "converged_MPa": round(conv, 3),
            "converged_xyz": [round(v, 3) for v in o["governing_xyz"]],
            "converged_where": o.get("governing_where", ""),
            "superposed_MPa": round(sup, 3),
            "superposed_xyz": sup_xyz,
            "difference_pct": round(100 * (sup - conv) / conv, 2),
            "levels": [round(lv["governing"], 3) for lv in o["levels"]],
            "nodes": [lv["n_nodes"] for lv in o["levels"]],
            "verdict": o["verdict"],
            "seconds": round(time.time() - t0),
        }
        print(f"  converged {conv:.3f} MPa at {row['converged_xyz']}, superposed {sup:.3f} MPa at {sup_xyz}, "
              f"{row['difference_pct']:+.2f}% ({row['seconds']} s)", flush=True)
        rows.append(row)
        prev = json.loads((out / "accuracy.json").read_text()) if (out / "accuracy.json").exists() else []
        keep = [r for r in prev if r["angle_deg"] != deg] + [row]
        (out / "accuracy.json").write_text(json.dumps(sorted(keep, key=lambda r: r["angle_deg"]), indent=1) + "\n")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--example", type=Path, default=Path("examples/cast-motor-mount"))
    ap.add_argument("--out", type=Path, required=True, help="scratch folder for the unit cases and studies")
    ap.add_argument("--converged", default="", help="angles in degrees to run full studies for, e.g. 0,90,180,270")
    ap.add_argument("--skip-cases", action="store_true", help="reuse the unit cases already in --out")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    example = a.example.resolve()
    if not a.skip_cases:
        solve_cases(example, a.out)
    if a.converged:
        converged(example, a.out, [float(x) for x in a.converged.split(",")])


if __name__ == "__main__":
    main()
