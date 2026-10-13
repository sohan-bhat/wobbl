"""Solve the website's motor mount for a load on the motor shaft in any direction.

The hero lets visitors point the load on the shaft any way in 3D. The model is
linear elastic, so a load of F newtons along the unit direction u is exactly

    F * (u1 * case1 + u2 * case2 + u3 * case3)

where case1 and case2 are unit loads along two perpendicular directions in the
plane the pulley turns in and case3 is a unit load along the shaft. This script
runs the example's own study (the arguments are read from its run.sh), takes the
mesh Wobbl converged on for the default direction, and solves the three unit
loads on that same mesh, so the results line up node for node. Nothing else
about the setup changes.

    python scripts/solve_site_directions.py --example examples/cast-motor-mount --out /tmp/dirs
    python scripts/solve_site_directions.py --example examples/cast-motor-mount --out /tmp/dirs \\
        --skip-cases --converged 0,60,90,120,150,0/90,30/45,120/-30

A direction is an azimuth in the pulley plane, measured from the horizontal and
positive below it with the default pull's horizontal side as zero, and an
elevation toward the shaft axis, both in degrees: the example's pull "30 degrees
below horizontal" is 30/0, and a load straight along the shaft is any/90.
Opposite loads flip the sign of every stress, so von Mises, the peak and the
verdict are the same for u and -u; only the way the part bends turns around.

``--converged`` then runs Wobbl's real study, with its own mesh refinement, for
each listed direction, and compares the converged peak with the superposed one
on the default mesh. That is the honest check of how well a mesh refined for one
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


def direction(e1, e2, deg: float, e3=None, el: float = 0.0) -> np.ndarray:
    t, h = math.radians(deg), math.radians(el)
    d = math.cos(h) * (math.cos(t) * e1 + math.sin(t) * e2) + (math.sin(h) * e3 if e3 is not None else 0.0)
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
    for name, e in (("case1", e1), ("case2", e2), ("case3", n)):
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
        "e3": n.round(9).tolist(),
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


def superposed_peak(out: Path, d: np.ndarray, info: dict) -> tuple[float, list[float]]:
    """Governing von Mises for a load of load_N along d, from the unit cases, away from excluded zones."""
    import pyvista as pv

    basis = [np.asarray(info[k]) for k in ("e1", "e2", "e3")]
    S = 0.0
    for k, e in enumerate(basis, start=1):
        c = pv.read(out / f"case{k}.vtu")
        S = S + float(d @ e) * np.asarray(c.point_data["stress_MPa"])
        if k == 1:
            ok = np.asarray(c.point_data["excluded_zone"]) < 0.5
            pts = np.asarray(c.points)
    vm = von_mises(info["load_N"] * S)
    i = np.flatnonzero(ok)[np.argmax(vm[ok])]
    return float(vm[i]), pts[i].round(3).tolist()


def converged(example: Path, out: Path, specs: list[str]) -> list[dict]:
    info = json.loads((out / "cases.json").read_text())
    e1, e2, e3 = (np.asarray(info[k]) for k in ("e1", "e2", "e3"))
    rows = []
    for spec in specs:
        az, el = (float(v) for v in (spec.split("/") + ["0"])[:2])
        d = direction(e1, e2, az, e3, el)
        args = example_args(example)
        i = args.index("--load")
        text = args[i + 1].split()
        text[2] = ",".join(f"{v:.7f}".rstrip("0").rstrip(".") if v else "0" for v in d)
        args[i + 1] = " ".join(text)
        folder = out / f"converged-{az:g}-{el:g}"
        args = with_option(args, "-o", str(folder)) + ["--no-images", "--json", "--quiet"]
        print(f"converged study at {az:g}/{el:g} deg: {args[i + 1]}", flush=True)
        t0 = time.time()
        p = subprocess.run([sys.executable, "-m", "wobbl.cli.main", "run", *args], capture_output=True, text=True)
        if p.returncode:
            print(p.stdout[-2000:], p.stderr[-2000:])
            raise SystemExit(f"the study at {az:g}/{el:g} deg failed")
        o = json.loads((folder / "outcome.json").read_text())
        shutil.rmtree(folder / "work", ignore_errors=True)
        for f in ("result.vtu", "report.html"):
            (folder / f).unlink(missing_ok=True)
        sup, sup_xyz = superposed_peak(out, d, info)
        conv = float(o["governing_MPa"])
        row = {
            "azimuth_deg": az,
            "elevation_deg": el,
            "direction": d.round(7).tolist(),
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
        key = (az, el)
        keep = [r for r in prev if (r["azimuth_deg"], r["elevation_deg"]) != key] + [row]
        keep.sort(key=lambda r: (r["elevation_deg"], r["azimuth_deg"]))
        (out / "accuracy.json").write_text(json.dumps(keep, indent=1) + "\n")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--example", type=Path, default=Path("examples/cast-motor-mount"))
    ap.add_argument("--out", type=Path, required=True, help="scratch folder for the unit cases and studies")
    ap.add_argument("--converged", default="", help="directions to run full studies for, azimuth/elevation in degrees, "
                                                    "e.g. 0,90,0/90,30/45")
    ap.add_argument("--skip-cases", action="store_true", help="reuse the unit cases already in --out")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    example = a.example.resolve()
    if not a.skip_cases:
        solve_cases(example, a.out)
    if a.converged:
        converged(example, a.out, a.converged.split(","))


if __name__ == "__main__":
    main()
