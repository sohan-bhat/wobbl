"""CalculiX adapter: linear static stress and heat transfer with 10 node tetrahedra.

Writes an Abaqus style .inp, runs ``ccx``, and reads the ASCII .frd results.
Units are Wobbl's internal system (mm, N, MPa, t, s, mW, degC).
"""
from __future__ import annotations

import io
import re
import subprocess
import time
from pathlib import Path

import numpy as np

from ..fe import HeatModel, HeatResult, StaticModel, StaticResult
from .base import SolverAdapter, SolverError, register
from .locate import find_tool, tool_env

CCX_NAMES = ["ccx", "ccx_2.23", "ccx2.23", "ccx_2.22", "ccx_2.21", "ccx_static", "calculix"]


class CalculiX(SolverAdapter):
    name = "CalculiX"
    domains = ("structural", "thermal")

    def locate(self) -> Path | None:
        return find_tool(CCX_NAMES, "WOBBL_CCX")

    def version(self) -> str:
        p = self.locate()
        if not p:
            return ""
        try:
            r = subprocess.run([str(p), "-v"], capture_output=True, text=True, timeout=20, env=tool_env(p))
            m = re.search(r"[Vv]ersion\s+([\d.]+)", r.stdout + r.stderr)
            return m.group(1) if m else ""
        except Exception:  # noqa: BLE001
            return ""

    # ------------------------------------------------------------------ static
    def solve_static(self, model: StaticModel, workdir: str | Path, job: str = "static",
                     timeout: float = 3600) -> StaticResult:
        ccx = self.require()
        work = Path(workdir)
        work.mkdir(parents=True, exist_ok=True)
        n = model.mesh.n_nodes
        extra = write_static_inp(work / f"{job}.inp", model)
        log, t = _run(ccx, work, job, timeout)
        blocks = read_frd(work / f"{job}.frd", n + 2 * len(model.rigid))
        if "DISP" not in blocks or "STRESS" not in blocks:
            raise SolverError("CalculiX finished without displacement or stress results", log=log[-3000:])
        U_all = blocks["DISP"][-1][1][:, :3]
        S = blocks["STRESS"][-1][1][:n, :6]
        RF = blocks["FORC"][-1][1][:n, :3] if "FORC" in blocks else np.zeros((n, 3))
        U = U_all[:n]
        size = float(np.linalg.norm(model.mesh.X.max(0) - model.mesh.X.min(0)))
        if not np.isfinite(U).all() or np.abs(U).max() > 1e3 * size:
            raise SolverError("the part is not held in place (it can move or spin freely)",
                              "Add fixtures that stop sliding and rotation, for example fix the bolt holes "
                              "or a whole flat face", log[-2000:])
        res = StaticResult(U, S, RF, t_solve=t, log=log, n_dof=3 * n, solver=f"CalculiX {self.version()}")
        for name, (ref, rot) in extra.items():
            res.extra_U[name] = np.concatenate([U_all[ref], U_all[rot]])
        return res

    # -------------------------------------------------------------------- heat
    def solve_heat(self, model: HeatModel, workdir: str | Path, job: str = "heat",
                   timeout: float = 3600) -> HeatResult:
        ccx = self.require()
        work = Path(workdir)
        work.mkdir(parents=True, exist_ok=True)
        write_heat_inp(work / f"{job}.inp", model)
        log, t = _run(ccx, work, job, timeout)
        blocks = read_frd(work / f"{job}.frd", model.mesh.n_nodes)
        if "NDTEMP" not in blocks:
            raise SolverError("CalculiX finished without temperature results", log=log[-3000:])
        steps = blocks["NDTEMP"]
        times = np.array([s[0] for s in steps])
        hist = np.array([s[1][:, 0] for s in steps])
        flux = blocks["FLUX"][-1][1][:, :3] if "FLUX" in blocks else None
        rfl = blocks["RFL"][-1][1][:, 0] if "RFL" in blocks else None
        return HeatResult(hist[-1], times, hist, flux, t, log, f"CalculiX {self.version()}", rfl)


def _run(ccx: Path, work: Path, job: str, timeout: float) -> tuple[str, float]:
    t0 = time.time()
    for ext in (".frd", ".dat", ".sta", ".cvg"):
        (work / f"{job}{ext}").unlink(missing_ok=True)
    try:
        r = subprocess.run([str(ccx), "-i", job], cwd=work, capture_output=True, text=True, timeout=timeout,
                           env=tool_env(ccx))
    except subprocess.TimeoutExpired:
        raise SolverError(f"CalculiX did not finish within {timeout:.0f} s",
                          "Use a coarser mesh (--quality fast) or a longer --budget") from None
    log = (r.stdout or "") + (r.stderr or "")
    errs = [ln.strip() for ln in log.splitlines() if "*ERROR" in ln]
    if r.returncode != 0 or errs or not (work / f"{job}.frd").exists():
        raise _explain(errs, log, r.returncode)
    return log, time.time() - t0


def _explain(errs: list[str], log: str, code: int) -> SolverError:
    text = " ".join(errs) or log[-500:]
    low = text.lower()
    if "jacobian" in low:
        return SolverError("some mesh elements are inverted (nonpositive jacobian)",
                           "Wobbl will retry with straight sided elements; if this persists, simplify tiny "
                           "features or pass --mesh-size", log[-3000:])
    if "singular" in low or "zero pivot" in low or "rigid body" in low:
        return SolverError("the stiffness matrix is singular: the part is not fully held",
                           "Add fixtures that stop all sliding and rotation", log[-3000:])
    if "memory" in low or "alloc" in low:
        return SolverError("CalculiX ran out of memory", "Use --quality fast or a larger --mesh-size", log[-3000:])
    return SolverError(f"CalculiX failed (exit {code}): {text[:400]}", log=log[-3000:])


# ---------------------------------------------------------------------- writers
def _write_mesh(f, model_mesh, extra_nodes: list[np.ndarray] | None = None):
    X = model_mesh.X
    f.write("*NODE, NSET=NALL\n")
    buf = io.StringIO()
    np.savetxt(buf, np.column_stack([np.arange(1, len(X) + 1), X]), fmt="%d,%.10g,%.10g,%.10g")
    f.write(buf.getvalue())
    for i, p in enumerate(extra_nodes or []):
        f.write(f"{len(X) + i + 1},{p[0]:.10g},{p[1]:.10g},{p[2]:.10g}\n")
    etype = "C3D10" if model_mesh.tets.shape[1] == 10 else "C3D4"
    f.write(f"*ELEMENT, TYPE={etype}, ELSET=EALL\n")
    buf = io.StringIO()
    T = model_mesh.tets + 1
    np.savetxt(buf, np.column_stack([np.arange(1, len(T) + 1), T]), fmt="%d", delimiter=",")
    f.write(buf.getvalue())


def _nset(f, name: str, nodes: np.ndarray):
    f.write(f"*NSET, NSET={name}\n")
    ids = np.asarray(nodes, dtype=np.int64) + 1
    for i in range(0, len(ids), 12):
        f.write(", ".join(map(str, ids[i:i + 12])) + "\n")


def write_static_inp(path: Path, model: StaticModel) -> dict[str, tuple[int, int]]:
    """Write the .inp. Returns rigid body name -> (ref node index, rot node index), 0 based."""
    mesh = model.mesh
    n = mesh.n_nodes
    extra_pts, extra = [], {}
    for k, r in enumerate(model.rigid):
        extra_pts += [np.asarray(r.ref_point, float), np.asarray(r.ref_point, float)]
        extra[r.name] = (n + 2 * k, n + 2 * k + 1)
    with open(path, "w") as f:
        f.write("*HEADING\nWobbl static stress\n")
        _write_mesh(f, mesh, extra_pts)
        for c in model.constraints:
            _nset(f, _setname(c.name), c.nodes)
        for k, r in enumerate(model.rigid):
            _nset(f, f"RB{k + 1}", r.nodes)
        f.write("*MATERIAL, NAME=MAT\n*ELASTIC\n" + f"{model.E:.10g},{model.nu:.10g}\n")
        f.write(f"*DENSITY\n{model.density:.10g}\n")
        f.write("*SOLID SECTION, ELSET=EALL, MATERIAL=MAT\n")
        for k, r in enumerate(model.rigid):
            ref, rot = extra[r.name]
            f.write(f"*RIGID BODY, NSET=RB{k + 1}, REF NODE={ref + 1}, ROT NODE={rot + 1}\n")
        fixed = {}
        for c in model.constraints:
            for d in c.dofs:
                for i in c.nodes:
                    fixed.setdefault(int(i), set()).add(d)
        eqs = []
        for nc in model.normal_constraints:
            nrm = np.asarray(nc.normal, float)
            nrm = nrm / np.linalg.norm(nrm)
            order = np.argsort(-np.abs(nrm))
            for i in nc.nodes:
                i = int(i)
                have = fixed.setdefault(i, set())
                dep = next((int(d) + 1 for d in order if abs(nrm[d]) > 1e-6 and (int(d) + 1) not in have), None)
                if dep is None:
                    continue
                terms = [(dep, nrm[dep - 1])] + [(d + 1, nrm[d]) for d in range(3)
                                                 if d + 1 != dep and abs(nrm[d]) > 1e-9]
                if len(terms) == 1:
                    have.add(dep)
                    continue
                have.add(dep)
                eqs.append((i, terms))
        if eqs:
            f.write("*EQUATION\n")
            for i, terms in eqs:
                f.write(f"{len(terms)}\n" + ",".join(f"{i + 1},{d},{c:.10g}" for d, c in terms) + "\n")
        f.write("*STEP\n*STATIC\n*BOUNDARY\n")
        for c in model.constraints:
            for d in c.dofs:
                f.write(f"{_setname(c.name)},{d},{d}\n")
        for nc in model.normal_constraints:  # single dof rollers that did not need an equation
            nrm = np.asarray(nc.normal, float)
            nz = np.nonzero(np.abs(nrm) > 1e-9)[0]
            if len(nz) == 1:
                for i in nc.nodes:
                    f.write(f"{int(i) + 1},{nz[0] + 1},{nz[0] + 1}\n")
        F = model.nodal_forces()
        rows = np.nonzero(np.abs(F).max(1) > 0)[0]
        if len(rows) or model.rigid:
            f.write("*CLOAD\n")
            for i in rows:
                for d in range(3):
                    if F[i, d] != 0.0:
                        f.write(f"{i + 1},{d + 1},{F[i, d]:.10e}\n")
            for r in model.rigid:
                ref, rot = extra[r.name]
                for d in range(3):
                    if r.force[d] != 0:
                        f.write(f"{ref + 1},{d + 1},{r.force[d]:.10e}\n")
                    if r.moment[d] != 0:
                        f.write(f"{rot + 1},{d + 1},{r.moment[d]:.10e}\n")
        if model.gravity is not None and np.linalg.norm(model.gravity) > 0:
            g = np.asarray(model.gravity, float)
            mag = np.linalg.norm(g)
            d = g / mag
            f.write(f"*DLOAD\nEALL,GRAV,{mag:.10g},{d[0]:.10g},{d[1]:.10g},{d[2]:.10g}\n")
        f.write("*NODE FILE\nU,RF\n*EL FILE\nS\n*END STEP\n")
    return extra


def write_heat_inp(path: Path, model: HeatModel):
    mesh = model.mesh
    with open(path, "w") as f:
        f.write("*HEADING\nWobbl heat transfer\n")
        _write_mesh(f, mesh)
        for k, (name, nodes, temp) in enumerate(model.fixed_temps):
            _nset(f, f"TFIX{k + 1}", nodes)
        f.write("*MATERIAL, NAME=MAT\n")
        f.write(f"*CONDUCTIVITY\n{model.conductivity:.10g}\n")
        f.write(f"*SPECIFIC HEAT\n{model.specific_heat:.10g}\n")
        f.write(f"*DENSITY\n{model.density:.10g}\n")
        f.write("*SOLID SECTION, ELSET=EALL, MATERIAL=MAT\n")
        f.write("*PHYSICAL CONSTANTS, ABSOLUTE ZERO=-273.15, STEFAN BOLTZMANN=5.670374E-11\n")
        f.write(f"*INITIAL CONDITIONS, TYPE=TEMPERATURE\nNALL,{model.initial_temp:.10g}\n")
        if model.transient:
            dt = model.time_step or model.duration / 100.0
            if model.output_times:
                f.write("*TIME POINTS, NAME=TP\n")
                tp = [t for t in model.output_times if 0 < t <= model.duration]
                for i in range(0, len(tp), 8):
                    f.write(", ".join(f"{t:.10g}" for t in tp[i:i + 8]) + "\n")
            f.write("*STEP, INC=100000\n*HEAT TRANSFER, DELTMX=2.0\n")
            f.write(f"{dt:.10g},{model.duration:.10g},{dt * 1e-4:.10g},{model.duration / 20.0:.10g}\n")
        else:
            f.write("*STEP\n*HEAT TRANSFER, STEADY STATE\n1.,1.\n")
        if model.fixed_temps:
            f.write("*BOUNDARY\n")
            for k, (name, nodes, temp) in enumerate(model.fixed_temps):
                f.write(f"TFIX{k + 1},11,11,{temp:.10g}\n")
        if model.films:
            f.write("*FILM\n")
            for name, mask, h, tinf in model.films:
                for e, fc in zip(mesh.tri_elem[mask], mesh.tri_face[mask]):
                    f.write(f"{e + 1},F{fc},{tinf:.10g},{h:.10g}\n")
        if model.fluxes or model.volumetric:
            f.write("*DFLUX\n")
            for name, mask, q in model.fluxes:
                for e, fc in zip(mesh.tri_elem[mask], mesh.tri_face[mask]):
                    f.write(f"{e + 1},S{fc},{q:.10g}\n")
            if model.volumetric:
                f.write(f"EALL,BF,{model.volumetric:.10g}\n")
        if model.radiation:
            f.write("*RADIATE\n")
            for name, mask, eps, tenv in model.radiation:
                for e, fc in zip(mesh.tri_elem[mask], mesh.tri_face[mask]):
                    f.write(f"{e + 1},R{fc},{tenv:.10g},{eps:.10g}\n")
        tp = ", TIME POINTS=TP" if (model.transient and model.output_times) else ""
        f.write(f"*NODE FILE{tp}\nNT,RFL\n*EL FILE{tp}\nHFL\n*END STEP\n")


def _setname(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_]", "_", name).upper()
    return ("N" + s)[:70]


# ----------------------------------------------------------------------- reader
def read_frd(path: str | Path, n: int) -> dict[str, list[tuple[float, np.ndarray]]]:
    """Parse an ASCII .frd into {block name: [(time, (n, k) array), ...]} for nodal result blocks."""
    out: dict[str, list[tuple[float, np.ndarray]]] = {}
    cur_name, cur_time, rows, ncomp = None, 0.0, None, 0
    with open(path, "r", errors="replace") as fh:
        for line in fh:
            if line.startswith("  100C"):
                parts = line.split()
                try:
                    cur_time = float(parts[2])
                except (IndexError, ValueError):
                    cur_time = 0.0
            elif line.startswith(" -4"):
                parts = line.split()
                cur_name, ncomp = parts[1], int(parts[2])
                rows = []
            elif line.startswith(" -3"):
                if cur_name is not None and rows is not None:
                    arr = np.zeros((n, max(ncomp, 1)))
                    if rows:
                        ids = np.array([r[0] for r in rows], dtype=np.int64) - 1
                        vals = [r[1] for r in rows]
                        width = max(len(v) for v in vals)
                        mat = np.zeros((len(vals), width))
                        for i, v in enumerate(vals):
                            mat[i, :len(v)] = v
                        keep = (ids >= 0) & (ids < n)
                        arr[ids[keep], :min(width, arr.shape[1])] = mat[keep, :arr.shape[1]]
                    out.setdefault(cur_name, []).append((cur_time, arr))
                cur_name, rows = None, None
            elif rows is not None and line.startswith(" -1"):
                nid = int(line[3:13])
                body = line[13:].rstrip("\n")
                vals = [float(body[i:i + 12]) for i in range(0, len(body) - 11, 12)]
                rows.append((nid, vals))
            elif rows is not None and line.startswith(" -2") and rows:
                body = line[13:].rstrip("\n")
                rows[-1] = (rows[-1][0], rows[-1][1] + [float(body[i:i + 12]) for i in range(0, len(body) - 11, 12)])
    return out


register(CalculiX())
