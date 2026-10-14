"""CalculiX for nonlinear statics: elastic plastic materials, load and unload steps, contact with friction and bolt
pretension.

The linear adapter in calculix.py writes one step and reads one result. A run here has several steps, each one
ramping the model's loads to a multiple of their value, and the solver may stop inside a step on purpose: a load
ramp toward collapse ends where CalculiX can no longer find equilibrium, and the increments it did finish are the
answer. Totals and monitor points go to the small .dat file at every increment, full fields to the .frd only where
they are needed, so the work folder stays small.
"""
from __future__ import annotations

import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..fe import StaticModel
from .base import SolverError
from .calculix import _explain, _nset, _setname, _write_mesh, read_frd
from .locate import tool_env

THREADS = 2   # nonlinear runs iterate many times on small meshes, where more threads barely help


@dataclass
class Material:
    """One material of a nonlinear run. ``hardening`` lists (true stress MPa, plastic strain) from yield on; an
    empty list keeps the material elastic. ``elements`` picks the tetrahedra it fills (all when None)."""
    name: str
    E: float
    nu: float
    density: float = 7.85e-9
    hardening: list[tuple[float, float]] = field(default_factory=list)
    elements: np.ndarray | None = None


@dataclass
class ContactPair:
    """Face to face penalty contact between two element face sets, with Coulomb friction when mu > 0."""
    name: str
    slave: np.ndarray            # boundary triangle mask of the slave side (the finer or smaller surface)
    master: np.ndarray           # boundary triangle mask of the master side
    stiffness: float             # penalty slope K, MPa/mm
    mu: float = 0.0
    stick: float = 0.0           # stick slope lambda, MPa/mm (shear stress per unit slip before sliding)
    tension: float = 0.0         # sigma_inf, the small tension CalculiX keeps across open gaps, MPa
    clearance: float | None = None   # treat the surfaces as this far apart, whatever the CAD says (0: touching)


@dataclass
class Pretension:
    """A bolt pretension section: a cut through a bolt whose two sides are pulled together by ``force``."""
    name: str
    faces: np.ndarray            # (k, 2) element index and CalculiX face number on one side of the cut
    normal: np.ndarray
    force: float                 # N


@dataclass
class Step:
    """One load step. Loads ramp linearly from the previous step's multiple to ``factor``."""
    factor: float
    nlgeom: bool = False
    first: float = 0.1           # first increment, as a fraction of the step
    smallest: float = 1e-4       # smallest increment before CalculiX gives up, as a fraction of the step
    largest: float = 0.25
    max_increments: int = 300
    every_increment: bool = False   # full fields at every increment rather than only at the end of the step
    prescribed: list[tuple[str, int, float]] = field(default_factory=list)   # (node set, dof, value) this step
    preload: bool = False        # pull the pretension sections to their force in this step
    lock: bool = False           # hold the pretension sections at their length from now on
    may_stop: bool = False       # a ramp toward collapse: stopping inside this step is the expected end


@dataclass
class NonlinearModel:
    static: StaticModel                      # mesh, constraints, nodal forces, rigid couplings, gravity
    materials: list[Material]
    steps: list[Step]
    contact: list[ContactPair] = field(default_factory=list)
    springs: list[tuple[np.ndarray, float]] = field(default_factory=list)   # (nodes, stiffness each) to ground
    pretension: list[Pretension] = field(default_factory=list)
    monitor: dict[str, np.ndarray] = field(default_factory=dict)            # node sets whose U goes to the .dat
    node_sets: dict[str, np.ndarray] = field(default_factory=dict)          # extra named sets (prescribed steps)
    fields: tuple[str, ...] = ("U", "RF", "S", "PEEQ")


@dataclass
class NonlinearResult:
    times: np.ndarray                        # total time of every finished increment
    reactions: dict[str, np.ndarray]         # constraint set -> (increments, 3) total reaction
    monitor: dict[str, np.ndarray]           # monitor set -> (increments, nodes, 3) displacement
    frames: list[dict]                       # saved fields: time, U, S, PEEQ, contact arrays
    finished: bool                           # every step ran to its end
    log: str = ""
    t_solve: float = 0.0
    solver: str = ""
    increments: int = 0
    iterations: int = 0
    pretension_disp: dict[str, np.ndarray] = field(default_factory=dict)   # name -> (increments,) opening


def write_nonlinear_inp(path: Path, model: NonlinearModel) -> dict:
    """Write the deck. Returns names the reader needs: rigid body nodes, pretension nodes, constraint sets."""
    sm = model.static
    mesh = sm.mesh
    n = mesh.n_nodes
    extra_pts, rigid = [], {}
    for k, r in enumerate(sm.rigid):
        extra_pts += [np.asarray(r.ref_point, float), np.asarray(r.ref_point, float)]
        rigid[r.name] = (n + 2 * k, n + 2 * k + 1)
    pre_nodes = {}
    for k, p in enumerate(model.pretension):
        c = mesh.X[mesh.tets[p.faces[:, 0], :4]].mean(1).mean(0)
        pre_nodes[p.name] = n + len(extra_pts)
        extra_pts.append(c)
    info = {"rigid": rigid, "pretension": pre_nodes, "sets": []}
    with open(path, "w") as f:
        f.write("*HEADING\nWobbl nonlinear static\n")
        _write_mesh(f, mesh, extra_pts)
        n_el = mesh.n_elems
        spring_sets = []
        for k, (nodes, kk) in enumerate(model.springs):
            for d in range(3):
                name = f"ESPR{k + 1}_{d + 1}"
                f.write(f"*ELEMENT, TYPE=SPRING1, ELSET={name}\n")
                for i in nodes:
                    n_el += 1
                    f.write(f"{n_el},{int(i) + 1}\n")
                spring_sets.append((name, d + 1, kk))
        for c in [*sm.constraints, *_axis_rollers(sm)]:
            _nset(f, _setname(c.name), c.nodes)
            info["sets"].append(_setname(c.name))
        for name, nodes in {**model.monitor, **model.node_sets}.items():
            _nset(f, name, nodes)
        if pre_nodes:
            _nset(f, _PRE_SET, np.array(list(pre_nodes.values())))
        for k, r in enumerate(sm.rigid):
            _nset(f, f"RB{k + 1}", r.nodes)
        for cp in model.contact:
            _surface(f, f"{cp.name}S", mesh, cp.slave)
            _surface(f, f"{cp.name}M", mesh, cp.master)
        for k, p in enumerate(model.pretension):
            f.write(f"*SURFACE, NAME=PT{k + 1}, TYPE=ELEMENT\n")
            for e, fc in p.faces:
                f.write(f"{int(e) + 1},S{int(fc)}\n")
        for k, mat in enumerate(model.materials):
            elset = "EALL"
            if mat.elements is not None and len(model.materials) > 1:
                elset = f"EMAT{k + 1}"
                ids = np.nonzero(mat.elements)[0] + 1 if mat.elements.dtype == bool else np.asarray(mat.elements) + 1
                f.write(f"*ELSET, ELSET={elset}\n")
                for i in range(0, len(ids), 12):
                    f.write(", ".join(map(str, ids[i:i + 12])) + "\n")
            f.write(f"*MATERIAL, NAME=MAT{k + 1}\n*ELASTIC\n{mat.E:.10g},{mat.nu:.10g}\n")
            f.write(f"*DENSITY\n{mat.density:.10g}\n")
            if mat.hardening:
                f.write("*PLASTIC\n" + "".join(f"{s:.10g},{e:.10g}\n" for s, e in mat.hardening))
            f.write(f"*SOLID SECTION, ELSET={elset}, MATERIAL=MAT{k + 1}\n")
        for name, dof, kk in spring_sets:
            f.write(f"*SPRING, ELSET={name}\n{dof}\n{kk:.10g}\n")
        for k, r in enumerate(sm.rigid):
            ref, rot = rigid[r.name]
            f.write(f"*RIGID BODY, NSET=RB{k + 1}, REF NODE={ref + 1}, ROT NODE={rot + 1}\n")
        for k, p in enumerate(model.pretension):
            nv = np.asarray(p.normal, float) / np.linalg.norm(p.normal)
            f.write(f"*PRE-TENSION SECTION, SURFACE=PT{k + 1}, NODE={pre_nodes[p.name] + 1}\n"
                    f"{nv[0]:.10g},{nv[1]:.10g},{nv[2]:.10g}\n")
        for cp in model.contact:
            f.write(f"*CONTACT PAIR, INTERACTION={cp.name}I, TYPE=SURFACE TO SURFACE\n{cp.name}S,{cp.name}M\n")
            f.write(f"*SURFACE INTERACTION, NAME={cp.name}I\n*SURFACE BEHAVIOR, PRESSURE-OVERCLOSURE=LINEAR\n"
                    f"{cp.stiffness:.6g},{cp.tension:.6g}\n")
            if cp.mu > 0:
                f.write(f"*FRICTION\n{cp.mu:.6g},{cp.stick:.6g}\n")
            if cp.clearance is not None:
                f.write(f"*CLEARANCE, MASTER={cp.name}M, SLAVE={cp.name}S, VALUE={cp.clearance:.6g}\n")
        _write_equations(f, sm)
        f.write("*TIME POINTS, NAME=TEND\n1.\n")
        f.write("*BOUNDARY\n")
        _write_boundaries(f, sm)
        prev = 0.0
        for s in model.steps:
            _write_step(f, model, s, prev, rigid, pre_nodes, info["sets"])
            prev = s.factor
    return info


def _surface(f, name: str, mesh, mask: np.ndarray):
    f.write(f"*SURFACE, NAME={name}, TYPE=ELEMENT\n")
    for e, fc in zip(mesh.tri_elem[mask], mesh.tri_face[mask]):
        f.write(f"{int(e) + 1},S{int(fc)}\n")


def _write_equations(f, sm: StaticModel):
    """Rollers on faces that are not square to an axis, as constraint equations (the same rule as the linear
    deck: the largest normal component of each node depends on the others)."""
    fixed = {}
    for c in sm.constraints:
        for d in c.dofs:
            for i in c.nodes:
                fixed.setdefault(int(i), set()).add(d)
    eqs = []
    for nc in sm.normal_constraints:
        nrm = np.asarray(nc.normal, float)
        nrm = nrm / np.linalg.norm(nrm)
        if np.count_nonzero(np.abs(nrm) > 1e-9) == 1:
            continue
        order = np.argsort(-np.abs(nrm))
        for i in nc.nodes:
            have = fixed.setdefault(int(i), set())
            dep = next((int(d) + 1 for d in order if abs(nrm[d]) > 1e-6 and (int(d) + 1) not in have), None)
            if dep is None:
                continue
            have.add(dep)
            eqs.append((int(i), [(dep, nrm[dep - 1])] + [(d + 1, nrm[d]) for d in range(3)
                                                          if d + 1 != dep and abs(nrm[d]) > 1e-9]))
    if eqs:
        f.write("*EQUATION\n")
        for i, terms in eqs:
            f.write(f"{len(terms)}\n" + ",".join(f"{i + 1},{d},{c:.10g}" for d, c in terms) + "\n")


def _write_boundaries(f, sm: StaticModel):
    for c in sm.constraints:
        for d in c.dofs:
            f.write(f"{_setname(c.name)},{d},{d}\n")
    for nc in sm.normal_constraints:
        nz = np.nonzero(np.abs(np.asarray(nc.normal, float)) > 1e-9)[0]
        if len(nz) == 1:
            f.write(f"{_setname(nc.name)},{nz[0] + 1},{nz[0] + 1}\n")


def _axis_rollers(sm: StaticModel) -> list:
    """Rollers square to an axis: held through a node set, so their reactions are totaled like any support."""
    return [nc for nc in sm.normal_constraints if np.count_nonzero(np.abs(np.asarray(nc.normal, float)) > 1e-9) == 1]


def _write_step(f, model: NonlinearModel, s: Step, prev: float, rigid: dict, pre_nodes: dict, sets: list[str]):
    sm = model.static
    head = f"*STEP, INC={s.max_increments}" + (", NLGEOM" if s.nlgeom else "")
    f.write(f"{head}\n*STATIC\n{s.first:.6g},1.,{s.smallest:.6g},{s.largest:.6g}\n")
    if s.may_stop:
        # near collapse every attempt fails slowly, so give up on an increment after 10 iterations instead of 16
        f.write("*CONTROLS, PARAMETERS=TIME INCREMENTATION\n4,8,9,10,10,4,,5\n")
    else:
        f.write("*CONTROLS, RESET\n")
    if s.prescribed:
        f.write("*BOUNDARY\n")
        for name, dof, val in s.prescribed:
            f.write(f"{name},{dof},{dof},{val:.10g}\n")
    if s.lock:
        f.write("*BOUNDARY, FIXED\n" + "".join(f"{i + 1},1,1\n" for i in pre_nodes.values()))
    F = sm.nodal_forces()
    rows = np.nonzero(np.abs(F).max(1) > 0)[0]
    loaded = len(rows) or sm.rigid or (s.preload and model.pretension) or (s.lock and model.pretension)
    if loaded:
        f.write("*CLOAD\n")
        for i in rows:
            for d in range(3):
                if F[i, d] != 0.0:
                    f.write(f"{i + 1},{d + 1},{s.factor * F[i, d]:.10e}\n")
        for r in sm.rigid:
            ref, rot = rigid[r.name]
            for d in range(3):
                if r.force[d] != 0:
                    f.write(f"{ref + 1},{d + 1},{s.factor * r.force[d]:.10e}\n")
                if r.moment[d] != 0:
                    f.write(f"{rot + 1},{d + 1},{s.factor * r.moment[d]:.10e}\n")
        for p in model.pretension:
            # the preload pulls the section in its own step; once locked, the section keeps its length and the
            # force on its node is released
            if s.preload:
                f.write(f"{pre_nodes[p.name] + 1},1,{p.force:.10e}\n")
            elif s.lock:
                f.write(f"{pre_nodes[p.name] + 1},1,0.\n")
    if sm.gravity is not None and np.linalg.norm(sm.gravity) > 0:
        g = np.asarray(sm.gravity, float)
        mag = np.linalg.norm(g)
        d = g / mag
        f.write(f"*DLOAD\nEALL,GRAV,{s.factor * mag:.10g},{d[0]:.10g},{d[1]:.10g},{d[2]:.10g}\n")
    # full fields only at the end of the step unless asked. CalculiX applies one output frequency to the .dat and
    # the .frd alike within a step, so a step that saves only its end prints no totals either
    freq = "" if s.every_increment else ", TIME POINTS=TEND"
    node = [x for x in model.fields if x in ("U", "RF")]
    el = [x for x in model.fields if x in ("S", "PEEQ", "E")]
    if node:
        f.write(f"*NODE FILE{freq}\n{','.join(node)}\n")
    if el:
        f.write(f"*EL FILE{freq}\n{','.join(el)}\n")
    if model.contact:
        f.write(f"*CONTACT FILE{freq}\nCDIS,CSTR\n")
    # totals and monitor points at every increment, for load paths and collapse curves
    if s.every_increment:
        for name in sets:
            f.write(f"*NODE PRINT, NSET={name}, TOTALS=ONLY\nRF\n")
        for name in model.monitor:
            f.write(f"*NODE PRINT, NSET={name}\nU\n")
        if pre_nodes:
            f.write(f"*NODE PRINT, NSET={_PRE_SET}\nU\n")
    f.write("*END STEP\n")


_PRE_SET = "NPRETENSION"


# ---------------------------------------------------------------------------------------------------- running
STOPPED = "increment size smaller than minimum"


def solve_nonlinear(model: NonlinearModel, workdir: str | Path, job: str = "nonlinear",
                    timeout: float = 3600) -> NonlinearResult:
    """Run the steps. A step marked ``may_stop`` that ends early because CalculiX cannot find equilibrium any more
    is a finished run (the ramp reached collapse); any other failure raises SolverError."""
    from . import base

    ccx = base.for_domain("structural").require()
    work = Path(workdir)
    work.mkdir(parents=True, exist_ok=True)
    for ext in (".frd", ".dat", ".sta", ".cvg"):
        (work / f"{job}{ext}").unlink(missing_ok=True)
    info = write_nonlinear_inp(work / f"{job}.inp", model)
    t0 = time.time()
    try:
        r = subprocess.run([str(ccx), "-i", job], cwd=work, capture_output=True, text=True, timeout=timeout,
                           env=tool_env(ccx, THREADS))
    except subprocess.TimeoutExpired:
        raise SolverError(f"CalculiX did not finish the nonlinear run within {timeout:.0f} s",
                          "Use --quality fast for a coarser mesh") from None
    log = (r.stdout or "") + (r.stderr or "")
    t = time.time() - t0
    errs = [ln.strip() for ln in log.splitlines() if "*ERROR" in ln]
    incs, iters, last_step = read_sta(work / f"{job}.sta")
    stopped = bool(errs) and all(STOPPED in e for e in errs)
    finished = r.returncode == 0 and not errs
    if not finished:
        step = model.steps[last_step - 1] if 0 < last_step <= len(model.steps) else None
        if not (stopped and step is not None and step.may_stop and incs > 0):
            if stopped:
                raise SolverError("CalculiX could not find equilibrium partway through the load (the increments "
                                  "shrank below the smallest allowed)",
                                  "Check that every part is held and that contact surfaces actually meet",
                                  log[-3000:])
            raise _explain(errs, log, r.returncode)
    n = model.static.mesh.n_nodes
    dat = read_dat(work / f"{job}.dat")
    frames = read_frames(work / f"{job}.frd", n + len(info["rigid"]) * 2 + len(info["pretension"]), n)
    if not finished:
        # CalculiX writes the attempt that failed too, for debugging; only converged increments are results
        done = [t for blocks in dat.values() for t, _ in blocks]
        if done:
            frames = [fr for fr in frames if fr["time"] <= max(done) + 1e-9]
    times = np.array(sorted({t for blocks in dat.values() for t, _ in blocks} | {fr["time"] for fr in frames}))
    reactions = {k: np.array([v[0] for _, v in dat.get(("total force", k), [])]) for k in info["sets"]}
    monitor = {}
    for name, nodes in model.monitor.items():
        rows = dat.get(("displacements", name), [])
        monitor[name] = np.array([v[:, 1:4] for _, v in rows]) if rows else np.zeros((0, len(nodes), 3))
    pre = {}
    for name, idx in info["pretension"].items():   # the first dof of a pretension node is how far the cut opens
        j = idx - n
        pre[name] = np.array([fr["U_extra"][j, 0] if len(fr.get("U_extra", ())) > j else np.nan for fr in frames])
    return NonlinearResult(times, reactions, monitor, frames, finished, log, t, f"CalculiX {_version()}",
                           incs, iters, pre)


def read_sta(path: Path) -> tuple[int, int, int]:
    """Finished increments, iterations over every attempt, and the step the run was in when it ended."""
    if not path.exists():
        return 0, 0, 0
    incs, iters, step = 0, 0, 0
    for line in path.read_text(errors="replace").splitlines():
        parts = line.split()
        if len(parts) >= 7 and parts[0].isdigit() and parts[1].isdigit():
            step = int(parts[0])
            iters += int(parts[3]) if parts[3].isdigit() else 0
            if not parts[2].endswith("U"):
                incs += 1
    return incs, iters, step


_DAT_HEAD = re.compile(r"^\s*(displacements|total force|forces)\s.*?for set (\S+) and time\s+(\S+)")


def read_dat(path: Path) -> dict[tuple[str, str], list[tuple[float, np.ndarray]]]:
    """NODE PRINT blocks of a .dat file: {(kind, set): [(time, rows), ...]}. Rows are the numbers on each line,
    the node id first for nodal values, or the three totals for TOTALS=ONLY."""
    out: dict = {}
    if not path.exists():
        return out
    lines = path.read_text(errors="replace").splitlines()
    i = 0
    while i < len(lines):
        m = _DAT_HEAD.match(lines[i])
        if not m:
            i += 1
            continue
        key, t = (m.group(1), m.group(2)), float(m.group(3))
        i += 1
        while i < len(lines) and not lines[i].strip():
            i += 1
        rows = []
        while i < len(lines) and lines[i].strip() and not _DAT_HEAD.match(lines[i]):
            try:
                rows.append([float(x) for x in lines[i].split()])
            except ValueError:
                break
            i += 1
        if rows:
            out.setdefault(key, []).append((t, np.array(rows)))
    return out


def read_frames(path: Path, n_all: int, n: int) -> list[dict]:
    """Saved fields by time: U (n, 3), RF, S (n, 6), PEEQ (n,), and contact opening, slip, pressure and shear."""
    if not path.exists():
        return []
    blocks = read_frd(path, n_all)
    frames: dict[float, dict] = {}
    for blk in ("DISP", "FORC", "STRESS", "PE", "CONTACT"):
        for t, arr in blocks.get(blk, []):
            fr = frames.setdefault(round(t, 9), {"time": t})
            if blk == "DISP":
                fr["U"], fr["U_extra"] = arr[:n, :3], arr[n:, :3]
            elif blk == "FORC":
                fr["RF"] = arr[:n, :3]
            elif blk == "STRESS":
                fr["S"] = arr[:n, :6]
            elif blk == "PE":
                fr["PEEQ"] = arr[:n, 0]
            else:
                fr["open"], fr["slip"] = arr[:n, 0], np.linalg.norm(arr[:n, 1:3], axis=1)
                fr["pressure"], fr["shear"] = arr[:n, 3], np.linalg.norm(arr[:n, 4:6], axis=1)
    return [frames[k] for k in sorted(frames)]


def _version() -> str:
    from .base import get

    try:
        return get("calculix").version()
    except Exception:  # noqa: BLE001
        return ""


def von_mises(S: np.ndarray) -> np.ndarray:
    sxx, syy, szz, sxy, syz, szx = np.asarray(S, float).T
    return np.sqrt(0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2)
                   + 3.0 * (sxy ** 2 + syz ** 2 + szx ** 2))
