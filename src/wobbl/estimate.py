"""Know before running: the time, peak memory and disk a solve needs, predicted before it starts.

Every solver Wobbl drives has a cost model here. Time is modeled as seconds on one core, spread over the cores
a run gets with Amdahl's law; memory as bytes at the peak; disk as the bytes the run writes.

CalculiX solves its equations with a sparse direct solver, whose work is set less by the number of nodes than
by how the mesh comes apart. A nested dissection cuts the mesh in two across its widest part, then cuts each
half, and so on down. The factorization costs about the cube of each cut and stores about its square, summed
over every cut, so a compact block costs far more than a thin plate with as many nodes, and a bar less still.
``dissection`` walks that recursion on a sample of node coordinates, taken from a mesh or streamed from a deck.
The coefficients were fit on gmsh meshes of blocks, plates and bars, and checked on structured meshes, solved
by CalculiX on one thread of a 2018 six core laptop (Intel i7-8750H). 'wobbl estimate --calibrate' fits this
machine's own factor in about a minute, and every finished job refines it.

Explicit dynamics (OpenRadioss) and CFD (OpenFOAM) cost a fixed amount per element and time step, or per cell
and iteration. Neither solver is installed on the reference machine, so their constants come from published
runs and every estimate says so until a finished job here measures them.
"""
from __future__ import annotations

import json
import math
import os
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

REFERENCE = "a 2018 six core laptop (Intel i7-8750H)"

# CalculiX on one thread: seconds = T_DOF * dof + T_FLOP * F, bytes = M0 + M_DOF * dof + M_FILL * M, where F and M
# are the dissection sums of the cut sizes cubed and squared, counted in degrees of freedom. A heat solve has its own
# time coefficients. The factorization spreads over threads far better than the work per node does.
CCX = {"t_dof": 1.27e-4, "t_flop": 4.5e-9, "t_dof_heat": 2.46e-4, "t_flop_heat": 2.84e-8, "m0": 3.08e6,
       "m_dof": 2775.0, "m_fill": 18.9, "m_transient": 7e7, "p_dof": 0.35, "p_flop": 0.95}
# bytes CalculiX writes: the .frd holds the mesh, then each saved result set
FRD_NODE, FRD_ELEMENT = 50.0, 150.0
FRD_SET = {3: 186.0, 1: 102.0}        # per node per result set: U, RF and S; or NT, RFL and HFL
INP_NODE, INP_ELEMENT = 60.0, 110.0
RESTART_NODE = 700.0    # a checkpoint (*RESTART, WRITE, OVERLAY): 350 per tet10 node measured static, 720 hex8 heat

# OpenRadioss: core seconds per element and cycle, from the CCSA's published LS-DYNA run of the 2010 Toyota Yaris
# (1.52 million elements, 200 ms in 5 h 58 min on 16 processors) at an assumed 1 microsecond step, taken back to
# one core with the same parallel fraction the estimate spreads it with, so the model gives that run back
EXPLICIT = {"per_element_cycle": 7.8e-7, "time_step": 1e-6, "parallel": 0.97, "bytes_element": 1600.0,
            "bytes_node": 400.0, "frame_node": 24.0, "frame_element": 8.0, "frames": 40}
# OpenFOAM: an order of magnitude for an incompressible steady solver on one core, per cell and iteration
CFD = {"per_cell_iteration": 4e-6, "parallel": 0.95, "bytes_cell": 1000.0, "write_cell": 120.0}
# ngspice: transient points times circuit nodes
SPICE = {"t0": 0.3, "per_point_node": 2e-6, "bytes": 6e7}

LEAF = 48            # sample points below which a piece of the mesh is treated as one compact block
LONG_JOB = 600.0     # seconds past which a solve is better run as a background job
DISK_RESERVE = 2e9   # bytes a run must leave free, since a full disk stops every other program on the machine too


# --------------------------------------------------------------------------------------------------- machine
def physical_cores() -> int:
    n = os.cpu_count() or 1
    try:
        if sys.platform == "darwin":
            import subprocess
            r = subprocess.run(["sysctl", "-n", "hw.physicalcpu"], capture_output=True, text=True, timeout=5)
            return max(1, int(r.stdout.strip()))
        if sys.platform.startswith("linux"):
            cores = set()
            phys = core = None
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("physical id"):
                    phys = line.split(":")[1].strip()
                elif line.startswith("core id"):
                    core = line.split(":")[1].strip()
                    cores.add((phys, core))
            return len(cores) or n
    except Exception:  # noqa: BLE001
        pass
    return n


def total_ram() -> float:
    """Physical memory in bytes."""
    try:
        if sys.platform == "win32":
            import ctypes

            class Status(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            s = Status()
            s.dwLength = ctypes.sizeof(Status)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s))
            return float(s.ullTotalPhys)
        return float(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
    except Exception:  # noqa: BLE001
        return 8e9


def default_threads() -> int:
    """The threads a solve gets, as tool_env sets them: OMP_NUM_THREADS if given, else up to four."""
    try:
        return max(1, int(os.environ["OMP_NUM_THREADS"]))
    except (KeyError, ValueError):
        return min(4, os.cpu_count() or 1)


@dataclass
class Machine:
    name: str
    cores: int
    threads: int                  # threads or MPI ranks a run gets
    ram: float                    # bytes
    disk_free: float              # bytes where the results go
    speed: float = 1.0            # seconds here per reference second, from calibration
    price_per_hour: float = 0.0   # cloud machines

    def to_dict(self) -> dict:
        return {"name": self.name, "cores": self.cores, "threads": self.threads,
                "ram_gb": round(self.ram / 1e9, 1), "disk_free_gb": round(self.disk_free / 1e9, 2),
                **({"price_per_hour": self.price_per_hour} if self.price_per_hour else {})}


def this_machine(where: str | Path = ".", threads: int | None = None) -> Machine:
    p = Path(where)
    while not p.exists() and p != p.parent:
        p = p.parent
    try:
        free = float(shutil.disk_usage(p).free)
    except OSError:
        free = float("inf")
    cores = physical_cores()
    return Machine("this machine", cores, threads or min(default_threads(), cores), total_ram(), free)


def amdahl(threads: int, parallel: float) -> float:
    """How many times faster a run is on ``threads`` than on one."""
    return 1.0 / ((1.0 - parallel) + parallel / max(1, threads))


# ------------------------------------------------------------------------------------------------ dissection
def dissection(sample: np.ndarray, n: int) -> tuple[float, float]:
    """The sums of cut sizes cubed and squared over a nested dissection of ``n`` nodes, in nodes.

    ``sample`` is any spread out subset of the node coordinates (every k-th node of a deck works). Each cut is a
    plane through the median across the widest extent, and its size is the nodes within a thin slab around it,
    scaled from the sample to the whole mesh. A piece holding fewer than LEAF sample points is counted as a compact
    block, whose own dissection sums to about twice its first cut cubed.
    """
    from scipy.spatial import cKDTree

    P = np.asarray(sample, float).reshape(-1, 3)
    if len(P) < 8 or n < 8:
        return float(n) ** 2, float(n) ** (4 / 3)
    phi = len(P) / n
    d, _ = cKDTree(P).query(P, k=2)
    a_s = float(np.median(d[:, 1]))          # spacing of the sample
    a = a_s * phi ** (1 / 3)                  # spacing of the whole mesh
    w = 3.0 * a_s                             # a slab a few sample layers wide
    F = M = 0.0
    stack = [P]
    while stack:
        Q = stack.pop()
        m = len(Q) / phi
        if len(Q) < LEAF:
            s = m ** (2 / 3)
            F += 2.0 * s ** 3
            M += 3.0 * s ** 2
            continue
        k = int(np.argmax(Q.max(0) - Q.min(0)))
        x = Q[:, k]
        mid = float(np.median(x))
        s = min(np.count_nonzero(np.abs(x - mid) <= w / 2) / phi * a / w, m / 2)
        F += s ** 3
        M += s ** 2
        left, right = Q[x < mid], Q[x >= mid]
        if len(left) and len(right):
            stack += [left, right]
    return F, M


def inside(pts: np.ndarray, X: np.ndarray, tris: np.ndarray, chunk: int = 2_000_000) -> np.ndarray:
    """Which points lie inside a closed triangulated surface, by the parity of crossings along a skew ray."""
    V0, V1, V2 = X[tris[:, 0]], X[tris[:, 1]], X[tris[:, 2]]
    e1, e2 = V1 - V0, V2 - V0
    d = np.array([0.5773, 0.5774, 0.5775])
    p = np.cross(d, e2)
    det = np.einsum("tk,tk->t", p, e1)
    ok = np.abs(det) > 1e-14
    inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
    out = np.zeros(len(pts), bool)
    step = max(1, chunk // max(len(tris), 1))
    for r0 in range(0, len(pts), step):
        s = pts[r0:r0 + step, None, :] - V0[None]
        u = np.einsum("rtk,tk->rt", s, p) * inv
        q = np.cross(s, e1[None])
        v = (q @ d) * inv
        t = np.einsum("tk,rtk->rt", e2, q) * inv
        hits = ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 1e-9)
        out[r0:r0 + step] = hits.sum(1) % 2 == 1
    return out


def solid_sample(prep, n_points: int = 4000) -> np.ndarray:
    """Points spread evenly through a part's solid, for the dissection of a mesh not built yet."""
    from .mesh.mesher import build_surface_mesh

    sm = build_surface_mesh(prep)
    lo, hi = np.asarray(prep.bbox[0], float), np.asarray(prep.bbox[1], float)
    box = float(np.prod(np.maximum(hi - lo, 1e-9)))
    fill = float(np.clip(prep.volume / box, 0.02, 1.0))
    pts = lo + np.random.default_rng(0).random((int(min(60000, 1.3 * n_points / fill)), 3)) * (hi - lo)
    return pts[inside(pts, sm.X, sm.tris[:, :3])][:n_points]


# --------------------------------------------------------------------------------------------- calibration
def calibration_path() -> Path:
    from .solvers.locate import wobbl_home

    return wobbl_home() / "calibration.json"


def load_calibration() -> dict:
    try:
        return json.loads(calibration_path().read_text())
    except (OSError, ValueError):
        return {"runs": []}


def factor(solver: str, what: str = "time") -> tuple[float, int]:
    """This machine's measured over modeled ratio for a solver, from its most recent runs, and how many runs
    it rests on. 1.0 with no runs."""
    runs = [r for r in load_calibration().get("runs", []) if r.get("solver") == solver
            and r.get(f"predicted_{what}") and r.get(f"measured_{what}")][-12:]
    if not runs:
        return 1.0, 0
    ratios = [r[f"measured_{what}"] / r[f"predicted_{what}"] for r in runs]
    return float(np.clip(math.exp(float(np.median(np.log(ratios)))), 0.1, 10.0)), len(runs)


def record(solver: str, predicted_time: float, measured_time: float, predicted_memory: float = 0.0,
           measured_memory: float = 0.0, size: dict | None = None, source: str = "job"):
    """Add one measured run. Times are one core seconds, so runs on any thread count compare."""
    cal = load_calibration()
    cal.setdefault("runs", []).append({"solver": solver, "predicted_time": predicted_time,
                                       "measured_time": measured_time, "predicted_memory": predicted_memory,
                                       "measured_memory": measured_memory, "size": size or {},
                                       "source": source, "when": time.time()})
    cal["runs"] = cal["runs"][-200:]
    p = calibration_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(cal, indent=1))
    tmp.replace(p)


def measured_run(argv: list[str], cwd: str | Path, env: dict | None = None,
                 timeout: float = 3600.0) -> tuple[float, float, int]:
    """Run a solver and return its wall seconds, its peak memory in bytes (0 where the platform cannot say) and
    its exit code."""
    import subprocess

    t0 = time.time()
    p = subprocess.Popen(argv, cwd=cwd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if sys.platform == "win32":
        code = p.wait(timeout=timeout)
        return time.time() - t0, 0.0, code
    _, status, ru = os.wait4(p.pid, 0)
    p.returncode = os.waitstatus_to_exitcode(status)
    rss = float(ru.ru_maxrss) * (1.0 if sys.platform == "darwin" else 1024.0)
    return time.time() - t0, rss, p.returncode


def calibrate(log=print, threads: int | None = None) -> dict:
    """Solve three small parts with CalculiX here, a block and a plate on gmsh meshes, and record each run's
    measured time and memory against the model. Takes about a minute on the reference machine."""
    import tempfile

    from .fe import NodeConstraint, StaticModel
    from .geometry.cad import prepare
    from .mesh.mesher import Sizing, build_mesh
    from .solvers import base as solvers
    from .solvers.calculix import write_static_inp
    from .solvers.locate import tool_env
    from .validation import geometry as G

    ccx = solvers.get("calculix").require()
    n_threads = threads or min(default_threads(), physical_cores())
    env = tool_env(ccx, n_threads)
    for k in ("OMP_NUM_THREADS", "CCX_NPROC_EQUATION_SOLVER", "CCX_NPROC_RESULTS", "CCX_NPROC_STIFFNESS"):
        env[k] = str(n_threads)
    runs = []
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        for shape, dims, h in (("block", (100, 40, 20), 6.0), ("plate", (200, 80, 5), 4.0),
                               ("block", (100, 40, 20), 4.0)):
            step = work / f"{shape}.step"
            if not step.exists():
                G.box(step, *dims)
            mesh = build_mesh(prepare(step, work / f"prep_{shape}"), Sizing(h=h))
            X = mesh.X
            lo = np.nonzero(X[:, 0] < 1e-6)[0]
            tip = np.nonzero(X[:, 0] > dims[0] - 1e-6)[0]
            F = np.zeros((len(X), 3))
            F[tip, 2] = -100.0 / len(tip)
            write_static_inp(work / "cal.inp", StaticModel(mesh, 200000.0, 0.3, 7.85e-9, [NodeConstraint("lo", lo)],
                                                          forces=F))
            wall, rss, code = measured_run([str(ccx), "-i", "cal"], work, env)
            if code != 0:
                raise RuntimeError(f"CalculiX failed on the calibration {shape} (exit {code})")
            cost = calculix_cost(len(X), len(mesh.tets), X)
            core = wall * amdahl(n_threads, cost.parallel)
            record("calculix", cost.core_seconds, core, cost.peak_bytes, rss,
                   {"nodes": len(X), "threads": n_threads, "wall_s": round(wall, 2)}, "calibration")
            runs.append({"shape": shape, "nodes": len(X), "wall_s": round(wall, 2),
                         "predicted_s": round(cost.core_seconds / amdahl(n_threads, cost.parallel), 2),
                         "memory_mb": round(rss / 1e6), "predicted_memory_mb": round(cost.peak_bytes / 1e6)})
            log(f"{shape} with {len(X):,} nodes: {wall:.1f} s against {runs[-1]['predicted_s']:.1f} s modeled, "
                f"{rss / 1e6:.0f} MB against {cost.peak_bytes / 1e6:.0f} MB")
            for ext in (".frd", ".dat", ".sta", ".cvg", ".12d"):
                (work / f"cal{ext}").unlink(missing_ok=True)
    f, n = factor("calculix")
    return {"threads": n_threads, "runs": runs, "time_factor": round(f, 3),
            "memory_factor": round(factor("calculix", "memory")[0], 3), "calibrated_from": n}


# --------------------------------------------------------------------------------------------- cost models
@dataclass
class Cost:
    """What one run costs on one core of the reference machine, before any machine is chosen."""
    solver: str
    procedure: str
    core_seconds: float
    parallel: float               # Amdahl's parallel fraction
    peak_bytes: float
    disk_bytes: float
    size: dict = field(default_factory=dict)
    basis: list[str] = field(default_factory=list)
    sums: tuple = ()              # CalculiX: the dissection sums, so a coarser mesh can be priced


def compact_sums(n: float) -> tuple[float, float]:
    """Dissection sums for a compact solid of ``n`` nodes, when no coordinates are at hand."""
    s = n ** (2 / 3)
    return 2.0 * s ** 3, 3.0 * s ** 2


def calculix_cost(nodes: int, elements: int = 0, sample: np.ndarray | None = None, sums: tuple | None = None,
                  dof: int = 3, solves: float = 1.0, outputs: int = 1, procedure: str = "static",
                  deck_bytes: float = 0.0, checkpoint: bool = False) -> Cost:
    """A CalculiX run of ``solves`` factorizations that writes ``outputs`` result sets."""
    basis = []
    if sums is None:
        if sample is not None and len(sample) >= 8:
            sums = dissection(sample, nodes)
        else:
            sums = compact_sums(nodes)
            basis.append("No node coordinates were given, so the mesh is taken as a compact solid. A thin walled "
                         "part costs less.")
    F, M = sums
    heat = "heat" in procedure
    t_node = (CCX["t_dof_heat"] if heat else CCX["t_dof"]) * dof * nodes
    t_fact = (CCX["t_flop_heat"] if heat else CCX["t_flop"]) * F * dof ** 3
    per = t_node + t_fact
    parallel = (t_node * CCX["p_dof"] + t_fact * CCX["p_flop"]) / max(per, 1e-30)
    mem = CCX["m0"] + CCX["m_dof"] * dof * nodes + CCX["m_fill"] * M * dof ** 2
    if heat and solves > 1:
        mem += CCX["m_transient"]          # a transient heat run holds about this much more, whatever its size
    elements = elements or int(nodes / 1.9)
    disk = (deck_bytes or INP_NODE * nodes + INP_ELEMENT * elements) + FRD_NODE * nodes + FRD_ELEMENT * elements \
        + FRD_SET.get(dof, 186.0) * nodes * max(1, outputs) + (RESTART_NODE * nodes if checkpoint else 0.0)
    basis.append(f"CalculiX cost from the mesh's nested dissection, fit on {REFERENCE}.")
    size = {"nodes": int(nodes), "elements": int(elements), "dof": int(dof * nodes), "solves": round(solves, 1),
            "result_sets": int(outputs)}
    return Cost("calculix", procedure, per * solves, parallel, mem, disk, size, basis, (F, M))


def explicit_cost(elements: int, nodes: int, end_time: float, time_step: float = 0.0, frames: int = 0) -> Cost:
    """An explicit dynamics run in OpenRadioss, where every element is updated every cycle."""
    basis = ["Explicit cost per element and cycle from the CCSA's published LS-DYNA run of the 2010 Toyota Yaris "
             "(1.52 million elements, 200 ms in 5 h 58 min on 16 processors), not yet measured on this machine."]
    if not time_step:
        basis.append("The deck sets no time step, so a 1 microsecond step is assumed, which mass scaling holds in "
                     "typical crash models.")
    dt = time_step or EXPLICIT["time_step"]
    cycles = max(1.0, end_time / dt)
    nodes = nodes or elements
    frames = frames or EXPLICIT["frames"]
    mem = EXPLICIT["bytes_element"] * elements + EXPLICIT["bytes_node"] * nodes
    disk = frames * (EXPLICIT["frame_node"] * nodes + EXPLICIT["frame_element"] * elements) + 2 * mem
    size = {"elements": int(elements), "nodes": int(nodes), "cycles": int(cycles), "time_step": dt,
            "end_time": end_time, "frames": int(frames)}
    return Cost("openradioss", "explicit", EXPLICIT["per_element_cycle"] * elements * cycles, EXPLICIT["parallel"],
                mem, disk, size, basis)


def cfd_cost(cells: int, iterations: float, writes: int = 1) -> Cost:
    """An OpenFOAM run, where every cell is visited every iteration or time step."""
    basis = ["CFD cost per cell and iteration is an order of magnitude for an incompressible solver, not yet "
             "measured on this machine."]
    mem = CFD["bytes_cell"] * cells
    disk = (200.0 + CFD["write_cell"] * max(1, writes)) * cells
    size = {"cells": int(cells), "iterations": int(iterations), "writes": int(writes)}
    return Cost("openfoam", "cfd", CFD["per_cell_iteration"] * cells * iterations, CFD["parallel"], mem, disk,
                size, basis)


def spice_cost(points: int, circuit_nodes: int = 10) -> Cost:
    """An ngspice transient. Circuits are small, so this only ever says it is quick."""
    return Cost("ngspice", "transient", SPICE["t0"] + SPICE["per_point_node"] * points * circuit_nodes, 0.0,
                SPICE["bytes"], 16.0 * points * circuit_nodes,
                {"points": int(points), "circuit_nodes": int(circuit_nodes)}, ["ngspice runs on one core."])


def step_increments(step) -> tuple[float, float, str]:
    """Increments, factorizations and a note for one CalculiX step. A transient adapts its increments between
    the first and the largest one it is given, so the count is their geometric mean, and the job's progress
    replaces it once the run starts."""
    period = step.period or 1.0
    if step.procedure in ("frequency", "buckle"):
        modes = step.modes or 10
        return 1.0, 1.5 + 0.05 * modes + (1.0 if step.procedure == "buckle" else 0.0), ""
    if step.procedure in ("heat", "dynamic", "explicit") or step.nlgeom:
        many = period / step.initial if step.initial > 0 else 100.0
        few = period / step.max_inc if step.max_inc > 0 else max(1.0, many / 10.0)
        inc = float(np.clip(math.sqrt(max(many, 1.0) * max(few, 1.0)), 1.0, 1e5))
        # a transient heat increment costs about 0.85 of a steady solve of the same mesh, measured on blocks and plates
        its = 4.0 if step.nlgeom else (0.85 if step.procedure == "heat" else 1.0)
        return inc, inc * its, f"about {inc:.0f} increments in a {step.procedure} step of {period:g}"
    return 1.0, 1.0, ""


def deck_cost(deck, checkpoint: bool = False) -> Cost:
    """The cost of a deck read by ``wobbl.decks``."""
    if deck.solver == "calculix":
        heat = bool(deck.steps) and all(s.procedure.startswith("heat") for s in deck.steps)
        solves, outputs, notes = 0.0, 0, []
        for s in deck.steps or []:
            inc, fac, note = step_increments(s)
            solves += fac
            outputs += int(inc)
            if note:
                notes.append(note)
        procs = sorted({s.procedure for s in deck.steps}) or ["static"]
        c = calculix_cost(deck.nodes, deck.n_elements, deck.sample, dof=1 if heat else 3, solves=max(1.0, solves),
                          outputs=max(1, outputs), procedure=", ".join(procs), deck_bytes=deck.bytes,
                          checkpoint=checkpoint)
        if notes:
            c.basis.append("The deck asks for " + ", then ".join(notes[:3]) + ". Results saved at every increment are "
                           "counted on disk.")
        return c
    if deck.solver == "openradioss":
        frames = int(deck.end_time / deck.output_interval) if deck.output_interval else 0
        return explicit_cost(deck.n_elements, deck.nodes, deck.end_time, deck.time_step, frames)
    if deck.solver == "openfoam":
        iters = deck.end_time / deck.time_step if deck.time_step else deck.end_time
        writes = int(deck.end_time / deck.output_interval) if deck.output_interval else 1
        return cfd_cost(deck.n_elements, iters, writes)
    raise ValueError(f"no cost model for {deck.solver}")


# ------------------------------------------------------------------------------------------------- estimates
def duration(s: float) -> str:
    if s < 90:
        return f"{s:.0f} s"
    if s < 5400:
        return f"{s / 60:.0f} min"
    if s < 172800:
        return f"{s / 3600:.1f} h"
    return f"{s / 86400:.1f} days"


def nbytes(b: float) -> str:
    if b < 1e9:
        return f"{b / 1e6:.0f} MB"
    if b < 1e12:
        return f"{b / 1e9:.1f} GB"
    return f"{b / 1e12:.1f} TB"


@dataclass
class Estimate:
    cost: Cost
    machine: Machine
    seconds: float                # wall clock on the machine
    peak_bytes: float
    disk_bytes: float
    verdict: str                  # fits, long, too long, does not fit
    problems: list[str] = field(default_factory=list)
    options: list[dict] = field(default_factory=list)
    basis: list[str] = field(default_factory=list)
    target: str = ""

    @property
    def fits(self) -> bool:
        return self.verdict != "does not fit"

    def to_dict(self) -> dict:
        return {"target": self.target, "solver": self.cost.solver, "procedure": self.cost.procedure,
                "size": self.cost.size, "machine": self.machine.to_dict(), "time_s": round(self.seconds, 1),
                "time": duration(self.seconds), "peak_memory_gb": round(self.peak_bytes / 1e9, 3),
                "disk_gb": round(self.disk_bytes / 1e9, 3), "verdict": self.verdict, "fits": self.fits,
                "problems": self.problems, "options": self.options, "basis": self.basis}

    def headline(self) -> str:
        m = self.machine
        where = f"{m.threads} cores of {m.name}" if m.threads > 1 else f"one core of {m.name}"
        return (f"About {duration(self.seconds)} on {where}, {nbytes(self.peak_bytes)} of memory at the peak and "
                f"{nbytes(self.disk_bytes)} on disk.")


def predict(cost: Cost, machine: Machine | None = None, max_hours: float | None = None, target: str = "",
            advise_options: bool = True) -> Estimate:
    """Put a cost on a machine: wall time on its threads with this machine's calibration, and whether it fits."""
    m = machine or this_machine()
    basis = list(cost.basis)
    tf = mf = 1.0
    if m.name == "this machine":
        tf, nt = factor(cost.solver, "time")
        mf, _ = factor(cost.solver, "memory")
        basis.append(f"Calibrated on this machine from {nt} measured run{'s' if nt != 1 else ''}." if nt else
                     "Not calibrated on this machine yet, so the reference machine's speed is used. "
                     "Run wobbl estimate --calibrate to measure it.")
    seconds = cost.core_seconds * tf * m.speed / amdahl(m.threads, cost.parallel)
    mem, disk = cost.peak_bytes * mf, cost.disk_bytes
    problems = []
    if mem > 0.85 * m.ram:
        problems.append(f"It needs about {nbytes(mem)} of memory and {m.name} has {nbytes(m.ram)}.")
    if disk > m.disk_free - DISK_RESERVE:
        problems.append(f"It writes about {nbytes(disk)} and {m.name} has {nbytes(m.disk_free)} free, which must "
                        f"keep {nbytes(DISK_RESERVE)} spare.")
    limit = (max_hours or 24.0) * 3600.0
    if problems:
        verdict = "does not fit"
    elif seconds > limit:
        verdict = "too long"
        problems.append(f"It would take about {duration(seconds)}, past the {duration(limit)} limit.")
    else:
        verdict = "long" if seconds > LONG_JOB else "fits"
    est = Estimate(cost, m, seconds, mem, disk, verdict, problems, [], basis, target)
    if advise_options:
        est.options = advise(est, max_hours)
    return est


def coarser(cost: Cost, f: float) -> Cost:
    """The same CalculiX run with elements ``f`` times larger: nodes fall as f cubed and each cut as f squared."""
    sz = cost.size
    n = max(8, int(sz["nodes"] / f ** 3))
    F, M = cost.sums
    solves = sz.get("solves", 1.0)
    c = calculix_cost(n, int(sz["elements"] / f ** 3), sums=(F / f ** 6, M / f ** 4), dof=max(1, sz["dof"] // sz["nodes"]),
                      solves=solves, outputs=sz.get("result_sets", 1), procedure=cost.procedure)
    c.basis = cost.basis
    return c


def advise(est: Estimate, max_hours: float | None = None) -> list[dict]:
    """What to do about a run that is long, too long or does not fit: a background job, a coarser first pass, a
    submodel, a remote machine the user owns or a cloud machine, each with its own predicted cost."""
    c, m = est.cost, est.machine
    target = est.target or "DECK"
    limit = (max_hours or 24.0) * 3600.0
    opts = []
    if est.verdict in ("long", "too long") and not any("memory" in p or "free" in p for p in est.problems):
        opts.append({"kind": "background", "command": f"wobbl jobs start {target}",
                     "summary": f"Run it in the background with wobbl jobs start {target}, which keeps going after "
                                "the terminal closes, shows progress and an ETA, and resumes from its last "
                                "checkpoint."})
    if est.verdict not in ("too long", "does not fit"):
        return opts
    if c.solver == "calculix" and c.sums:
        for f in (1.15, 1.3, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0, 5.0):
            e = predict(coarser(c, f), m, max_hours, advise_options=False)
            if e.fits and e.seconds <= min(limit, 7200.0) and e.peak_bytes <= 0.6 * m.ram:
                opts.append({"kind": "coarser", "element_factor": f, "nodes": e.cost.size["nodes"],
                             "time_s": round(e.seconds), "peak_memory_gb": round(e.peak_bytes / 1e9, 2),
                             "summary": f"A first pass with elements about {f:g} times larger, about "
                                        f"{e.cost.size['nodes']:,} nodes, takes {duration(e.seconds)} and needs "
                                        f"{nbytes(e.peak_bytes)}."})
                break
        if "static" in c.procedure:
            loc = predict(calculix_cost(50000, sums=compact_sums(50000)), m, advise_options=False)
            opts.append({"kind": "submodel", "time_s": round(loc.seconds),
                         "summary": "Then refine only around the peak. wobbl submodel takes the coarse result and "
                                    "drives a fine local model from its displacements, and a local model of 50,000 "
                                    f"nodes takes about {duration(loc.seconds)} here."})
    elif c.solver == "openradioss":
        half = predict(Cost(c.solver, c.procedure, c.core_seconds / 2, c.parallel, c.peak_bytes, c.disk_bytes,
                            c.size, c.basis), m, max_hours, advise_options=False)
        opts.append({"kind": "coarser", "time_s": round(half.seconds),
                     "summary": f"Mass scaling to a time step twice as long halves the run to about "
                                f"{duration(half.seconds)}. Check that the added mass stays a small share of the "
                                "model's."})
    from .jobs.remote import machines as remote_machines

    remotes = remote_machines()
    for name, rm in remotes.items():
        e = predict(c, rm, max_hours, advise_options=False)
        opts.append({"kind": "remote", "name": name, "time_s": round(e.seconds), "fits": e.fits,
                     "command": f"wobbl jobs start {target} --on {name}",
                     "summary": f"On {name} ({rm.cores} cores, {nbytes(rm.ram)}) it takes about "
                                f"{duration(e.seconds)}" + ("" if e.fits else " but does not fit there either") +
                                f". Run it there with wobbl jobs start {target} --on {name}."})
    if not remotes:
        opts.append({"kind": "remote", "command": "wobbl remote add NAME user@host",
                     "summary": "Add a workstation or server you own with wobbl remote add NAME user@host, and "
                                "Wobbl runs the job there over SSH, sending only the input and bringing back only "
                                "the results."})
    from .jobs.cloud import cheapest

    best = cheapest(c, limit)
    if best:
        opts.append({"kind": "cloud", **best,
                     "summary": f"On an AWS {best['instance']} ({best['cores']} cores, {best['ram_gb']:g} GB) it "
                                f"takes about {duration(best['time_s'])} for about ${best['cost_usd']:.2f} at on "
                                f"demand prices. wobbl jobs start {target} --cloud aws --instance "
                                f"{best['instance']} shows the full plan and starts nothing until you confirm it."})
    return opts


# ------------------------------------------------------------------------------------------------ entry points
def estimate_deck(path: str | Path, machine: Machine | None = None, max_hours: float | None = None,
                  checkpoint: bool = False) -> Estimate:
    """Read a deck or an OpenFOAM case without loading it and predict its run."""
    from .decks import read_deck

    deck = read_deck(path)
    est = predict(deck_cost(deck, checkpoint), machine or this_machine(Path(path).resolve().parent), max_hours,
                  str(path))
    est.basis += deck.notes
    return est


def estimate_size(solver: str, nodes: int = 0, elements: int = 0, end_time: float = 0.0, time_step: float = 0.0,
                  iterations: int = 0, procedure: str = "static", solves: float = 1.0,
                  machine: Machine | None = None, max_hours: float | None = None, target: str = "") -> Estimate:
    """Predict a run from its size alone, for a model that is not on this machine."""
    s = solver.lower()
    if s in ("calculix", "ccx"):
        heat = procedure.startswith("heat")
        cost = calculix_cost(nodes or int(elements * 1.9), elements, dof=1 if heat else 3, solves=solves,
                             procedure=procedure)
    elif s in ("openradioss", "radioss", "explicit"):
        if not end_time:
            raise ValueError("an explicit run needs its end time, for example --end-time 0.2")
        cost = explicit_cost(elements or nodes, nodes, end_time, time_step)
    elif s in ("openfoam", "cfd"):
        cost = cfd_cost(elements or nodes, iterations or 1000)
    elif s in ("ngspice", "spice", "circuit"):
        cost = spice_cost(iterations or 10000, nodes or 10)
    else:
        raise ValueError(f"no cost model for '{solver}'. Use calculix, openradioss, openfoam or ngspice")
    return predict(cost, machine, max_hours, target)

