"""Drop test: any part, any height, any orientation (or the worst of many), onto any floor.

"Will it survive a 1.2 m drop onto concrete?" becomes: mesh the part, give it the impact speed sqrt(2 g h) the
moment it touches the floor, and march its motion through the impact with the floor pushing back on the surface
nodes that go below it (explicit.py). The stress is read at about a hundred instants and the largest one, away
from the patch that hits the floor, is compared with the material's strength. An accelerometer anywhere on the
part (a battery mount, a sensor) reads the g it feels.

When the orientation is not given, every face, edge and corner landing is screened on a linear tet version of
the mesh, the worst few are run again with quadratic tets, and the worst of those is refined until its stress
settles. Momentum and energy are checked on every run.
"""
from __future__ import annotations

import itertools
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from ..units import fmt, to_internal
from ..verify.checks import Check, confidence, convergence_check
from . import tet10
from .explicit import Contact, History, integrate, omega_max
from .floors import FloorMaterial, get_floor

G = 9810.0                   # mm/s2
TOP = {"fast": 1, "normal": 2, "fine": 3}          # screened landings run again on quadratic tets
MAX_NODES = 80_000           # the largest solid mesh, estimated, a drop marches
TOL = 0.10                   # drop peaks settle to about 10% between meshes
RIGID_FLOOR = 10.0           # a hard floor is ten times stiffer than a layer of the part one element deep


@dataclass
class DropStudy:
    part: str
    material: str = "auto"
    height: str = "1 m"
    orientation: str = "worst"     # worst, faces, edges, corners, flat, or a direction such as -z or 1,1,-1
    floor: str = "concrete"
    masses: list[str] = field(default_factory=list)   # "45 g @ 0,0,5 r=8" or "45 g @ <faces>"
    watch: str = ""                # where the accelerometer sits: "x,y,z", faces, or the first added mass
    target_sf: float | None = None
    quality: str = "normal"
    question: str = ""
    scale: float = 1.0
    params: dict = field(default_factory=dict)
    mesh_size: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Model:
    """One mesh of the part with everything the integrator needs."""
    X: np.ndarray
    tets: np.ndarray
    tris: np.ndarray               # boundary faces (corner nodes first)
    tri_surf: np.ndarray
    K: object
    mass: np.ndarray               # lumped, with the added masses
    h: float
    wmax: float
    order: int
    mesh: object = None
    _rec: object = None
    scaled: float = 0.0            # share of mass added by mass scaling

    def recovery(self, E, nu):
        if self._rec is None:
            self._rec = tet10.StressRecovery(self.X, self.tets, E, nu)
        return self._rec


def parse_height(text: str) -> float:
    """Drop height in mm; a bare number is meters, as drop heights usually are."""
    return to_internal(str(text), "length", "m")


def make_model(prep, h: float, mat: dict, order: int, base=None) -> Model:
    from ..mesh.mesher import Sizing, build_mesh

    mesh = base.mesh if base is not None else build_mesh(prep, Sizing(h=h, order=2, curvature=12))
    tets, tris = (mesh.tets, mesh.tris) if order == 2 else _linear(mesh)
    X = mesh.X
    if order == 1:
        used = np.unique(tets)
        remap = -np.ones(len(X), np.int64)
        remap[used] = np.arange(len(used))
        X, tets, tris = X[used], remap[tets], remap[tris]
    K = tet10.stiffness(X, tets, mat["E"], mat["nu"])
    mass = tet10.lumped_mass(X, tets, mat["density"])
    return Model(X, tets, tris, mesh.tri_surf, K, mass, h, 0.0, order, mesh)


def finish(model: Model, extra: np.ndarray, scale: float = 0.02) -> Model:
    """Add the point masses, then a little mass where the smallest elements sit (selective mass scaling, at most
    scale of the part's mass) so they no longer hold the time step down, and find the mesh's top frequency.

    A node's frequency is bounded by the row sum of the stiffness over its mass. The few nodes of slivers that
    the mesher leaves sit far above the rest; raising their mass to bring them down to a common bound costs
    a percent or two of the mass and lengthens the step by a third or more.
    """
    model.mass = model.mass + extra
    if scale > 0:
        r = np.asarray(abs(model.K).sum(1)).ravel().reshape(-1, 3).max(1)
        m0, total = model.mass, model.mass.sum()
        lo, hi = 1.0, float(np.sqrt((r / m0).max()))
        for _ in range(50):
            mid = np.sqrt(lo * hi)
            if np.maximum(r / mid ** 2 - m0, 0.0).sum() > scale * total:
                lo = mid
            else:
                hi = mid
        model.mass = np.maximum(m0, r / hi ** 2)
        model.scaled = float(model.mass.sum() / total - 1.0)
    model.wmax = omega_max(model.K, np.repeat(model.mass, 3))
    return model


def _linear(mesh):
    return mesh.tets[:, :4], mesh.tris[:, :3]


def surface_weights(model: Model) -> np.ndarray:
    """Each surface node's share of the surface area, mm2: HRZ for six node faces, thirds for three node ones."""
    t = model.tris
    P = model.X[t[:, :3]]
    area = 0.5 * np.linalg.norm(np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]), axis=1)
    w = np.zeros(len(model.X))
    if t.shape[1] == 6:
        for k in range(3):
            np.add.at(w, t[:, k], area * 3 / 57)
            np.add.at(w, t[:, 3 + k], area * 16 / 57)
    else:
        for k in range(3):
            np.add.at(w, t[:, k], area / 3)
    return w


# --- orientations ---------------------------------------------------------------------------------------------

def candidates(spec: str, X: np.ndarray, mass: np.ndarray, quality: str = "normal") -> list[tuple[str, np.ndarray]]:
    """The landings to try: (name, direction that points down) in the part's own axes."""
    from ..geometry.select import parse_vector

    s = (spec or "worst").strip().lower()
    com = (mass[:, None] * X).sum(0) / mass.sum()
    lo, hi = X.min(0), X.max(0)
    out: list[tuple[str, np.ndarray]] = []
    want = set()
    for part in [p.strip() for p in s.replace(" and ", ",").split(";") if p.strip()] or ["worst"]:
        if part in ("worst", "any", "auto"):
            want |= {"face", "corner"} | ({"edge"} if quality == "fine" else set())
        elif part == "all":
            want |= {"face", "edge", "corner"}
        elif part.rstrip("s") in ("face", "edge", "corner"):
            want.add(part.rstrip("s"))
        elif part == "flat":
            want.add("flat")
        else:
            out.append((f"{part} down", parse_vector(part)))
    ax = "xyz"
    if "face" in want or "flat" in want:
        thin = int(np.argmin(hi - lo))
        for a in range(3):
            if "face" not in want and a != thin:
                continue
            for sgn in (-1, 1):
                d = np.zeros(3)
                d[a] = sgn
                out.append((f"{'+' if sgn > 0 else '-'}{ax[a]} face down", d))
    if "edge" in want:
        for a, b in itertools.combinations(range(3), 2):
            for sa, sb in itertools.product((-1, 1), repeat=2):
                p = com.copy()
                p[a] = hi[a] if sa > 0 else lo[a]
                p[b] = hi[b] if sb > 0 else lo[b]
                name = f"{'+' if sa > 0 else '-'}{ax[a]}{'+' if sb > 0 else '-'}{ax[b]} edge down"
                out.append((name, (p - com) / np.linalg.norm(p - com)))
    if "corner" in want:
        for sg in itertools.product((-1, 1), repeat=3):
            p = np.where(np.array(sg) > 0, hi, lo)
            name = "".join(f"{'+' if g > 0 else '-'}{ax[i]}" for i, g in enumerate(sg)) + " corner down"
            out.append((name, (p - com) / np.linalg.norm(p - com)))
    return out


def mirror_axes(X: np.ndarray, extra: np.ndarray, h: float) -> list[int]:
    """Axes the part is mirror symmetric across, through its center: the mirrored mesh lands on the mesh within
    about half an element everywhere (the mesh itself is not symmetric), and the added masses mirror too."""
    from scipy.spatial import cKDTree

    c = 0.5 * (X.min(0) + X.max(0))
    tree = cKDTree(X)
    out = []
    for a in range(3):
        Y = X.copy()
        Y[:, a] = 2 * c[a] - Y[:, a]
        d, _ = tree.query(Y)
        if np.quantile(d, 0.995) > 0.6 * h:
            continue
        if extra.any():
            w = extra / extra.sum()
            p = (w[:, None] * X).sum(0)
            if abs(p[a] - c[a]) > 0.6 * h:
                continue
        out.append(a)
    return out


def unique_landings(cands, axes: list[int]) -> tuple[list, dict]:
    """Drop the landings that a mirror symmetry of the part makes the same as one already listed."""
    seen, keep, same = {}, [], {}
    for name, d in cands:
        key = d.copy()
        for a in axes:
            key[a] = abs(key[a])
        k = tuple(np.round(key, 2))      # the center of mass of a mesh sits a hair off the mirror plane
        if k in seen:
            same.setdefault(seen[k], []).append(name)
            continue
        seen[k] = name
        keep.append((name, d))
    return keep, same


# --- one run ----------------------------------------------------------------------------------------------------

def contact_for(model: Model, d: np.ndarray, floor: FloorMaterial, E: float, band: float) -> Contact:
    """The floor under the part landing along d: the surface nodes within band of the lowest point, each with its
    patch of floor."""
    n = -d / np.linalg.norm(d)
    surf = np.unique(model.tris.ravel())
    hgt = model.X[surf] @ n
    level = float(hgt.min())
    cand = surf[hgt - level <= band]
    area = surface_weights(model)[cand]
    m = model.mass[cand]
    # the stiffest patch the step can follow: half the mesh's own top frequency on the node's mass
    cap = 0.5 * m * model.wmax ** 2
    k_rigid = np.minimum(RIGID_FLOOR * E / model.h * area, cap)
    if floor.hard:
        return Contact(n, level, cand, np.maximum(k_rigid, 1e-12))
    k = np.minimum(floor.modulus / floor.thickness * area, k_rigid)
    crush = floor.plateau * area if floor.plateau else None
    return Contact(n, level, cand, np.maximum(k, 1e-12), crush, np.full(len(cand), floor.bottom * floor.thickness),
                   k_rigid)


def simulate(model: Model, d: np.ndarray, v0: float, floor: FloorMaterial, mat: dict, watch: np.ndarray,
             frames: int = 100, ring: float = 0.3, progress=None) -> tuple[History, Contact]:
    band = 0.15 * float(np.ptp(model.X @ d)) + 2 * model.h
    c = contact_for(model, d, floor, mat["E"], band)
    # long enough for a soft floor: a spring and mass pulse on the softest floor patch, plus ringing
    m = model.mass.sum()
    k_all = float(c.k.sum())
    pulse = np.pi * np.sqrt(m / max(k_all * 0.05, 1e-30))
    cw = np.sqrt(mat["E"] / mat["density"])
    t_max = min(max(3 * pulse, 20 * float(np.ptp(model.X, axis=0).max()) / cw), 0.05)
    hist = integrate(model.K, model.mass, model.X, c, v0 * d, G * d, watch, t_max, frames, model.wmax,
                     ring=ring, progress=progress)
    return hist, c


def stress_history(model: Model, hist: History, mat: dict, ductile: bool) -> np.ndarray:
    """(frames, n) the stress compared with the strength at every kept instant: von Mises for ductile materials,
    the largest principal stress for brittle ones, each averaged over the time a wave takes to cross two elements.
    That average keeps every peak the part really feels and drops the overshoot of a sudden wave front, which no
    mesh resolves."""
    rec = model.recovery(mat["E"], mat["nu"])
    S = np.array([(tet10.von_mises if ductile else tet10.max_principal)(rec.stress(u.astype(float)))
                  for u in hist.frames_u])
    if len(hist.frames_t) > 2:
        step = float(np.median(np.diff(hist.frames_t)))
        cw = np.sqrt(mat["E"] / mat["density"])
        k = int(round(2 * model.h / cw / max(step, 1e-30)))
        if k >= 2:
            from scipy.ndimage import uniform_filter1d
            S = uniform_filter1d(S, k, axis=0, mode="nearest")
    return np.maximum(S, 0.0)


def lowpass(x: np.ndarray, dt: float, fc: float, before: float = 0.0) -> np.ndarray:
    """Zero phase fourth order Butterworth low pass, the way accelerometer records are filtered. The record is
    padded with the free fall reading before it and its last reading after it, three filter periods each, so the
    filter starts and ends at rest instead of inside the impact."""
    from scipy.signal import butter, sosfiltfilt

    nyq = 0.5 / dt
    if fc >= 0.95 * nyq or len(x) < 3:
        return x
    pad = int(np.ceil(3.0 / (fc * dt)))
    y = np.concatenate([np.full(pad, before), x, np.full(pad, x[-1])])
    sos = butter(4, fc / nyq, output="sos")
    return sosfiltfilt(sos, y, padtype=None)[pad:pad + len(x)]


def evaluate(model: Model, hist: History, c: Contact, mat: dict, ductile: bool, v0: float, watch: np.ndarray,
             g_cutoff: float | None, zone: float | None = None) -> dict:
    """The run's numbers: peak stress away from the contact patch (the touched nodes and everything within zone of
    them, two elements of the first mesh, the same on every mesh so the levels compare), the g, the pulse and
    what the momentum and energy checks need."""
    S = stress_history(model, hist, mat, ductile)
    peak = S.max(0)
    from scipy.spatial import cKDTree

    r = zone or 2.0 * model.h
    zone = np.zeros(len(model.X), bool)
    if hist.touched is not None and len(hist.touched):
        dist, _ = cKDTree(model.X[hist.touched]).query(model.X)
        zone = dist <= r
    away = ~zone if (~zone).any() else np.ones(len(zone), bool)
    gi = int(np.flatnonzero(away)[np.argmax(peak[away])])
    ci = int(np.argmax(peak))
    fi = int(np.argmax(S[:, gi]))
    up = c.normal
    acc = hist.accel @ up + G             # what an accelerometer reads: zero in free fall
    acc_f = lowpass(acc, hist.dt, g_cutoff, before=0.0) if g_cutoff else acc
    m = model.mass.sum()
    vz = hist.com_v @ up
    t0, t1 = hist.t_contact
    impulse = float(np.trapezoid(hist.force, hist.t))
    grav = m * G * float(hist.t[-1])
    return {"S": S, "peak": peak, "zone": zone, "zone_mm": r, "node": gi, "frame": fi, "governing": float(peak[gi]),
            "contact_peak": float(peak[ci]), "contact_node": ci,
            "g_watch": float(acc_f.max()) / G, "g_watch_raw": float(acc.max()) / G,
            "g_rigid": float(hist.force.max()) / (m * G), "force": float(hist.force.max()),
            "pulse_ms": 1e3 * (t1 - t0), "impulse": impulse,
            "momentum_change": m * (float(vz[-1]) + v0), "gravity_impulse": grav,
            "rebound": float(vz[-1]), "restitution": float(vz[-1]) / v0 if v0 else 0.0,
            "energy_drift": float(np.abs(hist.energy).max()) / max(hist.kinetic0, 1e-30),
            "penetration": hist.max_penetration, "acc_f": acc_f, "steps": hist.steps, "dt": hist.dt,
            "t_run": hist.t_run, "reason": hist.reason}


# --- the study --------------------------------------------------------------------------------------------------

def run_drop(study: DropStudy, out_dir: str | Path, progress=None, pre=None, render: bool = True):
    from ..materials import resolve_material
    from ..outcome_generic import GenericOutcome
    from ..structural.run import describe_part
    from ..structural.spec import SpecError

    t_start = time.time()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    say = progress or (lambda a, b: None)
    material = resolve_material(study.material if study.material != "auto" else "abs")
    mat = material.internal()
    H = parse_height(study.height)
    if H <= 0:
        raise SpecError(f"the drop height must be above zero, not '{study.height}'")
    v0 = float(np.sqrt(2 * G * H))
    floor = get_floor(study.floor)
    if pre is None:
        say("geometry", f"reading {Path(study.part).name}")
        pre = describe_part(study.part, out / "work", study.scale, None, study.params)
    prep, feats = pre
    t_wall = feats.thickness_typical or prep.diag / 10
    h0 = study.mesh_size or min(prep.diag / 14.0, 1.5 * t_wall)
    # a ten node tet mesh has about 9.4 V / h^3 nodes, and no element can be larger than the wall it sits in
    estimate = 9.4 * max(prep.volume, 1e-9) / min(h0, t_wall) ** 3
    if estimate > MAX_NODES and not study.mesh_size:
        raise SpecError(f"walls of {fmt(t_wall)} mm on a part {fmt(prep.diag)} mm across need about "
                        f"{estimate / 1000:.0f} thousand nodes as a solid, more than an explicit drop can march "
                        f"here (about {MAX_NODES // 1000} thousand). Pass --mesh-size to force a coarser mesh, or "
                        "drop a shorter piece of the part")
    ductile = material.ductile
    say("mesh", f"meshing with {h0:.3g} mm tets")
    lin = None
    quad0 = make_model(prep, h0, mat, 2)
    extra, watch, mass_notes = added_masses(study, quad0, feats)
    finish(quad0, extra)
    cands = candidates(study.orientation, quad0.X, quad0.mass, study.quality)
    axes = mirror_axes(quad0.X, extra, h0)
    cands, same = unique_landings(cands, axes)
    g_cut = 10_000.0
    zone = 2.0 * h0
    screen = []
    if len(cands) > 1:
        lin = make_model(prep, h0, mat, 1, base=quad0)
        finish(lin, _restrict(extra, quad0, lin), scale=0.05)
        lw = _restrict_nodes(watch, quad0, lin)
        say("screen", f"{len(cands)} landings on linear tets")
        screen = screen_landings(lin, cands, v0, floor, mat, lw, ductile, g_cut, say)
        order = np.argsort([-s["governing"] for s in screen])
        best = screen[int(order[0])]["governing"]
        # the runners up go on to quadratic tets only when the screen puts them within 15% of the worst
        top = [int(i) for i in order[:TOP.get(study.quality, 2)] if screen[int(i)]["governing"] >= 0.85 * best]
    else:
        top = [0]
    # the worst few again on quadratic tets, then the worst of those refined
    runs = {}
    for i in top:
        name, d = cands[i]
        say("solve", f"{name} on quadratic tets")
        hist, c = simulate(quad0, d, v0, floor, mat, watch)
        runs[i] = (hist, c, evaluate(quad0, hist, c, mat, ductile, v0, watch, g_cut, zone))
    worst = max(runs, key=lambda i: runs[i][2]["governing"])
    name, d = cands[worst]
    levels = [_level(0, quad0, runs[worst][2])]
    last = (quad0, *runs[worst])
    if study.quality != "fast":
        # refine where the peak is: a finer patch of mesh around it, the rest as it was
        from ..mesh.mesher import Refinement, Sizing, build_mesh

        g0 = quad0.X[runs[worst][2]["node"]]
        say("mesh", f"level 1: {0.5 * h0:.3g} mm tets around the peak for {name}")
        mesh = build_mesh(prep, Sizing(h=h0, order=2, curvature=12,
                                       refinements=[Refinement(0.5 * h0, points=[tuple(g0)], radius=3 * h0,
                                                               label="drop peak")]))
        mk = make_model(prep, h0, mat, 2, base=_Base(mesh))
        mk.h = 0.5 * h0
        ex, wk, _ = added_masses(study, mk, feats)
        finish(mk, ex)
        say("solve", f"level 1: {len(mk.X):,} nodes")
        hist, c = simulate(mk, d, v0, floor, mat, wk)
        ev = evaluate(mk, hist, c, mat, ductile, v0, wk, g_cut, zone)
        levels.append(_level(1, mk, ev))
        last = (mk, hist, c, ev)
        watch = wk
    model, hist, c, ev = last
    gov = ev["governing"]
    allow = material.allowable
    sf = allow / gov if gov > 0 else float("inf")
    target = study.target_sf or material.recommended_sf
    verdict = "holds" if sf >= target else ("marginal" if sf >= 1.0 else "fails")
    gxyz = model.X[ev["node"]]
    checks = _checks(levels, ev, hist, c, model, floor, prep.diag, allow, material)
    level, reasons = confidence(checks)
    measure = "von Mises" if ductile else "largest principal"
    head = (f"{'Survives' if verdict == 'holds' else 'Marginal on' if verdict == 'marginal' else 'Fails'} a "
            f"{fmt(H / 1000)} m drop onto {floor.name.lower()}, worst landing {name}: peak {measure} stress "
            f"{fmt(gov)} MPa at ({', '.join(fmt(v) for v in gxyz)}) mm, safety factor {sf:.2f} against "
            f"{fmt(allow)} MPa (target {target:g}). An accelerometer there reads {fmt(ev['g_watch'])} g filtered at "
            f"{fmt(g_cut / 1000)} kHz; the impact itself peaks at {fmt(ev['g_rigid'])} g over {fmt(ev['pulse_ms'])} ms.")
    if level != "high":
        head += f" Confidence is {level}."
    metrics = [("peak stress", f"{fmt(gov)} MPa"), ("safety factor", f"{sf:.2f}"), ("target", f"{target:g}"),
               ("worst landing", name), ("peak g", f"{fmt(ev['g_watch'])} g"),
               ("impact speed", f"{fmt(v0 / 1000)} m/s"), ("pulse", f"{fmt(ev['pulse_ms'])} ms")]
    fixes = _fixes(material, gov, target, sf, floor, H) if verdict != "holds" else []
    assumptions = _assumptions(material, floor, H, v0, name, same, mass_notes, model, g_cut, axes)
    img = {}
    if render:
        say("render", "drawing the stress and the g record")
        img = _images(out / "images", model, hist, ev, gxyz, gov, screen, allow)
    values = {"height_mm": H, "impact_speed_mm_s": v0, "floor": floor.key, "orientation": name,
              "down": [float(x) for x in d], "governing_MPa": gov, "governing_xyz": gxyz.tolist(),
              "contact_zone_peak_MPa": ev["contact_peak"], "safety_factor": sf, "allowable_MPa": allow,
              "g_watch": ev["g_watch"], "g_watch_unfiltered": ev["g_watch_raw"], "g_center_of_mass": ev["g_rigid"],
              "peak_force_N": ev["force"], "pulse_ms": ev["pulse_ms"], "restitution": ev["restitution"],
              "screening": screen, "levels": levels, "mass_kg": float(model.mass.sum()) * 1000,
              "accelerometer_cutoff_Hz": g_cut, "measure": measure,
              "_figures": [["stress", "Peak stress over the impact, worst landing"],
                           ["g", "Acceleration at the accelerometer and the floor force"],
                           ["screen", "Peak stress of every landing screened"]]}
    o = GenericOutcome("drop", str(study.part), study.question, verdict, head, metrics, level, reasons, checks,
                       assumptions, values, img, fixes, list(prep.notes), solver="Wobbl explicit (numpy)",
                       t_total=time.time() - t_start)
    try:
        playback(out / "playback.json", model, hist, ev, d, H, v0, floor, name, watch, gov)
    except Exception as e:  # noqa: BLE001  a bonus file, never a reason to fail
        o.notes.append(f"playback.json was not written ({str(e)[:120]})")
    try:
        from .scene import write_scene
        write_scene(out, model, hist, ev, d, H, v0, floor, name, watch, gov, allow, verdict, head, study.question,
                    study.part, measure)
    except Exception as e:  # noqa: BLE001  the 3D view is a bonus, never a reason to fail
        import os
        if os.environ.get("WOBBL_DEBUG"):
            raise
        o.notes.append(f"the 3D view was not written ({str(e)[:120]})")
    shutil.rmtree(out / "work", ignore_errors=True)
    return o.write(out)


class _Base:
    """A mesh made elsewhere, for make_model."""
    def __init__(self, mesh):
        self.mesh = mesh


def screen_landings(lin: Model, cands, v0, floor, mat, watch, ductile, g_cut, say) -> list[dict]:
    """Every landing on the linear tet model, a few at a time in worker processes when the machine has cores to
    spare, stopping each run when the part leaves the floor."""
    import os
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing as mp

    light = Model(lin.X, lin.tets, lin.tris, lin.tri_surf, lin.K, lin.mass, lin.h, lin.wmax, lin.order)
    jobs = [(light, name, d, v0, floor, mat, watch, ductile, g_cut) for name, d in cands]
    workers = min(len(jobs), max(1, (os.cpu_count() or 2) // 3), 4)
    if workers > 1:
        try:
            with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn")) as ex:
                out = []
                for k, r in enumerate(ex.map(_screen_one, jobs)):
                    say("screen", f"{k + 1}/{len(jobs)}: {r['name']}, {fmt(r['governing'])} MPa")
                    out.append(r)
                return out
        except Exception:  # noqa: BLE001  no worker processes here: run them one after another
            pass
    out = []
    for k, job in enumerate(jobs):
        out.append(_screen_one(job))
        say("screen", f"{k + 1}/{len(jobs)}: {out[-1]['name']}, {fmt(out[-1]['governing'])} MPa")
    return out


def _screen_one(job) -> dict:
    model, name, d, v0, floor, mat, watch, ductile, g_cut = job
    hist, c = simulate(model, d, v0, floor, mat, watch, frames=40, ring=0.0)
    ev = evaluate(model, hist, c, mat, ductile, v0, watch, None)     # ranked by stress; the g is read later
    return {"name": name, "d": [float(x) for x in d], "governing": ev["governing"], "g_watch": ev["g_watch"],
            "g_rigid": ev["g_rigid"], "pulse_ms": ev["pulse_ms"]}


def _level(k: int, model: Model, ev: dict) -> dict:
    return {"level": k, "h": model.h, "n_nodes": len(model.X), "governing": ev["governing"],
            "g_watch": ev["g_watch"], "g_rigid": ev["g_rigid"], "pulse_ms": ev["pulse_ms"], "steps": ev["steps"],
            "dt": ev["dt"], "t_run": ev["t_run"], "xyz": model.X[ev["node"]].tolist()}


def _restrict(extra: np.ndarray, quad: Model, lin: Model) -> np.ndarray:
    """Added masses moved from the quadratic mesh to the linear one, onto the nearest corner nodes."""
    from scipy.spatial import cKDTree

    out = np.zeros(len(lin.X))
    nz = np.flatnonzero(extra)
    if len(nz):
        _, j = cKDTree(lin.X).query(quad.X[nz])
        np.add.at(out, j, extra[nz])
    return out


def _restrict_nodes(nodes: np.ndarray, quad: Model, lin: Model) -> np.ndarray:
    from scipy.spatial import cKDTree

    _, j = cKDTree(lin.X).query(quad.X[nodes])
    return np.unique(j)


def added_masses(study: DropStudy, model: Model, feats) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Point masses (a battery, a board) spread over the nodes they sit on, in proportion to the nodes' own mass,
    and the nodes the accelerometer averages over."""
    extra = np.zeros(len(model.X))
    notes, first = [], None
    for text in study.masses:
        if "@" not in text:
            raise ValueError(f"an added mass reads '45 g @ 0,0,5' or '45 g @ <faces>', not '{text}'")
        amount, where = (x.strip() for x in text.split("@", 1))
        m = to_internal(amount, "mass", "g")
        nodes = _nodes_at(where, model, feats)
        w = model.mass[nodes] / model.mass[nodes].sum()
        extra[nodes] += m * w
        first = nodes if first is None else first
        notes.append(f"{amount} spread over {len(nodes)} nodes at {where}")
    if study.watch:
        watch = _nodes_at(study.watch, model, feats)
    elif first is not None:
        watch = first
    else:
        com = (model.mass[:, None] * model.X).sum(0) / model.mass.sum()
        watch = np.argsort(np.linalg.norm(model.X - com, axis=1))[:12]
    return extra, watch, notes


def _nodes_at(where: str, model: Model, feats) -> np.ndarray:
    """Nodes at a point ('x,y,z' or 'x,y,z r=5') or on faces picked the usual way."""
    import re

    m = re.match(r"^\s*\(?\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*\)?\s*(?:r\s*=\s*([\d.]+))?\s*$", where)
    if m:
        p = np.array([float(m.group(i)) for i in (1, 2, 3)])
        r = float(m.group(4)) if m.group(4) else 1.5 * model.h
        d = np.linalg.norm(model.X - p, axis=1)
        nodes = np.flatnonzero(d <= r)
        return nodes if len(nodes) >= 3 else np.argsort(d)[:8]
    from ..geometry.select import Selector

    sel = Selector(feats.faces, feats.bbox).select(where)
    mask = np.isin(model.tri_surf, list(sel.surfaces))
    nodes = np.unique(model.tris[mask].ravel())
    if not len(nodes):
        raise ValueError(f"'{where}' picks no part of the surface")
    return nodes


def _checks(levels, ev, hist, c, model, floor, diag, allow, material) -> list[Check]:
    vals = [L["governing"] for L in levels]
    if len(vals) == 1:
        ch = [Check("mesh convergence", "warn", f"one mesh only (fast quality): {fmt(vals[0])} MPa; run at normal "
                    "quality to refine it and see how far it moves", {"values": vals})]
    else:
        ch = [convergence_check(vals, TOL, "peak drop stress", "MPa",
                                "converged" if abs(vals[-1] - vals[-2]) <= TOL * vals[-1] else "level limit",
                                [L["g_watch"] for L in levels])]
    # momentum: the floor's impulse plus gravity's equals the change of momentum
    lhs = ev["impulse"]
    rhs = ev["momentum_change"] + ev["gravity_impulse"]
    err = abs(lhs - rhs) / max(abs(rhs), 1e-30)
    ch.append(Check("momentum", "pass" if err <= 0.02 else "warn" if err <= 0.05 else "fail",
                    f"the floor's impulse, {fmt(lhs)} N s, against the change of momentum less gravity's impulse, "
                    f"{fmt(rhs)} N s ({100 * err:.2g}% apart); the part leaves at {fmt(-ev['rebound'] / 1000)} m/s, a "
                    f"coefficient of restitution of {abs(ev['restitution']):.2f}", {"error": err}))
    e = ev["energy_drift"]
    ch.append(Check("energy", "pass" if e <= 0.02 else "warn" if e <= 0.05 else "fail",
                    f"kinetic plus strain energy, less the work of the floor and gravity and the energy damped at the "
                    f"mesh's own frequency, stays within {100 * e:.2g}% of the impact energy", {"drift": e}))
    pen = ev["penetration"]
    if floor.hard:
        r = pen / model.h
        ch.append(Check("contact", "pass" if r <= 0.1 else "warn",
                        f"the part sinks at most {fmt(pen)} mm into the floor, {100 * r:.2g}% of an element; the "
                        "floor's stiffness is a penalty that keeps that small", {"penetration_mm": pen}))
    else:
        ch.append(Check("contact", "info", f"the floor gives by up to {fmt(pen)} mm "
                        f"({100 * pen / floor.thickness:.0f}% of its {fmt(floor.thickness)} mm layer)",
                        {"penetration_mm": pen}))
    cz = ev["contact_peak"]
    if cz > ev["governing"]:
        ch.append(Check("contact stress", "info",
                        f"{fmt(cz)} MPa right where the part meets the floor, within {fmt(ev['zone_mm'])} mm of it, is left out "
                        "of the answer: it is the local crush of the contact patch, set by the mesh and the "
                        "floor's idealized stiffness. Expect a scuff or a dent there", {"value": cz}))
    rel = ev["governing"] / allow if allow else 0
    st = "pass" if 1e-4 < rel < 30 and hist.reason != "unstable" else "warn"
    ch.append(Check("units and sanity", st, f"part {fmt(diag)} mm across, {fmt(model.mass.sum() * 1e6)} g, peak g at "
                    f"the center of mass {fmt(ev['g_rigid'])}, pulse {fmt(ev['pulse_ms'])} ms; the run stopped "
                    f"because {hist.reason}", {"stress_ratio": rel}))
    return ch


def _fixes(material, gov, target, sf, floor, H) -> list[dict]:
    need = gov * target
    out = [{"kind": "material", "auto": False, "title": f"A material good for {fmt(need)} MPa",
            "summary": f"The worst landing peaks at {fmt(gov)} MPa; reaching the {target:g} target takes a material "
                       f"whose {material.criterion} strength is at least {fmt(need)} MPa at a similar stiffness, "
                       "or a tougher grade that yields instead of cracking."}]
    if floor.hard:
        out.append({"kind": "bumper", "auto": False, "title": "Soften the landing",
                    "summary": "A rubber bumper or corner guard lengthens the impact and lowers both the stress and "
                               "the g. Check it by running the same drop with --floor 'layer:E=5,t=3', a 3 mm "
                               "elastomer skin."})
    out.append({"kind": "wall", "auto": False, "title": "Spread the load near the worst spot",
                "summary": f"A rib or a thicker wall around the peak spreads the impact. The stress has to fall by "
                           f"{100 * (1 - sf / target):.0f}% to reach the target."})
    return out


def _assumptions(material, floor, H, v0, name, same, mass_notes, model, g_cut, axes) -> list[str]:
    a = [f"Dropped from {fmt(H / 1000)} m, so it lands at {fmt(v0 / 1000)} m/s (no air drag), without spin, "
         f"landing {name}.",
         f"Floor: {floor.describe()}.",
         f"{material.name}: E {fmt(material.E)} MPa, Poisson {material.nu:g}, density {fmt(material.density)} kg/m3, "
         f"{material.criterion} {fmt(material.allowable)} MPa ({material.source}). Strength at impact rates is "
         "often higher than the datasheet's static value, so this errs on the safe side.",
         "Linear elastic part, small strains, no friction with the floor. The first impact only: a bounce and a "
         "second landing on another corner are not followed.",
         "Stress is averaged over the time a wave takes to cross two elements, and the accelerometer record is "
         f"low pass filtered at {fmt(g_cut / 1000)} kHz, the bandwidth of a drop test accelerometer.",
         "The contact patch itself, within two elements of the first mesh of where the part meets the floor, is "
         "reported apart: its stress is the local crush of an ideally sharp landing.",
         f"{100 * model.scaled:.2g}% of the part's mass is added at the nodes of the smallest elements so they do not "
         "hold the time step down (selective mass scaling); the momentum and energy checks include it."]
    a += [f"Added mass: {n}." for n in mass_notes]
    if same:
        a.append("Mirror symmetry of the part made these landings the same as one already run: " +
                 "; ".join(f"{k} stands for {', '.join(v)}" for k, v in same.items()) + ".")
    return a


def _images(out, model, hist, ev, gxyz, gov, screen, allow) -> dict:
    from ..report.render import line_plot
    from ..shells.render import bar_chart, render_shell

    img = {}
    try:
        img["stress"] = render_shell(out / "stress.png", model.X, model.tris, ev["peak"], "peak stress", "MPa",
                                     mark_xyz=gxyz, mark_label=f"{fmt(gov)} MPa", clim=[0.0, max(gov, 1e-9)])
        t = 1e3 * (hist.t - hist.t_contact[0])
        img["g"] = line_plot(out / "g.png", t, {"accelerometer, g": ev["acc_f"] / G,
                                                "floor force / weight, g": hist.force / (model.mass.sum() * G)},
                             "time from first touch, ms", "g")
        if screen:
            img["screen"] = bar_chart(out / "screen.png", [s["name"].replace(" down", "") for s in screen],
                                      [s["governing"] for s in screen], "peak stress, MPa (linear tets)", allow,
                                      "strength", int(np.argmax([s["governing"] for s in screen])))
    except Exception as e:  # noqa: BLE001  pictures are a bonus
        img["error"] = str(e)[:200]
    return img


def playback(path, model: Model, hist: History, ev: dict, d: np.ndarray, H: float, v0: float, floor, name: str,
             watch: np.ndarray, gov: float, frames: int = 50) -> str:
    """playback.json: the part hitting the floor, frame by frame, with the stress flashing through it.

    Positions are x + u(t): u is the solved displacement from the moment of first touch, sent as the center of
    mass's travel plus an elastic part compressed onto a few shapes by a singular value decomposition (u_elastic =
    sum_k coeff[frame, k] * shape[k]), which keeps the file small and the motion exact to well under 1%. The fall
    before the impact is free fall from the height, which a viewer draws itself.
    """
    from ..life.playback import encode, mesh_block, quantize, surface, write

    pts, faces, nodes = surface(model.mesh if model.order == 2 else model, max_vertices=1500)
    idx = np.linspace(0, len(hist.frames_t) - 1, min(frames, len(hist.frames_t))).round().astype(int)
    U = np.array([hist.frames_u[i][nodes] for i in idx], float)              # (f,v,3)
    S = ev["S"][idx][:, nodes]
    m = model.mass[nodes]
    trans = (U * m[None, :, None]).sum(1) / m.sum()                          # (f,3)
    el = (U - trans[:, None, :]).reshape(len(idx), -1)
    uu, ss, vt = np.linalg.svd(el, full_matrices=False)
    keep = int(np.searchsorted(np.cumsum(ss ** 2) / max((ss ** 2).sum(), 1e-300), 0.9999) + 1)
    keep = max(1, min(keep, 10))
    coeff = uu[:, :keep] * ss[:keep]
    shapes = vt[:keep].reshape(keep, len(nodes), 3)
    top = max(float(S.max()), 1e-30)
    sq = np.round(np.clip(S / top, 0, 1) * 255).astype("uint8")
    t = hist.t - hist.t_contact[0]
    pick = np.linspace(0, len(t) - 1, min(400, len(t))).round().astype(int)
    doc = {"format": "wobbl-playback", "version": 1, "kind": "drop",
           "about": "A drop test: the part falls from the height, lands along 'down' and the floor pushes back. "
                    "Frames start at first touch; position = vertex + translation[frame] + sum_k coeff[frame,k] * "
                    "shape[k]. The elastic part is tiny, so scale it up to see it. Stress is the peak measure at "
                    "each vertex, the g record is what an accelerometer at the watched spot reads, low pass "
                    "filtered.",
           "units": {"length": "mm", "time": "s", "stress": "MPa", "acceleration": "g", "force": "N"},
           "mesh": mesh_block(pts, faces),
           "fall": {"height_mm": round(H, 3), "impact_speed_mm_s": round(v0, 3), "down": [round(float(x), 6)
                                                                                          for x in d],
                    "floor": floor.key, "floor_name": floor.name, "landing": name,
                    "up": [round(float(-x), 6) for x in d],
                    "floor_level_mm": round(float((model.X @ (-d)).min()), 4)},
           "frames": {"time_s": [float(f"{x:.6g}") for x in (hist.frames_t[idx] - hist.t_contact[0])],
                      "translation": quantize(trans, "int16"),
                      "coeff": quantize(coeff, "int16"),
                      "stress": encode(sq, "uint8", scale=top / 255.0, offset=0.0)},
           "shapes": quantize(shapes, "int16"),
           "history": {"time_s": [float(f"{x:.6g}") for x in t[pick]],
                       "g_watch": [round(float(x) / G, 3) for x in ev["acc_f"][pick]],
                       "force_N": [round(float(x), 3) for x in hist.force[pick]]},
           "watch_mm": model.X[watch].mean(0).round(4).tolist(),
           "governing": {"stress_MPa": round(gov, 4), "at_mm": model.X[ev["node"]].round(4).tolist(),
                         "time_s": float(f"{hist.frames_t[ev['frame']] - hist.t_contact[0]:.6g}")}}
    return write(path, doc)


__all__ = ["DropStudy", "run_drop", "candidates", "simulate", "make_model", "parse_height"]
