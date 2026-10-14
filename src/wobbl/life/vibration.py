"""Vibration: natural frequencies against what shakes the part, and the stress a sine sweep or random input causes.

CalculiX's *FREQUENCY step finds the modes on meshes that get finer until the frequencies that matter settle. With
those modes Wobbl

    - compares every natural frequency with the rotor speeds and tones the part has to live with,
    - sums the response to a sine sweep of the base and finds its peak, then has CalculiX's *STEADY STATE DYNAMICS
      step, reading the same modes from the .eig file, solve that peak again for the reported stress, and
    - sums the response to a random base input from its PSD (CalculiX has no random step) as a 3 sigma stress,
      with Steinberg's three band fatigue estimate for the test duration.

Checks: frequencies and response stress on successive meshes, the modal mass against the part's mass, Rayleigh's
quotient from the static sag under 1 g, a single mode estimate of the response (Miles' equation for random input),
and the two harmonic solutions against each other.
"""
from __future__ import annotations

import math
import shutil
import time
from pathlib import Path

import numpy as np

from ..geometry.select import Selector
from ..mesh.mesher import build_mesh
from ..solvers import base as solvers
from ..structural.build import build_static, tri_area_weights
from ..structural.spec import SpecError
from ..units import fmt
from . import ccx, modal
from .common import QUALITY, change, excluded_nodes, fit_first, governing, level_sizes, next_sizing, prepare
from .outcome import LifeOutcome, clean_work, save_study
from .vibspec import (VibrationStudy, default_damping, default_duration, parse_mass, parse_sine, seconds,
                      unit_vector)

MAX_MODES = 40


def run_vibration(study: VibrationStudy, out_dir: str | Path, progress=None, pre=None,
                  render: bool = True) -> LifeOutcome:
    from . import vibchecks as vc

    t0 = time.time()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    work = out / "work"
    base = study.base
    say = progress or (lambda a, b: None)
    adapter = solvers.for_domain("structural")
    exe = adapter.require()
    sine = parse_sine(study.sine) if study.sine else None
    if base.loads and sine is None:
        raise SpecError("loads in a vibration study shake the part as harmonic forces, so give the band they sweep "
                        "over, for example --sine '10-500 Hz'")
    if sine is not None and sine.accel == 0 and not base.loads:
        raise SpecError("a sweep needs a base acceleration, as in '2 g z 10-500 Hz', or loads that shake the part")
    p = prepare(base, work, progress, pre)
    mat = p.material
    zeta, zeta_why = (study.damping, f"{100 * study.damping:g}% of critical, as given") if study.damping else \
        default_damping(mat)
    if not 0 < zeta < 0.5:
        raise SpecError("damping is a fraction of critical, for example 0.02 for 2%")
    masses = [parse_mass(m) for m in study.masses]
    excite = study.excitations()
    psd = modal.parse_psd(study.psd) if study.psd else None
    axis = unit_vector(study.axis)
    top = max([e.hi for e in excite] + ([sine.hi] if sine else []) + ([float(psd.f[-1])] if psd else []) + [0.0])
    n_modes = int(min(max(2, study.modes), MAX_MODES))
    q = QUALITY.get(base.quality, QUALITY["normal"])
    sizes = level_sizes(p, base.quality, base.mesh_size)
    sel = Selector(p.features.faces, p.features.bbox)
    levels, last, stopped_by = [], None, "level limit"
    for k, planned in enumerate(sizes):
        sz = planned if last is None else next_sizing(planned, last[5], levels[-1]["n_nodes"], q["nodes"])
        if sz is None:
            stopped_by = "node limit"
            break
        mesh = build_mesh(p.prep, sz)
        if last is None:
            sz, mesh = fit_first(p, sz, mesh, q["nodes"])
        say("mesh", f"level {k}: element size {sz.h:.3g} mm, {mesh.n_nodes:,} nodes")
        built = build_static(mesh, p.loads, p.fixtures, mat, p.features.faces)
        hold_walls(built)
        attached = _attach(masses, sel, mesh)
        wl = work / f"level{k}"
        deck = ccx.make_deck(built.model, wl / "static.inp", attached)
        say("solve", f"level {k}: {n_modes} modes of {mesh.n_nodes:,} nodes with {adapter.name}")
        ccx.run(exe, wl, "modes", ccx.frequency_inp(deck, n_modes))
        modes = _read(wl, mesh, built.model.density, deck)
        if modes.count == 0 or modes.freq[0] <= 1e-3 * max(modes.freq[-1], 1e-9):
            raise SpecError("the supports leave the part free to move as a rigid body (a natural frequency of zero). "
                            "Hold it the way it is mounted, for example --fix 'bolt holes d=3.2'")
        (wl / "modes.frd").unlink(missing_ok=True)
        applied = _applied(exe, wl, deck, modes, mesh, built) if p.loads else None
        resp = _response(modes, mesh, built, p, sz, zeta, sine, psd, axis, applied)
        watch = _watched(modes, excite, axis)
        levels.append({"level": k, "n_nodes": mesh.n_nodes, "h": sz.h, "freq": modes.freq[:8].tolist(),
                       "watch_mode": watch + 1, "watch_Hz": float(modes.freq[watch]),
                       "stress": resp["stress"] if resp else None})
        if last is not None:
            shutil.rmtree(last[-1], ignore_errors=True)   # an earlier level's solver files are not needed again
        last = (mesh, built, deck, modes, resp, sz, wl)
        say("check", f"level {k}: mode {watch + 1} at {fmt(modes.freq[watch])} Hz"
                     + (f", response {fmt(resp['stress'])} MPa" if resp else "")
                     + (f" ({100 * change([L['watch_Hz'] for L in levels]):.1f}% change)" if k else ""))
        if k >= 1 and change([L["watch_Hz"] for L in levels]) <= q["tol"] and (
                resp is None or change([L["stress"] for L in levels]) <= 2.5 * q["tol"]):
            stopped_by = "converged"
            break
    mesh, built, deck, modes, resp, sz, wl = last
    if modes.freq[-1] < 1.5 * top and modes.count == n_modes and n_modes < MAX_MODES:
        say("check", f"the {n_modes} modes found reach {fmt(modes.freq[-1])} Hz, short of 1.5 times the "
                     f"{fmt(top)} Hz input, and more modes would cover it (--modes)")

    harmonic = None
    if sine is not None and resp is not None:
        say("solve", f"steady state dynamics at {fmt(resp['harmonic']['f_peak'])} Hz with {adapter.name}")
        harmonic = _ccx_harmonic(exe, wl, deck, modes, resp, sine, zeta, mesh)
    rayleigh = _rayleigh(exe, wl, deck, modes, mesh, built)
    dur, dur_why = (seconds(study.duration), "given") if study.duration else default_duration(study.psd)
    say("check", "checking mass, Rayleigh's quotient and the single mode estimates")
    checks, extra, verdict, sf, headline, values = vc.evaluate(
        study, mat, modes, mesh, built, levels, stopped_by, q["tol"], zeta, excite, sine, psd, axis, resp,
        harmonic, rayleigh, dur, p)
    say("render", "drawing the modes and the response")
    images = vc.images(out / "images", mesh, modes, excite, sine, psd, axis, resp, harmonic, study.margin) \
        if render else {}
    from ..verify.checks import confidence

    level, reasons = confidence(checks, extra)
    if level != "high":
        headline += f" Confidence is {level}."
    values.update({"damping": zeta, "damping_reason": zeta_why, "duration_s": dur, "duration_reason": dur_why,
                   "levels": levels})
    if vc.playback(out / "playback.json", mesh, modes, excite, resp, values):
        values["playback"] = "playback.json"
    o = LifeOutcome("vibration", str(base.part), base.question, verdict, headline, vc.metrics(values), level,
                    reasons, checks, vc.assumptions(mat, zeta_why, masses, deck, sine, psd, base), values, images,
                    vc.fixes(values, mat, p.features), deck.notes + p.prep.notes,
                    solver=f"CalculiX {adapter.version()}", t_total=time.time() - t0)
    o.write(out)
    save_study(study, out)
    clean_work(work, study.keep)
    return o


def _attach(masses, sel: Selector, mesh) -> list:
    out = []
    for k, m in enumerate(masses):
        try:
            mask = sel.select(m.faces).tri_mask(mesh)
        except Exception as e:  # noqa: BLE001
            raise SpecError(f"mass {k + 1} ({m.describe()}): {e}") from None
        if not mask.any():
            raise SpecError(f"mass {k + 1} ({m.describe()}) sits on no part of the mesh, so widen the selection")
        w, _ = tri_area_weights(mesh, mask)
        nodes = np.nonzero(w > 0)[0]
        point = np.asarray(m.point, float) if m.point is not None else (mesh.X[nodes] * w[nodes, None]).sum(0) / \
            w[nodes].sum()
        out.append(ccx.Attached(f"mass {k + 1}", nodes, point, m.mass, m.rigid, w[nodes]))
    return out


def _read(wl: Path, mesh, density: float, deck: ccx.Deck) -> modal.Modes:
    """The modes of one mesh. Participation factors and effective masses are Wobbl's own, from its mass matrix:
    CalculiX's sums count the rotation node of a rigid attachment as if it moved along x, y and z, which inflates
    them whenever an attached mass sits off its faces."""
    X, n = mesh.X, mesh.n_nodes
    e = ccx.read_eigen(wl / "modes.dat")
    _, U, S = ccx.read_modes(wl / "modes.frd", n, 2 * len(deck.extra))
    U = ccx.fill_rigid(U, X, deck).astype(np.float32) if deck.rigid_sets and len(U) else U
    m = min(len(e["freq"]), len(U))
    modes = modal.Modes(e["freq"][:m], U[:m], None if S is None else S[:m], e["participation"][:m],
                        e["effective_mass"][:m], e["total_mass"], n, deck.mass_nodes, deck.mass_values)
    modes.solver_total_mass = e["total_mass"]
    X_all = np.zeros((U.shape[1], 3))
    X_all[:n] = X
    for ref, (_, point) in deck.rigid_sets.items():
        X_all[ref] = X_all[ref + 1] = point
    modes.participation, modes.total_mass = modal.participation(mesh, density, modes, X_all)
    modes.effective_mass = modes.participation ** 2
    return modes


def _watched(modes: modal.Modes, excite, axis) -> int:
    """The mode whose frequency the answer hangs on: the one nearest an excitation, else the first mode that moves a
    tenth or more of the mass along the input axis, else the first."""
    if excite:
        mid = [math.sqrt(e.lo * e.hi) for e in excite]
        return int(np.argmin([min(abs(math.log(f / x)) for x in mid) for f in modes.freq]))
    frac = (modes.gamma(axis) ** 2) / max(float(modes.total_mass[:3].mean()), 1e-300)
    big = np.nonzero(frac >= 0.1)[0]
    return int(big[0]) if len(big) else 0


def hold_walls(built) -> None:
    """A part resting on a wall stays against it while it vibrates, as long as whatever presses it there outweighs
    the vibration, so each wall holds its faces along the wall's normal and lets them slide along it."""
    from ..fe import NormalConstraint

    for w in built.walls:
        built.model.normal_constraints.append(NormalConstraint(w.name, w.nodes, w.normal))
        built.zones[w.name] = w.nodes


def _applied(exe, wl: Path, deck, modes, mesh, built) -> dict:
    """Modal forces of the study's loads, from CalculiX's static solution under them. With K phi = w^2 M phi the
    modal force phi^T f equals w^2 phi^T M u, which covers point loads, pressures and loads carried in through
    couplings alike. The share of the static deflection the modes carry, sum (phi^T M u)^2 / u^T M u, says how much
    the modes left out would add."""
    text = deck.head + "*STEP\n*STATIC\n" + deck.supports + deck.loads + "*NODE FILE\nU\n*END STEP\n"
    ccx.run(exe, wl, "loads", text)
    n_all = mesh.n_nodes + 2 * len(deck.extra)
    u = ccx.read_blocks(wl / "loads.frd", n_all, ("DISP",))["DISP"][-1][1][:, :3].astype(float)
    u = ccx.fill_rigid(u, mesh.X, deck)
    (wl / "loads.frd").unlink(missing_ok=True)
    Mu = modal.mass_times(mesh, built.model.density, u, modes.mass_nodes, modes.mass_values)
    proj = np.einsum("mnd,nd->m", modes.shapes.astype(float), Mu)
    w2 = (2 * math.pi * modes.freq) ** 2
    uMu = float(np.sum(u * Mu))
    return {"force": w2 * proj, "share": float(np.sum(proj ** 2) / uMu) if uMu > 0 else 0.0,
            "static_mm": float(np.linalg.norm(u[:mesh.n_nodes], axis=1).max())}


def sweep_force(modes, sine, applied) -> np.ndarray:
    """Modal forces of a sweep: the base acceleration's inertia and the applied loads, in phase."""
    force = np.zeros(modes.count)
    if sine.accel:
        force = force + modal.base_force(modes, sine.accel, sine.direction)
    if applied is not None:
        force = force + applied["force"]
    return force


def _response(modes, mesh, built, p, sz, zeta, sine, psd, axis, applied=None) -> dict | None:
    """Governing response stress on this mesh, from the modes: the sine sweep's peak and the random 3 sigma."""
    if (sine is None and psd is None) or modes.stress is None:
        return None
    out = {}
    if psd is not None:
        R = modal.random_modal(modes, zeta, psd, axis)
        rms = modal.vm_rms(modes, R.C0)
        excluded, r_ex = excluded_nodes(mesh, built, p.features, sz, rms)
        gi = governing(rms, excluded)
        out["random"] = {"rms": rms, "excluded": excluded, "node": gi, "xyz": mesh.X[gi].tolist(),
                         "sigma_rms": float(rms[gi]), "raw_max": float(rms.max()), "rate": modal.zero_crossing_rate(
                             modes, R, gi), "R": R, "exclusion_radius": r_ex}
        out["stress"] = 3 * float(rms[gi])
    if sine is not None:
        out["harmonic"] = _sweep(modes, mesh, built, p, sz, zeta, sine, sweep_force(modes, sine, applied))
        if applied is not None:
            out["harmonic"]["load_share"] = applied["share"]
            out["harmonic"]["static_mm"] = applied["static_mm"]
        out["stress"] = max(out.get("stress", 0.0), out["harmonic"]["peak"])
    return out


def _sweep(modes, mesh, built, p, sz, zeta, sine, force) -> dict:
    """The largest von Mises amplitude over a sweep: at each mode in the band and around it, then the best
    frequency refined, all from the modal sum."""
    near = [f * (1 + zeta * np.linspace(-4, 4, 17)) for f in modes.freq if sine.lo <= f <= sine.hi]
    grid = np.unique(np.concatenate(near + [np.array([sine.lo, sine.hi])]))
    grid = grid[(grid >= sine.lo) & (grid <= sine.hi)]
    S = modes.stress.astype(float)

    def field(f):
        qf = modal.harmonic_coords(np.array([f]), modes, zeta, force)[0]
        return modal.vm_amplitude(np.einsum("m,mnc->nc", qf, S))

    first = field(grid[0])
    excluded, r_ex = excluded_nodes(mesh, built, p.features, sz, first)
    best = (-1.0, grid[0], first)
    for f in grid:
        vm = field(f)
        v = float(vm[~excluded].max()) if (~excluded).any() else float(vm.max())
        if v > best[0]:
            best = (v, f, vm)
    _, f_peak, vm = best
    excluded, r_ex = excluded_nodes(mesh, built, p.features, sz, vm)
    gi = governing(vm, excluded)
    # refine the peak frequency at the governing node with a golden section search across the neighbouring points
    lo, hi = f_peak * (1 - zeta), f_peak * (1 + zeta)
    node = lambda f: float(modal.vm_amplitude(np.einsum(   # noqa: E731
        "m,mc->c", modal.harmonic_coords(np.array([f]), modes, zeta, force)[0], S[:, gi])[None, :])[0])
    g = (math.sqrt(5) - 1) / 2
    a, b = max(lo, sine.lo), min(hi, sine.hi)
    for _ in range(40):
        c, d = b - g * (b - a), a + g * (b - a)
        if node(c) > node(d):
            b = d
        else:
            a = c
    f_peak = 0.5 * (a + b)
    vm = field(f_peak)
    gi = governing(vm, excluded)
    curve_f = np.unique(np.concatenate([np.geomspace(sine.lo, sine.hi, 300), grid]))
    q = modal.harmonic_coords(curve_f, modes, zeta, force)
    curve = modal.vm_amplitude(np.einsum("fm,mc->fc", q, S[:, gi]))
    return {"f_peak": float(f_peak), "peak": float(vm[gi]), "node": gi, "xyz": mesh.X[gi].tolist(), "vm": vm,
            "excluded": excluded, "raw_max": float(vm.max()), "curve_f": curve_f, "curve": curve,
            "exclusion_radius": r_ex, "force": force, "masses": mass_motion(modes, zeta, force, f_peak, sine)}


def mass_motion(modes: modal.Modes, zeta: float, force: np.ndarray, f: float, sine) -> list[dict]:
    """How far each attached mass moves relative to the base at frequency f, and its acceleration, which against a
    base input is the transmissibility: a = a_base - w^2 u_relative."""
    qf = modal.harmonic_coords(np.array([f]), modes, zeta, force)[0]
    w2 = (2 * math.pi * f) ** 2
    d = np.asarray(sine.direction, float) / np.linalg.norm(sine.direction)
    out = []
    for name, nodes in modes.mass_nodes.items():
        m = np.asarray(modes.mass_values[name], float)
        u = np.einsum("m,mkd->kd", qf, modes.shapes[:, nodes].astype(float))
        u = (u * m[:, None]).sum(0) / m.sum()
        accel = sine.accel * d - w2 * u
        a = float(np.linalg.norm(np.abs(accel)))
        out.append({"name": name, "relative_mm": float(np.linalg.norm(np.abs(u))), "accel_g": a / modal.G0,
                    "transmissibility": a / sine.accel if sine.accel else None})
    return out


def _ccx_harmonic(exe, wl: Path, deck, modes, resp, sine, zeta, mesh) -> dict | None:
    """Solve the sweep again with CalculiX's steady state dynamics, reading the same modes from the .eig file: the
    motion of the governing node across the band, and the whole field at Wobbl's peak frequency."""
    h = resp["harmonic"]
    gi = h["node"]
    loads = (ccx.body_load(deck, sine.accel, sine.direction) if sine.accel else "") + \
        (deck.loads if "load_share" in h else "")
    fp = h["f_peak"]
    try:
        text = ccx.harmonic_inp(deck, loads, [(sine.lo, sine.hi, 6, 3.0), (fp, fp * (1 + 1e-6), 2, 1.0)], zeta,
                                modes.count, np.array([gi]))
        ccx.run(exe, wl, "harmonic", text, eig_from="modes")
        printed = ccx.read_node_print(wl / "harmonic.dat")
        fieldset = ccx.read_harmonic_field(wl / "harmonic.frd", mesh.n_nodes + 2 * len(deck.extra))
        (wl / "harmonic.frd").unlink(missing_ok=True)
    except Exception as e:  # noqa: BLE001  the cross check is a bonus, never a reason to fail
        return {"error": str(e)[:200]}
    if not fieldset:
        return {"error": "CalculiX wrote no harmonic field"}
    f_ccx = min(fieldset, key=lambda f: abs(f - fp))
    U, S = fieldset[f_ccx]
    vm = modal.vm_amplitude(S[:mesh.n_nodes])
    excl = h["excluded"]
    gj = governing(vm, excl)
    force = h["force"]
    sweep_err = 0.0
    fs = sorted(f for f in printed if sine.lo <= f <= sine.hi)
    if fs:
        q = modal.harmonic_coords(np.array(fs), modes, zeta, force)
        mine = np.abs(np.einsum("fm,md->fd", q, modes.shapes[:, gi].astype(float)))
        theirs = np.array([np.abs(printed[f][1][0] + 1j * printed[f][2][0]) for f in fs])
        scale = max(float(theirs.max()), 1e-30)
        sweep_err = float(np.max(np.abs(mine - theirs)) / scale)
    return {"f": float(f_ccx), "peak": float(vm[gj]), "node": gj, "xyz": mesh.X[gj].tolist(), "vm": vm,
            "at_wobbl_node": float(vm[gi]), "sweep_points": len(fs), "sweep_error": sweep_err}


def _rayleigh(exe, wl: Path, deck, modes, mesh, built) -> dict | None:
    """Rayleigh's quotient from the static sag under 1 g along the direction the first mode moves most mass."""
    eff = modes.effective_mass[:, :3]
    if not len(eff):
        return None
    i = 0 if eff[0].max() >= 0.05 * float(modes.total_mass[:3].mean()) else int(np.argmax(eff.max(1)))
    d = np.eye(3)[int(np.argmax(eff[i]))]
    text = deck.head + "*STEP\n*STATIC\n" + deck.supports + ccx.body_load(deck, modal.G0, d) + \
        "*NODE FILE\nU\n*END STEP\n"
    try:
        ccx.run(exe, wl, "sag", text)
        n_all = mesh.n_nodes + 2 * len(deck.extra)
        u = ccx.read_blocks(wl / "sag.frd", n_all, ("DISP",))["DISP"][-1][1][:, :3].astype(float)
        u = ccx.fill_rigid(u, mesh.X, deck)
        (wl / "sag.frd").unlink(missing_ok=True)
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)[:200]}
    density = built.model.density
    f = modal.rayleigh_frequency(mesh, density, u, d, modal.G0, modes.mass_nodes, modes.mass_values)
    return {"frequency": f, "direction": "xyz"[int(np.argmax(d))], "mode": i + 1,
            "mode_frequency": float(modes.freq[i]), "first": float(modes.freq[0]),
            "sag_mm": float(np.abs(u[:, int(np.argmax(d))]).max())}
