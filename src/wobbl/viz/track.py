"""A scene's motion: the physical event at the input value the question asked about, played on its own clock.

A scene's timeline is often an input the reader sweeps with a slider (a drop height, a heat load, a wind). What
happens at one value of it is an event in time: the part falls, hits and bounces, a rocket lifts off and flies. The
motion holds that event: its physical times, the playback second each one is shown at (so a 0.5 s fall can play in
real time and the millisecond of its impact slowed thousands of times), and per object tracks of what moves.

    s.motion(t, play, label="time", unit="s")                 # M frames of physical time and their playback seconds
    part.track(pos=P, quat=Q, field=S, deform=U)              # each M first: rigid pose, color field, elastic shape
    flame.track(pos=P, quat=Q, size=L, glow=g)                # a body grows and glows with thrust
    motor.track(pos=P, quat=Q, field=v)                       # a body colored by one value at each moment
    path.track(drawn=tau)                                     # a line traces itself by motion time
    s.readout("peak stress", per_input, motion=per_motion_frame)

Apart from that event, what keeps moving at whatever value the slider stands at runs on the scene's steady clock,
real time shown some times slower, continuously and at once when the slider moves:

    s.steady(slow, note=...)                                  # one number, or one per timeline frame
    rotor.spin(rpm / 60, axis=(0, 0, 1))                      # turns per second at each timeline frame
    part.swing(f_drive)                                       # its deformation at the slider, times a cosine
    link.cycle(P, Q, rate)                                    # one cycle of poses, at cycles per second

A field or deformation over many frames is stored as a low rank sum, a few shapes per vertex and a weight for each
frame (a truncated SVD to a tenth of a percent of its largest value), so a millisecond of a solved impact on
thousands of vertices stays tens of kilobytes. The viewer adds the shapes back at the moment shown. The rank and
the largest error of that sum go in the header, so the stored motion can be checked against the solve.
"""
from __future__ import annotations

import numpy as np

from .objects import _Enc, _f, _fl, _quats

TOL = 1e-3          # largest error of a low rank field or shape, against its largest value
MAX_RANK = 16


def lowrank(A: np.ndarray, tol: float = TOL, max_rank: int = MAX_RANK) -> tuple[np.ndarray, np.ndarray, float]:
    """A (frames x n) as coef (frames x r) @ basis (r x n), with each basis row scaled to a largest value of 1 and
    r the fewest rows that keep every entry within tol of A's largest value. NaN entries (no value there) are
    left out of the fit and come back as NaN. Returns coef, basis and the largest error over the largest value."""
    A = np.asarray(A, float)
    none = ~np.isfinite(A).all(0)
    B = np.where(np.isfinite(A), A, 0.0)
    top = float(np.abs(B).max()) or 1.0
    U, s, Vt = np.linalg.svd(B, full_matrices=False)
    r, err = 1, np.inf
    for r in range(1, min(max_rank, len(s)) + 1):
        err = float(np.abs(B - (U[:, :r] * s[:r]) @ Vt[:r]).max()) / top
        if err <= tol:
            break
    coef, basis = U[:, :r] * s[:r], Vt[:r].copy()
    big = np.maximum(np.abs(basis).max(1), 1e-30)
    basis /= big[:, None]
    coef *= big[None, :]
    basis[:, none] = np.nan
    return coef, basis, err


def _rank_desc(enc: _Enc, A: np.ndarray, idx, c: int) -> dict:
    """A frames x vertices (x c) array kept on the vertices idx, as its low rank sum."""
    M = A.shape[0]
    flat = A[:, idx].reshape(M, -1)
    coef, basis, err = lowrank(flat)
    r = len(basis)
    return {"coef": enc.floats(coef), "basis": enc.floats(basis.reshape(r, len(idx), c)), "rank": r,
            "err": _f(err, 3)}


class Track:
    """What one object does over the motion's frames. Arrays have the motion's frame count first."""

    def __init__(self, obj):
        self.obj = obj
        self.pos = self.quat = self.field = self.deform = self.glow = self.size = self.drawn = self.value = None
        self.when = None

    def check(self, a, name: str, inner: int | None = None) -> np.ndarray:
        a = np.asarray(a, float)
        M = self.obj.s.mo_n
        if M == 0:
            raise ValueError("call scene.motion(t, play) before giving an object a track")
        if a.shape[0] != M:
            raise ValueError(f"{self.obj.kind} {self.obj.name!r} track {name}: {a.shape[0]} frames given, the "
                             f"motion has {M}")
        return a

    def encode(self, enc: _Enc, idx=None) -> dict:
        d: dict = {}
        if self.pos is not None:
            d["pos"] = enc.floats(self.pos)
        if self.quat is not None:
            d["quat"] = enc.floats(_quats(self.quat.copy()))
        if self.size is not None:
            d["size"] = enc.floats(self.size)
        if self.glow is not None:
            d["glow"] = _fl(self.glow, 3)
        if self.field is not None:
            f, v = self.field
            d["field"] = {"f": f, **_rank_desc(enc, v, idx, 1)}
        if self.value is not None:
            f, v = self.value
            d["value"] = {"f": f, "values": _fl(v, 5)}
        if self.deform is not None:
            U, ex = self.deform
            d["deform"] = {**_rank_desc(enc, U, idx, 3), "exaggerate": ex}
        if self.drawn is not None:
            d["drawn"] = enc.floats(self.drawn)
        if self.when is not None:
            d["when"] = [None if v is None else _f(v, 9) for v in self.when]
        return d


class Steady:
    """What one object keeps doing at the slider's value on the scene's steady clock (Scene.steady): a spin about an
    axis, a cycle of rigid poses or a swing through its shape, at a rate in turns or cycles per second of real
    time, one number or one per timeline frame. The viewer reads the rate at the slider's value and turns each
    object's phase on by that rate over the clock's slowed time, so moving the slider changes the motion at once
    and nothing jumps."""

    def __init__(self, obj, rate, what: str, **how):
        r = np.asarray(rate, float).ravel()
        n = obj.s.n
        if len(r) not in (1, n):
            raise ValueError(f"{obj.kind} {obj.name!r} {what}: {len(r)} rates for {n} timeline frames")
        if not np.isfinite(r).all():
            raise ValueError(f"{obj.kind} {obj.name!r} {what}: every rate must be a number")
        self.obj, self.rate, self.what, self.how = obj, r, what, how

    def encode(self, enc: _Enc, keep) -> dict:
        r = self.rate if len(self.rate) == 1 else self.rate[keep]
        d: dict = {"rate": _fl(r, 6)}
        if self.what == "spin":
            ax = np.asarray(self.how["axis"], float)
            d["spin"] = {"axis": _fl(ax / max(float(np.linalg.norm(ax)), 1e-30)), "center": _fl(self.how["center"])}
        elif self.what == "cycle":
            d["cycle"] = {"pos": enc.floats(self.how["pos"]), "quat": enc.floats(_quats(self.how["quat"].copy()))}
        else:
            d["swing"] = True
        return d


def steady_header(s, keep) -> dict | None:
    """The steady clock's header: how many times slower than real time it runs (one number or one per timeline
    frame) and what the view says about it. Checks that every object that moves on it has the clock."""
    moving = [o for o in s.objects if getattr(o, "std", None) is not None]
    if not s.st:
        if moving:
            raise ValueError(f"{moving[0].kind} {moving[0].name!r} moves on the steady clock: call "
                             "scene.steady(slow) first")
        return None
    d = dict(s.st)
    sl = np.asarray(d.pop("slow"), float)
    d["slow"] = _fl(sl if len(sl) == 1 else sl[keep], 6)
    return d


def motion_header(s) -> dict | None:
    if not s.mo_n:
        return None
    m = dict(s.mo)
    m["frames"] = s.mo_n
    return m


__all__ = ["Steady", "Track", "lowrank", "motion_header", "steady_header"]
