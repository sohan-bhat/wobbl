"""The scene: one compact file that holds a run's 3D simulation, written through one small API by every domain.

    from wobbl.viz import Scene

    s = Scene("Shelf bracket under its shelf", domain="strength", length="mm")
    s.timeline(load_n, unit="N", label="load", seconds=6, between="exact")
    part = s.mesh(X, tris, keep=[peak_node])
    part.color_by("von Mises", vm, unit="MPa", scale=ramp, limit=yield_mpa, limit_label="yield")
    part.deform(U, scale=ramp, exaggerate=40)
    s.readout("load", load_n, unit="N")
    s.event(36.8, "first yield", "at the inside corner", kind="fail")
    s.verdict("FAILS", "The inside corner yields at 92% of the shelf's weight.", tone="fail")
    s.save(out_dir)                     # out_dir/scene.wobbl

File layout, gzipped as a whole: the four bytes WBSC, a little endian uint32 version, a uint32 header length, the
header as UTF-8 JSON, zero padding to a multiple of 8, then the binary buffers, each starting on a multiple of 4.
The header lists the buffers as {"o": offset, "n": bytes, "t": type} with types u8, u16, u32 and f32. A float
array is stored as unsigned integers on a per component range, value = lo + q / M * (hi - lo) with M = 255 or
65535, and when "nan" is set the top code M means no value. Its descriptor reads {"b": buffer, "c": components,
"shape": [...], "lo": [...], "hi": [...]}. Integer arrays such as triangles read {"b", "shape"}.

Every per frame array has the timeline's frame count as its first axis. Times, never frame numbers, place labels,
events and line runs, so thinning frames to fit the budget leaves them where they were. The API decimates meshes
and thins frames itself until the gzipped file fits BUDGET, so no domain can blow it.
"""
from __future__ import annotations

import gzip
import json
import struct
from pathlib import Path

import numpy as np

from .track import motion_header
from .objects import DT, Arrows, Atoms, Body, Ground, Lines, Mesh, Points, _Enc, _f, _fl, _Obj

MAGIC = b"WBSC"
VERSION = 1
FILE = "scene.wobbl"
BUDGET = 1_000_000                 # bytes on disk
MAX_FRAMES = 240
TONES = ("pass", "warn", "fail", "info")


class Scene:
    """A run's 3D simulation, built up object by object and saved as one small file. See the module docstring."""

    def __init__(self, title: str, *, domain: str = "", question: str = "", length: str = "mm", up: str = "z",
                 source: str = ""):
        self.head = {"title": title, "domain": domain, "question": question, "units": {"length": length},
                     "up": up, "source": source}
        self.n = 1
        self.t = np.zeros(1)
        self.tl = {"unit": "", "label": "", "seconds": 6.0, "between": "step", "loop": True}
        self.fields: list[dict] = []
        self._fdata: list[list] = []
        self.objects: list[_Obj] = []
        self.readouts: list[dict] = []
        self.events: list[dict] = []
        self.labels: list[dict] = []
        self.notes: list[str] = []
        self.cam: dict = {}
        self.result: dict | None = None
        self.mag: dict | None = None
        self.mo: dict = {}
        self.mo_n = 0

    # time
    def timeline(self, t, *, unit: str = "s", label: str = "time", seconds: float = 8.0,
                 between: str = "interpolated", loop: bool = True, note: str = "", kind: str = "time",
                 design=None) -> "Scene":
        """The saved frames' times (or loads, or cycles). between says what the viewer may show between two saved
        frames: interpolated (smooth, and labeled as interpolated), exact (in between is exact, as for a linear
        model scaled by load) or step (each saved frame held until the next).

        kind "input" makes the frames a sweep over something the user controls, such as a load, a supply voltage,
        a payload or a wind speed. The viewer then shows a slider labeled with it instead of play controls, never
        plays on its own, and opens at design, the value the question asked about."""
        if between not in ("interpolated", "exact", "step"):
            raise ValueError("between is interpolated, exact or step")
        if kind not in ("time", "input"):
            raise ValueError("kind is time or input")
        self.t = np.asarray(t, float).ravel()
        self.n = len(self.t)
        self.tl = {"unit": unit, "label": label, "seconds": float(seconds), "between": between, "loop": loop,
                   "kind": kind}
        if design is not None:
            self.tl["design"] = float(design)
        if note:
            self.tl["note"] = note
        return self

    def motion(self, t, play, *, label: str = "time", unit: str = "s", note: str = "") -> "Scene":
        """The physical event at the input value the question asked about (the timeline's design value), played on
        its own clock: t holds each motion frame's physical time and play the playback second it is shown at, rising
        from 0, so a slow motion stretch is one where play moves faster than t. Objects follow it through their
        track() and readouts through readout(..., motion=). See track.py."""
        t, play = np.asarray(t, float).ravel(), np.asarray(play, float).ravel()
        if len(t) != len(play) or len(t) < 2:
            raise ValueError("motion: t and play need the same length, at least 2")
        if np.any(np.diff(t) <= 0) or np.any(np.diff(play) <= 0) or play[0] != 0:
            raise ValueError("motion: t and play must rise frame to frame, and play must start at 0")
        self.mo_n = len(t)
        self.mo = {"t": _fl(t, 9), "play": _fl(play, 6), "label": label, "unit": unit}
        if note:
            self.mo["note"] = note
        return self

    def field(self, name: str, unit: str = "", *, colormap: str = "turbo", range=None, limit=None,
              limit_label: str = "", digits: int = 3, data=None, target=None) -> int:
        """Declare a color field (or find one already declared by name) and return its index. Over limit the part
        shows white and the viewer's verdict reads fails; over target (0.8 of limit when not given) it reads
        marginal, as a target safety factor would make it."""
        for i, f in enumerate(self.fields):
            if f["name"] == name:
                if data is not None:
                    self._fdata[i].append(np.ravel(np.asarray(data, float)))
                return i
        f = {"name": name, "unit": unit, "colormap": colormap, "digits": digits}
        if range is not None:
            f["range"] = _fl(range)
        if limit is not None:
            f["limit"] = _f(limit)
            if limit_label:
                f["limit_label"] = limit_label
            if target is not None:
                f["target"] = _f(target)
        self.fields.append(f)
        self._fdata.append([] if data is None else [np.ravel(np.asarray(data, float))])
        return len(self.fields) - 1

    # objects
    def mesh(self, X, tris, *, name: str = "", color="#b9bdc4", opacity: float = 1.0, prop: bool = False,
             max_vertices: int = 12000, keep=(), smooth: bool = True, edges: bool = False) -> Mesh:
        """A triangle mesh (vertices x 3, triangles x 3, or six node triangles, which are drawn as four)."""
        m = Mesh(self, X, tris, name, color, opacity, prop, max_vertices, keep, smooth, edges)
        self.objects.append(m)
        return m

    def body(self, shape: str, size, *, pos=(0, 0, 0), quat=(1, 0, 0, 0), color="#9aa0a6", name: str = "",
             prop: bool = False, opacity: float = 1.0) -> Body:
        b = Body(self, shape, size, pos, quat, color, name, prop, opacity)
        self.objects.append(b)
        return b

    def atoms(self, elements, positions, *, bonds=None, cell: dict | None = None, name: str = "",
              radius: float = 0.35) -> Atoms:
        """Atoms by element symbol at positions (atoms x 3, or frames x atoms x 3) with bonds as index pairs. A
        crystal passes cell={"origin": o, "a": a, "b": b, "c": c} to draw its unit cell."""
        a = Atoms(self, elements, positions, bonds, cell, name, radius)
        self.objects.append(a)
        return a

    def points(self, positions, *, values=None, field: str = "", unit: str = "", colormap: str = "viridis",
               range=None, color="#ffd400", size: float = 1.0, name: str = "") -> Points:
        """Particles (points x 3, or frames x points x 3), colored by a field when values are given."""
        f = None
        if values is not None:
            f = self.field(field, unit, colormap=colormap, range=range, data=values)
        p = Points(self, positions, values, f, color, size, name)
        self.objects.append(p)
        return p

    def lines(self, paths, *, color="#ffd400", width: float = 2.0, values=None, field: str = "", unit: str = "",
              colormap: str = "viridis", range=None, drawn=None, at=None, fade: bool = False, flow=None,
              spacing: float | None = None, name: str = "", prop: bool = False) -> Lines:
        """Polylines given as a list of (points x 3) arrays. drawn gives each point the time it is drawn by, so a
        trajectory traces itself. at gives each line the time it appears, as crack fronts do, and fade dims the
        older ones. flow moves a glow along each line, as current along a trace: one speed for all, one per line,
        or frames x lines, in scene units per second of playback; spacing is the distance between glows."""
        f = None
        if values is not None:
            f = self.field(field, unit, colormap=colormap, range=range,
                           data=np.concatenate([np.ravel(v) for v in values]))
        ln = Lines(self, paths, color, width, values, f, drawn, at, flow, spacing, name, prop, fade)
        self.objects.append(ln)
        return ln

    def arrows(self, origins, vectors, *, scale=None, color="#ffd400", radius: float | None = None, labels=None,
               name: str = "", prop: bool = True) -> Arrows:
        """Arrows from origins along vectors (both fixed, or frames x arrows x 3); scale is a multiple per frame.
        radius is the shaft's in scene units. Labels may name readouts in braces, as "{load}"."""
        a = Arrows(self, origins, vectors, None if scale is None else np.asarray(scale, float), color, radius or 0,
                   labels, name, prop)
        self.objects.append(a)
        return a

    def ground(self, z: float = 0.0, *, size: float, grid: float | None = None, color="#1a1c20",
               center=(0.0, 0.0)) -> Ground:
        g = Ground(self, z, size, grid, color, center)
        self.objects.append(g)
        return g

    # annotations
    def label(self, text: str, at, *, t0=None, t1=None, kind: str = "note", follow: _Obj | None = None) -> "Scene":
        """Text pinned to a 3D point, shown from t0 to t1. Braces name a readout whose value fills in live. With
        follow, the point is relative to that body and moves with it."""
        d = {"text": text, "at": _fl(at), "kind": kind}
        if t0 is not None or t1 is not None:
            d["when"] = [None if t0 is None else _f(t0), None if t1 is None else _f(t1)]
        if follow is not None:
            d["follow"] = self.objects.index(follow)
        self.labels.append(d)
        return self

    def readout(self, name: str, values, *, unit: str = "", digits: int = 3, limit=None, target=None,
                motion=None) -> "Scene":
        """A named number per frame that ticks with playback, such as load, peak stress or chamber pressure. Above
        limit (one number, or one per frame such as a motor's torque at its speed) it shows in red. Above target
        (0.8 of limit when not given, and it may come without a limit) the viewer calls the input marginal."""
        v = np.asarray(values, float).ravel()
        if len(v) != self.n:
            raise ValueError(f"readout {name!r}: {len(v)} values for {self.n} frames")
        d = {"name": name, "unit": unit, "digits": digits, "v": v}
        if limit is not None and np.ndim(limit):
            d["lim"] = np.asarray(limit, float).ravel()
        elif limit is not None:
            d["limit"] = _f(limit)
        if target is not None:
            d["target"] = _f(target)
        if motion is not None:
            mv = np.asarray(motion, float).ravel()
            if len(mv) != self.mo_n:
                raise ValueError(f"readout {name!r}: {len(mv)} motion values for {self.mo_n} motion frames")
            d["mv"] = mv
        self.readouts.append(d)
        return self

    def readout_motion(self, name: str, values) -> "Scene":
        """Give a readout already made its value at every motion frame, as readout(..., motion=) would."""
        mv = np.asarray(values, float).ravel()
        if len(mv) != self.mo_n:
            raise ValueError(f"readout {name!r}: {len(mv)} motion values for {self.mo_n} motion frames")
        for r in self.readouts:
            if r["name"] == name:
                r["mv"] = mv
                return self
        raise ValueError(f"no readout named {name!r}")

    def motion_readout(self, name: str, values, *, unit: str = "", digits: int = 3) -> "Scene":
        """A number at every motion frame that only the motion has, such as a rocket's altitude at each moment of
        its flight. A view says these beside the motion while it plays, in place of the readings along the input,
        which come back once it has played. NaN where it has no value."""
        mv = np.asarray(values, float).ravel()
        if not self.mo_n:
            raise ValueError("call scene.motion(t, play) before giving it a readout")
        if len(mv) != self.mo_n:
            raise ValueError(f"motion readout {name!r}: {len(mv)} values for {self.mo_n} motion frames")
        self.mo.setdefault("readouts", []).append({"name": name, "unit": unit, "digits": int(digits),
                                                   "values": _fl(mv, 5)})
        return self

    def event(self, t: float, name: str, detail: str = "", kind: str = "info") -> "Scene":
        """Something that happens at a time: burnout, first yield, impact. kind fail marks a failure."""
        self.events.append({"t": _f(t), "name": name, "detail": detail, "kind": kind})
        return self

    def note(self, text: str) -> "Scene":
        if text not in self.notes:
            self.notes.append(text)
        return self

    def verdict(self, word: str, line: str, tone: str = "info") -> "Scene":
        """The result in one plain line, shown when playback reaches the end."""
        if tone not in TONES:
            tone = "info"
        self.result = {"word": word, "line": line, "tone": tone}
        return self

    def magnify(self, objects, factor: float, *, label: str = "", anchor=None, pivot=None) -> "Scene":
        """Draw small objects factor times larger so they show at the scene's scale, as a rocket on a flight kilometres
        long. Each object grows about its own pose, or about anchor when given, which keeps a group such as a pad
        and its rail together. While the scene's motion plays, a group with a pivot (a point in each object's own
        frame) grows about that point instead, so a rocket grown from its nozzle stands on its pad and flies its
        path tail first. A view says so and lets the reader see them at their true size. The first call names the
        toggle and its factor; later groups may grow by less, as a pad beside a rocket."""
        if not self.mag:
            self.mag = {"factor": _f(factor), "label": label or "Drawn", "groups": []}
        self.mag["groups"].append({"objects": [self.objects.index(o) for o in objects], "factor": _f(factor),
                                   "anchor": None if anchor is None else _fl(anchor),
                                   **({} if pivot is None else {"pivot": _fl(pivot)})})
        names = label or ", ".join(o.name for o in objects if o.name)[:40] or "small parts"
        self.note(f"{names[0].upper() + names[1:]} drawn {factor:g} times larger so {'it shows' if len(objects) == 1 or label else 'they show'}")
        return self

    def camera(self, *, target=None, direction=None, distance=None, fov: float = 30.0, follow: _Obj | None = None,
               presets: list[dict] | None = None) -> "Scene":
        """Where the view starts: looking at target from direction (pointing from the target to the eye) at
        distance. follow keeps a moving body in the middle. presets are extra named views, each a dict of the same
        keys, which the viewer offers as buttons."""
        c: dict = {"fov": fov}
        if target is not None:
            c["target"] = _fl(target)
        if direction is not None:
            d = np.asarray(direction, float)
            c["dir"] = _fl(d / np.linalg.norm(d))
        if distance is not None:
            c["dist"] = _f(distance)
        if follow is not None:
            c["follow"] = self.objects.index(follow)
        if presets:
            c["presets"] = [{k: (self.objects.index(v) if k == "follow" else
                                 _fl(np.asarray(v) / np.linalg.norm(v)) if k == "dir" else
                                 _fl(v) if k == "target" else v) for k, v in p.items()} for p in presets]
        self.cam = c
        return self

    # output
    def _header(self, keep, vfac) -> tuple[dict, bytes]:
        enc = _Enc()
        objs = [o.encode(enc, keep, vfac) for o in self.objects]
        fields = []
        for f, data in zip(self.fields, self._fdata):
            f = dict(f)
            if "range" not in f and data:
                v = np.concatenate(data)
                v = v[np.isfinite(v)]
                f["range"] = _fl([v.min(), v.max()] if len(v) else [0, 1])
            fields.append(f)
        h = {"format": "wobbl-scene", "version": VERSION, **self.head,
             "timeline": {"frames": int(len(keep)), "t": _fl(self.t[keep], 7), **self.tl},
             "camera": self.cam, "fields": fields, "objects": objs,
             "readouts": [{**{k: v for k, v in r.items() if k not in ("v", "lim", "mv")}, "values": _fl(r["v"][keep], 5),
                           **({"limits": _fl(r["lim"][keep], 5)} if "lim" in r else {}),
                           **({"motion": _fl(r["mv"], 5)} if "mv" in r else {})} for r in self.readouts],
             "events": sorted(self.events, key=lambda e: e["t"] if e["t"] is not None else 0),
             "labels": self.labels, "notes": self.notes, "buffers": enc.bufs}
        if self.result:
            h["verdict"] = self.result
        if self.mag:
            h["magnify"] = self.mag
        if self.mo_n:
            h["motion"] = motion_header(self)
        if any(o.prop for o in self.objects):
            h["props_note"] = "Props in gray stripes show the setup and are not part of the analysis"
        return h, enc.blob()

    def to_bytes(self, budget: int = BUDGET) -> bytes:
        """The gzipped file, with frames thinned and meshes decimated as far as it takes to fit the budget."""
        n = self.n
        frames, vfac = min(n, MAX_FRAMES), 1.0
        for _ in range(12):
            keep = np.unique(np.rint(np.linspace(0, n - 1, frames)).astype(int))
            data = pack(*self._header(keep, vfac))
            if len(data) <= budget:
                return data
            # most of a large file is frames when there are many; otherwise it is mesh
            over = budget / len(data)
            if frames > 16:
                frames = max(16, int(frames * over * 0.9))
            else:
                vfac *= over * 0.8
        raise ValueError(f"the scene takes {len(data) / 1e3:.0f} kB even thinned, over the {budget / 1e3:.0f} kB "
                         "budget")

    def save(self, out_dir: str | Path, budget: int = BUDGET) -> Path:
        """Write out_dir/scene.wobbl and return its path."""
        path = Path(out_dir) / FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self.to_bytes(budget))
        return path


def pack(header: dict, blob: bytes) -> bytes:
    h = json.dumps(header, separators=(",", ":"), allow_nan=False).encode()
    head = MAGIC + struct.pack("<II", VERSION, len(h)) + h
    head += b"\0" * ((-len(head)) % 8)
    return gzip.compress(head + blob, compresslevel=9, mtime=0)


def load(src) -> tuple[dict, bytes]:
    """The header and binary part of a scene file (a path or the gzipped bytes)."""
    raw = gzip.decompress(Path(src).read_bytes() if not isinstance(src, (bytes, bytearray)) else bytes(src))
    if raw[:4] != MAGIC:
        raise ValueError("not a Wobbl scene")
    _, n = struct.unpack("<II", raw[4:12])
    header = json.loads(raw[12:12 + n])
    start = 12 + n + ((-(12 + n)) % 8)
    return header, raw[start:]


def array(header: dict, blob: bytes, d: dict) -> np.ndarray:
    """Decode one descriptor back to values, as the viewer does."""
    b = header["buffers"][d["b"]]
    q = np.frombuffer(blob, dtype=DT[b["t"]], count=b["n"] // np.dtype(DT[b["t"]]).itemsize, offset=b["o"])
    q = q.astype(float).reshape(d["shape"])
    if "lo" not in d:
        return q.astype(np.int64)
    M = 255 if b["t"] == "u8" else 65535
    top = M - 1 if d.get("nan") else M
    c = d["c"]
    lo, hi = np.asarray(d["lo"]), np.asarray(d["hi"])
    flat = q.reshape(-1, c)
    out = lo + flat / top * (hi - lo)
    if d.get("nan"):
        out[flat == M] = np.nan
    return out.reshape(d["shape"])
