"""The object kinds a scene holds and the encoder that packs their arrays: meshes, rigid bodies, atoms, particles,
lines, arrows and the ground. Scene in scene.py creates them; each encodes itself for the frames kept.
"""
from __future__ import annotations

import numpy as np

from .mesh import decimate, split_quadratic

DT = {"u8": np.uint8, "u16": "<u2", "u32": "<u4", "f32": "<f4"}
SHAPES = ("sphere", "box", "cylinder", "capsule", "cone", "plane", "hex", "ellipsoid", "torus")


def _f(x, sig: int = 7):
    """A plain float rounded to sig significant digits, None when it is not finite."""
    x = float(x)
    return float(f"{x:.{sig}g}") if np.isfinite(x) else None


def _fl(a, sig: int = 7) -> list:
    return [_f(v, sig) for v in np.asarray(a, float).ravel()]


def _color(c) -> str:
    if isinstance(c, str):
        return c
    r = np.clip(np.asarray(c, float)[:3], 0, 1)
    return "#" + "".join(f"{int(round(v * 255)):02x}" for v in r)


class _Enc:
    """Collects binary buffers and the descriptors the header points at."""

    def __init__(self):
        self.parts: list[bytes] = []
        self.bufs: list[dict] = []
        self.size = 0

    def raw(self, a, t: str) -> int:
        b = np.ascontiguousarray(a, dtype=DT[t]).tobytes()
        pad = (-self.size) % 4
        if pad:
            self.parts.append(b"\0" * pad)
            self.size += pad
        self.bufs.append({"o": self.size, "n": len(b), "t": t})
        self.parts.append(b)
        self.size += len(b)
        return len(self.bufs) - 1

    def floats(self, a, bits: int = 16) -> dict:
        a = np.asarray(a, float)
        c = a.shape[-1] if a.ndim > 1 else 1
        flat = a.reshape(-1, c)
        ok = np.isfinite(flat)
        nan = not ok.all()
        lo = np.array([_f(np.min(flat[ok[:, k], k])) if ok[:, k].any() else 0.0 for k in range(c)])
        hi = np.array([_f(np.max(flat[ok[:, k], k])) if ok[:, k].any() else 0.0 for k in range(c)])
        M = (1 << bits) - 1
        top = M - 1 if nan else M
        span = np.where(hi > lo, hi - lo, 1.0)
        q = np.rint((np.where(ok, flat, lo) - lo) / span * top).clip(0, top)
        if nan:
            q[~ok] = M
        d = {"b": self.raw(q, "u8" if bits == 8 else "u16"), "c": c, "shape": list(a.shape),
             "lo": lo.tolist(), "hi": hi.tolist()}
        if nan:
            d["nan"] = True
        return d

    def ints(self, a) -> dict:
        a = np.asarray(a, np.int64)
        t = "u16" if a.size == 0 or a.max() < 65535 else "u32"
        return {"b": self.raw(a, t), "shape": list(a.shape)}

    def blob(self) -> bytes:
        return b"".join(self.parts)


class _Obj:
    kind = ""

    def __init__(self, scene: "Scene", name: str, prop: bool):
        self.s, self.name, self.prop = scene, name, prop
        self.when: tuple | None = None
        self.guide = False

    def as_guide(self) -> "_Obj":
        """Mark the object as a guide drawn only to carry text labels, such as a height scale's pole and ticks. A
        view that shows no labels leaves guides out."""
        self.guide = True
        return self

    def show(self, t0=None, t1=None) -> "_Obj":
        """Show the object only between two times on the timeline (either end may be open)."""
        self.when = (t0, t1)
        return self

    def base(self) -> dict:
        d = {"kind": self.kind}
        if self.name:
            d["name"] = self.name
        if self.prop:
            d["prop"] = True
        if self.guide:
            d["guide"] = True
        if self.when:
            d["when"] = [None if v is None else _f(v) for v in self.when]
        return d

    def _track(self):
        if getattr(self, "trk", None) is None:
            from .track import Track
            self.trk = Track(self)
        return self.trk

    def frames(self, a, keep, inner=1):
        """A per frame array checked against the timeline, thinned to the frames kept."""
        a = np.asarray(a, float)
        if a.shape[0] != self.s.n:
            raise ValueError(f"{self.kind} {self.name!r}: {a.shape[0]} frames given, the timeline has {self.s.n}")
        return a[keep]


class Mesh(_Obj):
    """A triangle mesh: optionally colored by a field, deformed, moved as a rigid body and revealed over time."""
    kind = "mesh"

    def __init__(self, scene, X, tris, name, color, opacity, prop, max_vertices, keep, smooth, edges):
        super().__init__(scene, name, prop)
        self.X = np.asarray(X, float)
        self.T = split_quadratic(np.asarray(tris, np.int64))
        self.color, self.opacity, self.max_vertices, self.keep = _color(color), opacity, max_vertices, keep
        self.smooth, self.edges = smooth, edges
        self.fld = self.dfm = self.pose = self.rev = None
        self.glow = None

    def color_by(self, field: str, values, *, scale=None, unit: str = "", colormap: str = "turbo", range=None,
                 limit=None, limit_label: str = "", digits: int = 3, target=None) -> "Mesh":
        """Color by a field. values is one value per vertex, either for every frame (shape frames x vertices), or
        once with scale giving each frame's multiple of it, which is exact for a linear model."""
        v = np.asarray(values, float)
        f = self.s.field(field, unit, colormap=colormap, range=range, limit=limit, limit_label=limit_label,
                         digits=digits, data=v if scale is None else np.outer(np.asarray(scale, float), v),
                         target=target)
        self.fld = (f, v, None if scale is None else np.asarray(scale, float))
        return self

    def deform(self, U, *, scale=None, exaggerate: float = 1.0) -> "Mesh":
        """Displacement per vertex, for every frame (frames x vertices x 3) or once with a scale per frame. The
        viewer draws it exaggerate times larger and says so."""
        self.dfm = (np.asarray(U, float), None if scale is None else np.asarray(scale, float), float(exaggerate))
        if exaggerate != 1:
            self.s.note(f"Deformation drawn {exaggerate:.3g} times larger so it shows")
        return self

    def move(self, pos, quat=(1, 0, 0, 0)) -> "Mesh":
        """A rigid pose: position and quaternion (w, x, y, z), fixed or one per frame."""
        self.pose = (np.asarray(pos, float), np.asarray(quat, float))
        return self

    def track(self, *, pos=None, quat=None, field=None, deform=None, glow=None, when=None) -> "Mesh":
        """What the mesh does over the scene's motion (Scene.motion), each array with the motion's frames first:
        a rigid pose (pos, quat), the field it is colored by at every vertex (frames x vertices, NaN where it has
        none) and its elastic shape (frames x vertices x 3), drawn as much larger as its deform() says."""
        t = self._track()
        if pos is not None:
            t.pos = t.check(pos, "pos").reshape(-1, 3)
            t.quat = t.check((1, 0, 0, 0) * np.ones((len(t.pos), 1)) if quat is None else quat, "quat").reshape(-1, 4)
        if field is not None:
            if not self.fld:
                raise ValueError(f"mesh {self.name!r}: color_by() first, so the track's field has its scale")
            t.field = (self.fld[0], t.check(field, "field"))
        if deform is not None:
            t.deform = (t.check(deform, "deform"), self.dfm[2] if self.dfm else 1.0)
        if glow is not None:
            t.glow = t.check(glow, "glow")
        if when is not None:
            t.when = tuple(when)
        return self

    def reveal(self, values, thresholds) -> "Mesh":
        """Draw only where a per vertex value is at most the frame's threshold, as a print rising layer by layer
        (values are heights) or a mold filling (values are fill times)."""
        self.rev = (np.asarray(values, float), np.asarray(thresholds, float))
        return self

    def encode(self, enc: _Enc, keep, vfac: float) -> dict:
        idx, T = decimate(self.X, self.T, max(200, int(self.max_vertices * vfac)), keep=self.keep)
        d = self.base()
        d.update(pos=enc.floats(self.X[idx]), tri=enc.ints(T), color=self.color)
        if self.opacity < 1:
            d["opacity"] = self.opacity
        if not self.smooth:
            d["flat"] = True
        if self.edges:
            d["edges"] = True
        if self.fld:
            f, v, sc = self.fld
            if sc is None and v.ndim == 2:
                d["field"] = {"f": f, "frames": enc.floats(self.frames(v, keep)[:, idx], 8)}
            elif sc is None:
                d["field"] = {"f": f, "values": enc.floats(v[idx])}
            else:
                d["field"] = {"f": f, "values": enc.floats(v[idx]), "scale": _fl(self.frames(sc, keep), 6)}
        if self.dfm:
            U, sc, ex = self.dfm
            if sc is None:
                d["deform"] = {"frames": enc.floats(self.frames(U, keep)[:, idx]), "exaggerate": ex}
            else:
                d["deform"] = {"values": enc.floats(U[idx]), "scale": _fl(self.frames(sc, keep), 6), "exaggerate": ex}
        if self.pose is not None:
            d["pose"] = _pose(self, enc, keep, *self.pose)
        if self.rev:
            v, th = self.rev
            d["reveal"] = {"values": enc.floats(v[idx]), "at": _fl(self.frames(th, keep))}
        if self.glow is not None:
            d["glow"] = _fl(self.frames(self.glow, keep), 3)
        if getattr(self, "trk", None) is not None:
            d["track"] = self.trk.encode(enc, idx)
        return d


def _quats(q: np.ndarray) -> np.ndarray:
    """Unit quaternions with each one on the same side as the one before, so blending between frames is short."""
    q = q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-30)
    if q.ndim == 2:
        for i in range(1, len(q)):
            if np.dot(q[i], q[i - 1]) < 0:
                q[i] = -q[i]
    return q


def _pose(obj: _Obj, enc: _Enc, keep, pos, quat) -> dict:
    out = {}
    out["pos"] = enc.floats(obj.frames(pos, keep)) if pos.ndim == 2 else _fl(pos)
    out["quat"] = enc.floats(_quats(obj.frames(quat, keep))) if quat.ndim == 2 else _fl(_quats(quat), 6)
    return out


class Body(_Obj):
    """A rigid primitive. Sizes follow MuJoCo: sphere [r], box [half x, half y, half z], cylinder, capsule, cone and
    hex [r, half length] along their own z, plane [half x, half y], ellipsoid [rx, ry, rz], torus [R, r]."""
    kind = "body"

    def __init__(self, scene, shape, size, pos, quat, color, name, prop, opacity):
        super().__init__(scene, name, prop)
        if shape not in SHAPES:
            raise ValueError(f"unknown shape {shape!r}; use one of {', '.join(SHAPES)}")
        self.shape, self.size, self.color, self.opacity = shape, _fl(size, 6), _color(color), opacity
        self.pos, self.quat = np.asarray(pos, float), np.asarray(quat, float)
        self.glow = self.fld = None

    def light(self, level) -> "Body":
        """How brightly it glows at each frame, 0 to 1, as an LED's light or a hot element's."""
        self.glow = np.asarray(level, float)
        return self

    def track(self, *, pos=None, quat=None, size=None, glow=None, field=None, when=None) -> "Body":
        """What the body does over the scene's motion: a pose, its size (frames x the size's numbers, as a flame
        that grows with thrust), its glow and the one value it is colored by (color_by() first, as a motor's
        torque against its limit at each moment), each with the motion's frames first."""
        t = self._track()
        if field is not None:
            if not self.fld:
                raise ValueError(f"body {self.name!r}: color_by() first, so the track's value has its scale")
            t.value = (self.fld[0], t.check(field, "field").ravel())
        if pos is not None:
            t.pos = t.check(pos, "pos").reshape(-1, 3)
            t.quat = t.check(np.tile(self.quat if self.quat.ndim == 1 else self.quat[0], (len(t.pos), 1))
                             if quat is None else quat, "quat").reshape(-1, 4)
        if size is not None:
            t.size = t.check(size, "size").reshape(t.obj.s.mo_n, -1)
        if glow is not None:
            t.glow = t.check(glow, "glow")
        if when is not None:
            t.when = tuple(when)
        return self

    def color_by(self, field: str, values, *, unit: str = "", colormap: str = "inferno", range=None, limit=None,
                 limit_label: str = "", digits: int = 3, target=None) -> "Body":
        """Color the whole body by one value per frame, as a lumped cell temperature."""
        v = np.asarray(values, float)
        f = self.s.field(field, unit, colormap=colormap, range=range, limit=limit, limit_label=limit_label,
                         digits=digits, data=v, target=target)
        self.fld = (f, v)
        return self

    def encode(self, enc, keep, vfac):
        d = self.base()
        d.update(shape=self.shape, size=self.size, color=self.color, **_pose(self, enc, keep, self.pos, self.quat))
        if self.opacity < 1:
            d["opacity"] = self.opacity
        if self.glow is not None:
            d["glow"] = _fl(self.frames(self.glow, keep), 3)
        if self.fld:
            d["field"] = {"f": self.fld[0], "values": _fl(self.frames(self.fld[1], keep), 5)}
        if getattr(self, "trk", None) is not None:
            d["track"] = self.trk.encode(enc)
        return d


class Atoms(_Obj):
    kind = "atoms"

    def __init__(self, scene, elements, positions, bonds, cell, name, scale):
        super().__init__(scene, name, False)
        self.el = [str(e) for e in elements]
        self.P = np.asarray(positions, float)
        self.bonds = None if bonds is None else np.asarray(bonds, np.int64).reshape(-1, 2)
        self.cell, self.scale = cell, scale

    def encode(self, enc, keep, vfac):
        d = self.base()
        P = self.frames(self.P, keep) if self.P.ndim == 3 else self.P
        d.update(el=self.el, pos=enc.floats(P), radius=self.scale)
        if self.bonds is not None and len(self.bonds):
            d["bonds"] = enc.ints(self.bonds)
        if self.cell is not None:
            d["cell"] = {k: _fl(v) for k, v in self.cell.items()}
        return d


class Points(_Obj):
    kind = "points"

    def __init__(self, scene, positions, values, field, color, size, name):
        super().__init__(scene, name, False)
        self.P, self.v, self.f = np.asarray(positions, float), values, field
        self.color, self.size = _color(color), float(size)

    def encode(self, enc, keep, vfac):
        d = self.base()
        every = max(1, int(np.ceil(1 / vfac)))          # a tight budget draws every few particles, never moves one
        P = (self.frames(self.P, keep) if self.P.ndim == 3 else self.P)[..., ::every, :]
        d.update(pos=enc.floats(P), color=self.color, size=_f(self.size))
        if self.v is not None:
            v = np.asarray(self.v, float)
            d["field"] = {"f": self.f, "frames" if v.ndim == 2 else "values":
                          enc.floats((self.frames(v, keep) if v.ndim == 2 else v)[..., ::every], 8)}
        return d


class Lines(_Obj):
    """Polylines: trajectories, crack fronts, field lines and streamlines, traces that carry current."""
    kind = "lines"

    def __init__(self, scene, paths, color, width, values, field, times, at, flow, spacing, name, prop, fade):
        super().__init__(scene, name, prop)
        self.paths = [np.asarray(p, float).reshape(-1, 3) for p in paths]
        self.color, self.width, self.values, self.f = _color(color), float(width), values, field
        self.times, self.at, self.flow, self.spacing, self.fade = times, at, flow, spacing, fade
        self.speed = self.scale = None
        self.offset = 0.0
        self.through = False

    def inside(self) -> "Lines":
        """The lines run inside a solid (a coolant's channel through a cold plate), so a view draws what moves along
        them over the solid that hides them, as a section would show it."""
        self.through = True
        return self

    def moving(self, speed, *, scale=None, base: float = 0.0) -> "Lines":
        """The fluid's own speed at every point of every line (a list of arrays like the paths, in scene units
        per second of real time), so the viewer can send tracers along each line as fast as the fluid goes there,
        slowed by a factor it says. scale (one per timeline frame) and base make the lines' values base + scale x
        values at each frame, as a fluid whose warming grows with the heat load."""
        sp = [np.asarray(v, float).ravel() for v in speed]
        if len(sp) != len(self.paths) or any(len(v) != len(p) for v, p in zip(sp, self.paths)):
            raise ValueError(f"lines {self.name!r}: one speed per point of every line")
        self.speed = sp
        if scale is not None:
            self.scale = np.asarray(scale, float).ravel()
            self.offset = float(base)
        return self

    def track(self, *, drawn=None, when=None) -> "Lines":
        """Each point's motion time, so the line traces itself as the motion plays (a flight path)."""
        if when is not None:
            self._track().when = tuple(when)
        if drawn is not None:
            t = self._track()
            t.drawn = np.concatenate([np.ravel(v) for v in drawn])
            if len(t.drawn) != sum(len(p) for p in self.paths):
                raise ValueError(f"lines {self.name!r}: one motion time per point")
        return self

    def encode(self, enc, keep, vfac):
        d = self.base()
        d.update(pos=enc.floats(np.concatenate(self.paths)), runs=[len(p) for p in self.paths], color=self.color,
                 width=self.width)
        if self.values is not None:
            d["field"] = {"f": self.f, "values": enc.floats(np.concatenate([np.ravel(v) for v in self.values]))}
            if self.scale is not None:
                d["field"].update(scale=_fl(self.frames(self.scale, keep), 6), base=_f(self.offset))
        if self.speed is not None:
            d["speed"] = enc.floats(np.concatenate(self.speed))
        if self.through:
            d["inside"] = True
        if getattr(self, "trk", None) is not None:
            d["track"] = self.trk.encode(enc)
        if self.times is not None:
            d["drawn"] = enc.floats(np.concatenate([np.ravel(v) for v in self.times]))
        if self.at is not None:
            d["at"] = _fl(self.at)
            if self.fade:
                d["fade"] = True
        if self.flow is not None:
            fl = np.asarray(self.flow, float)
            d["flow"] = {"spacing": _f(self.spacing or 1.0)}
            if fl.ndim == 0:
                d["flow"]["speed"] = [_f(fl)]
            elif fl.ndim == 1:
                d["flow"]["line_speed"] = _fl(fl, 4)
            else:
                d["flow"]["speed"] = [_fl(r, 4) for r in self.frames(fl, keep)]
        return d


class Arrows(_Obj):
    kind = "arrows"

    def __init__(self, scene, origins, vectors, scale, color, radius, labels, name, prop):
        super().__init__(scene, name, prop)
        self.o, self.v = np.asarray(origins, float), np.asarray(vectors, float)
        self.scale, self.color, self.radius, self.labels = scale, _color(color), radius, labels

    def encode(self, enc, keep, vfac):
        d = self.base()
        o = self.frames(self.o, keep) if self.o.ndim == 3 else self.o.reshape(-1, 3)
        v = self.frames(self.v, keep) if self.v.ndim == 3 else self.v.reshape(-1, 3)
        d.update(origin=enc.floats(o), vec=enc.floats(v), color=self.color, radius=_f(self.radius))
        if self.scale is not None:
            d["scale"] = _fl(self.frames(self.scale, keep), 5)
        if self.labels:
            d["labels"] = list(self.labels)
        return d


class Ground(_Obj):
    kind = "ground"

    def __init__(self, scene, z, size, grid, color, center):
        super().__init__(scene, "ground", True)
        self.z, self.size, self.grid, self.color, self.center = z, size, grid, _color(color), center

    def encode(self, enc, keep, vfac):
        d = self.base()
        d.update(z=_f(self.z), size=_f(self.size), color=self.color, center=_fl(self.center))
        if self.grid:
            d["grid"] = _f(self.grid)
        return d
