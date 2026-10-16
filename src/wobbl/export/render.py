"""Frames of a run drawn with Wobbl's own numpy rasterizer, the one the report images use (wobbl.report.render).

Each frame is rasterized at twice the final size with a depth buffer and averaged down, lit by the same three
camera lights as the report, with see through objects blended over the rest, the ground's grid drawn into the
ground itself, and lines and dots drawn where nothing in front hides them. A small caption in Wobbl's colors
carries the verdict, the key number and the live readouts.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from ..report.render import Camera, _edges, _interpolate, _lighting, _rasterize, _touching
from .sample import Posed, Scene, Surface, rgb

BG = "#0e0f11"
FG = "#f2f2ee"
ACCENT = "#ffd400"
DIM = "#9aa0a6"
SS = 2
FONT = Path(__file__).resolve().parents[1] / "viz" / "fonts" / "ibm-plex-mono-500-latin.woff2"


@dataclass
class View:
    target: np.ndarray
    direction: np.ndarray       # from the target toward the eye
    distance: float
    fov: float = 30.0


def start_view(sc: Scene, aspect: float) -> View:
    """The scene's own opening view, or one that takes in everything drawn at its first frame."""
    lo, hi = sc.bounds(0.0)
    c = sc.h.get("camera") or {}
    center = 0.5 * (lo + hi)
    radius = max(float(np.linalg.norm(hi - lo)) / 2, 1e-6)
    fov = float(c.get("fov") or 30.0)
    target = np.asarray(c.get("target") or center, float)
    up = sc.up
    default = (np.array([0.55, -0.75, 0.45]) if up[2] else np.array([0.55, 0.45, 0.75]))
    d = np.asarray(c.get("dir") or default, float)
    d /= np.linalg.norm(d)
    half = np.radians(fov / 2)
    fit = radius / np.sin(min(half, np.arctan(np.tan(half) * aspect))) * 1.08
    return View(target, d, float(c.get("dist") or fit), fov)


def orbit(v: View, up: np.ndarray, angle: float) -> View:
    """The view turned by angle degrees about the up axis through its target."""
    a = np.radians(angle)
    k = up / np.linalg.norm(up)
    d = v.direction
    d = d * np.cos(a) + np.cross(k, d) * np.sin(a) + k * np.dot(k, d) * (1 - np.cos(a))
    return View(v.target, d, v.distance, v.fov)


def camera(v: View, up: np.ndarray, size: tuple[int, int]) -> Camera:
    W, H = size
    eye = v.target + v.direction * v.distance
    fwd = -v.direction
    right = np.cross(fwd, up)
    if np.linalg.norm(right) < 1e-9:
        right = np.cross(fwd, [0.0, 1.0, 0.0])
    right /= np.linalg.norm(right)
    upv = np.cross(right, fwd)
    f = 0.5 * H / np.tan(np.radians(v.fov / 2))
    return Camera(eye, right, upv, fwd, f, 0.5 * W, 0.5 * H)


def _normals(P: np.ndarray, T: np.ndarray) -> np.ndarray:
    n = np.cross(P[T[:, 1]] - P[T[:, 0]], P[T[:, 2]] - P[T[:, 0]])
    N = np.zeros_like(P)
    for c in range(3):
        np.add.at(N, T[:, c], n)
    return N / np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-30)


def _lit(cam: Camera, s: Surface) -> np.ndarray:
    """Each vertex's color under the lights, both sides of a surface lit from the side the camera sees."""
    N = _normals(s.P, s.T)
    N *= np.where(np.einsum("ij,ij->i", N, cam.eye - s.P) < 0, -1.0, 1.0)[:, None]
    diff, spec = _lighting(cam, s.P, N, 28.0)
    col = s.C * (0.32 + 0.72 * diff)[:, None] + (0.16 * spec)[:, None]
    if s.glow:
        g = min(max(s.glow * 0.85, 0.0), 1.0)
        col = col * (1 - g) + s.gcol[None, :] * g
    return np.clip(col, 0, 1)


def _join(cam: Camera, surfaces: list[Surface]):
    """The surfaces as one set of points and triangles with their lit color and whether each is a prop."""
    Ps, Ts, As, n = [], [], [], 0
    for s in surfaces:
        if not len(s.T):
            continue
        As.append(np.column_stack([_lit(cam, s), np.full(len(s.P), 1.0 if s.prop else 0.0)]))
        Ps.append(s.P)
        Ts.append(s.T + n)
        n += len(s.P)
    if not n:
        return None
    P, T, A = np.concatenate(Ps), np.concatenate(Ts), np.concatenate(As).astype(np.float32)
    depth = (P - cam.eye) @ cam.fwd
    near = max(float(np.percentile(np.abs(depth), 50)) * 1e-3, 1e-6)
    T = T[(depth[T] > near).all(1)]
    return P, T, A


def _ground(cam: Camera, g: Surface, sx, sy, scale: float):
    """Where the ray through each sample (x, y in pixels of an image scale times the final size) meets the ground
    plane: its color with a faint grid and a fade into the background far away, whether it hits, and the inverse
    depth there. Worked out per sample, never rasterized."""
    axis, level, (c0, c1), half = g.plane
    others = [i for i in range(3) if i != axis]
    f = np.float32(cam.f * scale)
    a = (np.asarray(sx, np.float32) - np.float32(cam.cx * scale)) / f
    b = -(np.asarray(sy, np.float32) - np.float32(cam.cy * scale)) / f
    d = [np.float32(cam.fwd[k]) + a * np.float32(cam.right[k]) + b * np.float32(cam.up[k]) for k in range(3)]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (np.float32(level) - np.float32(cam.eye[axis])) / d[axis]
    u = np.float32(cam.eye[others[0]]) + t * d[others[0]]
    v = np.float32(cam.eye[others[1]]) + t * d[others[1]]
    hit = (t > 0) & (np.abs(u - c0) <= half) & (np.abs(v - c1) <= half)
    iz = np.where(hit, 1.0 / np.where(hit, t, 1.0), 0.0).astype(np.float32)
    col = np.empty(t.shape + (3,), np.float32)
    col[:] = g.C[0] * 0.92
    if g.grid:
        sq = np.float32(g.grid)
        size = t * np.sqrt(d[0] ** 2 + d[1] ** 2 + d[2] ** 2) / f / sq * np.float32(scale)   # a pixel, in squares
        gu, gv = u / sq, v / sq
        near = np.minimum(np.abs(gu - np.rint(gu)), np.abs(gv - np.rint(gv)))
        line = np.clip(1.2 - near / np.maximum(size, 1e-9), 0, 1)
        col += (1.0 - col) * (line * 0.12)[..., None]
    r = np.sqrt((u - c0) ** 2 + (v - c1) ** 2) / np.float32(half)
    fog = np.clip((r - 0.35) / 0.6, 0, 1)[..., None] * 0.9
    return col * (1 - fog) + rgb(BG).astype(np.float32) * fog, hit, iz


def _layer(cam: Camera, surfaces: list[Surface], grounds: list[Surface], w: int, h: int):
    """Surfaces and grounds drawn into a w x h picture: color, coverage and inverse depth per pixel.

    Every pixel is sampled once at its center. Where the picture can step sharply from one pixel to the next, at
    an outline or a jump in color, the pixel is sampled again SS x SS times from the triangles around it, as the
    report images are, so edges come out smooth while the smooth inside costs one sample a pixel."""
    img = np.empty((h * w, 3), np.float32)
    img[:] = rgb(BG)
    cover = np.zeros(h * w, bool)
    zbuf = np.zeros(h * w, np.float32)
    joined = _join(cam, surfaces)
    if joined is not None:
        P, T, A = joined
        x, y, iz = cam.project(P)
        iz32 = iz.astype(np.float32)
        tri, w1, w2, zbuf = _rasterize(x, y, iz, T, w, h)
        pix = np.nonzero(tri >= 0)[0]
        at = _interpolate(A, iz32, T, tri[pix], w1[pix], w2[pix])
        img[pix] = at[:, :3]
        cover[pix] = True
        prop = np.zeros(h * w, bool)
        prop[pix] = at[:, 3] > 0.5
    gz = np.zeros(h * w, np.float32)
    gcov = np.zeros(h * w, bool)
    jj, ii = np.divmod(np.arange(h * w), w)
    for g in grounds:
        col, hit, z = _ground(cam, g, ii + 0.5, jj + 0.5, 1.0)
        show = hit & (~cover | (z > zbuf))
        img[show] = col[show]
        gz = np.where(hit, z, gz)
        gcov |= hit
    if joined is None:
        return img, cover | gcov, np.maximum(zbuf, gz)
    lum = img @ np.array([0.3, 0.55, 0.15], np.float32)
    edge = _edges(h, w, cover, [lum / 0.05])
    many = np.nonzero(edge)[0]
    if len(many):
        W2 = w * SS
        k = np.arange(SS * SS)
        sub = ((many // w) * SS * W2 + (many % w) * SS)[:, None] + ((k // SS) * W2 + k % SS)[None, :]
        near = edge.reshape(h, w).copy()
        near[1:] |= near[:-1].copy()
        near[:-1] |= near[1:].copy()
        near[:, 1:] |= near[:, :-1].copy()
        near[:, :-1] |= near[:, 1:].copy()
        cand = _touching(x, y, T, near)
        t2, v1, v2, z2 = _rasterize(x * SS, y * SS, iz, T[cand], W2, h * SS)
        got = t2[sub]
        hit = got >= 0
        sy, sx = np.divmod(sub, W2)
        samples = np.empty(sub.shape + (3,), np.float32)
        samples[:] = rgb(BG)
        sz = np.zeros(sub.shape, np.float32)
        if hit.any():
            smp = sub[hit]
            at2 = _interpolate(A, iz32, T, cand[got[hit]], v1[smp], v2[smp])
            samples[hit] = at2[:, :3]
            sz[hit] = z2[smp]
        for g in grounds:
            col, ghit, z = _ground(cam, g, sx + 0.5, sy + 0.5, SS)
            show = ghit & (~hit | (z > sz))
            samples[show] = col[show]
        img[many] = samples.mean(1)
    if prop.any():   # props, which only set the scene, carry the viewer's faint diagonal stripes
        on = np.nonzero(prop)[0]
        img[on] *= np.where((jj[on] + ii[on]) % 12 < 6, 0.9, 1.0)[:, None].astype(np.float32)
    return img, cover | gcov, np.maximum(zbuf, np.where(cover, 0, gz))


def frame(sc: Scene, u: float, view: View, size: tuple[int, int]) -> np.ndarray:
    """The scene at position u as an (H, W, 3) uint8 picture."""
    w, h = size
    cam = camera(view, sc.up, size)
    posed: Posed = sc.posed(u)
    grounds = [s for s in posed.surfaces if s.ground and s.plane]
    solid = [s for s in posed.surfaces if s.alpha >= 0.999 and not s.ground]
    clear = [s for s in posed.surfaces if s.alpha < 0.999 and not s.ground]
    img, cover, zbuf = _layer(cam, solid, grounds, w, h)
    if clear:
        cimg, ccov, cz = _layer(cam, clear, [], w, h)
        a = float(np.mean([s.alpha for s in clear]))
        front = ccov & (cz >= zbuf)
        img[front] = img[front] * (1 - a) + cimg[front] * a
    pic = (np.clip(img.reshape(h, w, 3), 0, 1) * 255 + 0.5).astype(np.uint8)
    if posed.strokes:
        pic = _strokes(pic, posed, cam, zbuf.reshape(h, w))
    return pic


def _strokes(pic: np.ndarray, posed: Posed, cam: Camera, zbuf: np.ndarray) -> np.ndarray:
    """Lines and dots drawn at twice the size and shrunk onto the picture, where nothing in front hides them."""
    from PIL import Image, ImageDraw

    h, w = zbuf.shape
    over = Image.new("RGBA", (w * SS, h * SS), (0, 0, 0, 0))
    dr = ImageDraw.Draw(over)
    for st in posed.strokes:
        for path, col in zip(st.paths, st.colors):
            if not len(path):
                continue
            x, y, iz = cam.project(path, SS)
            depth = (path - cam.eye) @ cam.fwd
            xi = np.clip((x / SS).astype(int), 0, w - 1)
            yi = np.clip((y / SS).astype(int), 0, h - 1)
            seen = (depth > 0) & (iz >= zbuf[yi, xi] * 0.995)
            c = (np.clip(np.atleast_2d(col), 0, 1) * 255).astype(int)
            alpha = int(255 * st.alpha)
            if st.dots:
                r = max(1.5, st.width * SS * 1.5)
                for i in np.nonzero(seen)[0]:
                    dr.ellipse([x[i] - r, y[i] - r, x[i] + r, y[i] + r], fill=tuple(c[min(i, len(c) - 1)]) + (alpha,))
                continue
            width = max(1, int(round(st.width * SS)))
            for i in range(len(path) - 1):
                if seen[i] or seen[i + 1]:
                    dr.line([(x[i], y[i]), (x[i + 1], y[i + 1])], fill=tuple(c[min(i, len(c) - 1)]) + (alpha,),
                            width=width)
    small = over.resize((w, h), Image.Resampling.BOX)
    return np.asarray(Image.alpha_composite(Image.fromarray(pic).convert("RGBA"), small).convert("RGB"))


# ---------------------------------------------------------------- caption
@lru_cache(maxsize=8)
def font(size: int):
    from PIL import ImageFont

    try:
        return ImageFont.truetype(str(FONT), size)
    except OSError:
        try:
            return ImageFont.load_default(size)
        except TypeError:
            return ImageFont.load_default()


def caption(pic: np.ndarray, lines: list[tuple[str, str]], scale: float = 1.0) -> np.ndarray:
    """A small caption in the lower left: each line as (text, color), the first one larger, on Wobbl's dark."""
    from PIL import Image, ImageDraw

    lines = [(t, c) for t, c in lines if t]
    if not lines:
        return pic
    im = Image.fromarray(pic).convert("RGBA")
    over = Image.new("RGBA", im.size, (0, 0, 0, 0))
    dr = ImageDraw.Draw(over)
    big, small = font(int(round(22 * scale))), font(int(round(17 * scale)))
    pad, gap = int(16 * scale), int(7 * scale)
    fonts = [big] + [small] * (len(lines) - 1)
    boxes = [dr.textbbox((0, 0), t, font=f) for (t, _), f in zip(lines, fonts)]
    w = int(max(dr.textlength(t, font=f) for (t, _), f in zip(lines, fonts))) + 2 * pad + int(6 * scale)
    h = sum(b[3] - b[1] for b in boxes) + gap * (len(lines) - 1) + 2 * pad
    x0, y0 = int(24 * scale), im.size[1] - h - int(24 * scale)
    bg = tuple(int(v * 255) for v in rgb(BG))
    dr.rectangle([x0, y0, x0 + w, y0 + h], fill=bg + (225,))
    dr.rectangle([x0, y0, x0 + int(4 * scale), y0 + h], fill=tuple(int(v * 255) for v in rgb(ACCENT)) + (255,))
    y = y0 + pad
    for (t, c), f, b in zip(lines, fonts, boxes):
        dr.text((x0 + pad + int(6 * scale), y - b[1]), t, font=f, fill=tuple(int(v * 255) for v in rgb(c)) + (255,))
        y += b[3] - b[1] + gap
    return np.asarray(Image.alpha_composite(im, over).convert("RGB"))
