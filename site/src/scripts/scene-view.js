// Where the camera stands for a scene run, shared by the example page's live view (live-run.ts) and the stills
// drawn at build time (run-poster.js), so both open on the same picture. The rules come from how product and
// science viewers frame a model: aim at the subject and fill the frame with it (model-viewer fits the model's
// own extent, not the room it stands in), let props run out of the frame as context, and when the subject is
// tiny next to the path it travels (a rocket on a 3 km flight), follow it up close at its true size and show the
// whole scene in a small inset, as NASA's Eyes keeps a follow view beside its trajectory view.
import { Orbit, corners, fitCentered, fitTo } from './orbit.js';

const radiusOf = (b) => Math.hypot(b.hi[0] - b.lo[0], b.hi[1] - b.lo[1], b.hi[2] - b.lo[2]) / 2;
const centerOf = (b) => [0, 1, 2].map((i) => (b.lo[i] + b.hi[i]) / 2);
const angles = (v) => ({ az: Math.atan2(v[1], v[0]), el: Math.asin(Math.max(-1, Math.min(1, v[2] / (Math.hypot(...v) || 1)))) });

/** The follow target at frame position p (a frame and the share to the next), or null when the view does not follow. */
export function followAt(run, a, b, k) {
  const f = run.follow;
  if (!f) return null;
  const A = f.c[a], B = f.c[b] ?? A;
  return A.map((x, i) => x + (B[i] - x) * k);
}

/**
 * The main camera of a scene run in a view vw by vh CSS pixels, at frame a, b, k: following the subject when the
 * scene magnified it, else fitted to the subject, else to the whole box.
 */
export function mainOrbit(run, vw, vh, a = 0, b = 0, k = 0, { search = false } = {}) {
  const all = run.bounds, R = radiusOf(all), aspect = vw / vh;
  if (run.follow) {
    const { az, el } = angles(run.follow.dir || run.camera.view), r = run.follow.r;
    const o = new Orbit({ target: followAt(run, a, b, k), az, el, dist: r * 4, fov: run.camera.fov, aspect,
      near: r / 40, far: R * 40, minDist: r * 1.2, maxDist: R * 6 });
    fitSphere(o, r * (run.follow.fill || 1.2));
    return o;
  }
  const box = run.subject || run.hold || all, r = radiusOf(box);
  const pts = run.subject && run.points ? run.points : corners(box.lo, box.hi);
  const base = angles(run.camera.view);
  // a subject framed up close keeps the rest of the scene behind it in view, so the planes still come from the whole box
  const make = (az, el) => {
    const o = new Orbit({ target: centerOf(box), az, el, dist: r * 4, fov: run.camera.fov, aspect, near: Math.min(r, R) / 50,
      far: R * 40, minDist: r * 0.3, maxDist: Math.max(R, r) * 8 });
    fitCentered(o, pts, r * 0.3);
    return o;
  };
  let o = make(base.az, base.el);
  if (search && run.subject && run.points) {
    // a long thin subject (a linkage) is turned until it fills the frame, as a photographer turns a part to
    // run corner to corner: around the scene's own view, never from below or straight down
    let best = fillOf(o, pts);
    for (let da = -60; da <= 60; da += 15) {
      for (const de of [-10, 0, 12, 24]) {
        const el = Math.min(Math.max(base.el + (de * Math.PI) / 180, (15 * Math.PI) / 180), (68 * Math.PI) / 180);
        const c = make(base.az + (da * Math.PI) / 180, el), f = fillOf(c, pts);
        if (f > best * 1.04) { best = f; o = c; }
      }
    }
  }
  // a little air round it, so what stands beside the subject is not cut at the edge
  o.dist *= 1.06;
  o.update();
  return o;
}

/** How much of the view the points' outline on screen covers, 0 to 1. */
function fillOf(o, pts) {
  let x0 = 1, x1 = -1, y0 = 1, y1 = -1;
  for (const p of pts) {
    const [x, y] = o.project(p);
    x0 = Math.min(x0, x); x1 = Math.max(x1, x); y0 = Math.min(y0, y); y1 = Math.max(y1, y);
  }
  return Math.max(0, x1 - x0) * Math.max(0, y1 - y0) / 4;
}

/** Back the camera off until a sphere of radius r round its target fills the narrower side of the view. */
function fitSphere(o, r) {
  const half = (o.fov * Math.PI) / 360, h = Math.atan(Math.tan(half) * Math.min(1, o.aspect));
  o.dist = Math.max(o.minDist, r / Math.sin(h));
  o.update();
}

/**
 * The inset that shows the whole scene beside a follow view: its rectangle in the view (x, y, w, h from the top
 * left, in CSS pixels) and its camera, turned with the main one. Null when the view does not follow.
 */
export function insetView(run, vw, vh, az, el) {
  if (!run.follow) return null;
  const all = run.bounds, R = radiusOf(all);
  // shaped to the scene as the camera sees it, tall for a flight, wide for a long throw
  const probe = new Orbit({ target: centerOf(all), az, el, dist: R * 4, fov: run.camera.fov, aspect: 1, near: R / 50, far: R * 40 });
  let sx = 0, sy = 0;
  for (const p of corners(all.lo, all.hi)) {
    const d = [p[0] - probe.target[0], p[1] - probe.target[1], p[2] - probe.target[2]];
    sx = Math.max(sx, Math.abs(d[0] * probe.x[0] + d[1] * probe.x[1] + d[2] * probe.x[2]));
    sy = Math.max(sy, Math.abs(d[0] * probe.y[0] + d[1] * probe.y[1] + d[2] * probe.y[2]));
  }
  const shape = Math.min(1.5, Math.max(0.62, sx / (sy || 1)));
  const big = Math.min(vw, vh) * (vw < 480 ? 0.46 : 0.42);
  const w = Math.round(shape >= 1 ? big : big * shape), h = Math.round(shape >= 1 ? big / shape : big);
  const m = Math.round(Math.max(8, Math.min(vw, vh) * 0.03));
  // bottom left, so a peak marked inside it has room for its label on the right
  const rect = [m, Math.round(vh - h - m), w, h];
  const o = new Orbit({ target: centerOf(all), az, el, dist: R * 4, fov: run.camera.fov, aspect: w / h, near: R / 50, far: R * 40 });
  fitTo(o, corners(all.lo, all.hi), R * 1.05);
  return { rect, orbit: o };
}

/**
 * Aim a follow view at center with the subject in the middle of the part of the view the inset leaves free:
 * the camera slides sideways and up by half the inset's size, as a pan, so the subject never sits under it.
 */
export function aimAt(o, center, vw, vh, rect) {
  o.target = [...center];
  o.update();
  if (!rect) return;
  const visH = 2 * o.dist * Math.tan((o.fov * Math.PI) / 360), visW = visH * o.aspect;
  const side = rect[0] + rect[2] / 2 < vw / 2 ? -1 : 1;
  const sx = side * (rect[2] / vw) * visW * 0.5, sy = (rect[3] / vh) * visH * 0.2;
  o.target = [0, 1, 2].map((i) => center[i] + o.x[i] * sx - o.y[i] * sy);
  o.update();
}

/**
 * How much a scene's lines thicken or thin in a view, so a path keeps about the width of a line on screen
 * (2.6 CSS pixels across) however close the camera is: lines are made as tubes sized for the whole scene.
 */
export function tubeScale(run, o, vh) {
  const R = radiusOf(run.bounds), perPx = (2 * o.dist * Math.tan((o.fov * Math.PI) / 360)) / vh;
  return Math.min(1, (1.3 * perPx) / (2.4 * R * 0.0016));
}

/** An orbit's numbers as the drawing stage takes them. */
export const cameraOf = (o) => ({ position: o.eye, target: o.target, fov: o.fov, aspect: o.aspect, near: o.near, far: o.far });
