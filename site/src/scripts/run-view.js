// The camera and the frame a run opens on in its example page's view, shared by the live view (live-run.ts) and
// the stills drawn at build time (run-poster.js), so a still is the very frame the live view first shows. A
// strength or heat part is fitted with its load arrow inside the view; a scene frames or follows its subject
// (scene-view.js). Each frame is the description the drawing stage takes (play-draw.js frame).
import { Orbit, corners, fitTo } from './orbit.js';
import { aimAt, cameraOf, followAt, insetView, mainOrbit, tubeScale } from './scene-view.js';

export const clamp01 = (x) => Math.min(Math.max(x, 0), 1);

/** y at x on a piecewise linear curve. */
export function interp(xs, ys, x) {
  if (x <= xs[0]) return ys[0];
  if (x >= xs[xs.length - 1]) return ys[ys.length - 1];
  let lo = 0, hi = xs.length - 1;
  while (hi - lo > 1) { const mid = (lo + hi) >> 1; if (xs[mid] <= x) lo = mid; else hi = mid; }
  return ys[lo] + ((ys[hi] - ys[lo]) * (x - xs[lo])) / (xs[hi] - xs[lo]);
}

/** A scene's frame position p as the saved frames either side of it and the share of the way to the next. */
export function frameAt(run, p) {
  const F = run.input.t.length, a = Math.min(Math.max(Math.floor(p), 0), F - 1), b = Math.min(a + 1, F - 1);
  return { a, b, k: Math.min(Math.max(p - a, 0), 1) };
}

/** The rectangle of each part in CSS pixels: side by side when the view is wider than tall. */
export const paneViews = (n, w, h) => (n === 1 ? [[0, 0, w, h]]
  : w >= h ? [[0, 0, w / 2, h], [w / 2, 0, w / 2, h]] : [[0, 0, w, h / 2], [0, h / 2, w, h / 2]]);

/**
 * What the view of a run has to hold, from the run in its first pane (lead): its box, its radius, and the points
 * the camera keeps inside the view, the corners of the box and a strength run's arrow at full load.
 */
export function runView(kind, lead) {
  const scene = kind === 'scene';
  const lo = scene ? lead.bounds.lo : lead.mesh.bounds_min, hi = scene ? lead.bounds.hi : lead.mesh.bounds_max;
  const radius = Math.hypot(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]) / 2;
  const box = corners(lo, hi);
  if (kind === 'strength' && lead.arrow) {
    const a = lead.arrow, len = 0.42 * radius * (a.pull ? 1 : -1);
    box.push(a.point, a.point.map((x, i) => x + len * a.dir[i]));
  }
  return { kind, lead, lo, hi, radius, box };
}

/** The view's orbiting camera, before it is put where the run opens. */
export function newOrbit(rv) {
  const { lo, hi, radius, lead } = rv, target = [0, 1, 2].map((i) => (lo[i] + hi[i]) / 2);
  return new Orbit({ target, az: 0, el: 0, dist: radius * 4, fov: rv.kind === 'scene' ? lead.camera.fov : 30, aspect: 1, near: radius / 50,
    far: radius * 40, minDist: radius * 1.1, maxDist: radius * 8 });
}

/** Back the camera off until everything the view holds is inside a pane vw by vh, from where it now looks. */
export function fitView(orbit, rv, vw, vh, p) {
  const { lead } = rv;
  orbit.aspect = vw / vh;
  if (rv.kind === 'scene') {
    // a scene frames its subject, or follows it, as its stills do (scene-view.js), from where the camera now looks
    const { a, b, k } = frameAt(lead, p);
    const v = [Math.cos(orbit.el) * Math.cos(orbit.az), Math.cos(orbit.el) * Math.sin(orbit.az), Math.sin(orbit.el)];
    const o = mainOrbit({ ...lead, camera: { ...lead.camera, view: v }, follow: lead.follow && { ...lead.follow, dir: v } }, vw, vh, a, b, k);
    Object.assign(orbit, { target: o.target, dist: o.dist, near: o.near, far: o.far, minDist: o.minDist, maxDist: o.maxDist });
    orbit.update();
    return;
  }
  fitTo(orbit, rv.box, rv.radius * 1.2);
}

/** Turn the camera to the side the run opens on and fit it, for a pane vw by vh with a scene at frame position p. */
export function homeView(orbit, rv, vw, vh, p) {
  const { lead } = rv, scene = rv.kind === 'scene';
  const v = scene && lead.follow?.dir ? lead.follow.dir : lead.camera.view, l = Math.hypot(v[0], v[1], v[2]) || 1;
  orbit.az = Math.atan2(v[1], v[0]);
  orbit.el = Math.asin(Math.max(-1, Math.min(1, v[2] / l)));
  if (scene && lead.subject) {
    // the side the subject fills the frame from, as the stills open on it (scene-view.js)
    const { a, b, k } = frameAt(lead, p);
    const o = mainOrbit(lead, vw, vh, a, b, k, { search: true });
    orbit.az = o.az;
    orbit.el = o.el;
  }
  fitView(orbit, rv, vw, vh, p);
}

/**
 * The frame the stage draws at playback progress u (0 to 1): a strength run's share of its load, a heat run's
 * clock over span seconds. A scene's frame is filled in by sceneFrame.
 */
export function runFrame(rv, orbit, views, u, bend, span) {
  const { lead } = rv;
  const f = { camera: cameraOf(orbit), views, k: u, radius: rv.radius, bend };
  if (rv.kind === 'scene') {
    f.k = 1;
  } else if (rv.kind === 'heat') {
    if (lead.transient) {
      const s = u * span, ts = lead.times_s;
      let i = 0;
      while (i < ts.length - 2 && ts[i + 1] <= s) i++;
      f.heat = { i0: i, i1: i + 1, mix: clamp01((s - ts[i]) / (ts[i + 1] - ts[i])) };
      f.k = 1;
    } else {
      f.heat = { i0: 0, i1: 0, mix: 0 };
      f.k = u >= 1 ? 1 : interp(lead.warmup.t_s, lead.warmup.theta, u * span);
    }
  }
  return f;
}

/**
 * A scene's part of a frame: each pane's run (runs) at its own frame position (pos), and, for a view that follows
 * its subject, each pane's own camera and the inset beside it. insets is filled with each pane's inset view.
 */
export function sceneFrame(f, orbit, rv, runs, pos, views, look, insets) {
  const follows = !!rv.lead.follow;
  f.scenes = runs.map((r, row) => {
    const { a, b, k } = frameAt(r, pos[row]), T = r.input.t;
    return { a, b, k, t: k ? T[a] + (T[b] - T[a]) * k : T[a], bend: look.bend, mag: follows ? false : look.mag };
  });
  if (!follows || !orbit) return f;
  f.cameras = [];
  f.insets = [];
  insets.length = 0;
  runs.forEach((r, row) => {
    const { a, b, k } = frameAt(r, pos[row]), [, , vw, vh] = views[row];
    const ins = insetView(r, vw, vh, orbit.az, orbit.el);
    aimAt(orbit, followAt(r, a, b, k) ?? orbit.target, vw, vh, ins?.rect);
    f.cameras.push(cameraOf(orbit));
    insets.push(ins);
    f.scenes[row].tube = tubeScale(r, orbit, vh);
    if (!ins) { f.insets.push(null); return; }
    f.insets.push({ rect: ins.rect, camera: cameraOf(ins.orbit), scene: { ...f.scenes[row], mag: look.mag, tube: 1 } });
  });
  return f;
}
