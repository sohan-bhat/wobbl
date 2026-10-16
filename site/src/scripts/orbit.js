// A camera that orbits a target with z up, and the pointer and wheel handling that moves it, as
// OrbitControls would. It is a few numbers and a little math, so the page's main thread never loads
// three.js; the drawing worker gets the camera's numbers with every frame.

const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const unit = (a) => { const l = Math.hypot(a[0], a[1], a[2]) || 1; return [a[0] / l, a[1] / l, a[2] / l]; };
const LIMIT = Math.PI / 2 - 1e-3;

export class Orbit {
  /** target in mm, az and el in radians (azimuth from +x toward +y, elevation above the xy plane), fov in degrees */
  constructor({ target, az, el, dist, fov = 30, aspect = 1, near = 1, far = 5000, minDist = 0, maxDist = Infinity }) {
    Object.assign(this, { target: [...target], az, el, dist, fov, aspect, near, far, minDist, maxDist });
    this.view = null; // [fullWidth, fullHeight, offsetX, offsetY, width, height], as PerspectiveCamera.setViewOffset
    this.update();
  }

  position() {
    const c = Math.cos(this.el), t = this.target;
    return [t[0] + this.dist * c * Math.cos(this.az), t[1] + this.dist * c * Math.sin(this.az), t[2] + this.dist * Math.sin(this.el)];
  }

  /** Recompute the camera's axes and frustum after any change. */
  update() {
    this.el = Math.min(LIMIT, Math.max(-LIMIT, this.el));
    this.dist = Math.min(this.maxDist, Math.max(this.minDist, this.dist));
    const eye = this.position();
    const z = unit([eye[0] - this.target[0], eye[1] - this.target[1], eye[2] - this.target[2]]);
    const x = unit(cross([0, 0, 1], z)), y = cross(z, x);
    Object.assign(this, { eye, x, y, z });
    // the frustum at the near plane, as PerspectiveCamera.updateProjectionMatrix lays it out
    let top = this.near * Math.tan((this.fov * Math.PI) / 360), height = 2 * top, width = this.aspect * height, left = -0.5 * width;
    const v = this.view;
    if (v) {
      left += (v[2] * width) / v[0];
      top -= (v[3] * height) / v[1];
      width *= v[4] / v[0];
      height *= v[5] / v[1];
    }
    Object.assign(this, { left, top, width, height });
  }

  /** A point in mm to normalized device coordinates: x and y from -1 to 1, z below 1 when in front. */
  project(p) {
    const d = [p[0] - this.eye[0], p[1] - this.eye[1], p[2] - this.eye[2]];
    const cx = dot(d, this.x), cy = dot(d, this.y), cz = dot(d, this.z), n = this.near, f = this.far;
    const r = this.left + this.width, b = this.top - this.height;
    const w = -cz;
    const px = ((2 * n) / (r - this.left)) * cx + ((r + this.left) / (r - this.left)) * cz;
    const py = ((2 * n) / (this.top - b)) * cy + ((this.top + b) / (this.top - b)) * cz;
    const pz = (-(f + n) / (f - n)) * cz - (2 * f * n) / (f - n);
    return [px / w, py / w, w > 0 ? pz / w : 2];
  }

  /** The ray from the eye through a point given in normalized device coordinates. */
  ray(nx, ny) {
    const cx = this.left + ((nx + 1) / 2) * this.width, cy = this.top - this.height + ((ny + 1) / 2) * this.height;
    const d = unit([0, 1, 2].map((k) => cx * this.x[k] + cy * this.y[k] - this.near * this.z[k]));
    return { origin: this.eye, dir: d };
  }
}

/**
 * Back the camera off along its view until every point of box is inside the view with a margin, and never
 * nearer than least. The live playback and the posters drawn at build time frame a part the same way.
 */
export function fitTo(o, box, least) {
  o.update();
  const tv = Math.tan((o.fov * Math.PI) / 360) * 0.93, th = tv * o.aspect;
  let d = least;
  for (const p of box) {
    const r = [p[0] - o.target[0], p[1] - o.target[1], p[2] - o.target[2]];
    const cx = Math.abs(dot(r, o.x)), cy = Math.abs(dot(r, o.y)), cz = dot(r, o.z);
    d = Math.max(d, cz + cx / th, cz + cy / tv);
  }
  o.dist = d;
  o.update();
}

/** The eight corners of the box lo to hi. */
export const corners = (lo, hi) => [0, 1, 2, 3, 4, 5, 6, 7].map((c) => [c & 1 ? hi[0] : lo[0], c & 2 ? hi[1] : lo[1], c & 4 ? hi[2] : lo[2]]);

/**
 * Drag to turn the camera round its target and pinch to move in and out, as OrbitControls does. A
 * plain wheel scrolls the page; a pinch on a trackpad, which browsers send as a wheel with ctrl held,
 * zooms. onStart runs once a drag or pinch really moves, never for a press that turns into a page
 * scroll, and onChange after every move. Returns a function that takes the listeners off again.
 */
export function orbitControls(el, orbit, { onStart = () => {}, onChange = () => {}, rotateSpeed = 1 } = {}) {
  const pointers = new Map();
  let span = 0, started = false;
  const spread = () => {
    const [a, b] = [...pointers.values()];
    return Math.hypot(a.x - b.x, a.y - b.y);
  };
  const down = (e) => {
    if (e.pointerType === 'mouse' && e.button > 0) return;
    el.setPointerCapture(e.pointerId);
    pointers.set(e.pointerId, { x: e.clientX, y: e.clientY, x0: e.clientX, y0: e.clientY, touch: e.pointerType !== 'mouse' });
    if (pointers.size === 2) span = spread();
    if (pointers.size === 1) started = false;
  };
  const move = (e) => {
    const p = pointers.get(e.pointerId);
    if (!p) return;
    const dx = e.clientX - p.x, dy = e.clientY - p.y;
    if (!dx && !dy) return;
    if (!started && pointers.size === 1 && p.touch) {
      // one finger: up and down belong to the page's scroll until the finger has clearly gone sideways
      const sx = e.clientX - p.x0, sy = e.clientY - p.y0;
      if (Math.abs(sx) < 8 || Math.abs(sx) < Math.abs(sy)) return;
    }
    p.x = e.clientX;
    p.y = e.clientY;
    if (!started) { started = true; onStart(); }
    if (pointers.size === 1) {
      const k = (2 * Math.PI * rotateSpeed) / (el.clientHeight || 1);
      orbit.az -= dx * k;
      orbit.el += dy * k;
    } else if (pointers.size === 2) {
      const s = spread();
      if (span > 0 && s > 0) orbit.dist *= span / s;
      span = s;
    } else {
      return;
    }
    orbit.update();
    onChange();
  };
  const up = (e) => { pointers.delete(e.pointerId); if (pointers.size === 2) span = spread(); };
  const wheel = (e) => {
    if (!e.ctrlKey) return;
    e.preventDefault();
    orbit.dist *= e.deltaY > 0 ? 1 / 0.95 : 0.95;
    orbit.update();
    onStart();
    onChange();
  };
  const menu = (e) => e.preventDefault();
  el.addEventListener('pointerdown', down);
  el.addEventListener('pointermove', move, { passive: true });
  el.addEventListener('pointerup', up);
  el.addEventListener('pointercancel', up);
  el.addEventListener('wheel', wheel, { passive: false });
  el.addEventListener('contextmenu', menu);
  return () => {
    el.removeEventListener('pointerdown', down);
    el.removeEventListener('pointermove', move);
    el.removeEventListener('pointerup', up);
    el.removeEventListener('pointercancel', up);
    el.removeEventListener('wheel', wheel);
    el.removeEventListener('contextmenu', menu);
  };
}
