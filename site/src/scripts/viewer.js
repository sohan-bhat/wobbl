// The pokeable part: a real Wobbl result in 3D. This side runs on the page and keeps what the visitor
// touches: the camera and its controls, the arrow that points the load on the motor shaft, the probe
// and the labels.
// The drawing itself happens in part-draw.js, inside a worker (draw-worker.js) where the browser can
// draw there, so loading the mesh, compiling shaders and every frame stay off the page's main thread.
// The readout and the peak marker come from the export's peak table over every direction.
import { Orbit, orbitControls } from './orbit.js';
import { fmt, peakNode, tablePerN, unit } from './reading';

const idle = () => new Promise((r) => ('requestIdleCallback' in window ? requestIdleCallback(() => r(), { timeout: 600 }) : setTimeout(r, 50)));
const nextFrame = () => new Promise((r) => requestAnimationFrame(() => r()));
const easeInOut = (x) => (x <= 0 ? 0 : x >= 1 ? 1 : x < 0.5 ? 4 * x * x * x : 1 - (-2 * x + 2) ** 3 / 2);
const easeOut = (x) => (x <= 0 ? 0 : x >= 1 ? 1 : 1 - (1 - x) ** 3);
const lerp = (a, b, k) => a + (b - a) * k;
const along = (p, d, t) => [p[0] + d[0] * t, p[1] + d[1] * t, p[2] + d[2] * t];

/** The drawing, in a worker on an OffscreenCanvas. */
function inWorker(canvas, man, file, w, h, px) {
  const off = canvas.transferControlToOffscreen();
  const worker = new Worker(new URL('./draw-worker.js', import.meta.url), { type: 'module' });
  const picks = new Map();
  let id = 0, drawn = null, fail = null;
  const ready = new Promise((resolve, reject) => {
    fail = reject;
    worker.onmessage = (e) => {
      const d = e.data;
      if (d.type === 'ready') resolve();
      else if (d.type === 'drawn') { drawn?.(); drawn = null; }
      else if (d.type === 'pick') { picks.get(d.id)?.(d.rgba); picks.delete(d.id); }
      else if (d.type === 'error') reject(new Error(d.message));
    };
    worker.onerror = (e) => reject(new Error(e.message || 'the drawing worker failed'));
  });
  worker.postMessage({ type: 'init', canvas: off, man, file, w, h, px }, [off]);
  return {
    ready,
    fail: (err) => fail(err),
    draw: (frame, ack) => worker.postMessage({ type: 'frame', frame, ack }),
    drawn: () => new Promise((r) => { drawn = r; }),
    size: (nw, nh, npx) => worker.postMessage({ type: 'size', w: nw, h: nh, px: npx }),
    pick: (x, y, vw, vh) => new Promise((r) => { picks.set(++id, r); worker.postMessage({ type: 'pick', id, x, y, vw, vh }); }),
    dispose() { worker.postMessage({ type: 'dispose' }); setTimeout(() => worker.terminate(), 200); },
  };
}

/** The drawing on the page itself, one step per idle moment, with the mesh decoded in a worker. */
function onPage(canvas, man, file, w, h, px) {
  const decoder = new Worker(new URL('./part-worker.js', import.meta.url), { type: 'module' });
  const mesh = new Promise((resolve, reject) => {
    decoder.onmessage = (e) => (e.data.error ? reject(new Error(e.data.error)) : resolve(e.data.mesh));
    decoder.onerror = (e) => reject(new Error(e.message || 'the part worker failed'));
  });
  decoder.postMessage({ file, man });
  let drawer = null, fail = null;
  const ready = Promise.race([
    import('./part-draw.js').then((m) => m.createDrawer({ canvas, man, mesh, w, h, px, pause: idle })).then((d) => { drawer = d; decoder.terminate(); }),
    new Promise((_, reject) => { fail = reject; }),
  ]);
  return {
    ready,
    fail: (err) => fail(err),
    draw: (frame) => drawer?.draw(frame),
    drawn: () => nextFrame(),
    size: (nw, nh, npx) => drawer?.size(nw, nh, npx),
    pick: async (x, y, vw, vh) => (drawer ? drawer.pick(x, y, vw, vh) : [0, 0, 0, 0]),
    dispose() { decoder.terminate(); drawer?.dispose(); },
  };
}

/**
 * Draw the part in root. ui.data is the manifest, ui.state() gives { load, angle, exag },
 * ui.onChange(fn) and ui.onReset(fn) subscribe and return an unsubscribe, ui.setDir(az, el, byKey)
 * points the load when the visitor drags the arrow or presses the arrow keys, ui.touched() says the
 * visitor really did something (a click, a tap, a drag or a key), ui.peaks() gives the peak table
 * once it has loaded, ui.worker says whether
 * the browser can draw in a worker, and ui.signal aborts a start that is still loading. Resolves,
 * once the part has come in, to { dispose, view }: dispose tears it all down; view.get() and
 * view.set() read and move the camera, view.hot() lights the arrow as if held and view.hand()
 * shows the hint pointer on its head, for the idle animation.
 */
export async function start(root, ui) {
  const trash = [];
  try {
    return await build(root, ui, trash);
  } catch (err) {
    drop(trash);
    throw err;
  }
}

/** Run what dispose and a failed start leave behind, newest first. */
const drop = (trash) => { for (const fn of trash.splice(0).reverse()) { try { fn(); } catch { /* already gone */ } } };

async function build(root, ui, trash) {
  const man = ui.data, signal = ui.signal;
  const status = root.querySelector('.viewer-status');
  const poster = root.querySelector('.viewer-poster');
  const alive = () => { if (signal.aborted) throw new DOMException('The page was left', 'AbortError'); };
  const teardown = () => drop(trash);
  status.textContent = 'Loading the 3D part…';

  // Phones get at most 1.5 device pixels per CSS pixel, desktops 2; multisampling only below 2.
  const coarse = matchMedia('(pointer: coarse)').matches || Math.min(screen.width, screen.height) < 600;
  const reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;
  const fullPx = Math.min(window.devicePixelRatio || 1, coarse ? 1.5 : 2);
  let px = fullPx;
  let vw = root.clientWidth || 1, vh = root.clientHeight || 1;
  const canvas = document.createElement('canvas');
  canvas.setAttribute('role', 'img');
  canvas.tabIndex = 0;
  canvas.setAttribute('aria-label', 'Interactive 3D stress map of the cast motor mount. Drag the yellow arrow, or press the arrow keys, to point the load on the motor shaft, five degrees a press or one with Shift. Drag elsewhere to turn the part, pinch or press plus and minus to zoom, and double click or double tap to reset the view.');
  canvas.style.opacity = '0';
  canvas.style.transition = 'opacity 0.3s ease';
  root.prepend(canvas);
  trash.push(() => canvas.remove());
  const file = new URL(root.dataset.file, document.baseURI).href;
  const back = (ui.worker ? inWorker : onPage)(canvas, man, file, vw, vh, px);
  trash.push(() => back.dispose());
  signal.addEventListener('abort', () => back.fail(new DOMException('The page was left', 'AbortError')), { once: true });

  // The camera lives here, with the controls that move it; each frame sends it to the drawing.
  const lo = man.mesh.bounds_min, hi = man.mesh.bounds_max;
  const radius = Math.hypot(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]) / 2;
  const target = [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2 - 6, (lo[2] + hi[2]) / 2 - 4];
  // home: looking from (0.62, 1.05, 0.62), far enough back that the part fills the view
  const HOME = { az: Math.atan2(1.05, 0.62), el: Math.atan2(0.62, Math.hypot(0.62, 1.05)), dist: radius / Math.sin(Math.PI / 12) };
  const orbit = new Orbit({ target, ...HOME, fov: 30, aspect: vw / vh, minDist: radius * 1.3, maxDist: radius * 6 });
  const pl = man.pull;
  const pAt = pl.point, e1 = pl.e1, e2 = pl.e2, e3 = pl.e3;
  let pDir = [...e1];

  // The part sits in the page's first screen: a vertical swipe still scrolls the page on phones,
  // a sideways drag turns the part, and a drag that starts on the arrow turns the pull. No damping:
  // the part stops when the pointer stops, so nothing keeps drawing after a drag.
  canvas.style.touchAction = 'pan-y';
  let onStart = () => {};
  trash.push(orbitControls(canvas, orbit, {
    rotateSpeed: 0.8, onStart: () => { onStart(); ui.touched(); }, onChange: () => { busy(); requestRender(); },
  }));
  function resetView() {
    Object.assign(orbit, HOME);
    orbit.update();
    requestRender();
  }

  await back.ready;
  alive();

  const state = { ...ui.state() };
  const peakTag = root.querySelector('.peak-tag');
  const peakTagV = root.querySelector('.peak-tag-v');
  const loadTag = root.querySelector('.load-tag');
  const grip = root.querySelector('.pull-grip');
  const hand = root.querySelector('.hint-hand');
  const probe = root.querySelector('.probe');
  let peakP = [0, 0, 0];
  let len = 0, u = [1, 0, 0], grow = 1, showTags = false, handOpacity = 0, limit = man.allowable_MPa;
  // how far the entrance has come (k) and how much of the bend it shows
  const enter = { k: 1, bend: 1 };
  let hovering = false, turning = null, auto = false;

  function apply() {
    u = unit(state.az, state.el);
    pDir = [0, 1, 2].map((k) => e1[k] * u[0] + e2[k] * u[1] + e3[k] * u[2]);
    len = ((46 * state.load) / man.load_N) * grow;
    // the peak for this direction, from the export's table
    const peaks = ui.peaks(), node = peakNode(man, peaks, state.az, state.el), f = state.exag * enter.bend * state.load;
    peakP = [0, 1, 2].map((k) => node[k] + f * (u[0] * node[3 + k] + u[1] * node[6 + k] + u[2] * node[9 + k]));
    peakTagV.textContent = `${(tablePerN(man, peaks, state.az, state.el) * state.load).toFixed(1)} MPa`;
    loadTag.textContent = `${fmt(state.load)} N`;
    if (probe.dataset.sticky) {
      probe.hidden = true;
      delete probe.dataset.sticky;
    }
    requestRender();
  }
  // everything the drawing needs for one frame
  const frameState = () => ({
      camera: { position: orbit.eye, target: orbit.target, fov: orbit.fov, aspect: orbit.aspect, view: orbit.view },
      load: state.load, u, exag: state.exag * enter.bend, limit, len, grow: Math.min(grow, 1),
      hot: hovering || turning !== null || auto, ring: hovering || turning !== null,
  });

  // a point in mm to CSS pixels in the view, its depth (below 1 in front of the camera) and whether it is in view
  const toScreen = (p) => { const [x, y, z] = orbit.project(p); return [((x + 1) / 2) * vw, ((1 - y) / 2) * vh, z, x, y]; };
  const inView = (s) => s[2] < 1 && Math.abs(s[3]) < 1.05 && Math.abs(s[4]) < 1.05;
  const GRIP = coarse ? 44 : 28; // the arrow's hit area across, in CSS pixels

  // The labels. Both have fixed widths in CSS, so their sizes are read once, and again only after a
  // resize; each frame then tries a few places for each, best first, and takes the first pair that
  // stays 8 px inside the view without touching. Nothing here reads the layout.
  const INSET = 8, GAP = 6;
  let sizes = null;
  const inside = (r) => r[0] >= INSET && r[1] >= INSET && r[0] + r[2] <= vw - INSET && r[1] + r[3] <= vh - INSET;
  const touch = (a, b) => !!(a && b && a[0] < b[0] + b[2] + 4 && b[0] < a[0] + a[2] + 4 && a[1] < b[1] + b[3] + 4 && b[1] < a[1] + a[3] + 4);
  const clampIn = (r) => [Math.min(Math.max(r[0], INSET), vw - INSET - r[2]), Math.min(Math.max(r[1], INSET), vh - INSET - r[3]), r[2], r[3]];
  // the peak's value beside its marker, on the side away from the pulley first, then the other side,
  // then above and below
  function peakSpots(at, awayLeft) {
    const { pw, ph } = sizes, right = at[0] + 15, left = at[0] - 15 - pw, top = at[1] - ph / 2;
    const out = [];
    for (const dy of [0, -(ph + GAP), ph + GAP]) {
      for (const x of awayLeft ? [left, right] : [right, left]) out.push([x, top + dy, pw, ph]);
    }
    return out;
  }
  // the load just past the arrow's tip, then to either side of it, then back toward the pulley
  function loadSpots(at, u) {
    const { lw, lh } = sizes;
    return [u, [-u[1], u[0]], [u[1], -u[0]], [-u[0], -u[1]]]
      .map((d) => [at[0] - lw / 2 + (d[0] * (lw + GAP)) / 2, at[1] - lh / 2 + (d[1] * (lh + GAP)) / 2, lw, lh]);
  }
  function labels(pk, hubS, ld, u) {
    peakTag.hidden = !pk;
    loadTag.hidden = !ld;
    if (!pk && !ld) return;
    if (!sizes) {
      sizes = { pw: peakTagV.offsetWidth, ph: peakTagV.offsetHeight, lw: loadTag.offsetWidth, lh: loadTag.offsetHeight };
    }
    const As = pk ? peakSpots(pk, pk[0] < hubS[0]) : [null], Bs = ld ? loadSpots(ld, u) : [null];
    let a = null, b = null;
    search: for (const ca of As) {
      if (ca && !inside(ca)) continue;
      for (const cb of Bs) {
        if (cb && !inside(cb)) continue;
        if (!touch(ca, cb)) { a = ca; b = cb; break search; }
      }
    }
    if (!a && !b) {
      // nowhere clean: keep the best places, inside the view, and move the load clear of the value
      a = As[0] && clampIn(As[0]);
      b = Bs[0] && clampIn(Bs[0]);
      if (touch(a, b)) {
        const below = a[1] + a[3] + GAP;
        b = clampIn([b[0], below + b[3] <= vh - INSET ? below : a[1] - GAP - b[3], b[2], b[3]]);
      }
    }
    if (pk) {
      peakTag.style.translate = `${pk[0]}px ${pk[1]}px`;
      peakTagV.style.translate = `${a[0] - pk[0]}px ${a[1] - pk[1]}px`;
    }
    if (ld) loadTag.style.translate = `${b[0]}px ${b[1]}px`;
  }

  function overlays() {
    const hubS = toScreen(pAt), pkS = toScreen(peakP);
    const pk = showTags && inView(pkS) ? pkS : null;
    const reach = Math.max(len, 16);
    if (len <= 0.5 || !showTags) { labels(pk, hubS, null, null); grip.hidden = true; hand.hidden = true; return; }
    // the hit area: a pill over the whole arrow, hub to tip, wider than the arrow itself
    const tipS = toScreen(along(pAt, pDir, reach));
    const dx = tipS[0] - hubS[0], dy = tipS[1] - hubS[1], l = Math.hypot(dx, dy) || 1, pad = GRIP / 2;
    grip.hidden = hubS[2] > 1;
    grip.style.width = `${l + 2 * pad}px`;
    grip.style.height = `${GRIP}px`;
    grip.style.transform = `translate(${hubS[0] - (pad * dx) / l}px, ${hubS[1] - (pad * dy) / l - pad}px) rotate(${Math.atan2(dy, dx)}rad)`;
    // the hint pointer rests on the arrow's head
    hand.hidden = handOpacity <= 0.01;
    if (!hand.hidden) {
      const h = toScreen(along(pAt, pDir, reach - 4));
      hand.style.translate = `${h[0]}px ${h[1]}px`;
      hand.style.opacity = String(handOpacity);
    }
    // the load label goes just past the arrow's tip, on the side the arrow points to
    const ldS = toScreen(along(pAt, pDir, reach + 3));
    labels(pk, hubS, inView(ldS) ? ldS : null, [dx / l, dy / l]);
  }

  // Draw on demand, at most once a frame, and not at all while the part is off screen. While the
  // visitor drags, a slow frame lowers the pixel ratio a step; it comes back when they let go.
  let pending = false, raf = 0, visible = true, dirty = false, busyUntil = 0, last = 0;
  const gaps = [];
  function frame(now, ack) {
    pending = false;
    raf = 0;
    dirty = false;
    // only the visitor's own drags count: the idle animation draws at its own pace on purpose
    if (now < busyUntil && last && !auto) {
      gaps.push(now - last);
      if (gaps.length > 6) gaps.shift();
      if (gaps.length === 6 && gaps.filter((g) => g > 24).length >= 4 && px > 1) {
        px = Math.max(1, px - 0.25);
        gaps.length = 0;
        back.size(vw, vh, px);
      }
    }
    last = now;
    back.draw(frameState(), ack);
    overlays();
  }
  function restore() {
    if (performance.now() < busyUntil || px === fullPx) return;
    px = fullPx;
    back.size(vw, vh, px);
    requestRender();
  }
  // The page updates inside an animation frame, so draw right there instead of a frame later.
  function drawNow() {
    if (!visible) { dirty = true; return; }
    if (pending) { cancelAnimationFrame(raf); pending = false; }
    frame(performance.now());
  }
  function requestRender() {
    dirty = true;
    if (!visible || pending) return;
    pending = true;
    raf = requestAnimationFrame(frame);
  }
  let restoreTimer = 0;
  const busy = () => {
    busyUntil = performance.now() + 300;
    clearTimeout(restoreTimer);
    if (px < fullPx) restoreTimer = window.setTimeout(restore, 350);
  };
  trash.push(() => clearTimeout(restoreTimer));
  trash.push(() => cancelAnimationFrame(raf));
  const seen = new IntersectionObserver(([e]) => { visible = e.isIntersecting; if (visible && dirty) requestRender(); });
  seen.observe(root);
  trash.push(() => seen.disconnect());

  // the canvas's place on the page, read again only after a scroll or a resize, never per move
  let rect = null;
  function resize() {
    rect = null;
    const w = root.clientWidth, h = root.clientHeight;
    if (!w || !h) return;
    vw = w;
    vh = h;
    sizes = null;
    back.size(w, h, px);
    if (enter.k >= 1) {
      orbit.aspect = w / h;
      orbit.update();
    }
    requestRender();
  }
  const ro = new ResizeObserver(resize);
  ro.observe(root);
  trash.push(() => ro.disconnect());

  // Listeners on elements outside the canvas, so dispose can take them off again.
  const on = (el, type, fn, opts) => { el.addEventListener(type, fn, opts); trash.push(() => el.removeEventListener(type, fn, opts)); };
  const where = () => rect || (rect = canvas.getBoundingClientRect());
  const moved = () => { rect = null; };
  on(window, 'scroll', moved, { passive: true });
  on(window, 'resize', moved, { passive: true });

  // The arrow lights up under the pointer, while it is dragged, and while the idle animation turns it.
  const light = () => requestRender();

  // Pointing the load: a drag that starts anywhere on the arrow moves its tip over a sphere around the
  // pulley, as long as the arrow. Where the pointer's ray meets the sphere, the tip goes to the meeting
  // point nearer to where it was, so it can swing round toward the viewer or away; where the ray
  // misses, it goes to the nearest point of the sphere's outline.
  const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  function dirAt(x, y) {
    const r = where(), R = Math.max(len, 16);
    const { origin: o, dir: d } = orbit.ray(((x - r.left) / r.width) * 2 - 1, -((y - r.top) / r.height) * 2 + 1);
    const oc = [o[0] - pAt[0], o[1] - pAt[1], o[2] - pAt[2]];
    const b = dot(d, oc), disc = b * b - (dot(oc, oc) - R * R);
    let w;
    if (disc >= 0) {
      const was = pDir, near = [0, 1, 2].map((k) => oc[k] + (-b - Math.sqrt(disc)) * d[k]);
      const far = [0, 1, 2].map((k) => oc[k] + (-b + Math.sqrt(disc)) * d[k]);
      w = dot(near, was) >= dot(far, was) ? near : far;
    } else {
      w = [0, 1, 2].map((k) => oc[k] - b * d[k]);
    }
    const l = Math.hypot(w[0], w[1], w[2]) || 1;
    w = w.map((v) => v / l);
    const el = (Math.asin(Math.max(-1, Math.min(1, dot(w, e3)))) * 180) / Math.PI;
    const az = (Math.atan2(dot(w, e2), dot(w, e1)) * 180) / Math.PI;
    return [Math.round(az), Math.round(el)];
  }
  on(grip, 'pointerenter', (e) => { if (e.pointerType === 'mouse') { hovering = true; light(); } });
  on(grip, 'pointerleave', (e) => { if (e.pointerType === 'mouse') { hovering = false; light(); } });
  on(grip, 'pointerdown', (e) => {
    if (!e.isPrimary || e.button > 0) return;
    turning = e.pointerId;
    grip.setPointerCapture(e.pointerId);
    grip.dataset.turning = '';
    probe.hidden = true;
    ui.touched();
    light();
  });
  // the page folds the moves into one update a frame, and this view draws in that same frame
  on(grip, 'pointermove', (e) => {
    if (turning !== e.pointerId) return;
    busy();
    const [az, el] = dirAt(e.clientX, e.clientY);
    ui.setDir(az, el, false);
  }, { passive: true });
  const endTurn = (e) => {
    if (turning !== e.pointerId) return;
    turning = null;
    delete grip.dataset.turning;
    hovering = e.pointerType === 'mouse' && grip.matches(':hover');
    light();
  };
  on(grip, 'pointerup', endTurn);
  on(grip, 'pointercancel', endTurn);

  // The probe reads the stress under the pointer from a one pixel render, one request at a time.
  let picking = false;
  async function probeAt(clientX, clientY, sticky) {
    if (picking) return;
    picking = true;
    const r = where();
    const x = clientX - r.left, y = clientY - r.top;
    const px1 = await back.pick(x, y, vw, vh);
    picking = false;
    if (signal.aborted) return;
    if (!px1[3]) { probe.hidden = true; delete probe.dataset.sticky; return; }
    const s = ((px1[0] + px1[1] / 255 + px1[2] / 65025) / 255) * 4096;
    const sf = s > 1e-6 ? man.allowable_MPa / s : Infinity;
    probe.innerHTML = `<b>${s.toFixed(1)} MPa</b><br>${sf > 99 ? 'safety factor above 99' : `safety factor ${sf.toFixed(2)}`}` +
      (px1[3] > 200 ? '<br><span class="x">excluded zone</span>' : '');
    probe.style.left = `${x}px`;
    probe.style.top = `${y}px`;
    probe.style.transform = x > r.width - 200 ? 'translate(calc(-100% - 14px), -50%)' : '';
    probe.hidden = false;
    if (sticky) probe.dataset.sticky = '1'; else delete probe.dataset.sticky;
  }
  let hoverAt = null, hoverRaf = 0;
  on(canvas, 'pointermove', (e) => {
    // while the idle animation plays the values under the pointer would not hold still, so no probe
    if (e.pointerType !== 'mouse' || e.buttons || enter.k < 1 || auto) return;
    hoverAt = [e.clientX, e.clientY];
    if (!hoverRaf) {
      hoverRaf = requestAnimationFrame(() => {
        hoverRaf = 0;
        if (hoverAt) probeAt(hoverAt[0], hoverAt[1], false);
      });
    }
  }, { passive: true });
  trash.push(() => cancelAnimationFrame(hoverRaf));
  on(canvas, 'pointerleave', (e) => { if (e.pointerType === 'mouse') { hoverAt = null; probe.hidden = true; } });
  let down = null;
  on(canvas, 'pointerdown', (e) => {
    down = { x: e.clientX, y: e.clientY };
    hoverAt = null;
    if (e.pointerType === 'mouse') probe.hidden = true;
  });
  // a press that the browser turns into a page scroll is no tap
  on(canvas, 'pointercancel', () => { down = null; });
  // A double click, or two taps in quick succession, puts the camera back.
  let lastTap = 0;
  on(canvas, 'pointerup', (e) => {
    const tap = down && Math.hypot(e.clientX - down.x, e.clientY - down.y) < 6;
    down = null;
    if (!tap) return;
    if (e.pointerType !== 'mouse' && e.timeStamp - lastTap < 320) {
      lastTap = 0;
      probe.hidden = true;
      resetView();
      return;
    }
    lastTap = e.timeStamp;
    ui.touched();
    probeAt(e.clientX, e.clientY, true);
  });
  on(canvas, 'dblclick', () => { probe.hidden = true; resetView(); });

  // Keys on the part: left and right turn the load round the shaft, up and down tip it along the
  // shaft, five degrees a press or one with Shift, and plus and minus zoom.
  on(canvas, 'keydown', (e) => {
    const turn = { ArrowRight: [1, 0], ArrowLeft: [-1, 0], ArrowUp: [0, 1], ArrowDown: [0, -1] }[e.key];
    if (turn) {
      e.preventDefault();
      ui.touched();
      const step = e.shiftKey ? 1 : 5, now = ui.state(); // the page's direction: several presses can land in one frame
      ui.setDir(Math.round(now.az) + turn[0] * step, Math.max(-90, Math.min(90, Math.round(now.el) + turn[1] * step)), true);
      return;
    }
    if (e.key === '+' || e.key === '=' || e.key === '-' || e.key === '_') {
      ui.touched();
      orbit.dist *= e.key === '-' || e.key === '_' ? 1.12 : 1 / 1.12;
    } else {
      return;
    }
    e.preventDefault();
    orbit.update();
    busy();
    requestRender();
  });

  // The idle animation moves the camera through these, as azimuth and elevation around the target.
  const view = {
    get: () => ({ az: orbit.az, el: orbit.el, dist: orbit.dist }),
    set(v) {
      Object.assign(orbit, v);
      orbit.update();
      requestRender();
    },
    hot(onIt) { if (auto !== onIt) { auto = onIt; light(); } },
    hand(opacity) { if (handOpacity !== opacity) { handOpacity = opacity; requestRender(); } },
  };

  trash.push(ui.onChange((s) => { Object.assign(state, s); busy(); apply(); drawNow(); }));
  trash.push(ui.onReset(resetView));
  Object.assign(state, ui.state());
  resize();
  resetView();

  // The entrance: the canvas fades in over the poster from the poster's own camera, colors and
  // unbent shape, then the view eases round to its home, the part takes its load and bends,
  // and the arrow grows out of the pulley. About a second in all, and a plain fade with reduced motion.
  const pc = poster && poster.dataset.camera ? JSON.parse(poster.dataset.camera) : null;
  const homeTarget = [...orbit.target];
  function entranceAt(k, g, ack) {
    enter.k = k;
    const e = easeInOut(k);
    const a0 = [0, 1, 2].map((i) => pc.position[i] - pc.target[i]), r0 = Math.hypot(...a0);
    const s0 = { az: Math.atan2(a0[1], a0[0]), el: Math.asin(a0[2] / r0), dist: r0 };
    let daz = HOME.az - s0.az;
    daz = ((daz + Math.PI * 3) % (Math.PI * 2)) - Math.PI;
    orbit.target = [0, 1, 2].map((i) => lerp(pc.target[i], homeTarget[i], e));
    orbit.az = s0.az + daz * e;
    orbit.el = lerp(s0.el, HOME.el, e);
    orbit.dist = lerp(s0.dist, HOME.dist, e);
    // the poster's frame: its render size and crop, shown contained in this box
    const [fw, fh] = pc.frame, [cx, cy, cw, ch] = pc.crop;
    const s = Math.min(vw / cw, vh / ch), ox = (vw - cw * s) / 2, oy = (vh - ch * s) / 2;
    orbit.fov = lerp(pc.fov, 30, e);
    orbit.aspect = lerp(fw / fh, vw / vh, e);
    orbit.view = k < 1 ? [lerp(fw * s, vw, e), lerp(fh * s, vh, e), lerp(cx * s - ox, 0, e), lerp(cy * s - oy, 0, e), vw, vh] : null;
    orbit.update();
    limit = lerp(pc.clim, man.allowable_MPa, e);
    enter.bend = e;
    grow = easeOut(g);
    apply();
    if (pending) { cancelAnimationFrame(raf); pending = false; }
    frame(performance.now(), ack);
  }
  if (pc && !reduce) {
    entranceAt(0, 0, true);
    await back.drawn();
    alive();
    canvas.style.opacity = '';
    status.textContent = '';
    const t0 = performance.now();
    // a visitor who grabs the part before it settles gets the settled view at once
    let cut = false;
    onStart = () => { cut = true; };
    await new Promise((done) => {
      const step = (now) => {
        if (signal.aborted) { done(); return; }
        const ms = cut ? 1e9 : now - t0;
        if (ms > 300 && root.dataset.state !== 'live') root.dataset.state = 'live';
        entranceAt(Math.min(Math.max((ms - 150) / 950, 0), 1), Math.min(Math.max((ms - 550) / 500, 0), 1));
        if (ms >= 1100) done(); else requestAnimationFrame(step);
      };
      requestAnimationFrame(step);
    });
    onStart = () => {};
    alive();
  } else {
    apply();
    frame(performance.now(), true);
    await back.drawn();
    alive();
    canvas.style.opacity = '';
    await new Promise((r) => setTimeout(r, 300));
    alive();
  }
  root.dataset.state = 'live';
  status.textContent = '';
  showTags = true;
  requestRender();

  // Leaving the page: stop drawing, take the listeners off and give the WebGL context back.
  return {
    dispose() {
      teardown();
      for (const el of root.querySelectorAll('.peak-tag, .load-tag, .probe, .pull-grip, .hint-hand')) el.hidden = true;
    },
    view,
  };
}
