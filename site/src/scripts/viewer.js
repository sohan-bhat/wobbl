// The pokeable part: a real Wobbl result drawn with three.js, which is bundled from npm.
// The data holds two unit belt pulls at right angles in the pulley plane, solved on one mesh
// (scripts/solve_site_directions.py, scripts/export_site_part.py). The model is linear elastic, so a
// pull of F newtons at angle t is exactly F (cos t case1 + sin t case2). The vertex shader combines
// the two stress tensors and takes von Mises from the result at every vertex, since von Mises does
// not add up across loads; turning or sizing the pull only changes uniforms. The readout and the
// peak marker come from the peak the export found for every whole degree.
// A worker fetches and lays out the mesh (part-worker.js), so the page's main thread stays free.
import {
  BufferAttribute, BufferGeometry, ConeGeometry, CylinderGeometry, DoubleSide, Group,
  MathUtils, Mesh, MeshBasicMaterial, PerspectiveCamera, PlaneGeometry, Scene, ShaderMaterial,
  SphereGeometry, Vector2, Vector3, Vector4, WebGLRenderTarget, WebGLRenderer,
} from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { fmt, peakNode, tablePerN } from './reading';

const VERT = /* glsl */ `
attribute vec4 stressA;
attribute vec4 stressB;
attribute vec4 shear;
attribute vec4 dispA;
attribute vec3 dispB;
uniform float uF;
uniform vec2 uCS;
uniform float uExag;
uniform float uLimit;
// each channel's largest value: the attributes arrive as -1 to 1
uniform vec4 uRA;
uniform vec4 uRB;
uniform vec4 uRS;
uniform vec3 uRDA;
uniform vec3 uRDB;
varying vec3 vC;
varying float vE;
varying vec3 vN;
varying vec3 vV;
#ifdef PICK
varying float vS;
#endif
// Turbo, as a polynomial (Google, Apache 2.0), so the color needs no texture lookup
vec3 turbo(float x) {
  const vec4 kr4 = vec4(0.13572138, 4.61539260, -42.66032258, 132.13108234);
  const vec4 kg4 = vec4(0.09140261, 2.19418839, 4.84296658, -14.18503333);
  const vec4 kb4 = vec4(0.10667330, 12.64194608, -60.58204836, 110.36276771);
  const vec2 kr2 = vec2(-152.94239396, 59.28637943);
  const vec2 kg2 = vec2(4.27729857, 2.82956604);
  const vec2 kb2 = vec2(-89.90310912, 27.34824973);
  x = clamp(x, 0.0, 1.0);
  vec4 v4 = vec4(1.0, x, x * x, x * x * x);
  vec2 v2 = v4.zw * v4.z;
  return vec3(dot(v4, kr4) + dot(v2, kr2), dot(v4, kg4) + dot(v2, kg2), dot(v4, kb4) + dot(v2, kb2));
}
void main() {
  // the stress tensor for this pull, then von Mises from it
  vec4 s = uCS.x * (stressA * uRA) + uCS.y * (stressB * uRB);
  vec4 h = shear * uRS;
  vec2 t = uCS.x * h.xy + uCS.y * h.zw;
  vec3 d = s.xyz - s.yzx;
  float vm = uF * sqrt(0.5 * dot(d, d) + 3.0 * (s.w * s.w + dot(t, t)));
  float x = vm / uLimit;
  vC = x > 1.0 ? vec3(0.949, 0.949, 0.933) : turbo(x);
  vE = dispA.w;
#ifdef PICK
  vS = vm;
#endif
  vec3 p = position + (uExag * uF) * (uCS.x * dispA.xyz * uRDA + uCS.y * dispB * uRDB);
  vec4 mv = modelViewMatrix * vec4(p, 1.0);
  vN = normalize(normalMatrix * normal);
  vV = -mv.xyz;
  gl_Position = projectionMatrix * mv;
}`;

const FRAG = /* glsl */ `
uniform float uHatch;
uniform float uPx;
varying vec3 vC;
varying float vE;
varying vec3 vN;
varying vec3 vV;
void main() {
  vec3 v = normalize(vV);
  vec3 n = normalize(vN);
  n = faceforward(n, -v, n);
  vec3 l1 = normalize(vec3(0.35, 0.55, 0.75));
  vec3 l2 = normalize(vec3(-0.7, -0.1, 0.45));
  float d = 0.34 + 0.6 * max(dot(n, l1), 0.0) + 0.2 * max(dot(n, l2), 0.0);
  float s = 0.16 * pow(max(dot(n, normalize(l1 + v)), 0.0), 28.0);
  vec3 col = vC * d + vec3(s);
  if (uHatch > 0.5 && vE > 0.5) {
    float k = mod(gl_FragCoord.x - gl_FragCoord.y, 9.0 * uPx);
    if (k < 2.5 * uPx) col = mix(col, vec3(0.055, 0.059, 0.067), 0.72);
  }
  gl_FragColor = vec4(col, 1.0);
}`;

// The probe reads von Mises under the pointer from a one pixel render: 24 bits of MPa / 4096 in
// red, green and blue, and alpha 1 in an excluded zone, a half elsewhere, 0 off the part.
const PICK_FRAG = /* glsl */ `
varying float vS;
varying float vE;
void main() {
  float a = clamp(vS / 4096.0, 0.0, 0.99999) * 255.0;
  float r = floor(a); a = (a - r) * 255.0;
  float g = floor(a); a = (a - g) * 255.0;
  gl_FragColor = vec4(r / 255.0, g / 255.0, floor(a) / 255.0, vE > 0.5 ? 1.0 : 0.5);
}`;

/** The belt pull: drawn solid, plus a faint copy that shows through the part when it is behind it. */
function arrow(color) {
  const g = new Group();
  const solid = new MeshBasicMaterial({ color });
  const ghost = new MeshBasicMaterial({ color, transparent: true, opacity: 0.35, depthTest: false, depthWrite: false });
  const parts = {};
  for (const [k, geo] of Object.entries({
    shaft: new CylinderGeometry(1.1, 1.1, 1, 12),
    head: new ConeGeometry(3.6, 10, 20),
    hub: new SphereGeometry(3.2, 16, 10),
  })) {
    const m = new Mesh(geo, solid);
    const x = new Mesh(geo, ghost);
    x.renderOrder = 2;
    m.add(x);
    parts[k] = m;
    g.add(m);
  }
  g.userData = parts;
  return g;
}

const SIZES = { position: 3, normal: 3, stressA: 4, stressB: 4, shear: 4, dispA: 4, dispB: 3 };
const idle = () => new Promise((r) => ('requestIdleCallback' in window ? requestIdleCallback(() => r(), { timeout: 600 }) : setTimeout(r, 50)));
const nextFrame = () => new Promise((r) => requestAnimationFrame(() => r()));

const COLD = 0xffd400;

/**
 * Draw the part in root. ui.data is the manifest, ui.state() gives { load, angle, exag },
 * ui.onChange(fn) and ui.onReset(fn) subscribe and return an unsubscribe, and ui.signal aborts a
 * start that is still loading. Resolves to a function that tears it all down.
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
  const alive = () => { if (signal.aborted) throw new DOMException('The page was left', 'AbortError'); };
  const teardown = () => drop(trash);

  // The mesh is fetched and laid out in a worker while the main thread sets up WebGL.
  const worker = new Worker(new URL('./part-worker.js', import.meta.url), { type: 'module' });
  trash.push(() => worker.terminate());
  const got = new Promise((resolve, reject) => {
    worker.onmessage = (e) => (e.data.error ? reject(new Error(e.data.error)) : resolve(e.data.mesh));
    worker.onerror = (e) => reject(new Error(e.message || 'the part worker failed'));
    signal.addEventListener('abort', () => reject(new DOMException('The page was left', 'AbortError')), { once: true });
  });
  worker.postMessage({ file: new URL(root.dataset.file, document.baseURI).href, man });

  status.textContent = 'Loading the 3D part…';
  // one step per idle moment, so none of them holds the page up for long
  await idle();
  alive();

  // Phones get at most 1.5 device pixels per CSS pixel, desktops 2; multisampling only below 2.
  const coarse = matchMedia('(pointer: coarse)').matches || Math.min(screen.width, screen.height) < 600;
  const fullPx = Math.min(window.devicePixelRatio || 1, coarse ? 1.5 : 2);
  let px = fullPx;
  const renderer = new WebGLRenderer({ antialias: fullPx < 2, alpha: true, powerPreference: 'default' });
  trash.push(() => { renderer.dispose(); renderer.forceContextLoss(); });
  renderer.setClearColor(0x000000, 0);
  renderer.setPixelRatio(px);
  const canvas = renderer.domElement;
  canvas.setAttribute('role', 'img');
  canvas.tabIndex = 0;
  canvas.setAttribute('aria-label', 'Interactive 3D stress map of the cast motor mount. Drag or use the arrow keys to orbit, scroll, pinch or press plus and minus to zoom, and double click or double tap to reset the view.');
  canvas.style.opacity = '0';
  root.prepend(canvas);
  trash.push(() => canvas.remove());

  await idle();
  alive();
  const scene = new Scene();
  const camera = new PerspectiveCamera(30, 1, 1, 5000);
  camera.up.set(0, 0, 1);

  const uniforms = {
    uF: { value: man.load_N },
    uCS: { value: new Vector2(1, 0) },
    uExag: { value: 60 },
    uLimit: { value: man.allowable_MPa },
    uHatch: { value: 1 },
    uPx: { value: px },
    uRA: { value: new Vector4() },
    uRB: { value: new Vector4() },
    uRS: { value: new Vector4() },
    uRDA: { value: new Vector3() },
    uRDB: { value: new Vector3() },
  };
  const material = new ShaderMaterial({ vertexShader: VERT, fragmentShader: FRAG, side: DoubleSide, uniforms });
  const pickMaterial = new ShaderMaterial({
    vertexShader: VERT, fragmentShader: PICK_FRAG, side: DoubleSide, uniforms, defines: { PICK: '' },
  });
  const geo = new BufferGeometry();
  const mesh = new Mesh(geo, material);
  mesh.frustumCulled = false;
  scene.add(mesh);
  const pickScene = new Scene();
  const pickMesh = new Mesh(geo, pickMaterial);
  pickMesh.frustumCulled = false;
  pickScene.add(pickMesh);
  const pickTarget = new WebGLRenderTarget(1, 1);
  trash.push(() => { geo.dispose(); material.dispose(); pickMaterial.dispose(); pickTarget.dispose(); });

  // The base is bolted flat to a frame: show that plane.
  const lo = new Vector3(...man.mesh.bounds_min), hi = new Vector3(...man.mesh.bounds_max);
  const size = new Vector3().subVectors(hi, lo);
  const center = new Vector3().addVectors(lo, hi).multiplyScalar(0.5);
  const floor = new Mesh(new PlaneGeometry(size.x * 1.4, size.y * 1.4), new MeshBasicMaterial({ color: 0x15171a }));
  floor.position.set(center.x, center.y, lo.z - 0.4);
  scene.add(floor);

  // The belt pull acts at the pulley, off the part, and turns in the pulley plane.
  const pl = man.pull;
  const pAt = new Vector3(...pl.point);
  const e1 = new Vector3(...pl.e1), e2 = new Vector3(...pl.e2);
  const pDir = new Vector3(), Y = new Vector3(0, 1, 0);
  const pull = arrow(COLD);
  scene.add(pull);
  trash.push(() => scene.traverse((o) => { o.geometry?.dispose?.(); o.material?.dispose?.(); }));

  // No damping: the part stops when the pointer stops, so nothing keeps drawing after a drag.
  const controls = new OrbitControls(camera, canvas);
  trash.push(() => controls.dispose());
  // The part sits in the page's first screen: a vertical swipe still scrolls the page on phones,
  // a sideways drag turns the part.
  canvas.style.touchAction = 'pan-y';
  controls.enableDamping = false;
  controls.rotateSpeed = 0.8;
  const radius = size.length() / 2;
  controls.minDistance = radius * 1.3;
  controls.maxDistance = radius * 6;
  controls.target.copy(center).add(new Vector3(0, -6, -4));
  const home = new Vector3(0.62, 1.05, 0.62).normalize();
  const homeDist = radius / Math.sin(MathUtils.degToRad(camera.fov / 2));
  function resetView() {
    camera.position.copy(controls.target).addScaledVector(home, homeDist);
    controls.update();
    requestRender();
  }

  // Compile the shaders before the mesh arrives; with KHR_parallel_shader_compile it does not block.
  await idle();
  alive();
  await Promise.all([renderer.compileAsync(scene, camera), renderer.compileAsync(pickScene, camera)]);
  alive();
  // first use of the probe's target and shader, on the still empty mesh
  await idle();
  alive();
  renderer.setRenderTarget(pickTarget);
  renderer.render(pickScene, camera);
  renderer.setRenderTarget(null);
  const { arrays: m, range } = await got;
  alive();
  worker.terminate();
  uniforms.uRA.value.fromArray(range.stressA);
  uniforms.uRB.value.fromArray(range.stressB);
  uniforms.uRS.value.fromArray(range.shear);
  uniforms.uRDA.value.fromArray(range.dispA);
  uniforms.uRDB.value.fromArray(range.dispB);
  // Send the mesh to the GPU a few attributes at a time, each in its own idle moment, by drawing
  // it into the probe's one pixel target; the first frame on screen then has nothing left to send.
  geo.setIndex(new BufferAttribute(m.index, 1));
  for (const group of [['position', 'normal'], ['stressA', 'stressB'], ['shear', 'dispA', 'dispB']]) {
    for (const name of group) geo.setAttribute(name, new BufferAttribute(m[name], SIZES[name], name !== 'position'));
    await idle();
    alive();
    renderer.setRenderTarget(pickTarget);
    renderer.render(pickScene, camera);
    renderer.setRenderTarget(null);
  }
  await idle();
  alive();

  const state = { ...ui.state() };
  const peakTag = root.querySelector('.peak-tag');
  const peakTagV = root.querySelector('.peak-tag-v');
  const loadTag = root.querySelector('.load-tag');
  const probe = root.querySelector('.probe');
  const tmp = new Vector3(), tip = new Vector3(), hubS = new Vector3(), peakP = new Vector3();
  let len = 0, cs = [1, 0];

  function apply() {
    const t = (state.angle * Math.PI) / 180;
    cs = [Math.cos(t), Math.sin(t)];
    uniforms.uF.value = state.load;
    uniforms.uCS.value.set(cs[0], cs[1]);
    uniforms.uExag.value = state.exag;
    pDir.copy(e1).multiplyScalar(cs[0]).addScaledVector(e2, cs[1]);
    len = (46 * state.load) / man.load_N;
    pull.visible = len > 0.5;
    const { shaft, head, hub } = pull.userData;
    pull.position.copy(pAt);
    pull.quaternion.setFromUnitVectors(Y, pDir);
    const body = Math.max(len - 10, 0);
    shaft.scale.set(1, Math.max(body, 0.01), 1);
    shaft.position.set(0, body / 2, 0);
    head.position.set(0, body + 5, 0);
    hub.position.set(0, 0, 0);
    // the peak for this angle, found by the export at every whole degree
    const node = peakNode(man, state.angle), f = uniforms.uExag.value * state.load;
    peakP.set(node[0] + f * (cs[0] * node[3] + cs[1] * node[6]), node[1] + f * (cs[0] * node[4] + cs[1] * node[7]),
      node[2] + f * (cs[0] * node[5] + cs[1] * node[8]));
    peakTagV.textContent = `${(tablePerN(man, state.angle) * state.load).toFixed(1)} MPa`;
    loadTag.textContent = `${fmt(state.load)} N belt pull`;
    if (probe.dataset.sticky) {
      probe.hidden = true;
      delete probe.dataset.sticky;
    }
    requestRender();
  }

  let vw = 1, vh = 1;
  function place(el, p) {
    tmp.copy(p).project(camera);
    const vis = tmp.z < 1 && Math.abs(tmp.x) < 1.05 && Math.abs(tmp.y) < 1.05;
    el.hidden = !vis;
    if (vis) el.style.translate = `${((tmp.x + 1) / 2) * vw}px ${((1 - tmp.y) / 2) * vh}px`;
    return vis;
  }
  function overlays() {
    hubS.copy(pAt).project(camera);
    if (place(peakTag, peakP)) {
      // the value reads away from the pulley, so the arrow stays clear of it
      const left = tmp.x < hubS.x;
      peakTag.style.flexDirection = left ? 'row-reverse' : '';
      peakTag.style.transform = left ? 'translate(calc(-100% + 9px), -50%)' : '';
    }
    if (!pull.visible) { loadTag.hidden = true; return; }
    // the label sits just past the arrow's tip, on the side the arrow points to
    if (place(loadTag, tip.copy(pAt).addScaledVector(pDir, Math.max(len, 16) + 8))) {
      const sx = tmp.x - hubS.x, sy = hubS.y - tmp.y, l = Math.hypot(sx, sy) || 1;
      loadTag.style.transform = `translate(${(-50 + (50 * sx) / l).toFixed(1)}%, ${(-50 + (50 * sy) / l).toFixed(1)}%)`;
    }
  }

  // Draw on demand, at most once a frame, and not at all while the part is off screen. While the
  // visitor drags, a slow frame lowers the pixel ratio a step; it comes back when they let go.
  let pending = false, raf = 0, visible = true, dirty = false, busyUntil = 0, last = 0;
  const gaps = [];
  function frame(now) {
    pending = false;
    raf = 0;
    dirty = false;
    if (now < busyUntil && last) {
      gaps.push(now - last);
      if (gaps.length > 6) gaps.shift();
      if (gaps.length === 6 && gaps.filter((g) => g > 24).length >= 4 && px > 1) {
        px = Math.max(1, px - 0.25);
        gaps.length = 0;
        renderer.setPixelRatio(px);
        renderer.setSize(vw, vh, false);
        uniforms.uPx.value = px;
      }
    }
    last = now;
    renderer.render(scene, camera);
    overlays();
  }
  function restore() {
    if (performance.now() < busyUntil || px === fullPx) return;
    px = fullPx;
    renderer.setPixelRatio(px);
    renderer.setSize(vw, vh, false);
    uniforms.uPx.value = px;
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
  controls.addEventListener('change', () => { busy(); requestRender(); });
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
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
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

  // The probe reads the stress under the pointer from a one pixel render, at most once a frame.
  const px1 = new Uint8Array(4);
  function probeAt(clientX, clientY, sticky) {
    const r = where();
    const x = clientX - r.left, y = clientY - r.top;
    camera.setViewOffset(vw, vh, x, y, 1, 1);
    renderer.setRenderTarget(pickTarget);
    renderer.clear();
    renderer.render(pickScene, camera);
    renderer.readRenderTargetPixels(pickTarget, 0, 0, 1, 1, px1);
    renderer.setRenderTarget(null);
    camera.clearViewOffset();
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
    if (e.pointerType !== 'mouse' || e.buttons) return;
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
    probeAt(e.clientX, e.clientY, true);
  });
  on(canvas, 'dblclick', () => { probe.hidden = true; resetView(); });

  // Keyboard orbit: arrows rotate about the part, plus and minus zoom.
  const Z = new Vector3(0, 0, 1), off = new Vector3(), side = new Vector3(), next = new Vector3();
  on(canvas, 'keydown', (e) => {
    const step = e.shiftKey ? 0.3 : 0.1;
    off.subVectors(camera.position, controls.target);
    if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') {
      off.applyAxisAngle(Z, e.key === 'ArrowLeft' ? -step : step);
    } else if (e.key === 'ArrowUp' || e.key === 'ArrowDown') {
      side.crossVectors(off, Z).normalize();
      next.copy(off).applyAxisAngle(side, e.key === 'ArrowUp' ? step : -step);
      if (Math.abs(next.z / next.length()) < 0.97) off.copy(next);
    } else if (e.key === '+' || e.key === '=' || e.key === '-' || e.key === '_') {
      const l = off.length() * (e.key === '-' || e.key === '_' ? 1.12 : 1 / 1.12);
      off.setLength(Math.min(controls.maxDistance, Math.max(controls.minDistance, l)));
    } else {
      return;
    }
    e.preventDefault();
    camera.position.copy(controls.target).add(off);
    camera.lookAt(controls.target);
    controls.update();
  });

  trash.push(ui.onChange((s) => { Object.assign(state, s); busy(); apply(); drawNow(); }));
  trash.push(ui.onReset(resetView));
  Object.assign(state, ui.state());
  resize();
  resetView();

  apply();
  // Swap the poster for the canvas once the first frame is on screen.
  await nextFrame();
  await nextFrame();
  alive();
  canvas.style.opacity = '';
  root.dataset.state = 'live';
  status.textContent = '';

  // Leaving the page: stop drawing, take the listeners off and give the WebGL context back.
  return function dispose() {
    teardown();
    for (const el of root.querySelectorAll('.peak-tag, .load-tag, .probe')) el.hidden = true;
  };
}
