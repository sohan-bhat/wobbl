// The pokeable part: a real Wobbl result drawn with three.js.
// Data comes from scripts/export_site_part.py. Stress and displacement are
// scaled by the load slider; the model is linear elastic, so that is exact.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const VERT = /* glsl */ `
attribute float stress;
attribute float excl;
uniform float uScale;
varying float vS;
varying float vE;
varying vec3 vN;
varying vec3 vV;
void main() {
  vS = stress * uScale;
  vE = excl;
  vec4 mv = modelViewMatrix * vec4(position, 1.0);
  vN = normalize(normalMatrix * normal);
  vV = -mv.xyz;
  gl_Position = projectionMatrix * mv;
}`;

const FRAG = /* glsl */ `
precision highp float;
uniform float uLimit;
uniform float uHatch;
uniform float uPx;
varying float vS;
varying float vE;
varying vec3 vN;
varying vec3 vV;
uniform sampler2D uLut;
void main() {
  float x = vS / uLimit;
  vec3 base = x > 1.0 ? vec3(0.949, 0.949, 0.933) : texture2D(uLut, vec2(clamp(x, 0.0, 1.0) * 0.99609375 + 0.001953125, 0.5)).rgb;
  vec3 v = normalize(vV);
  vec3 n = normalize(vN);
  n = faceforward(n, -v, n);
  vec3 l1 = normalize(vec3(0.35, 0.55, 0.75));
  vec3 l2 = normalize(vec3(-0.7, -0.1, 0.45));
  float d = 0.34 + 0.6 * max(dot(n, l1), 0.0) + 0.2 * max(dot(n, l2), 0.0);
  float s = 0.16 * pow(max(dot(n, normalize(l1 + v)), 0.0), 28.0);
  vec3 col = base * d + vec3(s);
  if (uHatch > 0.5 && vE > 0.5) {
    float k = mod(gl_FragCoord.x - gl_FragCoord.y, 9.0 * uPx);
    if (k < 2.5 * uPx) col = mix(col, vec3(0.055, 0.059, 0.067), 0.72);
  }
  gl_FragColor = vec4(col, 1.0);
}`;

// matplotlib's turbo colormap, the one Wobbl's own heatmaps use, as 256 RGB hex triplets.
const TURBO = '30123b32154333184a341b51351e5836215f37246638276d392a733a2d793b2f803c32863d358b3e38913f3b973f3e9c4040a24143a74146ac4249b1424bb5434eba4451bf4454c34456c74559cb455ccf455ed34661d64664da4666dd4669e0466be3476ee64771e94773eb4776ee4778f0477bf2467df44680f64682f84685fa4687fb458afc458cfd448ffe4391fe4294ff4196ff4099ff3e9bfe3d9efe3ba0fd3aa3fc38a5fb37a8fa35abf833adf731aff52fb2f42eb4f22cb7f02ab9ee28bceb27bee925c0e723c3e422c5e220c7df1fc9dd1ecbda1ccdd81bd0d51ad2d21ad4d019d5cd18d7ca18d9c818dbc518ddc218dec018e0bd19e2bb19e3b91ae4b61ce6b41de7b21fe9af20eaac22ebaa25eca727eea42aefa12cf09e2ff19b32f29835f39438f4913cf58e3ff68a43f78746f8844af8804ef97d52fa7a55fa7659fb735dfc6f61fc6c65fd6969fd666dfe6271fe5f75fe5c79fe597dff5680ff5384ff5188ff4e8bff4b8fff4992ff4796fe4499fe429cfe409ffd3fa1fd3da4fc3ca7fc3aa9fb39acfb38affa37b1f936b4f836b7f735b9f635bcf534bef434c1f334c3f134c6f034c8ef34cbed34cdec34d0ea34d2e935d4e735d7e535d9e436dbe236dde037dfdf37e1dd37e3db38e5d938e7d739e9d539ebd339ecd13aeecf3aefcd3af1cb3af2c93af4c73af5c53af6c33af7c13af8be39f9bc39faba39fbb838fbb637fcb336fcb136fdae35fdac34fea933fea732fea431fea130fe9e2ffe9b2dfe992cfe962bfe932afe9029fd8d27fd8a26fc8725fc8423fb8122fb7e21fa7b1ff9781ef9751df8721cf76f1af66c19f56918f46617f36315f26014f15d13f05b12ef5811ed5510ec530feb500eea4e0de84b0ce7490ce5470be4450ae2430ae14109df3f08dd3d08dc3b07da3907d83706d63506d43305d23105d02f05ce2d04cc2b04ca2a04c82803c52603c32503c12302be2102bc2002b91e02b71d02b41b01b21a01af1801ac1701a91601a71401a41301a112019e10019b0f01980e01950d01920b018e0a018b09028808028507028106027e05027a0403';
function turboTexture() {
  const data = new Uint8Array(256 * 4);
  for (let i = 0; i < 256; i++) {
    for (let k = 0; k < 3; k++) data[i * 4 + k] = parseInt(TURBO.substr(i * 6 + k * 2, 2), 16);
    data[i * 4 + 3] = 255;
  }
  const tex = new THREE.DataTexture(data, 256, 1, THREE.RGBAFormat);
  tex.magFilter = THREE.LinearFilter;
  tex.minFilter = THREE.LinearFilter;
  tex.needsUpdate = true;
  return tex;
}

async function loadPart(base, version) {
  const man = await (await fetch(`${base}.json?v=${version}`)).json();
  const v = man.variants.distributed;
  const res = await fetch(new URL(v.file, new URL(`${base}.json`, location.href)));
  if (!res.ok) throw new Error(`part data: HTTP ${res.status}`);
  let buf = await res.arrayBuffer();
  const head = new Uint8Array(buf, 0, 2);
  if (head[0] === 0x1f && head[1] === 0x8b) {
    const stream = new Blob([buf]).stream().pipeThrough(new DecompressionStream('gzip'));
    buf = await new Response(stream).arrayBuffer();
  }
  const n = v.vertices, t = v.triangles, L = v.layout, lo = v.bounds_min, hi = v.bounds_max;
  const qp = new Uint16Array(buf, L.position, n * 3);
  const qn = new Int8Array(buf, L.normal, n * 4);
  const qd = new Int16Array(buf, L.disp, n * 3);
  const qs = new Uint16Array(buf, L.stress, n);
  const qe = new Uint8Array(buf, L.excluded, n);
  const pos = new Float32Array(n * 3), nrm = new Float32Array(n * 3), disp = new Float32Array(n * 3);
  const stress = new Float32Array(n), excl = new Float32Array(n);
  const dk = v.disp_max_mm / 32767, sk = v.stress_max_MPa / 65535;
  for (let i = 0; i < n; i++) {
    for (let k = 0; k < 3; k++) {
      pos[i * 3 + k] = lo[k] + (qp[i * 3 + k] / 65535) * (hi[k] - lo[k]);
      nrm[i * 3 + k] = qn[i * 4 + k] / 127;
      disp[i * 3 + k] = qd[i * 3 + k] * dk;
    }
    stress[i] = qs[i] * sk;
    excl[i] = qe[i];
  }
  const index = v.index_type === 'uint16' ? new Uint16Array(buf, L.index, t * 3) : new Uint32Array(buf, L.index, t * 3);
  return { man, v, pos, nrm, disp, stress, excl, index };
}

function arrow(color) {
  const g = new THREE.Group();
  const mat = new THREE.MeshBasicMaterial({ color });
  const shaft = new THREE.Mesh(new THREE.CylinderGeometry(1.1, 1.1, 1, 16), mat);
  const head = new THREE.Mesh(new THREE.ConeGeometry(3.6, 10, 24), mat);
  const hub = new THREE.Mesh(new THREE.SphereGeometry(3.2, 20, 14), mat);
  g.add(shaft, head, hub);
  g.userData = { shaft, head, hub };
  return g;
}

export async function start(root, ui) {
  const status = root.querySelector('.viewer-status');
  status.textContent = 'Loading the 3D part (540 KB)…';
  const P = await loadPart(root.dataset.part || 'data/cast-motor-mount', root.dataset.v || 1);
  const { man, v } = P;
  const n = v.vertices;

  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, powerPreference: 'high-performance' });
  renderer.setClearColor(0x000000, 0);
  const px = Math.min(window.devicePixelRatio || 1, 2);
  renderer.setPixelRatio(px);
  const canvas = renderer.domElement;
  canvas.setAttribute('role', 'img');
  canvas.tabIndex = 0;
  canvas.setAttribute('aria-label', 'Interactive 3D stress map of the cast motor mount. Drag or use the arrow keys to orbit; scroll, pinch or press plus and minus to zoom. The sliders change the load and the shape exaggeration.');
  root.prepend(canvas);

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(30, 1, 1, 5000);
  camera.up.set(0, 0, 1);

  const geo = new THREE.BufferGeometry();
  const live = new Float32Array(P.pos);
  const posAttr = new THREE.BufferAttribute(live, 3);
  posAttr.setUsage(THREE.DynamicDrawUsage);
  geo.setAttribute('position', posAttr);
  geo.setAttribute('normal', new THREE.BufferAttribute(P.nrm, 3));
  geo.setAttribute('stress', new THREE.BufferAttribute(P.stress, 1));
  geo.setAttribute('excl', new THREE.BufferAttribute(P.excl, 1));
  geo.setIndex(new THREE.BufferAttribute(P.index, 1));

  const material = new THREE.ShaderMaterial({
    vertexShader: VERT,
    fragmentShader: FRAG,
    side: THREE.DoubleSide,
    uniforms: {
      uScale: { value: 1 },
      uLimit: { value: man.allowable_MPa },
      uHatch: { value: 1 },
      uPx: { value: px },
      uLut: { value: turboTexture() },
    },
  });
  const mesh = new THREE.Mesh(geo, material);
  scene.add(mesh);

  // The base is bolted flat to a frame: show that plane.
  const lo = new THREE.Vector3(...v.bounds_min), hi = new THREE.Vector3(...v.bounds_max);
  const size = new THREE.Vector3().subVectors(hi, lo);
  const center = new THREE.Vector3().addVectors(lo, hi).multiplyScalar(0.5);
  const floor = new THREE.Mesh(
    new THREE.PlaneGeometry(size.x * 1.4, size.y * 1.4),
    new THREE.MeshBasicMaterial({ color: 0x15171a }),
  );
  floor.position.set(center.x, center.y, lo.z - 0.4);
  scene.add(floor);

  // The belt pull acts at the pulley, off the part.
  const load = man.loads[0];
  const pull = arrow(0xffd400);
  scene.add(pull);
  const pDir = new THREE.Vector3(...load.direction).normalize();
  const pAt = new THREE.Vector3(...load.point);

  // Governing point: highest stress outside the excluded zones.
  let peakIdx = 0;
  for (let i = 0, best = -1; i < n; i++) {
    if (P.excl[i] < 0.5 && P.stress[i] > best) { best = P.stress[i]; peakIdx = i; }
  }
  const peakVal = P.stress[peakIdx];

  const controls = new OrbitControls(camera, canvas);
  const reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;
  controls.enableDamping = !reduce;
  controls.dampingFactor = 0.12;
  controls.rotateSpeed = 0.8;
  const radius = size.length() / 2;
  controls.minDistance = radius * 1.3;
  controls.maxDistance = radius * 6;
  controls.target.copy(center).add(new THREE.Vector3(0, -6, -4));
  const home = new THREE.Vector3(0.62, 1.05, 0.62).normalize();
  function resetView() {
    const dist = radius / Math.sin(THREE.MathUtils.degToRad(camera.fov / 2)) * 1.0;
    camera.position.copy(controls.target).addScaledVector(home, dist);
    controls.update();
    requestRender();
  }

  const state = { scale: 1, exag: 60, hatch: true };
  const peakTag = root.querySelector('.peak-tag');
  const peakTagV = root.querySelector('.peak-tag-v');
  const loadTag = root.querySelector('.load-tag');
  const probe = root.querySelector('.probe');
  const tmp = new THREE.Vector3();

  function apply() {
    const f = state.scale * state.exag;
    const b = P.pos, d = P.disp;
    for (let i = 0; i < n * 3; i++) live[i] = b[i] + d[i] * f;
    posAttr.needsUpdate = true;
    geo.computeBoundingSphere();
    material.uniforms.uScale.value = state.scale;
    material.uniforms.uHatch.value = state.hatch ? 1 : 0;
    const len = 46 * state.scale;
    pull.visible = len > 0.5;
    const { shaft, head, hub } = pull.userData;
    pull.position.copy(pAt);
    pull.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), pDir);
    shaft.scale.set(1, Math.max(len - 10, 0.01), 1);
    shaft.position.set(0, Math.max(len - 10, 0) / 2, 0);
    head.position.set(0, Math.max(len - 10, 0) + 5, 0);
    hub.position.set(0, 0, 0);
    peakTagV.textContent = `${(peakVal * state.scale).toFixed(1)} MPa`;
    loadTag.textContent = `${Math.round(load.magnitude * state.scale).toLocaleString('en-US')} N belt pull`;
    if (probe.dataset.sticky) {
      probe.hidden = true;
      delete probe.dataset.sticky;
    }
    requestRender();
  }

  function place(el, p, w, h) {
    tmp.copy(p).project(camera);
    const vis = tmp.z < 1 && Math.abs(tmp.x) < 1.05 && Math.abs(tmp.y) < 1.05;
    el.hidden = !vis;
    if (vis) el.style.translate = `${((tmp.x + 1) / 2) * w}px ${((1 - tmp.y) / 2) * h}px`;
  }

  let vw = 1, vh = 1;
  function overlays() {
    const w = vw, h = vh;
    place(peakTag, new THREE.Vector3(live[peakIdx * 3], live[peakIdx * 3 + 1], live[peakIdx * 3 + 2]), w, h);
    if (pull.visible) place(loadTag, pAt, w, h); else loadTag.hidden = true;
  }

  let pending = false;
  function frame() {
    pending = false;
    const moving = controls.enableDamping && controls.update();
    renderer.render(scene, camera);
    overlays();
    if (moving) requestRender();
  }
  function requestRender() {
    if (!pending) { pending = true; requestAnimationFrame(frame); }
  }
  controls.addEventListener('change', requestRender);

  function resize() {
    const w = root.clientWidth, h = root.clientHeight;
    if (!w || !h) return;
    vw = w;
    vh = h;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    requestRender();
  }
  new ResizeObserver(resize).observe(root);

  // Read the stress under the pointer.
  const ray = new THREE.Raycaster();
  const ndc = new THREE.Vector2();
  const A = new THREE.Vector3(), B = new THREE.Vector3(), C = new THREE.Vector3(), bc = new THREE.Vector3();
  function probeAt(clientX, clientY, sticky) {
    const r = canvas.getBoundingClientRect();
    ndc.set(((clientX - r.left) / r.width) * 2 - 1, -((clientY - r.top) / r.height) * 2 + 1);
    ray.setFromCamera(ndc, camera);
    const hit = ray.intersectObject(mesh, false)[0];
    if (!hit) { probe.hidden = true; delete probe.dataset.sticky; return; }
    const { a, b, c } = hit.face;
    A.fromArray(live, a * 3); B.fromArray(live, b * 3); C.fromArray(live, c * 3);
    THREE.Triangle.getBarycoord(hit.point, A, B, C, bc);
    const s = (P.stress[a] * bc.x + P.stress[b] * bc.y + P.stress[c] * bc.z) * state.scale;
    const ex = P.excl[a] + P.excl[b] + P.excl[c] > 1.5;
    const sf = s > 1e-6 ? man.allowable_MPa / s : Infinity;
    probe.innerHTML = `<b>${s.toFixed(1)} MPa</b><br>${sf > 99 ? 'safety factor above 99' : `safety factor ${sf.toFixed(2)}`}` +
      (ex ? '<br><span class="x">excluded zone</span>' : '');
    const x = clientX - r.left, y = clientY - r.top;
    probe.style.left = `${x}px`;
    probe.style.top = `${y}px`;
    probe.style.transform = x > r.width - 200 ? 'translate(calc(-100% - 14px), -50%)' : '';
    probe.hidden = false;
    if (sticky) probe.dataset.sticky = '1'; else delete probe.dataset.sticky;
  }
  let hoverQueued = null;
  canvas.addEventListener('pointermove', (e) => {
    if (e.pointerType !== 'mouse' || e.buttons) return;
    if (!hoverQueued) requestAnimationFrame(() => { probeAt(hoverQueued.x, hoverQueued.y, false); hoverQueued = null; });
    hoverQueued = { x: e.clientX, y: e.clientY };
  });
  canvas.addEventListener('pointerleave', (e) => { if (e.pointerType === 'mouse') probe.hidden = true; });
  let down = null;
  canvas.addEventListener('pointerdown', (e) => { down = { x: e.clientX, y: e.clientY }; if (e.pointerType === 'mouse') probe.hidden = true; });
  canvas.addEventListener('pointerup', (e) => {
    if (down && Math.hypot(e.clientX - down.x, e.clientY - down.y) < 6) probeAt(e.clientX, e.clientY, true);
    down = null;
  });

  // Keyboard orbit: arrows rotate about the part, plus and minus zoom.
  const Z = new THREE.Vector3(0, 0, 1), off = new THREE.Vector3(), side = new THREE.Vector3();
  canvas.addEventListener('keydown', (e) => {
    const step = e.shiftKey ? 0.3 : 0.1;
    off.subVectors(camera.position, controls.target);
    if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') {
      off.applyAxisAngle(Z, e.key === 'ArrowLeft' ? -step : step);
    } else if (e.key === 'ArrowUp' || e.key === 'ArrowDown') {
      side.crossVectors(off, Z).normalize();
      const next = off.clone().applyAxisAngle(side, e.key === 'ArrowUp' ? step : -step);
      if (Math.abs(next.clone().normalize().z) < 0.97) off.copy(next);
    } else if (e.key === '+' || e.key === '=' || e.key === '-' || e.key === '_') {
      const len = off.length() * (e.key === '-' || e.key === '_' ? 1.12 : 1 / 1.12);
      off.setLength(Math.min(controls.maxDistance, Math.max(controls.minDistance, len)));
    } else {
      return;
    }
    e.preventDefault();
    camera.position.copy(controls.target).add(off);
    camera.lookAt(controls.target);
    controls.update();
    requestRender();
  });

  ui.onChange((s) => { Object.assign(state, s); apply(); });
  ui.onReset(resetView);
  Object.assign(state, ui.current());

  resize();
  resetView();
  apply();
  root.dataset.state = 'live';
  status.textContent = '';
}
