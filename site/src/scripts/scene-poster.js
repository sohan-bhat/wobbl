// The still of a scene run's view, drawn at build time by scripts/posters.mjs in headless Chrome with the
// very stage the example page plays the scene on (play-draw.js), at the place its replay ends: the input at
// the value the question asked about, the camera where the page puts it. scripts/posters.mjs bundles this
// file on its own; the pages never load it.
import { createStage } from './play-draw.js';
import { decodeScene } from './play.js';
import { Orbit, corners, fitTo } from './orbit.js';

async function gunzip(bytes) {
  if (bytes[0] !== 0x1f || bytes[1] !== 0x8b) return bytes.buffer;
  return new Response(new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'))).arrayBuffer();
}

/** Draw a scene's opening view at w by h CSS pixels, px device pixels each, and return it as WebP data URLs. */
globalThis.drawScenePoster = async ({ b64, run, w, h, px, sizes }) => {
  const bin = atob(b64), bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  const { arrays } = decodeScene(run, await gunzip(bytes));
  const canvas = document.createElement('canvas');
  document.body.replaceChildren(canvas);
  const stage = createStage({ canvas, w, h, px, kind: 'scene' });
  stage.add({ run, arrays, top: 0, row: 0 });
  stage.compile();
  const { lo, hi } = run.bounds;
  const radius = Math.hypot(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]) / 2;
  const target = [0, 1, 2].map((i) => (lo[i] + hi[i]) / 2);
  const v = run.camera.view;
  const orbit = new Orbit({ target, az: Math.atan2(v[1], v[0]), el: Math.asin(Math.max(-1, Math.min(1, v[2]))), dist: radius * 4,
    fov: run.camera.fov, aspect: w / h, near: radius / 50, far: radius * 40, minDist: radius * 1.1, maxDist: radius * 8 });
  fitTo(orbit, corners(lo, hi), radius * 1.2);
  const T = run.input.t, p = run.input.home, a = Math.floor(p), b = Math.min(a + 1, T.length - 1);
  const k = run.input.between === 'step' ? 0 : p - a;
  stage.frame({
    camera: { position: orbit.eye, target: orbit.target, fov: orbit.fov, aspect: orbit.aspect },
    views: [[0, 0, w, h]], k: 1, radius,
    scene: { a, b, k, t: k ? T[a] + (T[b] - T[a]) * k : T[a], bend: true, mag: true },
  });
  // onto the page's own background, at each size asked for, straight after the frame is drawn
  const out = sizes.map((s) => {
    const c = document.createElement('canvas');
    c.width = s[0]; c.height = s[1];
    const g = c.getContext('2d');
    g.fillStyle = '#0e0f11';
    g.fillRect(0, 0, s[0], s[1]);
    g.imageSmoothingQuality = 'high';
    g.drawImage(canvas, 0, 0, s[0], s[1]);
    return c.toDataURL('image/webp', 0.86);
  });
  stage.dispose();
  return out;
};
