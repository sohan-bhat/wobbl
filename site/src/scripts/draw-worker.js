// Draw the hero part in a worker, on the canvas the page handed over, so loading, compiling and
// drawing never hold up the page. The page sends the camera and the pull as they change; this draws
// the latest of them once per display frame.
import { createDrawer } from './part-draw.js';
import { buildMesh } from './part.js';

let drawer = null, latest = null, ack = false, queued = false;
const later = self.requestAnimationFrame ? (fn) => self.requestAnimationFrame(fn) : (fn) => setTimeout(fn, 0);

async function fetchMesh(file, man) {
  const res = await fetch(file);
  if (!res.ok) throw new Error(`part data: HTTP ${res.status}`);
  let buf = await res.arrayBuffer();
  const head = new Uint8Array(buf, 0, 2);
  if (head[0] === 0x1f && head[1] === 0x8b) {
    buf = await new Response(new Blob([buf]).stream().pipeThrough(new DecompressionStream('gzip'))).arrayBuffer();
  }
  return buildMesh(man, buf);
}

function draw() {
  queued = false;
  if (!drawer || !latest) return;
  drawer.draw(latest);
  if (ack) { ack = false; self.postMessage({ type: 'drawn' }); }
}

self.onmessage = async (e) => {
  const m = e.data;
  try {
    if (m.type === 'init') {
      const mesh = fetchMesh(m.file, m.man);
      drawer = await createDrawer({ canvas: m.canvas, man: m.man, mesh, w: m.w, h: m.h, px: m.px, pause: () => Promise.resolve() });
      self.postMessage({ type: 'ready' });
    } else if (m.type === 'frame') {
      latest = m.frame;
      ack = ack || m.ack;
      if (!queued) { queued = true; later(draw); }
    } else if (m.type === 'size') {
      drawer?.size(m.w, m.h, m.px);
    } else if (m.type === 'pick') {
      self.postMessage({ type: 'pick', id: m.id, rgba: drawer ? drawer.pick(m.x, m.y, m.vw, m.vh) : [0, 0, 0, 0] });
    } else if (m.type === 'dispose') {
      drawer?.dispose();
      drawer = null;
      self.close();
    }
  } catch (err) {
    self.postMessage({ type: 'error', message: String(err && err.message ? err.message : err) });
  }
};
