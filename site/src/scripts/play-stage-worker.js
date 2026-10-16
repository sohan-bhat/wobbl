// Draw a run's playback in a worker, on the canvas the page handed over, so fetching and laying out the
// part, compiling the shaders and every frame stay off the page's main thread, as draw-worker.js does
// for the landing page's part. The page sends the camera and the moment to show as they change; this
// draws the latest of them once per display frame.
import { createStage } from './play-draw.js';
import { decodePlay, decodeScene } from './play.js';
import { loadModels } from './play-models.js';

let stage = null, latest = null, queued = false, lit = null;
const parts = [];
const later = self.requestAnimationFrame ? (fn) => self.requestAnimationFrame(fn) : (fn) => setTimeout(fn, 16);

async function load(file, run) {
  const res = await fetch(file);
  if (!res.ok) throw new Error(`playback data: HTTP ${res.status}`);
  let buf = await res.arrayBuffer();
  const head = new Uint8Array(buf, 0, 2);
  if (head[0] === 0x1f && head[1] === 0x8b) {
    buf = await new Response(new Blob([buf]).stream().pipeThrough(new DecompressionStream('gzip'))).arrayBuffer();
  }
  return run.kind === 'scene' ? decodeScene(run, buf) : decodePlay(run, buf);
}

function draw() {
  queued = false;
  if (stage && latest) stage.frame(latest);
}

self.onmessage = async (e) => {
  const m = e.data;
  try {
    if (m.type === 'init') {
      stage = createStage({ canvas: m.canvas, w: m.w, h: m.h, px: m.px, kind: m.kind });
    } else if (m.type === 'part') {
      // the studio light comes from the site's env folder, two levels above the data the part comes from
      const root = new URL('../../', m.file).href;
      lit ||= stage?.light(new URL('env/studio.rgbe', root).href);
      // the Blender models, fetched alongside; the part waits a moment for them, then goes on without
      const models = loadModels(root, m.run.example).catch(() => null);
      const [{ arrays }] = await Promise.all([load(m.file, m.run), lit]);
      if (!stage) throw new Error('the drawing worker has no WebGL');
      if (parts[m.row]) stage.remove(parts[m.row]);
      const part = stage.add({ run: m.run, arrays, top: m.top, row: m.row });
      parts[m.row] = part;
      const dressed = models.then((x) => {
        stage?.dress(part, x);
        stage?.compile();
        if (latest && !queued) { queued = true; later(draw); }
      });
      await Promise.race([dressed, new Promise((ok) => setTimeout(ok, 2500))]);
      stage.compile();
      self.postMessage({ type: 'part', id: m.id });
    } else if (m.type === 'frame') {
      latest = m.frame;
      if (!queued) { queued = true; later(draw); }
    } else if (m.type === 'probe') {
      // the scene as last drawn, so the reading matches what is on screen
      self.postMessage({ type: 'probe', id: m.id, hit: stage ? stage.probe(m.row, m.origin, m.dir) : null });
    } else if (m.type === 'size') {
      stage?.size(m.w, m.h, m.px);
      if (latest && !queued) { queued = true; later(draw); }
    } else if (m.type === 'dispose') {
      stage?.dispose();
      stage = null;
      self.close();
    }
  } catch (err) {
    self.postMessage({ type: 'error', id: m.id, message: String(err && err.message ? err.message : err) });
  }
};
