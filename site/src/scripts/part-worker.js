// Fetch, gunzip and lay out the hero part off the main thread, then hand the arrays over without copying.
import { buildMesh } from './part.js';

self.onmessage = async (e) => {
  try {
    const { file, man } = e.data;
    const res = await fetch(file);
    if (!res.ok) throw new Error(`part data: HTTP ${res.status}`);
    let buf = await res.arrayBuffer();
    const head = new Uint8Array(buf, 0, 2);
    if (head[0] === 0x1f && head[1] === 0x8b) {
      buf = await new Response(new Blob([buf]).stream().pipeThrough(new DecompressionStream('gzip'))).arrayBuffer();
    }
    const mesh = buildMesh(man, buf);
    self.postMessage({ mesh }, Object.values(mesh.arrays).map((a) => a.buffer));
  } catch (err) {
    self.postMessage({ error: String(err && err.message ? err.message : err) });
  }
};
