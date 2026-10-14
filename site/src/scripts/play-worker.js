// Fetch, gunzip and lay out one run's playback file off the main thread, then hand the arrays over
// without copying.
import { decodePlay } from './play.js';

self.onmessage = async (e) => {
  const { id, file, run } = e.data;
  try {
    const res = await fetch(file);
    if (!res.ok) throw new Error(`playback data: HTTP ${res.status}`);
    let buf = await res.arrayBuffer();
    const head = new Uint8Array(buf, 0, 2);
    if (head[0] === 0x1f && head[1] === 0x8b) {
      buf = await new Response(new Blob([buf]).stream().pipeThrough(new DecompressionStream('gzip'))).arrayBuffer();
    }
    const out = decodePlay(run, buf);
    self.postMessage({ id, ...out }, Object.values(out.arrays).map((a) => a.buffer));
  } catch (err) {
    self.postMessage({ id, error: String(err && err.message ? err.message : err) });
  }
};
