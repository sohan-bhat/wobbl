// Fetch, gunzip and lay out one run's playback file (or its scene) off the main thread, then hand the arrays over
// without copying.
import { loadRun, transfers } from './play.js';

self.onmessage = async (e) => {
  const { id, file, run } = e.data;
  try {
    const out = await loadRun(file, run);
    if (run.kind === 'scene') {
      self.postMessage({ id, ...out }, transfers(out.arrays));
      return;
    }
    self.postMessage({ id, ...out }, Object.values(out.arrays).map((a) => a.buffer));
  } catch (err) {
    self.postMessage({ id, error: String(err && err.message ? err.message : err) });
  }
};
