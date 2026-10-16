// The key a scene's Cycles gallery still is kept under (site/blender/stills/index.json): everything the picture
// depends on and nothing else, so a still is drawn again exactly when its scene, its framing, its context models
// or the way stills are rendered change. scripts/posters.mjs checks it at build time and site/blender/stills.mjs
// uses it to find the stills to redraw.
import { createHash } from 'node:crypto';
import { existsSync, readFileSync } from 'node:fs';
import { join, resolve } from 'node:path';

export const STILL_VERSION = 1;

/** The example key the models manifest uses: the run folder without its trailing output part. */
export const exampleOf = (rel) => rel.replace(/\/output(\/.*)?$/, '');

/** The parts of a run's play summary that move the camera or the objects in the still. */
const framing = (p) => ({
  input: { t: p.input.t, home: p.input.home, between: p.input.between }, camera: p.camera, bounds: p.bounds, subject: p.subject,
  points: p.points, follow: p.follow, hold: p.hold, magnify: p.magnify, exag: p.exag,
});

export function stillKey(site, rel, sceneBytes, play) {
  const h = createHash('sha1').update(String(STILL_VERSION)).update(sceneBytes).update(JSON.stringify(framing(play)));
  for (const f of ['blender/still.py', 'blender/still-scene.mjs']) h.update(readFileSync(resolve(site, f)));
  const mf = resolve(site, 'public/models/manifest.json');
  const entry = existsSync(mf) ? JSON.parse(readFileSync(mf, 'utf-8')).examples?.[exampleOf(rel)] : null;
  if (entry) {
    h.update(JSON.stringify(entry));
    for (const e of [...(entry.replace || []), ...(entry.add || [])]) {
      const p = join(resolve(site, 'public/models'), e.file);
      if (existsSync(p)) h.update(readFileSync(p));
    }
  }
  return h.digest('hex').slice(0, 16);
}
