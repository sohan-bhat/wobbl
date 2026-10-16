// Renders the gallery's Cycles stills that are missing or out of date, into site/blender/stills: one WebP at the
// card's size per scene run and index.json, which keeps each still's key (scripts/still-key.mjs). Stills are
// drawn here, on a machine with Blender, and committed; the pages build only checks the keys and falls back to
// the stage's own still for a scene whose still is out of date.
//
//   node site/blender/stills.mjs [--all] [--samples 64] [rel ...]
//   BLENDER=/path/to/blender GLTFPACK="node .../gltfpack/cli.js" node site/blender/stills.mjs
import { spawnSync } from 'node:child_process';
import { existsSync, mkdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { syncScenes } from '../scripts/scenes.mjs';
import { exampleOf, stillKey } from '../scripts/still-key.mjs';
import { layout } from './still-scene.mjs';

const site = resolve(new URL('..', import.meta.url).pathname);
const dir = resolve(site, 'blender/stills');
const work = resolve(site, 'blender/.cache/stills');
const args = process.argv.slice(2);
const all = args.includes('--all');
const si = args.indexOf('--samples');
const samples = si >= 0 ? Number(args[si + 1]) : 64;
const only = args.filter((a, i) => !a.startsWith('--') && args[i - 1] !== '--samples');
const blender = process.env.BLENDER || 'blender';

mkdirSync(dir, { recursive: true });
mkdirSync(work, { recursive: true });
const indexPath = join(dir, 'index.json');
const index = existsSync(indexPath) ? JSON.parse(readFileSync(indexPath, 'utf-8')) : {};
const scenes = syncScenes(site);
const out = resolve(site, 'public/data/scenes');
const slug = (rel) => rel.replace(/[^a-z0-9]+/gi, '-');

let drawn = 0;
for (const [rel, e] of Object.entries(scenes)) {
  if (!e.file || !e.play || (only.length && !only.includes(rel))) continue;
  const bytes = readFileSync(join(out, e.file));
  const key = stillKey(site, rel, bytes, e.play);
  const file = `${slug(rel)}.webp`;
  if (!all && index[rel]?.key === key && existsSync(join(dir, file))) continue;
  const t0 = Date.now();
  const base = join(work, slug(rel));
  const { json, bin } = layout(bytes, e.play, 640, 480);
  writeFileSync(`${base}.json`, JSON.stringify(json));
  writeFileSync(`${base}.bin`, bin);
  const r = spawnSync(blender, ['-b', '--factory-startup', '--python-exit-code', '1', '-P', resolve(site, 'blender/still.py'), '--',
    base, exampleOf(rel), `${base}.png`, String(samples), join(dir, file)], { stdio: ['ignore', 'pipe', 'inherit'], encoding: 'utf-8' });
  if (r.status !== 0) { console.warn(`[stills] ${rel} failed`); continue; }
  index[rel] = { key, file };
  writeFileSync(indexPath, `${JSON.stringify(Object.fromEntries(Object.entries(index).sort()), null, 1)}\n`);
  drawn++;
  console.log(`[stills] ${rel} in ${((Date.now() - t0) / 1000).toFixed(0)} s`);
}
// a scene that is gone takes its still with it
for (const rel of Object.keys(index)) {
  if (scenes[rel]) continue;
  rmSync(join(dir, index[rel].file), { force: true });
  delete index[rel];
}
writeFileSync(indexPath, `${JSON.stringify(Object.fromEntries(Object.entries(index).sort()), null, 1)}\n`);
console.log(`[stills] drew ${drawn}`);
