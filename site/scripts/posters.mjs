// Draws a still of every scene run's view at build time, in headless Chrome, with the stage the example
// page plays the scene on (src/scripts/play-draw.js through src/scripts/scene-poster.js, bundled here with
// esbuild). Each still is the view the page opens on once its replay has run: the input at the value the
// question asked about, the camera where the page puts it. The page shows the large one until its 3D view
// takes over and the gallery's card shows the small one, as a plain picture. Stills are kept in
// .cache/posters by the hash of the scene, its summary and the drawing code, so only a new or changed
// scene is drawn again. Without Chrome the page starts on its live view and the card on Wobbl's own image.
//
// The gallery's card is the photoreal Cycles still of the same view when one is committed for the scene in
// site/blender/stills and its key (scripts/still-key.mjs) still matches the scene, its framing and its context
// models. A scene whose still is out of date keeps the stage's card until site/blender/stills.mjs draws it again.
import { createHash } from 'node:crypto';
import { copyFileSync, existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { stillKey } from './still-key.mjs';

const VERSION = 3;                        // bump when the stills' look changes outside the drawing code
const W = 640, H = 480, PX = 2;           // the page's view in CSS pixels, drawn at twice that
const SIZES = [[W * PX, H * PX], [W, H]]; // the page's still, then the card's

async function browser() {
  let chromium;
  try { ({ chromium } = await import('playwright-core')); } catch { return null; }
  const args = ['--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'];
  const tries = [{ channel: 'chrome' }, {}];
  if (process.env.WOBBL_CHROME) tries.unshift({ executablePath: process.env.WOBBL_CHROME });
  for (const t of tries) {
    try { return await chromium.launch({ ...t, args }); } catch { /* the next way to find Chrome */ }
  }
  return null;
}

/** The drawing code as one script for the page Chrome opens. */
async function bundle(site) {
  const { build } = await import('esbuild');
  const r = await build({
    entryPoints: [resolve(site, 'src/scripts/scene-poster.js')], bundle: true, format: 'iife', write: false,
    minify: true, logLevel: 'silent', absWorkingDir: site,
  });
  return r.outputFiles[0].text;
}

/** Adds stills to every scene in the manifest scripts/scenes.mjs wrote, drawing the ones not in the cache. */
export async function drawPosters(site) {
  if (process.env.WOBBL_POSTERS === '0') return;
  const out = resolve(site, 'public/data/scenes'), cache = resolve(site, '.cache/posters');
  const mf = join(out, 'manifest.json');
  if (!existsSync(mf)) return;
  const m = JSON.parse(readFileSync(mf, 'utf-8'));
  // the page's studio light, so the stage's stills are lit as the live view is
  const envPath = resolve(site, 'public/env/studio.rgbe');
  const env = existsSync(envPath) ? readFileSync(envPath).toString('base64') : null;
  let js;
  try { js = await bundle(site); } catch (err) { console.warn(`[posters] could not bundle the drawing code: ${err.message ?? err}`); return; }
  mkdirSync(cache, { recursive: true });
  const todo = [];
  for (const [rel, e] of Object.entries(m.scenes)) {
    if (!e.file || !e.play) continue;
    const bytes = readFileSync(join(out, e.file));
    const key = createHash('sha1').update(String(VERSION)).update(js).update(env ?? '').update(bytes).update(JSON.stringify(e.play)).digest('hex').slice(0, 16);
    todo.push({ rel, e, bytes, key });
  }
  const missing = todo.filter((t) => !existsSync(join(cache, `${t.key}.webp`)) || !existsSync(join(cache, `${t.key}.card.webp`)));
  let b = null;
  if (missing.length) {
    b = await browser();
    if (!b) console.warn(`[posters] no Chrome found, so ${missing.length} scenes start without a still`);
  }
  if (b) {
    const t0 = Date.now();
    try {
      const page = await b.newPage({ viewport: { width: W, height: H }, deviceScaleFactor: 1 });
      page.on('pageerror', (err) => console.warn(`[posters] ${err.message}`));
      await page.setContent('<!doctype html><html><body style="margin:0;background:#0e0f11"></body></html>');
      await page.addScriptTag({ content: js });
      for (const t of missing) {
        try {
          const [big, small] = await page.evaluate((a) => globalThis.drawScenePoster(a),
            { b64: t.bytes.toString('base64'), run: t.e.play, w: W, h: H, px: PX, sizes: SIZES, env });
          writeFileSync(join(cache, `${t.key}.webp`), Buffer.from(big.split(',')[1], 'base64'));
          writeFileSync(join(cache, `${t.key}.card.webp`), Buffer.from(small.split(',')[1], 'base64'));
        } catch (err) {
          console.warn(`[posters] ${t.rel}: ${err.message ?? err}`);
        }
      }
    } finally {
      await b.close();
    }
    console.log(`[posters] drew ${missing.length} in ${((Date.now() - t0) / 1000).toFixed(1)} s`);
  }
  for (const t of todo) {
    if (!existsSync(join(cache, `${t.key}.webp`))) continue;
    t.e.poster = t.e.file.replace(/\.wobbl$/, '.webp');
    t.e.card = t.e.file.replace(/\.wobbl$/, '.card.webp');
    copyFileSync(join(cache, `${t.key}.webp`), join(out, t.e.poster));
    copyFileSync(join(cache, `${t.key}.card.webp`), join(out, t.e.card));
    Object.assign(t.e, { posterW: SIZES[0][0], posterH: SIZES[0][1] });
  }
  // the Cycles stills for the gallery's cards, where they are up to date
  const stills = resolve(site, 'blender/stills'), indexPath = join(stills, 'index.json');
  const index = existsSync(indexPath) ? JSON.parse(readFileSync(indexPath, 'utf-8')) : {};
  const stale = [];
  for (const t of todo) {
    if (!t.e.poster) continue;
    const s = index[t.rel];
    if (s && existsSync(join(stills, s.file)) && s.key === stillKey(site, t.rel, t.bytes, t.e.play)) {
      t.e.card = t.e.file.replace(/\.wobbl$/, '.still.webp');
      copyFileSync(join(stills, s.file), join(out, t.e.card));
    } else stale.push(t.rel);
  }
  if (stale.length) console.warn(`[posters] ${stale.length} Cycles stills are out of date, so their cards show the stage's still: ${stale.join(', ')}. node site/blender/stills.mjs draws them again.`);
  writeFileSync(mf, JSON.stringify(m));
}

if (import.meta.url === `file://${process.argv[1]}`) {
  await drawPosters(resolve(process.argv[2] ?? '.'));
}
