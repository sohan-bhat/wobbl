// Draws the stills of the example pages' live views at build time, in headless Chrome, with the stage, the studio
// light, the Blender models and the camera the live view itself uses (src/scripts/run-poster.js, bundled here with
// esbuild). A page's still is the first frame its live view shows, so the view takes over from it without a jump:
// a strength or heat part before its load or its clock starts, a scene at the start of its input. Every view on
// every page has its stills (src/lib/figures.ts lists them, as the pages build them), each pane drawn in the
// view's shape on a wide and on a narrow screen, since the panes of a two part view sit side by side on one and
// stacked on the other. A scene also has a card, its view once its replay has run, for the gallery and for a page
// that cannot run the 3D view. Everything is kept in .cache/posters by the hash of what it is drawn from: the run's
// files, the studio, the models and the drawing code, so only a new or changed view is drawn again. Without Chrome
// the pages show Wobbl's own heatmap or no still.
//
// The gallery's card is the photoreal Cycles still of the same view when one is committed for the scene in
// site/blender/stills and its key (scripts/still-key.mjs) still matches the scene, its framing and its context
// models. A scene whose still is out of date keeps the stage's card until site/blender/stills.mjs draws it again.
import { createHash } from 'node:crypto';
import { copyFileSync, existsSync, mkdirSync, readFileSync, readdirSync, rmSync, writeFileSync } from 'node:fs';
import { extname, join, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { stillKey } from './still-key.mjs';

const VERSION = 4;                        // bump when the stills' look changes outside the drawing code
const QUALITY = 0.86;                     // WebP quality
// the view in CSS pixels as the example pages lay it out in a 1280 and a 375 pixel wide window
// (styles/live-run.css), one pane at 4:3 or two side by side at 2:1 and stacked at 4:5, and the device pixels to a
// CSS pixel it is drawn at: two panes side by side a little under twice, so each stays about as large a file
const LAYOUTS = { wide: { 1: [566, 424, 2], 2: [1182, 590, 1.6] }, narrow: { 1: [341, 255, 2], 2: [341, 427, 2] } };
const CARD = [640, 480, 2];               // a scene's card, drawn at twice that and halved
const ORIGIN = 'https://wobbl.local';     // where the drawing page finds the site's public files
const TYPES = { '.json': 'application/json', '.rgbe': 'application/octet-stream', '.glb': 'model/gltf-binary', '.webp': 'image/webp' };

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
    entryPoints: [resolve(site, 'src/scripts/run-poster.js')], bundle: true, format: 'iife', write: false,
    minify: true, logLevel: 'silent', absWorkingDir: site,
  });
  return r.outputFiles[0].text;
}

/** Every live view on the example pages, as src/lib/figures.ts lists them, read with the pages' own code. */
async function listFigures(site, base) {
  const { build } = await import('esbuild');
  const outfile = resolve(site, '.cache/figures.mjs');
  await build({
    entryPoints: [resolve(site, 'src/lib/figures.ts')], bundle: true, platform: 'node', format: 'esm', outfile,
    logLevel: 'silent', absWorkingDir: site, define: { 'import.meta.env.BASE_URL': JSON.stringify(base) },
  });
  const was = process.cwd();
  process.chdir(site); // the pages' code reads the site from its working directory, as Astro runs it
  try {
    return (await import(`${pathToFileURL(outfile).href}?${Date.now()}`)).figures();
  } finally {
    process.chdir(was);
  }
}

/** A digest of the models the manifest lists for some examples, so their stills are drawn again when one changes. */
function modelsDigest(site, examples) {
  const mf = resolve(site, 'public/models/manifest.json');
  const h = createHash('sha1');
  const all = existsSync(mf) ? JSON.parse(readFileSync(mf, 'utf-8')).examples ?? {} : {};
  for (const ex of [...new Set(examples)].sort()) {
    const entry = all[ex];
    if (!entry) continue;
    h.update(ex).update(JSON.stringify(entry));
    for (const e of [...(entry.replace || []), ...(entry.add || [])]) {
      const p = resolve(site, 'public/models', e.file);
      if (existsSync(p)) h.update(readFileSync(p));
    }
  }
  return h.digest('hex');
}

const sha = (...parts) => { const h = createHash('sha1'); for (const p of parts) h.update(p); return h.digest('hex').slice(0, 16); };
const bytesOf = (url) => Buffer.from(url.split(',')[1], 'base64');

/** Adds stills to every view and a card to every scene, drawing the ones not in the cache. */
export async function drawPosters(site, base = '/wobbl/') {
  if (process.env.WOBBL_POSTERS === '0') return;
  base = base.endsWith('/') ? base : `${base}/`;
  const pub = resolve(site, 'public'), out = resolve(pub, 'data/scenes'), cache = resolve(site, '.cache/posters');
  const mf = join(out, 'manifest.json');
  if (!existsSync(mf)) return;
  const m = JSON.parse(readFileSync(mf, 'utf-8'));
  const envPath = resolve(pub, 'env/studio.rgbe');
  const env = existsSync(envPath) ? readFileSync(envPath) : Buffer.alloc(0);
  let js, figs;
  try {
    js = await bundle(site);
    figs = await listFigures(site, base);
  } catch (err) {
    console.warn(`[posters] could not bundle the drawing code: ${err.message ?? err}`);
    return;
  }
  mkdirSync(cache, { recursive: true });
  // the site's file for a link under the base path
  const fileOf = (link) => resolve(pub, link.replace(base, '').replace(/^\//, ''));
  const common = [String(VERSION), js, env, JSON.stringify({ LAYOUTS, CARD, QUALITY })];

  // each view's stills, and the card of each scene whose Cycles still (blender/stills) is missing or out of date
  const cycles = resolve(site, 'blender/stills'), indexPath = join(cycles, 'index.json');
  const ci = existsSync(indexPath) ? JSON.parse(readFileSync(indexPath, 'utf-8')) : {};
  const jobs = [];
  for (const f of figs) {
    const files = f.panes.map((p) => readFileSync(fileOf(p.file)));
    const key = sha(...common, modelsDigest(site, f.panes.map((p) => p.run.example)), JSON.stringify(f), ...files);
    jobs.push({ type: 'view', name: f.id, f, key });
  }
  for (const [rel, e] of Object.entries(m.scenes)) {
    if (!e.file || !e.play) continue;
    const bytes = readFileSync(join(out, e.file)), s = ci[rel];
    if (s && existsSync(join(cycles, s.file)) && s.key === stillKey(site, rel, bytes, e.play)) {
      e.card = e.file.replace(/\.wobbl$/, '.still.webp');
      copyFileSync(join(cycles, s.file), join(out, e.card));
      continue;
    }
    const key = sha(...common, 'card', modelsDigest(site, [e.play.example]), bytes, JSON.stringify(e.play));
    jobs.push({ type: 'card', name: rel, e, key });
  }
  const done = (j) => existsSync(join(cache, `${j.key}.json`));
  const missing = jobs.filter((j) => !done(j));
  let b = null;
  if (missing.length) {
    b = await browser();
    if (!b) console.warn(`[posters] no Chrome found, so ${missing.length} views start without a still`);
  }
  if (b) {
    const t0 = Date.now();
    try {
      const page = await b.newPage({ viewport: { width: 1280, height: 960 }, deviceScaleFactor: 1 });
      page.on('pageerror', (err) => console.warn(`[posters] ${err.message}`));
      // the site's public folder under the base path, as the pages fetch the runs, the studio and the models
      await page.route(`${ORIGIN}/**`, (route) => {
        const path = decodeURIComponent(new URL(route.request().url()).pathname);
        const file = path.startsWith(base) ? resolve(pub, path.slice(base.length)) : null;
        if (!file || !file.startsWith(pub) || !existsSync(file)) {
          return route.fulfill(path === `${base}poster` ? { contentType: 'text/html', body: `<!doctype html><body style="margin:0;background:#0e0f11"></body>` } : { status: 404, body: '' });
        }
        return route.fulfill({ body: readFileSync(file), contentType: TYPES[extname(file)] ?? 'application/octet-stream' });
      });
      await page.goto(`${ORIGIN}${base}poster`);
      await page.addScriptTag({ content: js });
      const draw = (a) => page.evaluate((x) => globalThis.drawRunStills(x), { base: `${ORIGIN}${base}`, quality: QUALITY, ...a });
      const save = (url) => {
        const buf = bytesOf(url), name = `${sha(buf)}.webp`;
        writeFileSync(join(cache, name), buf);
        return name;
      };
      for (const j of missing) {
        try {
          if (j.type === 'view') {
            const n = j.f.fix > 0 ? 2 : 1, panes = j.f.panes.map((p) => ({ file: p.file, run: p.run, home: p.home }));
            const shot = {};
            for (const [name, sizes] of Object.entries(LAYOUTS)) {
              const [w, h, px] = sizes[n];
              shot[name] = (await draw({ kind: j.f.kind, panes, fix: j.f.fix, w, h, px })).map((u) => u && save(u));
            }
            writeFileSync(join(cache, `${j.key}.json`), JSON.stringify(shot));
          } else {
            const [card] = await draw({ kind: 'scene', panes: [{ file: `data/scenes/${j.e.file}`, run: j.e.play }], fix: -1, w: CARD[0], h: CARD[1], px: CARD[2], at: 'home', scale: 1 / CARD[2] });
            writeFileSync(join(cache, `${j.key}.json`), JSON.stringify({ card: save(card) }));
          }
        } catch (err) {
          console.warn(`[posters] ${j.name}: ${err.message ?? err}`);
        }
      }
    } finally {
      await b.close();
    }
    console.log(`[posters] drew ${missing.length} in ${((Date.now() - t0) / 1000).toFixed(1)} s`);
  }

  // the views' stills, for the pages (src/lib/figures.ts stillsOf)
  const stills = resolve(pub, 'data/stills');
  rmSync(stills, { recursive: true, force: true });
  mkdirSync(stills, { recursive: true });
  const index = {};
  const put = (name) => { copyFileSync(join(cache, name), join(stills, name)); return name; };
  for (const j of jobs) {
    if (!done(j)) continue;
    if (j.type === 'card') continue;
    const shot = JSON.parse(readFileSync(join(cache, `${j.key}.json`), 'utf-8'));
    const n = j.f.fix > 0 ? 2 : 1;
    // each pane's size in pixels, as run-poster.js cuts it out of the view (run-view.js paneViews)
    const dims = (name) => {
      const [w, h, px] = LAYOUTS[name][n], [pw, ph] = n === 1 ? [w, h] : w >= h ? [w / 2, h] : [w, h / 2];
      return [Math.round(pw * px), Math.round(ph * px)];
    };
    index[j.name] = j.f.panes.map((_, i) => (shot.wide[i] && shot.narrow[i]
      ? { wide: put(shot.wide[i]), narrow: put(shot.narrow[i]), w: dims('wide')[0], h: dims('wide')[1], nw: dims('narrow')[0], nh: dims('narrow')[1] }
      : null));
  }
  writeFileSync(join(stills, 'index.json'), JSON.stringify(index));
  // what no view or card draws from any more leaves the cache
  const keep = new Set(jobs.filter(done).flatMap((j) => [`${j.key}.json`, ...Object.values(JSON.parse(readFileSync(join(cache, `${j.key}.json`), 'utf-8'))).flat()]));
  for (const f of readdirSync(cache)) if (!keep.has(f)) rmSync(join(cache, f), { force: true });

  // the stage's cards of the scenes whose Cycles still is out of date, beside the scene each was drawn from
  const stale = [];
  for (const j of jobs) {
    if (j.type !== 'card' || !done(j)) continue;
    j.e.card = j.e.file.replace(/\.wobbl$/, '.card.webp');
    copyFileSync(join(cache, JSON.parse(readFileSync(join(cache, `${j.key}.json`), 'utf-8')).card), join(out, j.e.card));
    stale.push(j.name);
  }
  if (stale.length) console.warn(`[posters] ${stale.length} Cycles stills are out of date, so their cards show the stage's still: ${stale.join(', ')}. node site/blender/stills.mjs draws them again.`);
  writeFileSync(mf, JSON.stringify(m));
}

if (import.meta.url === `file://${process.argv[1]}`) {
  await drawPosters(resolve(process.argv[2] ?? '.'), process.argv[3]);
}
