// Draws a poster of every example scene at build time, in Wobbl's own viewer in headless Chrome, so the
// examples gallery shows each card at once and starts the live viewer only when someone reaches for it.
// Each poster is the card's opening view as a 640 x 480 webp, beside the slider's label, value and place
// the card shows under it. Posters are kept in .cache/posters by the hash of the scene and the viewer, so
// only a new or changed scene is drawn again. Without Chrome the cards fall back to drawing in the page.
import { createHash } from 'node:crypto';
import { copyFileSync, existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join, resolve } from 'node:path';

const VERSION = 1;                        // bump when the poster's look changes
const W = 320, H = 240, SCALE = 2;        // the card's stage in CSS pixels, drawn at twice that
// the card's quiet look, as scripts/scenes.ts gives it
const QUIET = '.wv{border:0}.st{aspect-ratio:4/3;min-height:0;max-height:none}';

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

/** One poster: the opening view drawn by the viewer, and what its slider says. */
async function draw(page, bytes) {
  return page.evaluate(async ({ b64, W, QUIET }) => {
    const host = document.createElement('div');
    host.style.width = `${W}px`;
    document.body.replaceChildren(host);
    const v = await window.WobblView.mount(host, { src: b64, still: true });
    if (!v) throw new Error('the scene did not load');
    const st = document.createElement('style'); st.textContent = QUIET; host.shadowRoot.append(st);
    await new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(() => requestAnimationFrame(r))));
    const r = host.shadowRoot, c = r.querySelector('.st canvas'), i = r.querySelector('.ctl input');
    const out = {
      img: c.toDataURL('image/webp', 0.82), label: r.querySelector('.ctl label span')?.textContent ?? '',
      value: r.querySelector('.ctl output')?.textContent ?? '', at: +i.max > 0 ? +i.value / +i.max : 0,
      one: r.querySelector('.trk')?.hidden ?? true,
    };
    v.dispose();
    return out;
  }, { b64: bytes.toString('base64'), W, QUIET });
}

/** Adds a poster to every scene in the manifest scripts/scenes.mjs wrote, drawing the ones not in the cache. */
export async function drawPosters(site) {
  if (process.env.WOBBL_POSTERS === '0') return;
  const out = resolve(site, 'public/data/scenes'), cache = resolve(site, '.cache/posters');
  const mf = join(out, 'manifest.json');
  if (!existsSync(mf)) return;
  const m = JSON.parse(readFileSync(mf, 'utf-8'));
  const js = readFileSync(join(out, m.viewer));
  mkdirSync(cache, { recursive: true });
  const todo = [];
  for (const [rel, e] of Object.entries(m.scenes)) {
    if (!e.file) continue;
    const bytes = readFileSync(join(out, e.file));
    const key = createHash('sha1').update(String(VERSION)).update(js).update(bytes).digest('hex').slice(0, 16);
    e.poster = e.file.replace(/\.wobbl$/, '.webp');
    todo.push({ rel, e, bytes, key });
  }
  const missing = todo.filter((t) => !existsSync(join(cache, `${t.key}.json`)));
  let b = null;
  if (missing.length) {
    b = await browser();
    if (!b) console.warn(`[posters] no Chrome found, so ${missing.length} cards draw their scene in the page instead`);
  }
  if (b) {
    const t0 = Date.now();
    try {
      const page = await b.newPage({ viewport: { width: W + 40, height: H + 200 }, deviceScaleFactor: SCALE });
      await page.setContent('<!doctype html><html><body style="margin:0;background:#0e0f11"></body></html>');
      await page.addScriptTag({ content: js.toString('utf-8') });
      for (const t of missing) {
        try {
          const p = await draw(page, t.bytes);
          writeFileSync(join(cache, `${t.key}.webp`), Buffer.from(p.img.split(',')[1], 'base64'));
          writeFileSync(join(cache, `${t.key}.json`), JSON.stringify({ label: p.label, value: p.value, at: p.at, one: p.one }));
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
    const info = join(cache, `${t.key}.json`);
    if (!existsSync(info)) { delete t.e.poster; continue; }
    copyFileSync(join(cache, `${t.key}.webp`), join(out, t.e.poster));
    Object.assign(t.e, { posterW: W * SCALE, posterH: H * SCALE, slider: JSON.parse(readFileSync(info, 'utf-8')) });
  }
  writeFileSync(mf, JSON.stringify(m, null, 1));
}

if (import.meta.url === `file://${process.argv[1]}`) {
  await drawPosters(resolve(process.argv[2] ?? '.'));
}
