// Copies every example's scene.wobbl into public/data/scenes with a content hash in its name, bundles
// Wobbl's own viewer (src/wobbl/viz/viewer, the one run reports embed) beside them, and writes a manifest
// the pages read at build time. astro.config.mjs runs it before dev and build, so newer scenes are picked
// up on their own. The folder is generated and ignored by git.
import { createHash } from 'node:crypto';
import { existsSync, mkdirSync, readFileSync, readdirSync, rmSync, statSync, writeFileSync } from 'node:fs';
import { join, relative, resolve } from 'node:path';
import { gunzipSync } from 'node:zlib';

const hash = (b) => createHash('sha1').update(b).digest('hex').slice(0, 10);

/** Every scene.wobbl under examples/, by the run folder that holds it (shelf-bracket/output and so on). */
function find(dir, out = []) {
  for (const name of readdirSync(dir)) {
    if (name === 'work' || name.startsWith('.')) continue;
    const p = join(dir, name);
    const s = statSync(p);
    if (s.isDirectory()) find(p, out);
    else if (name === 'scene.wobbl') out.push(p);
  }
  return out;
}

/** The header of a scene, read the way the viewer reads it, or the reason it cannot be read. */
function header(bytes) {
  let raw = bytes[0] === 0x1f && bytes[1] === 0x8b ? gunzipSync(bytes) : bytes;
  if (raw.subarray(0, 4).toString('latin1') !== 'WBSC') throw new Error('not a Wobbl scene');
  const n = raw.readUInt32LE(8);
  return JSON.parse(raw.subarray(12, 12 + n).toString('utf-8'));
}

/** The viewer as one script, put together the way src/wobbl/viz/embed.py does it. */
function viewer(root) {
  const dir = resolve(root, 'src/wobbl/viz/viewer');
  const head = "(() => {\n'use strict';\nconst W = globalThis.WobblView = globalThis.WobblView || {};\nif (W.mount) return;\n";
  return head + ['core.js', 'objects.js', 'mount.js'].map((p) => readFileSync(join(dir, p), 'utf-8')).join('\n') + '})();\n';
}

export function syncScenes(site) {
  const root = resolve(site, '..');
  const out = resolve(site, 'public/data/scenes');
  rmSync(out, { recursive: true, force: true });
  mkdirSync(out, { recursive: true });
  const js = viewer(root);
  const viewerFile = `viewer.${hash(js)}.js`;
  writeFileSync(join(out, viewerFile), js);
  const scenes = {};
  const ex = resolve(root, 'examples');
  for (const p of existsSync(ex) ? find(ex) : []) {
    const rel = relative(ex, resolve(p, '..')).split('\\').join('/');
    const bytes = readFileSync(p);
    try {
      const h = header(bytes);
      const file = `${rel.replace(/[^a-z0-9]+/gi, '-')}.${hash(bytes)}.wobbl`;
      writeFileSync(join(out, file), bytes);
      scenes[rel] = { file, bytes: bytes.length, title: h.title ?? '', frames: h.timeline?.frames ?? 1,
        verdict: h.verdict ?? null, domain: h.domain ?? '' };
    } catch (e) {
      scenes[rel] = { broken: String(e.message ?? e) };
    }
  }
  writeFileSync(join(out, 'manifest.json'), JSON.stringify({ viewer: viewerFile, scenes }, null, 1));
  return scenes;
}

if (import.meta.url === `file://${process.argv[1]}`) {
  const s = syncScenes(resolve(process.argv[2] ?? '.'));
  for (const [rel, v] of Object.entries(s)) console.log(rel, v.broken ? `BROKEN ${v.broken}` : `${v.file} ${v.frames} frames`);
}
