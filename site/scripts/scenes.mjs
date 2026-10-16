// Copies every example's scene.wobbl into public/data/scenes with a content hash in its name and writes a
// manifest the pages read at build time. Each entry carries what the example page shows under its view, in
// the shape the strength and heat playback uses (lib/playback.ts): the input the slider sweeps and the value
// the question asked about, the reading the meter follows with its target and limit, where the main field
// peaks at every saved frame and the box the camera holds. The drawing reads the scene file itself, in the
// page's drawing worker (scripts/play.js). astro.config.mjs runs this before dev and build, so newer scenes are
// picked up on their own. The folder is generated and ignored by git.
import { createHash } from 'node:crypto';
import { existsSync, mkdirSync, readFileSync, readdirSync, rmSync, statSync, writeFileSync } from 'node:fs';
import { join, relative, resolve } from 'node:path';
import { gunzipSync } from 'node:zlib';
import { autoMagnify, fieldPeaks, readScene, sceneBounds, sceneFraming, sig } from '../src/scripts/scene-file.js';

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

/** Three significant figures without trailing zeros, as the strength meter's ticks: 165, 82.5, 6.1. */
const short = (x) => Number(x.toPrecision(3)).toLocaleString('en-US', { maximumFractionDigits: 6 });
const r6 = (x) => (Number.isFinite(x) ? +x.toPrecision(6) : null);
const cap = (s) => (s ? s[0].toUpperCase() + s.slice(1) : s);

/** What the example page shows for a scene, worked out once from its header and arrays. */
export function scenePlay(h, arr) {
  // small moving things grown so they show, worked out first since the peaks and the box depend on it
  h.magnify = autoMagnify(h, arr, sceneBounds(h, arr));
  const tl = h.timeline, F = tl.frames, t = tl.t;
  const tPos = (x) => { for (let i = 0; i < F - 1; i++) if (x <= t[i + 1]) return Math.min(Math.max(i + (x - t[i]) / ((t[i + 1] - t[i]) || 1), i), i + 1); return F - 1; };
  let home = tl.design !== undefined ? tPos(tl.design) : F - 1;
  if (tl.between !== 'exact') home = Math.round(home);
  const peaks = fieldPeaks(h, arr);
  const f0 = h.fields[0];
  const sameAsInput = (r) => r.name === tl.label || r.values.every((v, i) => Math.abs(v - (t[i] ?? NaN)) <= 1e-4 * Math.max(1, Math.abs(t[i])));
  const varies = (vs) => { const f = vs.filter(Number.isFinite); return f.length > 1 && Math.max(...f) - Math.min(...f) > 1e-6 * Math.max(1, ...f.map(Math.abs)); };
  // the reading the meter follows: the main field's peak when it has a limit, else a readout with a limit,
  // else the first readout that is not the input, else the field's peak
  let q = null;
  const fromField = (f) => ({
    name: /^(peak|max|largest|hottest)\b/i.test(f.name) ? f.name : `peak ${f.name}`, unit: f.unit || '', digits: f.digits ?? 3, values: peaks.map((p) => r6(p ? p.v : NaN)),
    lo: f.range[0], limit: f.limit ?? null, target: f.limit !== undefined ? f.target ?? f.limit * 0.8 : null,
    limitLabel: f.limit_label && f.limit_label.length <= 16 ? f.limit_label : 'limit', named: f.target !== undefined, field: true,
  });
  const fromReadout = (r) => ({
    name: r.name, unit: r.unit || '', digits: r.digits ?? 3, values: r.values.map(r6), lo: Math.min(0, ...r.values.filter(Number.isFinite)),
    limit: r.limit ?? null, target: r.target ?? (r.limit !== undefined ? r.limit * 0.8 : null), limitLabel: 'limit', named: r.target !== undefined, field: false,
  });
  if (f0 && f0.limit !== undefined && peaks && peaks.some(Boolean)) q = fromField(f0);
  else {
    const lim = h.readouts.find((r) => r.limit !== undefined && !sameAsInput(r));
    const big = h.readouts.find((r) => !sameAsInput(r));
    if (lim) q = fromReadout(lim);
    else if (big) q = fromReadout(big);
    else if (f0 && peaks && peaks.some(Boolean)) q = fromField(f0);
  }
  if (q) {
    const vs = q.values.filter(Number.isFinite);
    const hi = Math.max(q.limit ?? -Infinity, ...vs);
    q.top = q.limit !== null ? q.limit * 1.25 : hi + 0.12 * (hi - q.lo || Math.abs(hi) || 1);
    if (q.top <= q.lo) q.top = q.lo + 1;
    q.marks = [];
    // a target the scene named gets its tick; the 80% one it falls back to only colors the verdict
    if (q.named && q.target !== null && q.target > q.lo && q.target < q.limit) q.marks.push({ at: r6(q.target), text: `${short(q.target)} ${q.named ? 'target' : 'marginal'}` });
    if (q.limit !== null) q.marks.push({ at: r6(q.limit), text: `${short(q.limit)}${q.unit ? ` ${q.unit}` : ''} limit` });
  }
  // the reading's first crossing of its limit, along the input
  let event = null;
  if (q && q.limit !== null) {
    const v = q.values;
    for (let f = 1; f < F; f++) {
      if (!(v[f] > q.limit) || !(v[f - 1] <= q.limit)) continue;
      const x = t[f - 1] + ((q.limit - v[f - 1]) / (v[f] - v[f - 1])) * (t[f] - t[f - 1]);
      event = { at: r6(tPos(x)), text: `Passes the ${short(q.limit)}${q.unit ? ` ${q.unit}` : ''} limit at ${short(x)}${tl.unit ? ` ${tl.unit}` : ''}` };
      break;
    }
  }
  // up to two more readings that change along the input, said in a line under the meter
  // a readout that only repeats the meter's reading is left out
  const repeats = (r) => q && r.values.every((v, i) => !Number.isFinite(q.values[i]) || Math.abs(v - q.values[i]) <= 0.01 * Math.max(Math.abs(q.values[i]), 1e-9));
  const also = h.readouts.filter((r) => !sameAsInput(r) && varies(r.values) && !repeats(r) && (!q || r.name !== q.name.replace(/^peak /, '') && r.name !== q.name))
    .slice(0, 2).map((r) => ({ name: r.name, unit: r.unit || '', digits: r.digits ?? 3, values: r.values.map(r6) }));
  // named values along the input that the run compares, such as wet and dry glass or a box's modes: two or
  // more marked moments that are neither a failure nor a warning. The page shows them side by side.
  const named = (h.events || []).filter((e) => e.kind === 'info' && e.t !== null && e.t >= t[0] && e.t <= t[F - 1]);
  const alts = named.length >= 2 ? named.map((e) => {
    let at = tPos(e.t);
    if (tl.between !== 'exact') at = Math.round(at);
    return { name: cap(e.name), at: r6(at) };
  }) : null;
  const exag = Math.max(1, ...h.objects.filter((o) => o.deform && !o.guide).map((o) => o.deform.exaggerate || 1));
  const cam = h.camera || {};
  const d = cam.dir || [0.62, -1.05, 0.62], l = Math.hypot(...d);
  const bounds = sceneBounds(h, arr), { subject, points, follow, hold } = sceneFraming(h, arr, bounds, Math.round(home));
  return {
    kind: 'scene', title: h.title || '', domain: h.domain || '',
    input: { label: cap(tl.label || 'frame'), unit: tl.unit || '', t: t.map(r6), between: tl.between, kind: tl.kind || 'time', home: r6(home), design: tl.design ?? null },
    q, peaks: peaks && peaks.map((p) => p && { v: p.v, p0: p.p0, pe: p.pe }), event, also, alts,
    exag: +exag.toPrecision(3), magnify: h.magnify ? { factor: h.magnify.factor, label: h.magnify.label || 'Drawn', groups: h.magnify.groups } : null,
    camera: { view: d.map((x) => x / l), fov: cam.fov || 30 }, bounds, subject, points, follow, hold,
    verdict: h.verdict ? { word: h.verdict.word, tone: h.verdict.tone, line: h.verdict.line } : null,
  };
}

export function syncScenes(site) {
  const root = resolve(site, '..');
  const out = resolve(site, 'public/data/scenes');
  rmSync(out, { recursive: true, force: true });
  mkdirSync(out, { recursive: true });
  const scenes = {};
  const ex = resolve(root, 'examples');
  for (const p of existsSync(ex) ? find(ex) : []) {
    const rel = relative(ex, resolve(p, '..')).split('\\').join('/');
    const bytes = readFileSync(p);
    try {
      const raw = bytes[0] === 0x1f && bytes[1] === 0x8b ? gunzipSync(bytes) : bytes;
      const { h, arr } = readScene(new Uint8Array(raw.buffer, raw.byteOffset, raw.byteLength));
      const file = `${rel.replace(/[^a-z0-9]+/gi, '-')}.${hash(bytes)}.wobbl`;
      writeFileSync(join(out, file), bytes);
      scenes[rel] = { file, bytes: bytes.length, title: h.title ?? '', frames: h.timeline?.frames ?? 1, play: scenePlay(h, arr) };
    } catch (e) {
      scenes[rel] = { broken: String(e.stack ?? e.message ?? e) };
    }
  }
  writeFileSync(join(out, 'manifest.json'), JSON.stringify({ scenes }));
  return scenes;
}

if (import.meta.url === `file://${process.argv[1]}`) {
  const s = syncScenes(resolve(process.argv[2] ?? '.'));
  for (const [rel, v] of Object.entries(s)) {
    if (v.broken) { console.log(rel, `BROKEN ${v.broken}`); continue; }
    const p = v.play, q = p.q;
    console.log(`${rel.padEnd(32)} ${p.input.label} home=${p.input.home} | ${q ? `${q.name} ${q.values[Math.round(p.input.home)]} ${q.unit} lim=${q.limit}` : 'no reading'} | ${p.event ? p.event.text : ''} | also ${p.also.map((a) => a.name).join(', ')}`);
  }
}
