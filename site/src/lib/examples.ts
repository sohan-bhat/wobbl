// The examples, read straight from examples/*/: the README, run.sh and each run's outputs.
// Nothing here is written by hand. The question and command come from run.sh, the
// verdict, numbers and checks from outcome.json, the plain text from summary.txt,
// and the notes from the example's README.
import { existsSync, readFileSync } from 'node:fs';
import { execSync } from 'node:child_process';
import { resolve } from 'node:path';
import { ROOT } from './site';

// The newer kinds of study come first, since the landing page's hero already shows a stress result:
// motion, then circuits, then chemistry and the rest as they land, with strength and heat after them.
export const ORDER = ['rocket-launch', 'falling-rock', 'circuits', 'propane-flame', 'knsb-motor',
  'cast-motor-mount', 'shelf-bracket', 'motor-mount', 'enclosure', 'drone-arm', 'heat-sink', 'battery-pack'];

const EX = resolve(ROOT, 'examples');

/** Site images (public/img, made by scripts/build_site_images.py) for each run's output images. */
export const RUN_IMAGES: Record<string, Record<string, string>> = {
  'cast-motor-mount/output': { stress: 'cast-mount-stress' },
  'shelf-bracket/output': { stress: 'bracket-as-drawn' },
  'shelf-bracket/output/fix-fillet': { stress: 'bracket-fillet' },
  'shelf-bracket/output/fix-thicken': { stress: 'bracket-thicken' },
  'shelf-bracket/output/fix-gusset': { stress: 'bracket-gusset' },
  'motor-mount/output': { stress: 'motor-mount-ribs' },
  'motor-mount/output-noribs': { stress: 'motor-mount-noribs' },
  'enclosure/output': { stress: 'enclosure-stress' },
  'drone-arm/output': { stress: 'drone-arm-stress' },
  'heat-sink/output': { temperature: 'heat-sink-temperature' },
  'battery-pack/output': { temperature: 'battery-temperature', history: 'battery-history' },
  'circuits/buck': { waveform: 'buck-waveform' },
  'falling-rock/output': { flight: 'falling-rock-flight' },
  'rocket-launch/output': { flight: 'rocket-flight', path: 'rocket-path' },
  'propane-flame/output': { sweep: 'propane-sweep' },
  'knsb-motor/output': { burn: 'knsb-burn' },
};

const DOMAIN: Record<string, string> = {
  strength: 'Strength', thermal: 'Heat', circuit: 'Circuits', motion: 'Motion', combustion: 'Chemistry', motor: 'Motors',
};
export const COLOR: Record<string, string> = { holds: 'holds', ok: 'holds', marginal: 'marginal', fails: 'fails', hot: 'fails' };

/** An event a motion run marks, such as burnout or impact, with where and how fast. */
export interface FlightEvent { name: string; t_s: number; alt_m: number; x_m?: number; speed_mps?: number }
export interface Check { name: string; status: string; summary: string; detail?: Record<string, unknown> }
export interface Fix { kind: string; title: string; after?: { governing_MPa: number; safety_factor: number; verdict: string; report: string; params?: Record<string, number> } }
export interface Outcome {
  domain: string; verdict: string; headline: string; confidence: string; question?: string;
  governing_MPa?: number; safety_factor?: number; target_sf?: number;
  metrics?: [string, string][]; checks?: Check[]; fixes?: Fix[]; images?: Record<string, string>;
  // motion runs: what happened when, how long the run took and what the model included
  events?: FlightEvent[]; runtime_s?: number; model?: { states: number; effects: string[] };
}

export interface Run {
  rel: string;
  question: string;
  command: string;
  outcome: Outcome;
  summary: string;
  report: string | null;   // site path of the copied report, when the repository tracks it
  fixes: Fix[];            // fixes that were built and solved again
}

export interface Example {
  name: string; title: string; subtitle: string; subtitleFromBody: boolean; readme: string; runs: Run[]; domain: string;
}

const tracked: Set<string> | null = (() => {
  try {
    return new Set(execSync('git ls-files examples', { cwd: ROOT, encoding: 'utf-8' }).split('\n'));
  } catch {
    return null;
  }
})();

/** The command as a reader would type it: no --question or -o, wrapped at about 72 columns. */
function display(cmd: string): string {
  cmd = cmd.replace(/\s--question\s+"[^"]*"/, '').replace(/\s-o\s+\S+/, '');
  const words = cmd.match(/"[^"]*"|'[^']*'|\S+/g) ?? [];
  const groups: string[] = [];
  for (const w of words) {
    if (groups.length && !w.startsWith('-')) groups[groups.length - 1] += ` ${w}`;
    else groups.push(w);
  }
  const lines: string[] = [];
  let cur = '';
  for (const g of groups) {
    if (cur && cur.length + g.length + 1 > 72) { lines.push(`${cur} \\`); cur = `  ${g}`; }
    else cur = cur ? `${cur} ${g}` : g;
  }
  return [...lines, cur].join('\n');
}

/** Split a shell command line into words, honoring double and single quotes. */
function argv(line: string): string[] {
  return (line.match(/"[^"]*"|'[^']*'|\S+/g) ?? []).map((w) => w.replace(/^["']|["']$/g, ''));
}

/** The wobbl commands in a run.sh, with shell variables filled in. */
function parseRunSh(path: string): { argv: string[]; shown: string }[] {
  const text = readFileSync(path, 'utf-8').replace(/\\\n\s*/g, ' ');
  const env: Record<string, string> = {};
  const out: { argv: string[]; shown: string }[] = [];
  for (let line of text.split('\n')) {
    line = line.trim();
    const m = line.match(/^([A-Z_][A-Z0-9_]*)=(.+)$/);
    if (m) { env[m[1]] = argv(m[2])[0]; continue; }
    if (!line.startsWith('wobbl ')) continue;
    for (const [k, v] of Object.entries(env)) line = line.replace(new RegExp(`\\$\\{?${k}\\}?`, 'g'), () => v);
    out.push({ argv: argv(line), shown: display(line) });
  }
  return out;
}

const opt = (a: string[], ...names: string[]) => {
  const i = a.findIndex((x) => names.includes(x));
  return i >= 0 ? a[i + 1] : undefined;
};

function load(name: string): Example {
  const folder = resolve(EX, name);
  const readme = readFileSync(resolve(folder, 'README.md'), 'utf-8');
  // The title is the H1. A question after it, in the H1 or as the first short line, is the subtitle.
  const h1 = readme.split('\n')[0].replace(/^#\s*/, '').trim();
  const cut = h1.search(/[:.?]\s/);
  let title = cut > 0 ? h1.slice(0, cut) : h1;
  let sub = cut > 0 ? h1.slice(cut + 1).trim() : '';
  let subtitleFromBody = false;
  if (!sub) {
    const first = readme.split('\n').slice(1).join('\n').trim().split(/\n\s*\n/)[0] ?? '';
    if (first && !first.startsWith('```') && !first.includes('\n') && first.length < 140) {
      sub = first;
      subtitleFromBody = true;
    }
  }
  title = title.trim();
  const runs: Run[] = [];
  for (const { argv: a, shown } of parseRunSh(resolve(folder, 'run.sh'))) {
    if (a[1] === 'fix') {
      const target = runs.find((r) => r.rel === `${name}/${a[2]}`);
      if (target) {
        target.fixes = (target.outcome.fixes ?? []).filter((f) => f.after);
        target.command += `\n${shown}`;
      }
      continue;
    }
    const rel = `${name}/${opt(a, '-o', '--out') ?? 'output'}`;
    const outcome = JSON.parse(readFileSync(resolve(EX, rel, 'outcome.json'), 'utf-8')) as Outcome;
    const reportRel = `examples/${rel}/report.html`;
    const has = existsSync(resolve(EX, rel, 'report.html')) && (tracked === null || tracked.has(reportRel));
    runs.push({
      rel, question: opt(a, '--question') ?? outcome.question ?? '', command: shown, outcome,
      summary: readFileSync(resolve(EX, rel, 'summary.txt'), 'utf-8'), report: has ? reportRel : null, fixes: [],
    });
  }
  return {
    name, title, subtitle: sub ? sub[0].toUpperCase() + sub.slice(1) : '', subtitleFromBody, readme, runs,
    domain: DOMAIN[runs[0].outcome.domain] ?? '',
  };
}

let cache: Example[] | null = null;
export function examples(): Example[] {
  cache ??= ORDER.filter((n) => existsSync(resolve(EX, n, 'run.sh'))).map(load);
  return cache;
}

export function findRun(rel: string): { ex: Example; run: Run } {
  for (const ex of examples()) for (const run of ex.runs) if (run.rel === rel) return { ex, run };
  throw new Error(`no example run ${rel}`);
}

/** The headline numbers of a run, each value kept on one line with its unit. */
export function metrics(run: Run): [string, string][] {
  const o = run.outcome;
  const rows: [string, string][] = o.domain === 'strength'
    ? [['Peak stress', `${fmt3(o.governing_MPa!)} MPa`], ['Safety factor', o.safety_factor!.toFixed(2)], ['Target', `${o.target_sf}`]]
    : (o.metrics ?? []).slice(0, 3).map(([k, v]) => [k[0].toUpperCase() + k.slice(1), v]);
  return [...rows.map(([k, v]) => [k, v.replace(/ /g, ' ')] as [string, string]), ['Confidence', o.confidence]];
}

/** Three significant figures, the way Wobbl prints them: 124, 52.3, 9.82. */
export function fmt3(x: number): string {
  return Number(x.toPrecision(3)).toString();
}

/** A short name for an applied fix, from the parameters it was built with. */
export function fixLabel(f: Fix): string {
  const p = f.after?.params ?? {};
  if (f.kind === 'fillet' && 'fillet' in p) return `${p.fillet} mm fillet`;
  if (f.kind === 'thicken' && 'thickness' in p) return `Walls at ${p.thickness} mm`;
  if (f.kind === 'gusset') return 'Gusset across the corner';
  return f.title;
}

/** The image of a run on the site, or null for circuits without a plot. */
export function runImages(run: Run): { name: string; kind: string }[] {
  const names = RUN_IMAGES[run.rel] ?? {};
  return Object.entries(names).map(([kind, name]) => ({ kind, name }));
}

/** Text of the run's outputs, so a hand written answer can be checked against them. */
export function runText(run: Run): string {
  let text = run.summary + JSON.stringify(run.outcome);
  for (const f of run.fixes) {
    const s = resolve(EX, run.rel, f.after!.report.split('/')[0], 'summary.txt');
    if (existsSync(s)) text += readFileSync(s, 'utf-8');
  }
  return text;
}

/** Fail the build when a hand written line quotes a number that is not in the run's outputs. */
export function assertNumbers(line: string, run: Run): void {
  const text = runText(run);
  for (const n of line.match(/\d+(?:\.\d+)?/g) ?? []) {
    if (!new RegExp(`(?<![\\d.])${n.replace('.', '\\.')}(?!\\d)`).test(text)) {
      throw new Error(`the line for ${run.rel} says ${n}, which is not in its outputs: ${line}`);
    }
  }
}

/** Where README images point, mapped to the site's WebP copies. */
export function readmeImage(name: string, src: string): string | null {
  for (const [rel, kinds] of Object.entries(RUN_IMAGES)) {
    if (!rel.startsWith(`${name}/`)) continue;
    const out = rel.slice(name.length + 1);
    for (const [kind, img] of Object.entries(kinds)) {
      const run = examples().flatMap((e) => e.runs).find((r) => r.rel === rel);
      const file = run?.outcome.images?.[kind];
      if (file && src === `${out}/${file}`) return `/img/${img}.webp`;
    }
  }
  return null;
}

/**
 * The example's README for its page. The title (and a question line under it, when the
 * README has one) are shown in the page header, and the ./run.sh block in "Run it yourself",
 * so they are left out here. Images point at the site's copies, other relative links at GitHub.
 */
export function readmeBody(ex: Example): string {
  let lines = ex.readme.split('\n');
  lines = lines.slice(1); // the H1
  let body = lines.join('\n').replace(/```sh\n\.\/run\.sh\n```\n?/g, '');
  if (ex.subtitleFromBody) body = body.replace(/^\s*[^\n]+\n/, '');
  return body.replace(/(!?)\[([^\]]*)\]\(([^)\s]+)\)/g, (_m, bang, text, href) => {
    if (/^[a-z]+:/i.test(href) || href.startsWith('/')) return `${bang}[${text}](${href})`;
    if (bang) {
      const img = readmeImage(ex.name, href);
      if (img) return `![${text}](${img})`;
    }
    const rel = resolve(EX, ex.name, href).slice(ROOT.length + 1);
    const folder = existsSync(resolve(ROOT, rel)) && !/\.[a-z]+$/i.test(rel);
    return `${bang}[${text}](https://github.com/sohan-bhat/wobbl/${folder ? 'tree' : 'blob'}/main/${rel})`;
  });
}
