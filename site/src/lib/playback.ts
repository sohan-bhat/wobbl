// The live playback of strength and heat runs on the example pages. Everything here is read from
// public/data/play/<example>.json, which scripts/sitegen/export_playback.py writes from each run's own
// results; the page plays the run back from the binary file it names and, before any script runs or
// where the 3D view cannot run, shows the final state with the answer.
import { existsSync, readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { RUN_IMAGES, fixLabel, fmt3, type Example, type Run } from './examples';
import { url } from './site';
import BENDING from '../data/bending.json';

interface Mesh { triangles: number; corners: number; nodes: number; linear: boolean; bounds_min: number[]; bounds_max: number[] }
interface Base {
  label: string; file: string; bytes: number; mesh: Mesh; layout: Record<string, unknown>; verdict: string;
  camera: { view: number[]; center: number[]; radius: number };
}
export interface StrengthRun extends Base {
  /** the material the run's study names (pla, 6061-t6, custom:...,process=cast), for the part's surface finish */
  material?: string;
  /** the faces the run holds fixed, as its study writes them (z=min) */
  fixed?: string;
  kind: 'strength'; governing_MPa: number; governing_xyz: number[]; governing_disp: number[]; allowable_MPa: number;
  target_sf: number; safety_factor: number; measure: string; max_disp_mm: number; loads: string[]; exag: number;
  arrow: { point: number[]; dir: number[]; pull: boolean; disp: number[] } | null;
}
export interface HeatRun extends Base {
  material?: string;
  kind: 'heat'; air_C: number; limit_C: number; t_max_C: number; hot_xyz: number[]; scale_C: [number, number];
  marker?: boolean; transient: boolean; times_s?: number[]; hottest_C?: number[]; cross_s?: number | null;
  warmup?: { t_s: number[]; theta: number[]; lumped_C: number; tau_s: number };
  /** isotherm spacing in C, and the corner cut away at the hottest point when that point is inside the part */
  iso_C?: number;
  cut?: { point: number[]; sign: number[]; points: number; faces: { axis: number; at: number; u: number[]; v: number[] }[] };
}
export type PlayRun = StrengthRun | HeatRun;

const DIR = resolve(process.cwd(), 'public/data/play');
const cache = new Map<string, Record<string, PlayRun> | null>();

/** The runs of an example that have playback, by output folder (shelf-bracket/output and so on). */
export function playRuns(example: string): Record<string, PlayRun> | null {
  if (!cache.has(example)) {
    const f = resolve(DIR, `${example}.json`);
    const runs: Record<string, PlayRun> | null = existsSync(f) ? JSON.parse(readFileSync(f, 'utf-8')).runs : null;
    // a part the site draws elsewhere too bends by the one factor data/bending.json gives it
    const bend = (BENDING as Record<string, number | string>)[example];
    if (runs && typeof bend === 'number') for (const r of Object.values(runs)) if (r.kind === 'strength') r.exag = bend;
    // what the part is made of and where it is held, from each run's own study, so the stage can draw its surface
    for (const [rel, r] of Object.entries(runs ?? {})) {
      // the example as public/models/manifest.json names it, for the models drawn round the part
      (r as PlayRun & { example?: string }).example = rel.replace(/\/output[^/]*(\/.*)?$/, '');
      const st = resolve(process.cwd(), '../examples', rel, 'study.json');
      if (!existsSync(st)) continue;
      try {
        const study = JSON.parse(readFileSync(st, 'utf-8'));
        if (typeof study.material === 'string') r.material = study.material;
        const fx = study.fixtures?.[0]?.faces;
        if (r.kind === 'strength' && typeof fx === 'string') r.fixed = fx;
      } catch { /* a study that cannot be read leaves the default finish */ }
    }
    cache.set(example, runs);
  }
  return cache.get(example)!;
}

export function playFor(ex: Example, run: Run): PlayRun | null {
  return playRuns(ex.name)?.[run.rel] ?? null;
}

/** One part in the view: a run or one of its fixes. */
export interface Pane { rel: string; name: string; play: PlayRun; poster: string | null; file: string }

/** The site image Wobbl's own heatmap of a run was cut down to, for the poster and the fallback. */
function posterOf(rel: string): string | null {
  const imgs = RUN_IMAGES[rel];
  return imgs ? imgs.stress ?? imgs.temperature ?? null : null;
}

/** The run and, when it was fixed, every fix, each with what the page needs to draw it. */
export function panes(ex: Example, run: Run): Pane[] {
  const all = playRuns(ex.name) ?? {};
  const pane = (rel: string, name: string): Pane | null => all[rel]
    ? { rel, name, play: all[rel], poster: posterOf(rel), file: url(`data/play/${all[rel].file}`) } : null;
  const out = [pane(run.rel, 'As drawn')];
  for (const f of run.fixes) out.push(pane(`${run.rel}/${f.after!.report.split('/')[0]}`, fixLabel(f)));
  return out.filter((p): p is Pane => p !== null);
}

/** The fix shown beside the part at first: the first that holds, or else the one with the lowest peak. */
export function firstFix(list: Pane[]): number {
  const fixes = list.slice(1);
  if (!fixes.length) return -1;
  const holds = fixes.findIndex((p) => p.play.verdict === 'holds' || p.play.verdict === 'ok');
  if (holds >= 0) return holds + 1;
  let best = 1;
  fixes.forEach((p, i) => { if ((p.play as StrengthRun).governing_MPa < (list[best].play as StrengthRun).governing_MPa) best = i + 1; });
  return best;
}

/** 1234.5 as "1,234.5", as the page's script writes it too. */
export function fmt(x: number, d = 0): string {
  const s = Math.abs(x).toFixed(d);
  const dot = s.indexOf('.');
  const whole = (dot < 0 ? s : s.slice(0, dot)).replace(/\B(?=(\d{3})+(?!\d))/g, ',');
  return (x < 0 && Number(s) !== 0 ? '-' : '') + whole + (dot < 0 ? '' : s.slice(dot));
}

/** "21:04" for 1264 seconds. */
export function clock(s: number): string {
  const m = Math.floor(s / 60), r = Math.floor(s - m * 60);
  return `${m}:${String(r).padStart(2, '0')}`;
}

/** The meter under the view: where its ends and marks are, in the run's units. */
export function meter(p: PlayRun) {
  if (p.kind === 'strength') {
    const top = p.allowable_MPa * 1.25;
    return { lo: 0, top, marks: [{ at: p.allowable_MPa / p.target_sf, text: `${fmt3(p.allowable_MPa / p.target_sf)} target` },
      { at: p.allowable_MPa, text: `${fmt3(p.allowable_MPa)} MPa limit` }] };
  }
  const lo = p.air_C, hi = Math.max(p.t_max_C, p.limit_C);
  return { lo, top: hi + 0.15 * (hi - lo), marks: [{ at: p.limit_C, text: `${fmt3(p.limit_C)} C limit` }] };
}

/** The final reading of a pane, as the run reported it. */
export function finalValue(p: PlayRun): string {
  return p.kind === 'strength' ? fmt3(p.governing_MPa) : fmt3(p.t_max_C);
}

/** The bending factor, or how much faster than real time a heat run plays, and how long it plays. */
export function pace(p: PlayRun): { tag: string; seconds: number; speed: number; span: number } {
  if (p.kind === 'strength') return { tag: p.exag === 1 ? 'Bending to scale' : `Bending ×${p.exag}`, seconds: 4.5, speed: 1, span: 1 };
  const span = p.transient ? p.times_s![p.times_s!.length - 1] : p.warmup!.t_s[p.warmup!.t_s.length - 1];
  const nice = [50, 100, 150, 200, 300, 400, 500, 600, 800, 1000];
  const speed = nice.reduce((a, b) => (Math.abs(b - span / 8.5) < Math.abs(a - span / 8.5) ? b : a));
  const lines = p.iso_C ? `, lines every ${p.iso_C} C` : '';
  return { tag: `${speed}× speed${lines}`, seconds: span / speed, speed, span };
}
