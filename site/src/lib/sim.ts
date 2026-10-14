// A motion run as a scene to play back, laid out at build time from its trajectory.csv.
// Distance and altitude share one scale, so the path has its true shape, and the page plays
// it at real time or at a fixed speedup it shows. The scene is drawn at its last moment, with
// the answer, so it reads fully without script, and scripts/sim.ts plays it from the start.
import { existsSync, readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { ROOT } from './site';
import type { Run } from './examples';
import { clock, meters, mps, pace, secs } from '../scripts/flight-format';

interface Row { t: number; x: number; alt: number; speed: number; phase: string }

/** The stage in SVG units: its size, the top of the sky the path may use and the ground line. */
export const W = 600;
export const H = 450;
const TOP = 52;
const GROUND = H - 46;

const csvPath = (run: Run) => resolve(ROOT, 'examples', run.rel, 'trajectory.csv');
export const hasSim = (run: Run) => run.outcome.domain === 'motion' && existsSync(csvPath(run));

/** The rows of trajectory.csv. A row that repeats the time before it, the landed row at rest, is left out. */
function rows(run: Run): Row[] {
  const [head, ...lines] = readFileSync(csvPath(run), 'utf-8').trim().split(/\r?\n/);
  const col = Object.fromEntries(head.split(',').map((h, i) => [h.trim(), i]));
  for (const k of ['t_s', 'alt_m']) if (!(k in col)) throw new Error(`${run.rel}/trajectory.csv has no ${k} column`);
  const num = (c: string[], k: string) => (k in col ? Number(c[col[k]]) : 0);
  const all = lines.map((line) => {
    const c = line.split(',');
    return { t: num(c, 't_s'), x: num(c, 'x_m'), alt: num(c, 'alt_m'), speed: num(c, 'speed_mps'),
      phase: 'phase' in col ? c[col.phase].trim() : '' };
  }).filter((r) => Number.isFinite(r.t) && Number.isFinite(r.alt));
  return all.filter((r, i) => i === 0 || r.t > all[i - 1].t);
}

/** Ramer, Douglas and Peucker over any number of dimensions, always keeping the rows marked. */
function simplify(pts: number[][], tol: number, keep: Set<number>): number[] {
  const out = new Uint8Array(pts.length);
  out[0] = out[pts.length - 1] = 1;
  for (const k of keep) out[k] = 1;
  const marks = [...out.keys()].filter((i) => out[i]);
  const stack: [number, number][] = marks.slice(1).map((b, i) => [marks[i], b]);
  while (stack.length) {
    const [a, b] = stack.pop()!;
    let far = -1, dist = tol;
    const ab = pts[b].map((v, k) => v - pts[a][k]);
    const len2 = ab.reduce((t, v) => t + v * v, 0) || 1;
    for (let i = a + 1; i < b; i++) {
      // distance from the segment between a and b
      const f = Math.min(Math.max(pts[i].reduce((t, v, k) => t + (v - pts[a][k]) * ab[k], 0) / len2, 0), 1);
      const d = Math.hypot(...pts[i].map((v, k) => v - (pts[a][k] + ab[k] * f)));
      if (d > dist) { dist = d; far = i; }
    }
    if (far > 0) { out[far] = 1; stack.push([a, far], [far, b]); }
  }
  return [...out.keys()].filter((i) => out[i]);
}

/** Phase names as a reader would say them. */
export const PHASE: Record<string, string> = {
  rail: 'On the rail', boost: 'Under thrust', coast: 'Coasting', drogue: 'Under the drogue', main: 'Under the main',
  fall: 'Falling', flight: 'In flight', landed: 'Landed',
};
const cap = (s: string) => s[0].toUpperCase() + s.slice(1);

export interface Sim {
  drop: boolean;            // a body let go from a height, drawn beside a cliff
  rocket: boolean;          // a rocket on a rail, drawn with its flame and parachutes
  cliff: string;            // the cliff's outline, for a drop
  rail: string;             // the launch rail, for a rocket
  trail: string;            // the whole path, as the last moment shows it
  end: { x: number; y: number; a: number };
  chutes: string[];         // the parachutes open at the end
  ghost: { t: number; x: number; top: number } | null;  // the same drop without air, beside it
  frames: string;           // t, x, y, speed and phase index of the rows the playback steps through
  phases: string;           // the phases the frames point at, as key=label
  scale: number;            // stage units per meter, the same along the ground and up
  ground: number;           // the ground line on the stage
  events: string;           // when each event happens and its name, for the flags
  speedup: number;
  decimals: number;         // the clock's decimals, two for a run under 20 s and one otherwise
  bar: { pct: number; label: string };   // the scale bar, its width as a share of the stage
  answer: string;
  readouts: [string, string][];
  widths: Record<string, number>; // the widest value each readout takes, in characters, so nothing shifts
  said: string;             // the run in one sentence, for screen readers
}

const r1 = (v: number) => Math.round(v * 10) / 10;

export function sim(run: Run): Sim {
  const rs = rows(run);
  const o = run.outcome;
  const events = o.events ?? [];
  const tEnd = rs[rs.length - 1].t;
  const altMax = Math.max(...rs.map((r) => r.alt));
  const xs = rs.map((r) => r.x);
  const xMin = Math.min(0, ...xs), xMax = Math.max(0, ...xs);
  const drop = rs[0].speed === 0 && rs[0].alt > 0;
  const rocket = rs.some((r) => r.phase === 'rail' || r.phase === 'boost');
  // one scale for both axes, the larger of the two spans filling the stage
  const s = Math.min((GROUND - TOP) / (altMax * 1.04), (W * 0.8) / Math.max(xMax - xMin, 1));
  const x0 = drop ? W * 0.56 - s * (xMin + xMax) / 2 : W / 2 - s * (xMin + xMax) / 2;
  const X = (x: number) => x0 + x * s;
  const Y = (alt: number) => GROUND - Math.max(alt, 0) * s;

  // the rows the playback steps through: every event and phase change, and enough others that
  // straight steps between them stay within a few tenths of a unit on the stage, in time and speed
  const vMax = Math.max(...rs.map((r) => r.speed), 1);
  const keep = new Set<number>();
  rs.forEach((r, i) => {
    if (i && r.phase !== rs[i - 1].phase) { keep.add(i - 1); keep.add(i); }
    if (events.some((e) => Math.abs(e.t_s - r.t) < 1e-6)) keep.add(i);
  });
  const idx = simplify(rs.map((r) => [X(r.x), Y(r.alt), (r.t / tEnd) * W, (r.speed / vMax) * 80]), 0.4, keep);
  const phaseNames = [...new Set(rs.map((r) => r.phase))];
  const frames = idx.map((i) => {
    const r = rs[i];
    return [Number(r.t.toPrecision(5)), r1(X(r.x)), r1(Y(r.alt)), Number(r.speed.toPrecision(3)), phaseNames.indexOf(r.phase)].join(',');
  }).join(';');
  const trail = idx.map((i, k) => `${k ? 'L' : 'M'}${r1(X(rs[i].x))} ${r1(Y(rs[i].alt))}`).join('');

  // the scale bar: 1, 2 or 5 times a power of ten that draws between 60 and 150 units
  let len = 10 ** Math.floor(Math.log10(120 / s));
  for (const k of [1, 2, 5, 10]) if (k * len * s >= 60) { len *= k; break; }
  const bar = { pct: (len * s / W) * 100, label: meters(len) };

  const last = rs[rs.length - 1];
  const prev = rs[rs.length - 2] ?? last;
  const endPhase = last.phase;
  const a = rocket && !['drogue', 'main'].includes(endPhase) ? Math.atan2(X(last.x) - X(prev.x), Y(prev.alt) - Y(last.alt)) * 180 / Math.PI : 0;
  const chutes = rocket ? ['drogue', 'main'].filter((c) => rs.some((r) => r.phase === c)) : [];

  // a cliff for a drop: from the left edge to just beside the path, as tall as the drop
  const face = X(0) - 9;
  const top = Y(rs[0].alt);
  const cliff = drop ? `M0 ${GROUND}L0 ${r1(top)}L${r1(face - 14)} ${r1(top)}L${r1(face)} ${r1(top + 10)}L${r1(face - 4)} ${GROUND}Z` : '';
  // for a drop from rest, the same drop without air falls beside it: Wobbl's vacuum check flies it
  // with gravity held at its ground value, so it is the parabola that lands at the check's time
  const vac = Number((o.checks ?? []).find((c) => c.name === 'vacuum drop')?.detail?.time_s);
  const ghost = drop && vac > 0 ? { t: vac, x: r1(X(0) + 16), top: r1(top) } : null;
  const railAngle = rocket && rs.length > 2 ? Math.atan2(rs[2].x - rs[0].x, Math.max(rs[2].alt - rs[0].alt, 1e-6)) : 0;
  const rail = rocket ? `M${r1(X(0))} ${GROUND}L${r1(X(0) + Math.sin(railAngle) * 26)} ${r1(GROUND - Math.cos(railAngle) * 26)}` : '';

  // the answer at the end, from the run's own numbers
  const metric = (k: string) => o.metrics?.find(([n]) => n === k)?.[1];
  const decimals = tEnd < 20 ? 2 : 1;
  const time = clock(tEnd, decimals);
  const hit = metric('impact speed') ?? mps(last.speed);
  const answer = metric('apogee')
    ? `Reaches ${meters(parseFloat(metric('apogee')!))} and lands ${metric('drift') ? `${metric('drift')} from the pad ` : ''}after ${time}`
    : metric('range') ? `Lands ${metric('range')} away after ${time} at ${hit}` : `Lands at ${hit} after ${time}`;

  const readouts: [string, string][] = [['Time', time], ['Altitude', meters(0)], ['Speed', pace(last.speed)]];
  if (rocket) readouts.push(['Phase', PHASE.landed]);
  const widest = (vals: string[]) => Math.max(...vals.map((v) => v.length));
  const widths = {
    Time: widest(rs.map((r) => clock(r.t, decimals))), Altitude: widest(rs.map((r) => meters(r.alt))),
    Speed: widest(rs.map((r) => pace(r.speed))), Phase: widest([...phaseNames.map((p) => PHASE[p] ?? cap(p)), PHASE.landed]),
  };

  // one flag per moment: events at the same time, like apogee and the drogue, share it
  const moments: { t: number; names: string[] }[] = [];
  for (const e of events.filter((ev) => ev.t_s > 0.001 || drop)) {
    const m = moments[moments.length - 1];
    if (m && Math.abs(e.t_s - m.t) < 1e-3) m.names.push(e.name); else moments.push({ t: e.t_s, names: [e.name] });
  }
  const flagEvents = moments.map((m) => `${Number(m.t.toPrecision(5))},${cap(m.names.join(' and '))}`);
  if (ghost) flagEvents.push(`${Number(ghost.t.toPrecision(5))},Lands without air,ghost`);
  flagEvents.sort((p, q) => parseFloat(p) - parseFloat(q));
  return {
    drop, rocket, cliff, rail, trail, end: { x: r1(X(last.x)), y: r1(Y(last.alt)), a: Math.round(a) }, chutes, ghost, frames,
    phases: phaseNames.map((p) => `${p}=${PHASE[p] ?? cap(p)}`).join(';'), scale: s, ground: GROUND,
    events: flagEvents.join(';'),
    speedup: speedup(tEnd), decimals, bar, answer, readouts, widths,
    said: `${answer}. The run marks ${events.map((e) => `${e.name} at ${secs(e.t_s)}, ${meters(e.alt_m ?? 0)} up`).join(', then ')}.`
      + (ghost ? ` The same drop without air lands after ${secs(ghost.t)}.` : ''),
  };
}

/** Real time for a run under 20 s, and otherwise the smallest of 2, 5, 10, 20 or 50 times that plays it in about 20 s. */
export function speedup(tEnd: number): number {
  if (tEnd <= 20) return 1;
  return [2, 5, 10, 20, 50, 100].find((k) => tEnd / k <= 22) ?? 100;
}
