// A motion run's altitude over time, laid out at build time from its trajectory.csv.
// The plot is a 1000 x 1000 SVG stretched over a box of fixed aspect ratio, so the line
// scales with the page while every label stays HTML at its own size. Each event label
// takes the spot around its marker that stays clear of the line, the plot's edges and the
// other labels at a phone, a tablet and a desktop width at once.
import { existsSync, readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { ROOT } from './site';
import type { Run } from './examples';
import { meters, secs } from '../scripts/flight-format';

export { meters, mps, secs } from '../scripts/flight-format';

export interface Sample { t: number; alt: number; speed: number }
export interface FlightEvent { name: string; t_s: number; alt_m: number; x_m?: number; speed_mps?: number }

/** The plot's width over its height, and the phone sized plot that decides which events share a marker. */
export const ASPECT = 16 / 10;
const SMALL = { w: 290, h: 290 / ASPECT };

const csvPath = (run: Run) => resolve(ROOT, 'examples', run.rel, 'trajectory.csv');
export const hasFlight = (run: Run) => run.outcome.domain === 'motion' && existsSync(csvPath(run));

/**
 * The time, altitude and speed columns of trajectory.csv. A row that repeats the time of the one
 * before it, such as the landed row at rest after the impact, is left out.
 */
export function trajectory(run: Run): Sample[] {
  const [head, ...lines] = readFileSync(csvPath(run), 'utf-8').trim().split(/\r?\n/);
  const col = Object.fromEntries(head.split(',').map((h, i) => [h.trim(), i]));
  for (const k of ['t_s', 'alt_m']) if (!(k in col)) throw new Error(`${run.rel}/trajectory.csv has no ${k} column`);
  const num = (cells: string[], k: string) => (k in col ? Number(cells[col[k]]) : NaN);
  const rows = lines.map((line) => {
    const c = line.split(',');
    return { t: num(c, 't_s'), alt: num(c, 'alt_m'), speed: num(c, 'speed_mps') };
  }).filter((r) => Number.isFinite(r.t) && Number.isFinite(r.alt));
  return rows.filter((r, i) => i === 0 || r.t > rows[i - 1].t);
}

/**
 * For a drop from rest, the same drop without air: Wobbl's vacuum check flies it with gravity held at
 * its ground value, so it is the parabola from the start height that lands at the check's time.
 */
function withoutAir(run: Run, start: Sample): { tv: number; h: number } | null {
  const check = (run.outcome.checks ?? []).find((c) => c.name === 'vacuum drop');
  const tv = Number(check?.detail?.time_s);
  return check && start.speed === 0 && start.alt > 0 && tv > 0 ? { tv, h: start.alt } : null;
}

/** Ticks at 1, 2, 2.5 or 5 times a power of ten, at most `most` steps from 0 up to `max`. */
function step(max: number, most: number): number {
  const p = 10 ** Math.floor(Math.log10(max / most));
  for (const k of [1, 2, 2.5, 5, 10, 20]) if (Math.ceil(max / (k * p) - 1e-9) <= most) return k * p;
  return 20 * p;
}

/** A plain number for an axis: thousands separated, no trailing zeros. */
export const tick = (v: number) => Number(v.toPrecision(6)).toLocaleString('en-US', { maximumFractionDigits: 2 });

/** Ramer, Douglas and Peucker: drop points that stay within `tol` of the line through their neighbors. */
function simplify(pts: [number, number][], tol: number): [number, number][] {
  if (pts.length < 3) return pts;
  const keep = new Uint8Array(pts.length);
  keep[0] = keep[pts.length - 1] = 1;
  const stack: [number, number][] = [[0, pts.length - 1]];
  while (stack.length) {
    const [a, b] = stack.pop()!;
    const [ax, ay] = pts[a], [bx, by] = pts[b];
    const len = Math.hypot(bx - ax, by - ay) || 1;
    let far = -1, dist = tol;
    for (let i = a + 1; i < b; i++) {
      const d = Math.abs((bx - ax) * (ay - pts[i][1]) - (ax - pts[i][0]) * (by - ay)) / len;
      if (d > dist) { dist = d; far = i; }
    }
    if (far > 0) { keep[far] = 1; stack.push([a, far], [far, b]); }
  }
  return pts.filter((_, i) => keep[i]);
}

const label = (name: string) => name[0].toUpperCase() + name.slice(1);
/** Names read as a list: "Apogee and drogue", "Burnout, apogee and drogue". */
const joinNames = (names: string[]) => {
  const n = names.map((s, i) => (i ? s.toLowerCase() : label(s)));
  return n.length < 2 ? n[0] : `${n.slice(0, -1).join(', ')} and ${n[n.length - 1]}`;
};

export interface Marker {
  left: number; top: number;     // percent of the plot
  name: string; detail: string;
  ghost?: boolean;               // the end of the drop without air, drawn hollow
  label: string;                 // where the label sits around its marker, as inline CSS
  align: 'left' | 'right' | 'center';
  lead?: 'up' | 'down' | 'left' | 'right';   // a line from the label back to a marker set well away
  at: number;                    // share of the run's time, for the line drawing in
}

/**
 * A label spot around a marker: to its left, middle or right, above, level or below, and how far off.
 * A label straight above or below can be centered on its marker or end at it (ax), which lets a
 * marker on the plot's right edge keep its label inside the plot.
 */
interface Spot { hx: -1 | 0 | 1; vy: -1 | 0 | 1; gx: number; gy: number; ax: -1 | 0 | 1 }
const SPOTS: Spot[] = [];
for (const gx of [9, 20, 36, 52]) for (const hx of [1, -1] as const) SPOTS.push({ hx, vy: 0, gx, gy: 0, ax: 0 });
for (const gy of [9, 20, 36, 52, 68]) for (const vy of [-1, 1] as const) for (const ax of [0, -1, 1] as const) SPOTS.push({ hx: 0, vy, gx: 0, gy, ax });
for (const gx of [6, 20, 36, 52]) for (const gy of [4, 9, 20, 36]) for (const hx of [1, -1] as const) for (const vy of [-1, 1] as const) SPOTS.push({ hx, vy, gx, gy, ax: 0 });
/** Plot widths the labels must fit at: a phone, a tablet and a desktop column. */
const WIDTHS = [290, 420, 560];

export interface Chart {
  line: string; area: string;
  ghost: string;                 // the same drop without air, dashed, or empty
  yTicks: { v: string; top: number }[];
  xTicks: { v: string; left: number }[];
  yChars: number;                // widest y tick, in characters, for the gutter
  markers: Marker[];
  dryRun: boolean;               // whether the chart shows the drop without air
  samples: string;               // t, altitude and speed at about 80 instants, for the readout
  tEnd: number; top: number;
  events: FlightEvent[];
}

export function flightChart(run: Run): Chart {
  const rows = trajectory(run);
  const tEnd = rows[rows.length - 1].t;
  const dry = withoutAir(run, rows[0]);
  const altMax = Math.max(...rows.map((r) => r.alt));
  const ys = step(altMax, 4);
  const top = Math.ceil(altMax / ys - 1e-9) * ys;
  const xs = step(tEnd, 5);
  const X = (t: number) => (t / tEnd) * 1000;
  const Y = (a: number) => 1000 - (Math.max(a, 0) / top) * 1000;
  const r1 = (v: number) => Math.round(v * 10) / 10;

  // the line, with every point dropped that sits within about a tenth of a percent of the plot of it
  const pts = simplify(rows.map((r) => [X(r.t), Y(r.alt)] as [number, number]), 1.2);
  const line = pts.map(([x, y], i) => `${i ? 'L' : 'M'}${r1(x)} ${r1(y)}`).join('');
  const area = `${line}L1000 1000L0 1000Z`;
  const ghostPts: [number, number][] = dry
    ? Array.from({ length: 25 }, (_, i) => { const t = (dry.tv * i) / 24; return [X(t), Y(dry.h * (1 - (t / dry.tv) ** 2))]; })
    : [];
  const ghost = ghostPts.map(([x, y], i) => `${i ? 'L' : 'M'}${r1(x)} ${r1(y)}`).join('');

  const yTicks = [];
  for (let i = 0; i * ys <= top + 1e-9; i++) yTicks.push({ v: tick(i * ys), top: (1 - (i * ys) / top) * 100 });
  const xTicks = [];
  for (let i = 0; i * xs <= tEnd + 1e-9; i++) xTicks.push({ v: tick(i * xs), left: ((i * xs) / tEnd) * 100 });

  // events: the start is the corner of the chart, so it is not labeled; events that land on
  // the same spot of a phone sized chart share one marker and one label
  const events = (run.outcome.events ?? []) as FlightEvent[];
  const rel = (e: { t_s: number; alt_m: number }) => [X(e.t_s) / 1000, Y(e.alt_m) / 1000];
  const near = (a: number[], b: number[]) => Math.hypot((a[0] - b[0]) * SMALL.w, (a[1] - b[1]) * SMALL.h) < 14;
  const start = rel({ t_s: rows[0].t, alt_m: rows[0].alt });
  const groups: FlightEvent[][] = [];
  for (const e of events) {
    if (near(rel(e), start)) continue;
    const last = groups[groups.length - 1];
    if (last && near(rel(e), rel(last[0]))) last.push(e);
    else groups.push([e]);
  }
  if (dry) groups.push([{ name: 'without air', t_s: dry.tv, alt_m: 0 }]);

  // the line as points about every 0.2% of the plot along its length, to measure how much of it a label covers
  const trace: number[][] = [];
  for (const path of [pts, ghostPts]) {
    for (let i = 1; i < path.length; i++) {
      const [ax, ay] = path[i - 1], [bx, by] = path[i];
      const n = Math.max(1, Math.ceil(Math.hypot(bx - ax, by - ay) / 2));
      for (let k = 0; k <= n; k++) trace.push([(ax + ((bx - ax) * k) / n) / 1000, (ay + ((by - ay) * k) / n) / 1000]);
    }
  }

  // each label takes the spot that keeps it inside the plot, off the line, off the other labels
  // and markers and close to its own marker, at a phone, a tablet and a desktop width at once
  const placed: Spot[] = [];
  const sizes: number[][] = [];
  const boxAt = (s: Spot, p: number[], w: number, h: number, W: number) => {
    const H = W / ASPECT, px = p[0] * W, py = p[1] * H;
    const x = s.hx < 0 ? px - s.gx - w : s.hx > 0 ? px + s.gx : s.ax < 0 ? px + 5 - w : s.ax > 0 ? px - 5 : px - w / 2;
    const y = s.vy < 0 ? py - s.gy - h : s.vy > 0 ? py + s.gy : py - h / 2;
    return [x, y, w, h];
  };
  const markers: Marker[] = groups.map((g, gi) => {
    const e = g[0];
    const p = rel(e);
    const name = joinNames(g.map((x) => x.name));
    const detail = e.alt_m > 0.5 ? `${secs(e.t_s)}, ${meters(e.alt_m)}` : secs(e.t_s);
    const w = Math.max(name.length * 7.6, detail.length * 6.9) + 10, h = 38;
    sizes.push([w, h]);
    let best = { spot: SPOTS[0], score: Infinity };
    for (const spot of SPOTS) {
      let score = (spot.gx + spot.gy) * 1.5;
      for (const W of WIDTHS) {
        const H = W / ASPECT;
        const [bx, by, bw, bh] = boxAt(spot, p, w, h, W);
        // past the sides or below the ground is out; a little past the right edge, into the plot's
        // margin, and a little above the top are allowed
        score += 1000 * (Math.max(0, -bx) + Math.max(0, bx + bw - W - 6) + Math.max(0, -14 - by) + Math.max(0, by + bh - H));
        const ds = (2 / 1000) * W;
        for (const [lx, ly] of trace) {
          const x = lx * W, y = ly * H;
          if (x > bx - 3 && x < bx + bw + 3 && y > by - 3 && y < by + bh + 3) score += 6 * ds;
        }
        placed.forEach((o, oi) => {
          const [ox, oy, ow, oh] = boxAt(o, rel(groups[oi][0]), sizes[oi][0], sizes[oi][1], W);
          score += 2 * Math.max(0, Math.min(bx + bw, ox + ow) - Math.max(bx, ox)) * Math.max(0, Math.min(by + bh, oy + oh) - Math.max(by, oy));
        });
        groups.forEach((o, oi) => {
          if (oi === gi) return;
          const [qx, qy] = rel(o[0]);
          if (qx * W > bx - 7 && qx * W < bx + bw + 7 && qy * H > by - 7 && qy * H < by + bh + 7) score += 500;
        });
      }
      if (score < best.score - 1e-9) best = { spot, score };
    }
    const s = best.spot;
    placed.push(s);
    const css = [
      s.hx > 0 ? `left: ${s.gx}px` : s.hx < 0 ? `right: ${s.gx}px`
        : s.ax < 0 ? 'right: -5px' : s.ax > 0 ? 'left: -5px' : 'left: 0; --tx: -50%',
      s.vy > 0 ? `top: ${s.gy}px` : s.vy < 0 ? `bottom: ${s.gy}px` : 'top: 0; --ty: -50%',
    ];
    // a label set well away straight above, below or beside its marker gets a thin line back to it
    const gap = s.hx === 0 ? s.gy : s.vy === 0 ? s.gx : 0;
    const lead = gap > 12 ? (s.hx === 0 ? (s.vy < 0 ? 'down' : 'up') : (s.hx < 0 ? 'right' : 'left')) : undefined;
    if (lead) css.push(`--lead: ${gap - 6}px`);
    if (lead && s.hx === 0 && s.ax !== 0) css.push(`--lx: ${s.ax < 0 ? 'calc(100% - 5px)' : '5px'}`);
    return {
      left: p[0] * 100, top: p[1] * 100, name, detail, label: css.join('; '), lead, ghost: dry !== null && gi === groups.length - 1,
      align: s.hx < 0 || s.ax < 0 ? 'right' : s.hx > 0 || s.ax > 0 ? 'left' : 'center', at: e.t_s / tEnd,
    };
  });

  // the readout: real rows, about 80 of them evenly spread in time, plus the row at every event,
  // to four figures, which keeps them to about a kilobyte of the page
  const want = 80;
  const atEvent = (r: Sample) => events.some((e) => Math.abs(e.t_s - r.t) < 1e-6);
  const picked: Sample[] = [];
  let next = 0;
  rows.forEach((r, i) => {
    if (r.t >= next - 1e-9 || atEvent(r) || i === rows.length - 1) picked.push(r);
    if (r.t >= next - 1e-9) next = r.t + tEnd / want;
  });
  const samples = picked.map((r) => [r.t, r.alt, r.speed].map((v) => (Number.isFinite(v) ? Number(v.toPrecision(4)) : '')).join(',')).join(';');

  return {
    line, area, ghost, yTicks, xTicks, yChars: Math.max(...yTicks.map((t) => t.v.length)), markers, dryRun: dry !== null,
    samples, tEnd, top, events,
  };
}
