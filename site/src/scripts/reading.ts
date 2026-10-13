// The hero part's reading for a belt pull of a given size and direction. HeroPart.astro uses it
// at build time so the page reads right before any script runs, and hero.ts uses it live.

/** public/data/cast-motor-mount.json, written by scripts/export_site_part.py. */
export interface PartData {
  name: string;
  file: string;
  load_N: number;
  allowable_MPa: number;
  target_sf: number;
  pull: { point: number[]; axis: number[]; e1: number[]; e2: number[]; default_deg: number };
  mesh: { triangles: number; corners: number; nodes: number; bounds_min: number[]; bounds_max: number[] };
  layout: Record<string, { offset: number; step?: number; count?: number; channels?: number }>;
  /** governing von Mises in MPa at load_N for each whole degree, and where: an index into peak_nodes */
  peak_by_degree: number[];
  peak_at_degree: number[];
  /** x, y, z in mm, then case 1 and case 2 displacement in mm per newton */
  peak_nodes: number[][];
  /** Wobbl's own converged studies at other angles, against the combined unit pulls */
  converged: { angle_deg: number; converged_MPa: number; superposed_MPa: number; difference_pct: number }[];
}

export type Verdict = 'holds' | 'marginal' | 'fails';
export const WORD: Record<Verdict, string> = { holds: 'Holds', marginal: 'Marginal', fails: 'Fails' };

/** 1234.5 as "1,234.5". Plain string work: the first Intl call on a page can take a hundred
 * milliseconds on a slow phone while it loads its locale data. */
export function fmt(x: number, d = 0): string {
  const s = Math.abs(x).toFixed(d);
  const dot = s.indexOf('.');
  const whole = (dot < 0 ? s : s.slice(0, dot)).replace(/\B(?=(\d{3})+(?!\d))/g, ',');
  return (x < 0 && Number(s) !== 0 ? '-' : '') + whole + (dot < 0 ? '' : s.slice(dot));
}

/** A whole degree from 0 to 359. */
export const wrap = (deg: number) => ((Math.round(deg) % 360) + 360) % 360;

/** Peak, safety factor and verdict for a pull of load N, given the governing MPa per newton. */
export function reading(man: PartData, load: number, perN: number) {
  const peak = perN * load;
  const sf = peak > 0 ? man.allowable_MPa / peak : Infinity;
  const verdict: Verdict = sf >= man.target_sf ? 'holds' : sf >= 1 ? 'marginal' : 'fails';
  // the largest loads on the slider's 10 N steps that still meet the target and do not yield
  const target = Math.floor(man.allowable_MPa / (man.target_sf * perN) / 10) * 10;
  const yields = Math.floor(man.allowable_MPa / perN / 10) * 10;
  return { peak, sf, verdict, target, yields };
}

/** The table's governing MPa per newton, at a whole degree or between two, where the animation passes. */
export function tablePerN(man: PartData, deg: number): number {
  const d = ((deg % 360) + 360) % 360, i = Math.floor(d), f = d - i, t = man.peak_by_degree;
  return (f < 1e-9 ? t[i % 360] : t[i % 360] * (1 - f) + t[(i + 1) % 360] * f) / man.load_N;
}

/** The table's peak node at the whole degree nearest to deg. */
export const peakNode = (man: PartData, deg: number) => man.peak_nodes[man.peak_at_degree[wrap(deg)]];

/** Words for a direction as the dial and the default view show it, for screen readers. */
export function angleWords(deg: number): string {
  const d = wrap(deg);
  if (d === 0) return 'horizontal, to the right';
  if (d === 90) return 'straight down';
  if (d === 180) return 'horizontal, to the left';
  if (d === 270) return 'straight up';
  const below = d < 180;
  const a = below ? Math.min(d, 180 - d) : Math.min(360 - d, d - 180);
  return `${a} degrees ${below ? 'below' : 'above'} horizontal, to the ${d < 90 || d > 270 ? 'right' : 'left'}`;
}

/** The widest text each live value can take, in characters, so its box never changes size. */
export function widths(man: PartData, maxLoad: number) {
  let lo = Infinity, hi = 0;
  for (const p of man.peak_by_degree) { lo = Math.min(lo, p); hi = Math.max(hi, p); }
  const most = reading(man, maxLoad, hi / man.load_N).peak;
  const least = reading(man, 1, lo / man.load_N);
  return {
    peak: fmt(most, 1).length,
    limit: `${fmt(Math.max(least.target, least.yields))} N`.length,
  };
}

/**
 * Where a direction comes near one that Wobbl's own converged study puts more than 5% away from
 * the combined result, that study and the gap. Opposite pulls give the same stress, so angles
 * are compared half a turn apart.
 */
export function offCheck(man: PartData, deg: number) {
  const near = (a: number) => {
    const d = Math.abs((((deg - a) % 180) + 180) % 180);
    return Math.min(d, 180 - d) <= 15;
  };
  return man.converged.find((c) => Math.abs(c.difference_pct) > 5 && near(c.angle_deg)) ?? null;
}
