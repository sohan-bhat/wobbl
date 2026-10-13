// The hero part's reading for a load on the motor shaft of a given size and direction. HeroPart.astro
// uses it at build time so the reading is right before any script runs, and hero.ts uses it live.
//
// A direction is an azimuth in the pulley plane, measured from the horizontal and positive below it,
// with the example's pull on the horizontal side at zero, and an elevation toward the shaft axis, both
// in degrees. The example's pull, 30 degrees below horizontal, is azimuth 30 and elevation 0.

/** public/data/cast-motor-mount.json, written by scripts/export_site_part.py. */
export interface PartData {
  name: string;
  file: string;
  peaks: string;
  load_N: number;
  allowable_MPa: number;
  target_sf: number;
  pull: { point: number[]; axis: number[]; e1: number[]; e2: number[]; e3: number[]; default_deg: number };
  mesh: { triangles: number; corners: number; nodes: number; bounds_min: number[]; bounds_max: number[] };
  layout: Record<string, { offset: number; step?: number; count?: number; channels?: number }>;
  /** the default pull's governing von Mises, and the range over every direction, MPa at load_N */
  default_peak_MPa: number;
  peak_range_MPa: number[];
  /** x, y, z in mm, then case 1, 2 and 3 displacement in mm per newton */
  peak_nodes: number[][];
  /** Wobbl's own converged studies at other directions, against the combined unit loads */
  converged: { azimuth_deg: number; elevation_deg: number; converged_MPa: number; superposed_MPa: number; difference_pct: number }[];
}

/** The peak table, public/data/<name>.peaks.<hash>.json: half the sphere of directions, azimuth major. */
export interface Peaks { grid_deg: number; azimuths: number; elevations: number; mpa_x100: number[]; node: number[] }

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

/** The unit direction in the case basis for an azimuth and elevation in degrees. */
export function unit(az: number, el: number): [number, number, number] {
  const a = (az * Math.PI) / 180, e = (el * Math.PI) / 180;
  return [Math.cos(e) * Math.cos(a), Math.cos(e) * Math.sin(a), Math.sin(e)];
}

/** Peak, safety factor and verdict for a load of load N, given the governing MPa per newton. */
export function reading(man: PartData, load: number, perN: number) {
  const peak = perN * load;
  const sf = peak > 0 ? man.allowable_MPa / peak : Infinity;
  const verdict: Verdict = sf >= man.target_sf ? 'holds' : sf >= 1 ? 'marginal' : 'fails';
  return { peak, sf, verdict };
}

/**
 * The table's grid position for a direction. Opposite loads give the same von Mises, so a direction
 * in the other half of the sphere reads as its opposite: azimuth 0 up to 180, elevation -90 to 90.
 */
function cell(peaks: Peaks, az: number, el: number) {
  let a = ((az % 360) + 360) % 360, e = Math.max(-90, Math.min(90, el));
  if (a >= 180) { a -= 180; e = -e; }
  return { i: a / peaks.grid_deg, j: (e + 90) / peaks.grid_deg };
}
/** One table entry, wrapping past azimuth 180 onto the opposite direction. */
function entry(peaks: Peaks, i: number, j: number, of: number[]) {
  const n = peaks.elevations;
  if (i >= peaks.azimuths) { i -= peaks.azimuths; j = n - 1 - j; }
  return of[i * n + Math.max(0, Math.min(n - 1, j))];
}

/** The governing MPa per newton for any direction, read between the table's points. */
export function tablePerN(man: PartData, peaks: Peaks | null, az: number, el: number): number {
  if (!peaks) return man.default_peak_MPa / man.load_N;
  const { i, j } = cell(peaks, az, el), i0 = Math.floor(i), j0 = Math.floor(j), fi = i - i0, fj = j - j0;
  const v = (a: number, b: number) => entry(peaks, a, b, peaks.mpa_x100);
  const mpa = (v(i0, j0) * (1 - fi) + v(i0 + 1, j0) * fi) * (1 - fj) + (v(i0, j0 + 1) * (1 - fi) + v(i0 + 1, j0 + 1) * fi) * fj;
  return mpa / 100 / man.load_N;
}

/** The peak's node at the table point nearest to a direction, or the default one before the table loads. */
export function peakNode(man: PartData, peaks: Peaks | null, az: number, el: number): number[] {
  if (!peaks) return man.peak_nodes[0];
  const { i, j } = cell(peaks, az, el);
  return man.peak_nodes[entry(peaks, Math.round(i), Math.round(j), peaks.node)];
}

/** Words for a direction, as the default view shows it, for screen readers. */
export function angleWords(az: number, el: number): string {
  const a = ((Math.round(az) % 360) + 360) % 360, e = Math.round(el);
  if (Math.abs(e) >= 89) return e > 0 ? 'straight out along the shaft' : 'straight in along the shaft';
  const below = a > 0 && a < 180, side = a < 90 || a > 270 ? 'right' : 'left';
  const tilt = a < 180 ? Math.min(a, 180 - a) : Math.min(a - 180, 360 - a);
  const round = a % 180 === 0 ? `horizontal, to the ${side}` : a === 90 ? 'straight down' : a === 270 ? 'straight up'
    : `${tilt} degrees ${below ? 'below' : 'above'} horizontal, to the ${side}`;
  return e === 0 ? round : `${round}, tipped ${Math.abs(e)} degrees ${e > 0 ? 'out along' : 'in along'} the shaft`;
}

/** The widest text the big reading can take, in characters, so its box never changes size. */
export function widths(man: PartData, maxLoad: number) {
  return { peak: fmt((man.peak_range_MPa[1] * maxLoad) / man.load_N, 1).length };
}

/**
 * Where a direction comes within 15 degrees of one that Wobbl's own converged study puts more than 5%
 * away from the combined result, that study. Opposite loads give the same stress, so u and -u match.
 */
export function offCheck(man: PartData, az: number, el: number) {
  const u = unit(az, el);
  return man.converged.find((c) => {
    if (Math.abs(c.difference_pct) <= 5) return false;
    const w = unit(c.azimuth_deg, c.elevation_deg);
    return Math.abs(u[0] * w[0] + u[1] * w[1] + u[2] * w[2]) >= Math.cos((15 * Math.PI) / 180);
  }) ?? null;
}
