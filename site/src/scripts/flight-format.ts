// How the flight charts print times, heights and speeds, shared by the build (lib/flight.ts)
// and the pointer readout in the browser (app.ts).

/** Seconds with about three figures: 0.19 s, 2.21 s, 26.1 s, 229 s. */
export const secs = (t: number) => `${t >= 100 ? Math.round(t) : Number(t.toPrecision(3))} s`;

/** Meters, whole from 10 m up: 4.2 m, 300 m, 3,661 m. */
export const meters = (m: number) =>
  `${(Math.abs(m) >= 10 ? Math.round(m) : Number(m.toPrecision(2))).toLocaleString('en-US')} m`;

/** A running clock with a fixed number of decimals, so its width holds: 1.45 s, 14.5 s. */
export const clock = (t: number, decimals: number) => `${t.toFixed(decimals)} s`;

/** A running speed to a tenth: 9.7 m/s, 427.3 m/s. */
export const pace = (v: number) => `${v.toFixed(1)} m/s`;

/** Meters per second with three figures: 5.2 m/s, 58.6 m/s, 291 m/s. */
export const mps = (v: number) => `${Number(v.toPrecision(3)).toLocaleString('en-US')} m/s`;
