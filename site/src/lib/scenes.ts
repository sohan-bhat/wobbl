// The 3D scenes of the example runs, from the manifest scripts/scenes.mjs writes into public/data/scenes.
// A run with a scene.wobbl and no strength or heat playback of its own plays its scene in the same view as
// those runs (components/LiveRun.astro): the drawing reads the scene file, and the readings under the view
// come from the summary scripts/scenes.mjs worked out of it.
import { existsSync, readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { url } from './site';
import { playRuns } from './playback';
import type { Run } from './examples';

export interface Reading {
  name: string; unit: string; digits: number; values: (number | null)[]; lo: number; top: number;
  limit: number | null; target: number | null; limitLabel: string; field: boolean; named: boolean; marks: { at: number; text: string }[];
}
export interface ScenePlay {
  kind: 'scene'; title: string; domain: string;
  input: { label: string; unit: string; t: number[]; between: string; kind: string; home: number; design: number | null };
  q: Reading | null; peaks: ({ v: number; p0: number[]; pe: number[] } | null)[] | null;
  event: { at: number; text: string } | null;
  also: { name: string; unit: string; digits: number; values: (number | null)[] }[];
  alts: { name: string; at: number }[] | null;
  exag: number; magnify: { factor: number; label: string } | null;
  camera: { view: number[]; fov: number }; bounds: { lo: number[]; hi: number[] };
  verdict: { word: string; tone: string; line: string } | null;
}
interface Entry { file?: string; bytes?: number; title?: string; frames?: number; broken?: string; play?: ScenePlay; poster?: string; posterW?: number; posterH?: number; card?: string }

let cache: Record<string, Entry> | null = null;
function manifest(): Record<string, Entry> {
  if (!cache) {
    const f = resolve(process.cwd(), 'public/data/scenes/manifest.json');
    cache = existsSync(f) ? JSON.parse(readFileSync(f, 'utf-8')).scenes : {};
  }
  return cache!;
}

/** The scene of a run, by its folder under examples (circuits/led, rocket-launch/output), when it has a good one. */
export function sceneFor(rel: string) {
  const e = manifest()[rel];
  if (!e || !e.file || !e.play) return null;
  // the posters scripts/posters.mjs drew of the view the page opens on, once its replay has run
  const poster = e.poster ? { src: url(`data/scenes/${e.poster}`), card: url(`data/scenes/${e.card ?? e.poster}`),
    width: e.posterW ?? 1280, height: e.posterH ?? 960 } : null;
  return { src: url(`data/scenes/${e.file}`), file: e.file, title: e.title ?? '', play: e.play, poster };
}

/** The scene a run plays on its page and shows on its card. A strength or heat run with its own playback keeps it. */
export function sceneRun(run: Run) {
  const name = run.rel.split('/')[0];
  if (playRuns(name)?.[run.rel]) return null;
  return sceneFor(run.rel);
}
