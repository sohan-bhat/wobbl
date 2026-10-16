// The 3D scenes of the example runs, from the manifest scripts/scenes.mjs writes into public/data/scenes.
// Each run's scene.wobbl plays in Wobbl's own viewer, the same one its report embeds.
import { existsSync, readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { url } from './site';
import { hasSim } from './sim';
import { playRuns } from './playback';
import type { Run } from './examples';

interface Slider { label: string; value: string; at: number; one: boolean }
interface Entry { file?: string; bytes?: number; title?: string; frames?: number; broken?: string; poster?: string; posterW?: number; posterH?: number; slider?: Slider }
interface Manifest { viewer: string; scenes: Record<string, Entry> }

let cache: Manifest | null = null;
function manifest(): Manifest {
  if (!cache) {
    const f = resolve(process.cwd(), 'public/data/scenes/manifest.json');
    cache = existsSync(f) ? JSON.parse(readFileSync(f, 'utf-8')) : { viewer: '', scenes: {} };
  }
  return cache!;
}

/** The scene of a run, by its folder under examples (circuits/led, rocket-launch/output), when it has a good one. */
export function sceneFor(rel: string) {
  const e = manifest().scenes[rel];
  if (!e || !e.file) return null;
  // the poster scripts/posters.mjs drew of the opening view, and what the slider under it says
  const poster = e.poster ? { src: url(`data/scenes/${e.poster}`), width: e.posterW ?? 640, height: e.posterH ?? 480, slider: e.slider! } : null;
  return { src: url(`data/scenes/${e.file}`), title: e.title ?? '', frames: e.frames ?? 1, bytes: e.bytes ?? 0, poster };
}

/** The viewer script the pages load once the first scene comes near the screen. */
export function viewerSrc(): string {
  return url(`data/scenes/${manifest().viewer}`);
}

/** The scene a run shows in the viewer on its card and its page. A strength or heat run with its own playback
 * (LiveRun) and a motion run with its flight playback keep those, so they get none. */
export function viewerScene(run: Run) {
  const name = run.rel.split('/')[0];
  if (hasSim(run) || playRuns(name)?.[run.rel]) return null;
  return sceneFor(run.rel);
}
