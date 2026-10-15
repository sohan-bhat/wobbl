// The 3D scenes of the example runs, from the manifest scripts/scenes.mjs writes into public/data/scenes.
// Each run's scene.wobbl plays in Wobbl's own viewer, the same one its report embeds.
import { existsSync, readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { url } from './site';

interface Entry { file?: string; bytes?: number; title?: string; frames?: number; broken?: string }
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
export function sceneFor(rel: string): { src: string; title: string; frames: number; bytes: number } | null {
  const e = manifest().scenes[rel];
  if (!e || !e.file) return null;
  return { src: url(`data/scenes/${e.file}`), title: e.title ?? '', frames: e.frames ?? 1, bytes: e.bytes ?? 0 };
}

/** The viewer script the pages load once the first scene comes near the screen. */
export function viewerSrc(): string {
  return url(`data/scenes/${manifest().viewer}`);
}
