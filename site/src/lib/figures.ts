// What each example page plays in its live views: which runs share a view and which panes each view holds. The
// page reads it (pages/examples/[name].astro, components/LiveRun.astro) and so do the stills drawn at build time
// (scripts/posters.mjs, through figures()), so a still is drawn of exactly the panes and runs its view opens on.
import { examples, type Example, type Run } from './examples';
import { firstFix, panes as panesOf, playFor } from './playback';
import { sceneRun, type ScenePlay } from './scenes';
import { num } from '../scripts/scene-file.js';

/**
 * What the page shows for each run: a run of the same question over the same input plays beside the one before
 * it (a board laid out two ways), and a run whose scene repeats an earlier one's (two studies of one command)
 * adds its numbers under that one instead of a second view of the same thing.
 */
export interface Item { run: Run; beside: Run[]; also: Run[] }
export function pageItems(ex: Example): Item[] {
  const items: Item[] = [];
  const sceneKey = (r: Run) => { const x = sceneRun(r); return x ? JSON.stringify([x.play.title, x.play.q?.values]) : null; };
  for (const r of ex.runs) {
    const s = sceneRun(r), key = sceneKey(r);
    if (s) {
      const dup = items.find((it) => sceneKey(it.run) === key);
      if (dup) { dup.also.push(r); continue; }
      const pair = items.find((it) => { const x = sceneRun(it.run); return x && it.run.question === r.question && x.play.input.label === s.play.input.label; });
      if (pair) { pair.beside.push(r); continue; }
    }
    items.push({ run: r, beside: [], also: [] });
  }
  return items;
}

export interface ScenePane {
  rel: string; name: string; at?: string; src: string; play: ScenePlay; home: number; asked: boolean; verdict: string;
  poster: { src: string; card: string; width: number; height: number } | null;
}
/**
 * A scene run's panes: the run itself, or the run and the others of its example over the same input, or the
 * values along its input that the run compares (wet and dry glass), the one the question asked about first.
 */
export function scenePanes(run: Run, beside: Run[]): ScenePane[] {
  const sc = sceneRun(run);
  if (!sc) return [];
  const sp = sc.play, inp = sp.input;
  const lerpAt = (vs: (number | null)[], p: number) => {
    const a = Math.floor(p), b = Math.min(a + 1, vs.length - 1), k = inp.between === 'step' ? 0 : p - a;
    const x = vs[a] ?? NaN, y = vs[b] ?? NaN;
    return k ? x + (y - x) * k : x;
  };
  const inputText = (v: number) => `${num(v)}${inp.unit ? ` ${inp.unit}` : ''}`;
  const cap = (x: string) => x[0].toUpperCase() + x.slice(1);
  if (beside.length) {
    return [run, ...beside].map((r) => { const x = sceneRun(r)!; return { rel: r.rel, name: cap(r.rel.split('/').pop()!.replace(/-/g, ' ')), src: x.src, play: x.play,
      home: x.play.input.home, asked: true, verdict: r.outcome.verdict, poster: x.poster }; });
  }
  if (sp.alts) {
    const named = sp.alts.map((a) => ({ ...a, asked: Math.abs(a.at - inp.home) < 0.5 }));
    const list = named.some((a) => a.asked) ? [...named.filter((a) => a.asked), ...named.filter((a) => !a.asked)]
      : [{ name: 'As asked', at: inp.home, asked: true }, ...named];
    return list.map((a) => ({ rel: run.rel, name: a.name, at: inputText(lerpAt(inp.t, a.at)), src: sc.src, play: sp, home: a.at,
      asked: a.asked, verdict: run.outcome.verdict, poster: sc.poster }));
  }
  return [{ rel: run.rel, name: 'As asked', src: sc.src, play: sp, home: inp.home, asked: true, verdict: run.outcome.verdict, poster: sc.poster }];
}

/**
 * One live view as the page builds it (components/LiveRun.astro), for its stills: every pane it can show, each
 * with the run's file and summary, and the pane shown beside the first at the start (fix, or -1 for one pane).
 */
export interface Figure {
  id: string; kind: 'strength' | 'heat' | 'scene'; fix: number;
  panes: { rel: string; file: string; run: Record<string, any>; home?: number }[];
}
export function figures(): Figure[] {
  const out: Figure[] = [];
  for (const ex of examples()) {
    if (ex.into) continue;
    for (const { run, beside } of pageItems(ex)) {
      if (playFor(ex, run)) {
        const all = panesOf(ex, run);
        out.push({ id: run.rel, kind: all[0].play.kind, fix: firstFix(all), panes: all.map((p) => ({ rel: p.rel, file: p.file, run: p.play })) });
      } else if (sceneRun(run)) {
        const ps = scenePanes(run, beside);
        out.push({ id: run.rel, kind: 'scene', fix: ps.length > 1 ? 1 : -1, panes: ps.map((p) => ({ rel: p.rel, file: p.src, run: p.play, home: p.home })) });
      }
    }
  }
  return out;
}
