// Where a run's playback is drawn. Where the browser can draw WebGL on an OffscreenCanvas, everything
// happens in a worker (play-stage-worker.js): fetching and laying out the part, compiling the shaders
// and every frame, so the page's main thread only keeps the camera, the readings and the input, as
// the landing page's part does. Elsewhere the part is drawn on the page and its file is decoded in a
// worker (play-worker.js). Both take the same calls.

export interface Decoded { arrays: Record<string, any>; vertices: number }
export interface Back {
  /** Fetch a run's file and draw it in view row, in place of whatever was there. */
  add(row: number, file: string, run: Record<string, any>, top: number): Promise<void>;
  frame(f: Record<string, any>): void;
  /** What a scene's main field reads under a ray in view row, or null where it meets nothing colored. */
  probe(row: number, origin: number[], dir: number[]): Promise<Record<string, any> | null>;
  size(w: number, h: number, px: number): void;
  dispose(): void;
}
type Waiting<T> = Map<number, { ok: (v: T) => void; fail: (e: Error) => void }>;
const abs = (file: string) => new URL(file, document.baseURI).href;

/** WebGL that a graphics card draws, as the landing page's part asks for it; software WebGL counts as none. */
function context(make: () => HTMLCanvasElement | OffscreenCanvas): boolean {
  try {
    const o = { failIfMajorPerformanceCaveat: true };
    const c = make() as HTMLCanvasElement;
    const ctx = (c.getContext('webgl2', o) || c.getContext('webgl', o)) as WebGLRenderingContext | null;
    ctx?.getExtension('WEBGL_lose_context')?.loseContext();
    return !!ctx;
  } catch {
    return false;
  }
}
export const webgl = () => context(() => document.createElement('canvas'));
export const offscreen = () => 'transferControlToOffscreen' in HTMLCanvasElement.prototype && typeof OffscreenCanvas !== 'undefined'
  && context(() => new OffscreenCanvas(1, 1));

/** One decoding worker for the page, shared by every figure drawn on the page itself. */
export function decoder() {
  let worker: Worker | null = null, id = 0;
  const waiting: Waiting<Decoded> = new Map();
  return {
    decode(file: string, run: Record<string, any>): Promise<Decoded> {
      if (!worker) {
        worker = new Worker(new URL('./play-worker.js', import.meta.url), { type: 'module' });
        worker.onmessage = (e) => {
          const w = waiting.get(e.data.id);
          waiting.delete(e.data.id);
          if (e.data.error) w?.fail(new Error(e.data.error)); else w?.ok(e.data);
        };
        worker.onerror = (e) => { for (const w of waiting.values()) w.fail(new Error(e.message || 'the playback worker failed')); waiting.clear(); };
      }
      return new Promise((ok, fail) => { waiting.set(++id, { ok, fail }); worker!.postMessage({ id, file: abs(file), run }); });
    },
    dispose() { worker?.terminate(); worker = null; waiting.clear(); },
  };
}

/** The drawing in a worker, on the canvas handed over to it. */
export function inWorker(canvas: HTMLCanvasElement, kind: string, w: number, h: number, px: number): Back {
  const off = canvas.transferControlToOffscreen();
  const worker = new Worker(new URL('./play-stage-worker.js', import.meta.url), { type: 'module' });
  const waiting: Waiting<any> = new Map();
  let id = 0;
  worker.onmessage = (e) => {
    const d = e.data, w8 = waiting.get(d.id);
    waiting.delete(d.id);
    if (d.type === 'error') w8?.fail(new Error(d.message)); else w8?.ok(d.hit ?? null);
  };
  worker.onerror = (e) => { for (const w8 of waiting.values()) w8.fail(new Error(e.message || 'the drawing worker failed')); waiting.clear(); };
  worker.postMessage({ type: 'init', canvas: off, kind, w, h, px }, [off]);
  return {
    add: (row, file, run, top) => new Promise((ok, fail) => {
      waiting.set(++id, { ok, fail });
      worker.postMessage({ type: 'part', id, row, file: abs(file), run, top });
    }),
    frame: (f) => worker.postMessage({ type: 'frame', frame: f }),
    probe: (row, origin, dir) => new Promise((ok, fail) => {
      waiting.set(++id, { ok, fail });
      worker.postMessage({ type: 'probe', id, row, origin, dir });
    }),
    size: (nw, nh, npx) => worker.postMessage({ type: 'size', w: nw, h: nh, px: npx }),
    dispose() { worker.postMessage({ type: 'dispose' }); setTimeout(() => worker.terminate(), 200); },
  };
}

/** The drawing on the page itself, with each file decoded in the shared worker. */
export async function onPage(canvas: HTMLCanvasElement, kind: string, w: number, h: number, px: number,
  dec: ReturnType<typeof decoder>): Promise<Back> {
  const { createStage } = await import('./play-draw.js');
  const stage = createStage({ canvas, w, h, px, kind });
  const parts: any[] = [];
  let lit: Promise<void> | null = null;
  return {
    async add(row, file, run, top) {
      // the studio light comes from the site's env folder, two levels above the data the part comes from
      lit ||= stage.light(new URL('../../env/studio.rgbe', abs(file)).href);
      const [d] = await Promise.all([dec.decode(file, run), lit]);
      if (parts[row]) stage.remove(parts[row]);
      parts[row] = stage.add({ run, arrays: d.arrays, top, row });
      stage.compile();
    },
    frame: (f) => stage.frame(f),
    probe: async (row, origin, dir) => stage.probe(row, origin, dir),
    size: (nw, nh, npx) => stage.size(nw, nh, npx),
    dispose: () => stage.dispose(),
  };
}
