// Plays each run's scene.wobbl live in Wobbl's viewer. A scene loads when it comes near the screen and plays
// while it is on screen. Only a few WebGL views run at once (two on a phone): the rest show a still of
// their own last frame, and take a live view back when one frees up. A view that scrolls away keeps
// running state and camera until a scene on screen needs its context.
type View = { pause(): void; play(): void; dispose(): void; frames: number };
declare global { interface Window { WobblView?: { mount(el: Element, o: Record<string, unknown>): Promise<View | null> } } }

const LIMIT = matchMedia('(max-width: 719px)').matches ? 2 : 5;
// cards keep only the moving part and the time: the page around them already says the rest
const QUIET = `.bar,.nt,.ro,.lg,.vd,.ts,.ttl,.tag,.lb{display:none!important}.wv{border:0}.st{aspect-ratio:4/3;min-height:0;max-height:none}`;

interface Slot { el: HTMLElement; host: HTMLElement | null; view: View | null; seen: boolean; busy: boolean; bytes?: Promise<Uint8Array> }
const slots = new Map<HTMLElement, Slot>();
let viewer: Promise<void> | null = null;

function loadViewer(src: string): Promise<void> {
  viewer ??= new Promise((ok, no) => {
    if (window.WobblView?.mount) return ok();
    const s = document.createElement('script');
    s.src = src; s.async = true; s.onload = () => ok(); s.onerror = () => no(new Error('the viewer did not load'));
    document.head.append(s);
  });
  return viewer;
}

const live = () => [...slots.values()].filter((s) => s.view || s.busy).length;
const card = (s: Slot) => s.el.dataset.mode === 'card';

function bytes(s: Slot): Promise<Uint8Array> {
  s.bytes ??= fetch(s.el.dataset.scene!).then((r) => {
    if (!r.ok) throw new Error(`the scene did not load (${r.status})`);
    return r.arrayBuffer();
  }).then((b) => new Uint8Array(b));
  return s.bytes;
}

async function mount(s: Slot, autoplay: boolean, end = false): Promise<View | null> {
  await loadViewer(s.el.dataset.viewer!);
  const data = await bytes(s);
  const host = document.createElement('div');
  host.className = 'scene-host';
  s.el.querySelector('.scene-host')?.remove();
  s.el.prepend(host);
  s.host = host;
  const p = window.WobblView!.mount(host, { src: data, autoplay, still: true, end });
  if (card(s) && host.shadowRoot) { const st = document.createElement('style'); st.textContent = QUIET; host.shadowRoot.append(st); }
  return p;
}

/** A still of the view as it is now, kept in place of the live view once it is let go. */
function still(s: Slot) {
  const c = s.host?.shadowRoot?.querySelector('.st canvas') as HTMLCanvasElement | null;
  if (!c || !c.width) return;
  let img = s.el.querySelector<HTMLImageElement>('img.scene-still');
  if (!img) { img = document.createElement('img'); img.className = 'scene-still'; img.alt = ''; s.el.append(img); }
  try { img.src = c.toDataURL('image/webp', 0.8); } catch { /* a tainted or lost canvas leaves the old still */ }
  s.el.classList.add('has-still');
}

function release(s: Slot) {
  if (!s.view) return;
  still(s);
  s.view.dispose();
  s.view = null;
  s.host?.remove();
  s.host = null;
  s.el.classList.remove('is-live');
}

const frame = () => new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)));

/** The last frame as a still, where a loaded part shows its stress and a run its result, drawn in a view that is let go straight after, one at a time. */
let queue: Promise<void> = Promise.resolve();
function poster(s: Slot): Promise<void> {
  s.busy = true;
  queue = queue.then(() => drawPoster(s));
  return queue;
}
async function drawPoster(s: Slot) {
  try {
    const v = await mount(s, false, true);
    if (v) { await frame(); s.view = v; release(s); }
  } catch { /* the live attempt later says what went wrong */ }
  s.busy = false;
  s.host?.remove();
  s.host = null;
}

async function start(s: Slot) {
  s.busy = true;
  try {
    s.view = await mount(s, true);
    if (s.view) s.el.classList.add('is-live');
  } catch (e) {
    s.el.classList.add('is-broken');
    const m = s.el.querySelector('.scene-wait span:last-child');
    if (m) m.textContent = `The 3D scene did not load. ${(e as Error).message}.`;
  }
  s.busy = false;
  schedule();
}

/** Give free live views to scenes on screen, details pages' main views first, and stills to the rest. */
function schedule() {
  const want = [...slots.values()].filter((s) => s.seen && !s.view && !s.busy && !s.el.classList.contains('is-broken'));
  want.sort((a, b) => Number(card(a)) - Number(card(b)));
  for (const s of want) {
    // a live view off screen keeps its place and camera until a scene on screen needs its context
    if (live() >= LIMIT) { const idle = [...slots.values()].find((o) => o.view && !o.seen); if (idle) release(idle); }
    if (live() < LIMIT) start(s);
    else if (!s.el.classList.contains('has-still')) poster(s).then(schedule);
  }
}

const io = new IntersectionObserver((list) => {
  for (const e of list) {
    const s = slots.get(e.target as HTMLElement);
    if (!s) continue;
    s.seen = e.isIntersecting;
  }
  schedule();
}, { rootMargin: '120px 0px' });

function scan() {
  for (const el of document.querySelectorAll<HTMLElement>('.scene[data-scene]')) {
    if (slots.has(el)) continue;
    slots.set(el, { el, host: null, view: null, seen: false, busy: false });
    io.observe(el);
  }
}

// pages change without a reload, so the views of the page going away let go of their contexts first
document.addEventListener('astro:before-swap', () => {
  for (const s of slots.values()) { io.unobserve(s.el); s.view?.dispose(); }
  slots.clear();
});
document.addEventListener('astro:page-load', scan);
scan();
