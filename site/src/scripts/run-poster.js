// The stills of a run's view on its example page, drawn at build time by scripts/posters.mjs in headless Chrome
// with what the live view draws with: the stage (play-draw.js), lit by the studio and dressed with the Blender
// models as play-back.ts does it, and the camera and frames of run-view.js. A page's still is the first frame its
// live view shows, the frame the view sits on before its replay runs: a strength part unloaded, a heat part at the
// start of its clock, a scene at the start of its input, all from the camera the view opens on. A scene's card is
// the same view at the value its question asked about, for the gallery and a page that has no 3D view.
// scripts/posters.mjs bundles this file on its own; the pages never load it.
import { createStage } from './play-draw.js';
import { loadRun } from './play.js';
import { loadModels } from './play-models.js';
import { homeView, interp, newOrbit, paneViews, runFrame, runView, sceneFrame } from './run-view.js';

const BG = '#0e0f11'; // the page's own background, under the view as it is under the live canvas

/** One pane of the view drawn into the stage in its row, with its models, as play-back.ts's onPage adds it. */
async function addPane(stage, root, pane, top, row) {
  const models = loadModels(root, pane.run.example).catch(() => null);
  const { arrays } = await loadRun(pane.file, pane.run);
  const part = stage.add({ run: pane.run, arrays, top, row });
  if (pane.run.kind === 'heat') evenStart(part, pane.run, arrays);
  stage.dress(part, await models);
  return part;
}

/**
 * A heat part all at one temperature when its clock starts, as a part at the air's temperature is, drawn as a
 * graphics card draws it. The isotherms' test then finds a line of no width: on the card a temperature right on
 * an isotherm darkens the whole part as a line and any other leaves it clear, while software WebGL, which reads
 * the one temperature back a hair off across each face, would draw speckle. So the still gives the test exactly
 * zero where the card draws the line, and no isotherms where it draws none.
 */
function evenStart(part, run, arrays) {
  let rise = 0;
  if (run.transient) {
    const n = arrays.position.length / 3, f = arrays.frames;
    for (let i = 1; i < n; i++) if (f[i] !== f[0]) return;
    rise = f[0];
  } else if (interp(run.warmup.t_s, run.warmup.theta, 0) !== 0) return;
  const u = part.uniforms, iso = u.uIso.value, at = (run.air_C + rise) / iso;
  if (iso > 0 && Math.abs(at - Math.round(at)) < 1e-6) { u.uAir.value = -rise; u.uIso.value = 1; }
  else u.uIso.value = 0;
}

/**
 * Draw a view w by h CSS pixels, px device pixels each, with the panes the page can show in it: the first pane,
 * and beside it the one at fix (or each of the others in turn, as the page's buttons choose them). Returns a WebP
 * data URL of every pane in its own place in the view, scaled by scale. at is 'start' for the page's still, the
 * view before its replay runs, or 'home' for a scene's card, the view once it has run.
 */
globalThis.drawRunStills = async ({ base, kind, panes, fix, w, h, px, at = 'start', scale = 1, quality = 0.86 }) => {
  const canvas = document.createElement('canvas');
  document.body.replaceChildren(canvas);
  const stage = createStage({ canvas, w, h, px, kind });
  const abs = panes.map((p) => ({ ...p, file: new URL(p.file, base).href }));
  // the studio light and the models come from the site's root, two levels above the data the part comes from
  const root = new URL('../../', abs[0].file).href;
  await stage.light(new URL('env/studio.rgbe', root).href);
  const top = kind === 'strength' ? Math.max(...panes.map((p) => p.run.governing_MPa)) : 0;
  const rv = runView(kind, panes[0].run), scene = kind === 'scene';
  const look = { bend: true, mag: true };
  const homeOf = (i) => panes[i].home ?? panes[i].run.input.home;
  const out = new Array(panes.length).fill(null);
  const choices = fix > 0 ? [fix, ...panes.map((_, i) => i).filter((i) => i > 0 && i !== fix)] : [-1];
  let first = null, second = null;
  try {
    first = await addPane(stage, root, abs[0], top, 0);
    for (const c of choices) {
      const shown = c > 0 ? [0, c] : [0];
      if (second) stage.remove(second);
      second = c > 0 ? await addPane(stage, root, abs[c], top, 1) : null;
      stage.compile();
      const views = paneViews(shown.length, w, h), runs = shown.map((i) => panes[i].run);
      // the live view builds at the value each pane stands at once played, and draws that once before its replay
      // starts, which sets where the studio stands (play-draw.js), then the replay starts it from the beginning
      let pos = shown.map((i) => (scene ? homeOf(i) : 0));
      const orbit = newOrbit(rv);
      homeView(orbit, rv, views[0][2], views[0][3], pos[0]);
      const frame = () => {
        // at the start of a replay the load is zero and the clock at its start, however long the clock runs
        const f = runFrame(rv, orbit, views, 0, look.bend, 1);
        return scene ? sceneFrame(f, orbit, rv, runs, pos, views, look, []) : f;
      };
      stage._frame(frame());
      if (at === 'start') {
        pos = shown.map(() => 0);
        stage._frame(frame());
      }
      // each pane cut out of the view, onto the page's own background, straight after the frame is drawn
      shown.forEach((pi, row) => {
        if (out[pi]) return;
        const [x, y, vw, vh] = views[row];
        const c = document.createElement('canvas');
        c.width = Math.round(vw * px * scale);
        c.height = Math.round(vh * px * scale);
        const g = c.getContext('2d');
        g.fillStyle = BG;
        g.fillRect(0, 0, c.width, c.height);
        g.imageSmoothingQuality = 'high';
        g.drawImage(canvas, x * px, y * px, vw * px, vh * px, 0, 0, c.width, c.height);
        out[pi] = c.toDataURL('image/webp', quality);
      });
    }
  } finally {
    stage.dispose();
  }
  return out;
};
