// The models made in Blender for the example pages (site/blender, written to public/models with a manifest), put
// in place of the plain shapes a scene draws: a wiper's crank as a stamped steel link with its bushings, a
// motor with its gearbox. Each model is authored in its object's own frame, so hung under that object's pose it
// follows the simulated motion exactly, joints and all. The loader (three.js's GLTFLoader with the meshopt
// decoder) is only fetched by a page that has models. A model that is missing or fails leaves the plain shape.
//
// manifest.json: { version: 1, examples: { "<example>": {
//   replace: [{ object, index?, file, tint?: [material names], scale?, matrix? }],
//   add: [{ file, parent?, scale?, matrix? }],     world frame, or the named object's frame
//   hide: [object names] } } }
import {
  AlwaysStencilFunc, Box3, CustomBlending, Ray, Vector3, EqualStencilFunc, FrontSide, Group, ImageBitmapLoader, KeepStencilOp, NotEqualStencilFunc,
  OneFactor, OneMinusSrcAlphaFactor, ReplaceStencilOp, SRGBColorSpace,
} from 'three';
import { studioShader } from './play-look.js';

const manifests = new Map(), files = new Map();
let loader = null;

/** The loader, made once: meshopt for gltfpack's files, and image bitmaps for textures inside a worker. */
function gltfLoader() {
  loader ||= Promise.all([
    import('three/examples/jsm/loaders/GLTFLoader.js'),
    import('three/examples/jsm/libs/meshopt_decoder.module.js'),
  ]).then(([{ GLTFLoader }, { MeshoptDecoder }]) => {
    const l = new GLTFLoader();
    l.setMeshoptDecoder(MeshoptDecoder);
    // three.js picks a loader that needs a document on Safari; a worker has none, but has createImageBitmap
    if (typeof document === 'undefined' && typeof createImageBitmap !== 'undefined') {
      l.register((parser) => {
        parser.textureLoader = new ImageBitmapLoader(parser.options.manager);
        return { name: 'wobbl_worker_textures' };
      });
    }
    // WebP textures without three.js's test of support, which makes an Image and so fails in a worker: every
    // browser that can draw this stage reads WebP. Registered after three.js's own, it takes its place.
    l.register((parser) => ({
      name: 'EXT_texture_webp',
      loadTexture(index) {
        const def = parser.json.textures[index], ext = def.extensions && def.extensions.EXT_texture_webp;
        if (!ext) return null;
        return parser.loadTextureImage(index, ext.source, parser.textureLoader);
      },
    }));
    return l;
  });
  return loader;
}

/** The example a run belongs to, as the manifest names it: shelf-bracket for shelf-bracket/output/fix-fillet. */
export const exampleOf = (rel) => String(rel || '').replace(/\/output[^/]*(\/.*)?$/, '');

/**
 * Everything the manifest lists for an example, loaded: { spec, models: Map(file -> gltf or null) }, or null when
 * it lists nothing. root is the site's root URL.
 */
export async function loadModels(root, example) {
  if (!example) return null;
  const url = new URL('models/manifest.json', root).href;
  if (!manifests.has(url)) manifests.set(url, fetch(url).then((r) => (r.ok ? r.json() : null)).catch(() => null));
  const m = await manifests.get(url);
  const spec = m && m.examples && m.examples[example];
  if (!spec) return null;
  const want = [...new Set([...(spec.replace || []), ...(spec.add || [])].map((e) => e.file))];
  if (!want.length) return { spec, models: new Map() };
  const l = await gltfLoader();
  const got = await Promise.all(want.map((f) => {
    const u = new URL(`models/${f}`, root).href;
    if (!files.has(u)) files.set(u, l.loadAsync(u).catch(() => null));
    return files.get(u);
  }));
  return { spec, models: new Map(want.map((f, i) => [f, got[i]])) };
}

// ---- materials ----

const ready = new WeakMap();
/**
 * A model's material made ready for the stage: lit by the studio the way everything else is (play-look.js), and
 * glass drawn as the stage draws glass, see through with whole reflections, instead of three.js's transmission,
 * which would draw the whole scene again every frame.
 */
function prepare(m) {
  if (ready.has(m)) return ready.get(m);
  let out = m;
  if (m.transmission > 0) {
    out = m.clone();
    const alpha = Math.min(0.3, Math.max(0.08, 1 - m.transmission + 0.08));
    out.transmission = 0;
    Object.assign(out, { transparent: true, depthWrite: false, side: FrontSide, blending: CustomBlending, blendSrc: OneFactor, blendDst: OneMinusSrcAlphaFactor });
    // clear, so it is left out over the part the run is about rather than washing its colors with reflections
    Object.assign(out, { stencilWrite: true, stencilRef: 1, stencilFunc: NotEqualStencilFunc, stencilWriteMask: 0, stencilZPass: KeepStencilOp });
    out.onBeforeCompile = (sh) => {
      studioShader(sh);
      sh.fragmentShader = sh.fragmentShader.replace('#include <opaque_fragment>', `
  outgoingLight = totalDiffuse * ${alpha.toFixed(3)} + totalSpecular;
  diffuseColor.a = clamp(${alpha.toFixed(3)} + dot(totalSpecular, vec3(0.3333)), 0.0, 1.0);
#include <opaque_fragment>`);
    };
    out.customProgramCacheKey = () => `wobbl-model-glass-${alpha.toFixed(3)}`;
  } else {
    out.onBeforeCompile = studioShader;
    // drawn whole only where the part the run is about is not behind it (see subject below)
    Object.assign(out, { stencilWrite: true, stencilRef: 1, stencilFunc: NotEqualStencilFunc, stencilWriteMask: 0, stencilZPass: KeepStencilOp });
  }
  ready.set(m, out);
  return out;
}

// ---- the part stays in sight ----
// A model in front of the part a run is about (a motor on its mount, a heater on a heat sink, a box on a shelf
// bracket) is drawn faint where it covers the part, so the stress or the temperature still reads through it, as
// an X-ray view in an engineering program does. The part marks the pixels it covers in the stencil buffer;
// every model is drawn whole outside them and as a faint copy over them.

/** Make a material mark the pixels its part covers: the part's surface, drawn before any model. */
export function subject(m) {
  Object.assign(m, { stencilWrite: true, stencilRef: 1, stencilFunc: AlwaysStencilFunc, stencilZPass: ReplaceStencilOp });
  return m;
}

const ghosts = new WeakMap(), veils = new WeakMap();
function ghostOf(p) {
  if (ghosts.has(p)) return ghosts.get(p);
  const g = p.clone();
  g.onBeforeCompile = studioShader;
  Object.assign(g, { transparent: true, opacity: 0.2, depthWrite: false, stencilFunc: EqualStencilFunc });
  ghosts.set(p, g);
  return g;
}
// a model standing between the camera and the part (a plate the part hangs under, seen from above) is drawn
// faint all over, as a sheet of glass, so the view of the part is not a view of a wall
function veilOf(p) {
  if (veils.has(p)) return veils.get(p);
  const g = p.clone();
  g.onBeforeCompile = studioShader;
  Object.assign(g, { transparent: true, opacity: 0.07, depthWrite: false, stencilWrite: false });
  veils.set(p, g);
  return g;
}

const _ray = new Ray(), _hit = new Vector3();

/**
 * One copy of a model, turned from glTF's y up back to the scene's z up, with its own copies of the materials the
 * entry tints. Returns { group, meshes, tint(rgb bytes) }.
 */
export function instance(gltf, entry) {
  const root = gltf.scene.clone(true);
  root.rotation.x = Math.PI / 2;
  if (entry.scale) root.scale.setScalar(entry.scale);
  const group = new Group();
  group.add(root);
  if (entry.matrix) {
    group.matrixAutoUpdate = false;
    group.matrix.fromArray(entry.matrix);
  }
  const tinted = [], meshes = [], extra = [];
  const names = new Set(entry.tint || []);
  // a tinted material's faint copy is its own, so it takes the same color
  const tintedGhost = (m) => {
    if (!tinted.includes(m)) return ghostOf(m);
    const g = ghostOf(m).clone();
    g.onBeforeCompile = studioShader;
    tinted.push(g);
    return g;
  };
  root.traverse((o) => {
    if (!o.isMesh) return;
    o.castShadow = true;
    o.receiveShadow = true;
    o.frustumCulled = false;
    meshes.push(o);
    const swap = (m) => {
      let p = prepare(m);
      if (names.has(m.name)) {
        p = p.clone();
        p.onBeforeCompile = studioShader;
        tinted.push(p);
      }
      if (p.transparent) o.castShadow = false;
      return p;
    };
    o.material = Array.isArray(o.material) ? o.material.map(swap) : swap(o.material);
    o.renderOrder = 1;
    // the faint copy over the part, sharing the mesh's geometry and its place
    const mats = Array.isArray(o.material) ? o.material : [o.material];
    if (mats.every((m) => m.transparent)) return;
    const ghost = o.clone();
    ghost.material = Array.isArray(o.material) ? mats.map((m) => (m.transparent ? m : tintedGhost(m))) : tintedGhost(o.material);
    ghost.userData.over = ghost.material;
    ghost.userData.veil = Array.isArray(o.material) ? mats.map((m) => (m.transparent ? m : veilOf(m))) : veilOf(o.material);
    Object.assign(ghost, { castShadow: false, receiveShadow: false, renderOrder: 2 });
    ghost.children.length = 0;
    extra.push([o, ghost]);
  });
  for (const [o, ghost] of extra) o.parent.add(ghost);
  let last = '', veiled = false, box = null;
  return {
    group, meshes,
    /**
     * Veil the model when it stands between the camera at from and the part's center at to (Vector3s, in the
     * scene's frame), and draw it whole again when it does not.
     */
    veil(from, to) {
      if (!box) { group.updateWorldMatrix(true, true); box = new Box3().setFromObject(group); }
      const d = from.distanceTo(to);
      _ray.origin.copy(from);
      _ray.direction.subVectors(to, from).normalize();
      const on = !!_ray.intersectBox(box, _hit) && _hit.distanceTo(from) < d * 0.98 && !box.containsPoint(to);
      if (on === veiled) return;
      veiled = on;
      for (const [o, g] of extra) { o.visible = !on; g.material = on ? g.userData.veil : g.userData.over; }
    },
    /** Paint the tinted materials a color given as sRGB bytes. */
    tint(rgb) {
      const key = rgb.join(',');
      if (key === last) return;
      last = key;
      for (const m of tinted) m.color.setRGB(rgb[0] / 255, rgb[1] / 255, rgb[2] / 255, SRGBColorSpace);
    },
  };
}
