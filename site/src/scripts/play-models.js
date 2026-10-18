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
  AlwaysStencilFunc, CustomBlending, FrontSide, Group, ImageBitmapLoader, KeepStencilOp, NotEqualStencilFunc,
  OneFactor, OneMinusSrcAlphaFactor, ReplaceStencilOp, SRGBColorSpace,
} from 'three';
import { keepLight, studioShader } from './play-look.js';

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

// a model's textures filtered sharp at a glancing angle, as a desk's grain or a box's wood seen across the view
let anisotropy = 1;
/** The anisotropic filtering every model texture takes, from the stage's renderer (play-draw.js). */
export function setAnisotropy(n) { anisotropy = Math.max(anisotropy, n || 1); }
const MAPS = ['map', 'normalMap', 'roughnessMap', 'metalnessMap', 'aoMap', 'bumpMap', 'emissiveMap', 'clearcoatNormalMap'];

const ready = new WeakMap();
/**
 * A model's material made ready for the stage: lit by the studio the way everything else is (play-look.js), and
 * glass drawn as the stage draws glass, see through with whole reflections, instead of three.js's transmission,
 * which would draw the whole scene again every frame.
 */
function prepare(m) {
  if (ready.has(m)) return ready.get(m);
  // a metallic roughness map holds one metal value, which blender/lib/looks.py writes into its blue channel and its
  // name (r_0.32_0.46_m0, Concrete034_r0). Lossy WebP keeps the green roughness but blurs the blue toward it, which
  // would draw a plastic or a wooden desk half metal, so the metal is taken from the name, as the model was made.
  const metal = m.metalnessMap && /_[mr]([01])$/.exec(m.metalnessMap.name || '');
  if (metal) { m.metalness *= Number(metal[1]); m.metalnessMap = null; }
  for (const k of MAPS) if (m[k] && m[k].anisotropy < anisotropy) { m[k].anisotropy = anisotropy; m[k].needsUpdate = true; }
  let out = m;
  if (m.transmission > 0) {
    out = m.clone();
    const alpha = Math.min(0.3, Math.max(0.08, 1 - m.transmission + 0.08));
    out.transmission = 0;
    Object.assign(out, { transparent: true, depthWrite: false, side: FrontSide, blending: CustomBlending, blendSrc: OneFactor, blendDst: OneMinusSrcAlphaFactor });
    // clear, so it is left out over the part the run is about rather than washing its colors with reflections
    Object.assign(out, { stencilWrite: true, stencilRef: 1, stencilFunc: NotEqualStencilFunc, stencilWriteMask: 0, stencilZPass: KeepStencilOp });
    out.onBeforeCompile = (sh) => {
      // glass shows what is behind it, which the occlusion worked out on that surface would darken
      studioShader(sh, false);
      sh.fragmentShader = sh.fragmentShader.replace('#include <opaque_fragment>', `
  outgoingLight = totalDiffuse * ${alpha.toFixed(3)} + totalSpecular;
  diffuseColor.a = clamp(${alpha.toFixed(3)} + dot(totalSpecular, vec3(0.3333)), 0.0, 1.0);
#include <opaque_fragment>`);
    };
    out.customProgramCacheKey = () => `wobbl-model-glass-${alpha.toFixed(3)}`;
  } else {
    out.onBeforeCompile = studioShader;
  }
  ready.set(m, out);
  return out;
}

// ---- the part under glass ----
// A model is drawn whole, as the Cycles stills draw it: a motor in front of its mount hides the mount there, and the
// view turns to show what it hides. Glass is the one exception: it is left out over the part the run is about, which
// marks the pixels it covers in the stencil buffer, so its reflections never wash the field's colors.

/** Make a material mark the pixels its part covers: the part's surface, drawn before any model. */
export function subject(m) {
  Object.assign(m, { stencilWrite: true, stencilRef: 1, stencilFunc: AlwaysStencilFunc, stencilZPass: ReplaceStencilOp });
  return m;
}

/**
 * Keep something colored by a field that is not the part itself (a line or a body colored by its value, a model
 * painted with its field's color) or something that glows out of the resting view's ambient occlusion
 * (play-refine.js), so its colors stay as the legend gives them. Its shader is built again without it.
 */
export function keepColor(m) {
  if (m.userData.keepColor) return m;
  m.userData.keepColor = true;
  const was = m.onBeforeCompile, key = m.customProgramCacheKey();
  m.onBeforeCompile = function (sh, renderer) {
    was.call(this, sh, renderer);
    keepLight(sh);
  };
  m.customProgramCacheKey = () => `${key}|keep`;
  m.needsUpdate = true;
  return m;
}

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
  const tinted = [], meshes = [];
  const names = new Set(entry.tint || []);
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
  });
  let last = '';
  return {
    group, meshes,
    /** Paint the tinted materials a color given as sRGB bytes. */
    tint(rgb) {
      const key = rgb.join(',');
      if (key === last) return;
      last = key;
      for (const m of tinted) m.color.setRGB(rgb[0] / 255, rgb[1] / 255, rgb[2] / 255, SRGBColorSpace);
    },
    /** The tinted materials carry a field's color: keep it out of the occlusion (keepColor). */
    keepTint() { for (const m of tinted) keepColor(m); },
  };
}
