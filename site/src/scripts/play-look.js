// The studio every playback surface is lit in, the way model-viewer and Sketchfab light a product: a real
// photographed studio (Poly Haven's Studio Small 08, CC0, see public/env/README.md) as image based light,
// prefiltered with three.js's PMREM so a rough cast face and a polished pin each reflect it as they would,
// on three.js's own physically based materials (MeshStandardMaterial and MeshPhysicalMaterial), tone mapped
// with Khronos PBR Neutral so a stress map keeps its hues. The studio and a key light stand where the gallery's
// Cycles stills put theirs for the view the part opens on (blender/still.py), so the live view lights a part as
// its card does. Every material here is patched in the same ways (onBeforeCompile): the turned studio, the key
// light's fall off, PBR Neutral, and a surface finish worked out from the run's material.
import {
  DataTexture, DataUtils, DirectionalLight, EquirectangularReflectionMapping, HalfFloatType, LinearFilter, Matrix3,
  MeshDepthMaterial, PMREMGenerator, RGBADepthPacking, RGBAFormat, ShaderChunk, Vector3, Vector4,
} from 'three';

/** Uniforms every patched material shares: the studio's turn, its strength and the exposure. */
export const LOOK = {
  uEnvRot: { value: new Matrix3() },
  // the studio's diffuse light as nine spherical harmonics, worked out once from the photograph: a part's diffuse
  // light then costs a few multiplies instead of a second lookup in the prefiltered map on every pixel
  uSH: { value: Array.from({ length: 9 }, () => new Vector3()) },
  uEnvI: { value: 0.9 },
  uExposure: { value: 1.3 },
  // the key light as the Cycles stills place it (blender/still.py light): where it stands in the world and the
  // square of its distance to the view's target, and the way it faces, so it falls off with distance and off its
  // axis as their disc light does while its shadow comes from the stage's directional light
  uKey: { value: new Vector4(0, 0, 0, 0) },
  uKeyAxis: { value: new Vector3(0, 0, -1) },
};

/** A page color as the studio's view gives it: exposed and tone mapped with PBR Neutral, as sRGB bytes. */
export function toned(rgb) {
  const lin = rgb.map((c) => { const x = c / 255; return (x <= 0.04045 ? x / 12.92 : ((x + 0.055) / 1.055) ** 2.4) * LOOK.uExposure.value; });
  const x = Math.min(...lin), off = x < 0.08 ? x - 6.25 * x * x : 0.04;
  // dark colors only: the highlight roll off never reaches them
  return lin.map((c) => { const y = Math.max(c - off, 0); return Math.round(255 * (y <= 0.0031308 ? y * 12.92 : 1.055 * y ** (1 / 2.4) - 0.055)); });
}

// The big octabox, the photograph's brightest pixel, sits at u 0.646. Turned to u 0.361 it stands 50 degrees round
// from the camera toward its right, where the Cycles stills turn it (blender/still.py hdri_rotation), with the
// second box behind the camera on the left as a fill.
const KEY_U = 0.646, KEY_AT = 0.361;

/**
 * Set the studio for a view from azimuth az (about z, the camera's side of the part): z is up in every scene
 * and y in the photograph, and the big softbox goes up to the front right of a camera on that side. The stage
 * sets it from the view a part opens on and keeps it as the part is orbited, as a studio stays put while a
 * photographer walks round a product.
 */
export function turnStudio(az) {
  const c = Math.cos(-az), s = Math.sin(-az);
  const f = 2 * Math.PI * (KEY_U - KEY_AT), cf = Math.cos(f), sf = Math.sin(f);
  // world (z up) turned by -az about z, then into the photograph's frame (x, z, -y), then turned in its xz plane
  // rows of Rz(-az): [c, -s, 0], [s, c, 0], [0, 0, 1]; C maps (x, y, z) to (x, z, -y)
  const r = [c, -s, 0, 0, 0, 1, -s, -c, 0];
  // in the photograph's xz plane: x' = x cf - z sf, z' = x sf + z cf
  const m = [
    r[0] * cf - r[6] * sf, r[1] * cf - r[7] * sf, r[2] * cf - r[8] * sf,
    r[3], r[4], r[5],
    r[0] * sf + r[6] * cf, r[1] * sf + r[7] * cf, r[2] * sf + r[8] * cf,
  ];
  LOOK.uEnvRot.value.set(m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7], m[8]);
}

// ---- the studio photograph --------------------------------------------------------------------------
// public/env/studio.rgbe is the HDRI cut to 512 by 256 and stored as RGBE in four planes (red, green, blue,
// exponent), each row stored as the change from the pixel before, gzipped: about 200 kB for the whole light.

async function gunzip(buf) {
  const h = new Uint8Array(buf, 0, 2);
  if (h[0] !== 0x1f || h[1] !== 0x8b) return buf;
  return new Response(new Blob([buf]).stream().pipeThrough(new DecompressionStream('gzip'))).arrayBuffer();
}

function decodeStudio(buf) {
  const b = new Uint8Array(buf), W = b[0] | (b[1] << 8), H = b[2] | (b[3] << 8), N = W * H;
  const p = new Uint8Array(N * 4);
  for (let pl = 0; pl < 4; pl++) {
    for (let y = 0; y < H; y++) {
      let prev = 0;
      for (let x = 0; x < W; x++) { const i = y * W + x; prev = (prev + b[4 + pl * N + i]) & 255; p[pl * N + i] = prev; }
    }
  }
  const half = new Uint16Array(N * 4), one = DataUtils.toHalfFloat(1), sh = new Float64Array(27);
  // the photograph's first row is its top; three.js's equirectangular lookup starts at the bottom
  for (let y = 0; y < H; y++) {
    // the direction of each pixel as three.js's equirectangular lookup reads it, and its solid angle
    const lat = ((H - 1 - y + 0.5) / H - 0.5) * Math.PI, cl = Math.cos(lat), dw = cl * (Math.PI / H) * ((2 * Math.PI) / W);
    for (let x = 0; x < W; x++) {
      const i = y * W + x, o = ((H - 1 - y) * W + x) * 4, e = p[3 * N + i];
      const f = e ? 2 ** (e - 136) : 0, r = p[i] * f, g = p[N + i] * f, b2 = p[2 * N + i] * f;
      half[o] = DataUtils.toHalfFloat(r);
      half[o + 1] = DataUtils.toHalfFloat(g);
      half[o + 2] = DataUtils.toHalfFloat(b2);
      half[o + 3] = one;
      const phi = ((x + 0.5) / W - 0.5) * 2 * Math.PI, dx = cl * Math.cos(phi), dy = Math.sin(lat), dz = cl * Math.sin(phi);
      const basis = [0.282095, 0.488603 * dy, 0.488603 * dz, 0.488603 * dx, 1.092548 * dx * dy, 1.092548 * dy * dz,
        0.315392 * (3 * dz * dz - 1), 1.092548 * dx * dz, 0.546274 * (dx * dx - dy * dy)];
      for (let k = 0; k < 9; k++) { const w = basis[k] * dw; sh[k * 3] += r * w; sh[k * 3 + 1] += g * w; sh[k * 3 + 2] += b2 * w; }
    }
  }
  return { W, H, half, sh };
}

/** The photograph from its stored bytes, gzipped or not. */
export async function decodeStudioBytes(buf) {
  return decodeStudio(await gunzip(buf));
}

let studio = null;
/** The photograph, fetched and decoded once for every stage in this thread. */
export function loadStudio(url) {
  studio ||= fetch(url).then((r) => {
    if (!r.ok) throw new Error(`studio light: HTTP ${r.status}`);
    return r.arrayBuffer();
  }).then(gunzip).then(decodeStudio);
  return studio;
}

/** The studio prefiltered for a renderer, as scene.environment takes it. */
export function studioEnvironment(renderer, data) {
  for (let k = 0; k < 9; k++) LOOK.uSH.value[k].set(data.sh[k * 3], data.sh[k * 3 + 1], data.sh[k * 3 + 2]);
  LOOK.uSH.ready = true;
  const tex = new DataTexture(data.half, data.W, data.H, RGBAFormat, HalfFloatType);
  tex.mapping = EquirectangularReflectionMapping;
  tex.magFilter = LinearFilter;
  tex.minFilter = LinearFilter;
  tex.generateMipmaps = false;
  tex.needsUpdate = true;
  const pm = new PMREMGenerator(renderer);
  const rt = pm.fromEquirectangular(tex);
  tex.dispose();
  pm.dispose();
  return rt;
}

// ---- patches -----------------------------------------------------------------------------------------

export const SRGB = /* glsl */ `
vec3 toLinear(vec3 c) {
  return mix(c / 12.92, pow((c + 0.055) / 1.055, vec3(2.4)), step(0.04045, c));
}`;

// Khronos PBR Neutral, as model-viewer draws a product: base colors pass through, only highlights roll off
const NEUTRAL = /* glsl */ `
uniform float uExposure;
vec3 pbrNeutral(vec3 c) {
  const float start = 0.76, desat = 0.15;
  c *= uExposure;
  float x = min(c.r, min(c.g, c.b));
  c -= x < 0.08 ? x - 6.25 * x * x : 0.04;
  float peak = max(c.r, max(c.g, c.b));
  if (peak < start) return c;
  const float d = 1.0 - start;
  float top = 1.0 - d * d / (peak + d - start);
  c *= top / peak;
  float g = 1.0 - 1.0 / (desat * (peak - top) + 1.0);
  return mix(c, vec3(top), g);
}`;

// three.js's own image based light, with the studio turned with the camera and scaled
const IBL = /* glsl */ `
#ifdef USE_ENVMAP
uniform mat3 uEnvRot;
uniform float uEnvI;
uniform vec3 uSH[9];
// three.js's own irradiance from nine harmonics (lights_pars_begin), here since this chunk comes before it
vec3 shIrradiance(vec3 n) {
  float x = n.x, y = n.y, z = n.z;
  vec3 r = uSH[0] * 0.886227 + uSH[1] * 1.023328 * y + uSH[2] * 1.023328 * z + uSH[3] * 1.023328 * x;
  r += uSH[4] * 0.858086 * x * y + uSH[5] * 0.858086 * y * z + uSH[6] * (0.743125 * z * z - 0.247708);
  r += uSH[7] * 0.858086 * x * z + uSH[8] * 0.429043 * (x * x - y * y);
  return max(r, 0.0);
}
vec3 getIBLIrradiance(const in vec3 normal) {
#ifdef ENVMAP_TYPE_CUBE_UV
  vec3 worldNormal = uEnvRot * inverseTransformDirection(normal, viewMatrix);
#ifdef WOBBL_SH
  return shIrradiance(worldNormal) * envMapIntensity * uEnvI;
#else
  vec4 envMapColor = textureCubeUV(envMap, worldNormal, 1.0);
  return PI * envMapColor.rgb * envMapIntensity * uEnvI;
#endif
#else
  return vec3(0.0);
#endif
}
vec3 getIBLRadiance(const in vec3 viewDir, const in vec3 normal, const in float roughness) {
#ifdef ENVMAP_TYPE_CUBE_UV
  vec3 reflectVec = reflect(-viewDir, normal);
  reflectVec = normalize(mix(reflectVec, normal, roughness * roughness));
  reflectVec = uEnvRot * inverseTransformDirection(reflectVec, viewMatrix);
#ifdef WOBBL_SOFT_REFLECTION
  // a part colored by a field reflects the studio as the harmonics give it while the view moves, a few multiplies
  // where the map costs two lookups on each of its many small triangles, and the map itself as the view comes to
  // rest, faded in with the resting view's occlusion (uSSAO.z), so a resting view and its still show it sharp
  vec3 soft = shIrradiance(reflectVec) * (1.0 / PI) * envMapIntensity * uEnvI;
  if (uSSAO.z <= 0.0) return soft;
  return mix(soft, textureCubeUV(envMap, reflectVec, roughness).rgb * envMapIntensity * uEnvI, uSSAO.z);
#else
  vec4 envMapColor = textureCubeUV(envMap, reflectVec, roughness);
  return envMapColor.rgb * envMapIntensity * uEnvI;
#endif
#else
  return vec3(0.0);
#endif
}
#ifdef USE_ANISOTROPY
vec3 getIBLAnisotropyRadiance(const in vec3 viewDir, const in vec3 normal, const in float roughness, const in vec3 bitangent, const in float anisotropy) {
  vec3 bentNormal = cross(bitangent, viewDir);
  bentNormal = normalize(cross(bentNormal, bitangent));
  bentNormal = normalize(mix(bentNormal, normal, pow2(pow2(1.0 - anisotropy * (1.0 - roughness)))));
  return getIBLRadiance(viewDir, bentNormal, roughness);
}
#endif
#endif`;

// the key light falls off as a disc light at its place does: with the square of the distance and with the cosine
// off its axis, from where it is toward each point, at the strength the stage gives it at the view's target
const KEY = /* glsl */ `
uniform vec4 uKey;
uniform vec3 uKeyAxis;
void keyAt(inout IncidentLight l, const in vec3 p) {
  if (uKey.w <= 0.0) return;
  vec3 d = (viewMatrix * vec4(uKey.xyz, 1.0)).xyz - p;
  float d2 = max(dot(d, d), 1e-12);
  l.direction = d * inversesqrt(d2);
  l.color *= uKey.w / d2 * max(dot(-l.direction, normalize((viewMatrix * vec4(uKeyAxis, 0.0)).xyz)), 0.0);
}`;
const LIGHTS = ShaderChunk.lights_fragment_begin.replace('getDirectionalLightInfo( directionalLight, directLight );',
  'getDirectionalLightInfo( directionalLight, directLight );\n\t\tkeyAt( directLight, geometryPosition );');

/**
 * The ambient occlusion a resting view works out from its own depth (play-refine.js): tSSAO holds it one texel
 * per pixel of the stage's drawing, with the view depth it was worked out at, and uSSAO is one over the drawing's
 * size and the strength it is laid with, zero while the view moves. It dims only the light that comes from all
 * round, the studio's diffuse light, as three.js's own aoMap dims it: the key light, its shadow and the reflections
 * stay as they are, so a resting view keeps the modelling, the highlights and the cast shadows the moving one has.
 */
export const SSAO = {
  tSSAO: { value: null },
  uSSAO: { value: new Vector3(0, 0, 0) },
};

// in three.js's standard and physical materials (STANDARD), the ones lit by the studio's diffuse light
const OCCLUSION = /* glsl */ `
#include <aomap_fragment>
#if defined(WOBBL_AO) && defined(STANDARD)
if (uSSAO.z > 0.0) {
  vec2 ssao = texture2D(tSSAO, gl_FragCoord.xy * uSSAO.xy).rg;
  // only where it was worked out on this surface (it holds the view depth, negative, beside the occlusion): an
  // edge whose neighbour lies far behind keeps its own light
  float ssaoK = abs(ssao.g + vViewPosition.z) < 0.03 * abs(vViewPosition.z) ? uSSAO.z : 0.0;
  // the diffuse light alone: reflections, the key light and its shadow keep their strength, so an edge still
  // catches its highlight and a face lit by the key light is not greyed
  reflectedLight.indirectDiffuse *= mix(1.0, ssao.r, ssaoK);
}
#endif`;

/**
 * The studio's light and PBR Neutral on a three.js lit material's shader, in onBeforeCompile. The renderer's own
 * tone mapping stays off, so nothing else on the page (the arrows, the backdrop) is touched. The surface takes
 * the resting view's ambient occlusion (SSAO above) unless ao is false: a part colored by a field, glass and
 * anything that glows keep their colors exactly (keepLight).
 */
export function studioShader(shader, ao) {
  Object.assign(shader.uniforms, LOOK);
  // the harmonics stand in for the map once the photograph has been read (RoomEnvironment has none)
  if (LOOK.uSH.ready) shader.fragmentShader = `#define WOBBL_SH\n${shader.fragmentShader}`;
  shader.fragmentShader = shader.fragmentShader
    .replace('#include <envmap_physical_pars_fragment>', IBL)
    .replace('#include <tonemapping_fragment>', 'gl_FragColor.rgb = pbrNeutral(gl_FragColor.rgb);')
    .replace('#include <common>', `#include <common>\n${NEUTRAL}\nuniform sampler2D tSSAO;\nuniform vec3 uSSAO;`)
    .replace('#include <lights_pars_begin>', `#include <lights_pars_begin>\n${KEY}`)
    .replace('#include <lights_fragment_begin>', LIGHTS)
    .replace('#include <aomap_fragment>', OCCLUSION);
  Object.assign(shader.uniforms, SSAO);
  // three.js hands onBeforeCompile the renderer as its second argument, which counts as asking for the occlusion
  if (ao !== false) shader.fragmentShader = `#define WOBBL_AO\n${shader.fragmentShader}`;
}

/** Leave a patched shader out of the resting view's occlusion. */
export function keepLight(shader) {
  shader.fragmentShader = shader.fragmentShader.replace('#define WOBBL_AO\n', '');
}

/** The studio as studioShader gives it, kept out of the occlusion, its reflections sharp at rest: for parts colored by a field. */
export function fieldStudioShader(shader) {
  studioShader(shader, false);
  if (LOOK.uSH.ready) shader.fragmentShader = `#define WOBBL_SOFT_REFLECTION\n${shader.fragmentShader}`;
}

// ---- finishes ----------------------------------------------------------------------------------------
// What a part's surface is, from the material its run names. A part colored by a field keeps the colors as its
// paint under a clear coat, as the Cycles stills paint their parts (roughness 0.42, a quarter coat, no metal), so
// the map reads on every face the same way in the view and its card. A fine surface texture, cast grain,
// machining or print layers, still comes from the real material.

/**
 * finish: { name, metalness, roughness, clearcoat, grain (bump strength), grainMm (feature size in mm),
 * layerMm (layer lines along z in mm, 0 for none), sheen }
 */
export const FINISHES = {
  aluminum_cast: { metalness: 0, roughness: 0.42, clearcoat: 0.25, grain: 0.25, grainMm: 0.9, layerMm: 0 },
  aluminum_machined: { metalness: 0, roughness: 0.42, clearcoat: 0.25, grain: 0.12, grainMm: 0.08, layerMm: 0 },
  steel: { metalness: 0, roughness: 0.42, clearcoat: 0.25, grain: 0.1, grainMm: 0.1, layerMm: 0 },
  iron_cast: { metalness: 0, roughness: 0.42, clearcoat: 0.25, grain: 0.6, grainMm: 0.4, layerMm: 0 },
  copper: { metalness: 0, roughness: 0.42, clearcoat: 0.25, grain: 0.1, grainMm: 0.08, layerMm: 0 },
  plastic_printed: { metalness: 0, roughness: 0.42, clearcoat: 0.25, grain: 0.12, grainMm: 0.25, layerMm: 0.2 },
  nylon_sls: { metalness: 0, roughness: 0.42, clearcoat: 0.25, grain: 0.5, grainMm: 0.12, layerMm: 0 },
  plastic_molded: { metalness: 0, roughness: 0.42, clearcoat: 0.25, grain: 0.04, grainMm: 0.2, layerMm: 0 },
  composite: { metalness: 0, roughness: 0.42, clearcoat: 0.25, grain: 0.08, grainMm: 0.3, layerMm: 0 },
  painted: { metalness: 0, roughness: 0.42, clearcoat: 0.25, grain: 0.06, grainMm: 0.3, layerMm: 0 },
  // what a scene's props and parts without a field are made of
  steel_bare: { metalness: 1, roughness: 0.26, clearcoat: 0, grain: 0, grainMm: 1, layerMm: 0 },
  aluminum_bare: { metalness: 1, roughness: 0.38, clearcoat: 0, grain: 0, grainMm: 1, layerMm: 0 },
  paint_gloss: { metalness: 0, roughness: 0.32, clearcoat: 0.7, grain: 0, grainMm: 1, layerMm: 0 },
  paint_black: { metalness: 0, roughness: 0.42, clearcoat: 0.3, grain: 0, grainMm: 1, layerMm: 0 },
  rubber: { metalness: 0, roughness: 0.82, clearcoat: 0, grain: 0, grainMm: 1, layerMm: 0 },
  matte: { metalness: 0, roughness: 0.88, clearcoat: 0, grain: 0, grainMm: 1, layerMm: 0 },
  floor: { metalness: 0, roughness: 0.55, clearcoat: 0, grain: 0, grainMm: 1, layerMm: 0 },
  glass: { metalness: 0, roughness: 0.04, clearcoat: 0, grain: 0, grainMm: 1, layerMm: 0 },
  emissive: { metalness: 0, roughness: 1, clearcoat: 0, grain: 0, grainMm: 1, layerMm: 0 },
};

/** The finish a run's material name gives, as Wobbl writes it (pla, 6061-t6, custom:E=72400,...,process=cast). */
export function finishOf(name) {
  const m = String(name || '').toLowerCase();
  if (!m) return 'painted';
  const cast = /process=cast|\bcast\b|a356|a380|die/.test(m);
  const E = +((m.match(/e=([\d.]+)/) || [])[1] || 0);
  if (/\bpla\b|petg|\babs\b|\basa\b|tpu|fdm|printed/.test(m)) return 'plastic_printed';
  if (/pa12|pa11|nylon|sls|mjf/.test(m)) return 'nylon_sls';
  if (/\bpc\b|polycarbonate|acetal|pom|delrin|hdpe|uhmw|ptfe/.test(m)) return 'plastic_molded';
  if (/carbon|cfrp|gfrp|fr4|composite|glass fib/.test(m)) return 'composite';
  if (/copper|brass|bronze|c110|c360/.test(m)) return 'copper';
  if (/iron/.test(m)) return 'iron_cast';
  if (/steel|stainless|\b30[0-9]\b|\b31[0-9]\b|1018|1045|4130|4140|s235|s355|a36|17-4/.test(m)) return cast ? 'iron_cast' : 'steel';
  if (/alu|\bal\b|6061|6063|7075|2024|5052|al-/.test(m)) return cast ? 'aluminum_cast' : 'aluminum_machined';
  if (E) {
    // a custom material by its stiffness: a metal over 40 GPa, a polymer under 10
    if (E > 150000) return cast ? 'iron_cast' : 'steel';
    if (E > 40000) return cast ? 'aluminum_cast' : 'aluminum_machined';
    if (E < 10000) return 'plastic_printed';
  }
  if (/li-ion|cell|battery/.test(m)) return 'steel';
  return 'painted';
}

// Value noise in 3D for the surface texture, and its gradient by finite differences
const GRAIN = /* glsl */ `
uniform float uGrain;
uniform float uGrainMm;
uniform float uLayerMm;
uniform float uMm;
varying vec3 vLocal;
float hash3(vec3 p) { p = fract(p * 0.3183099 + 0.1); p *= 17.0; return fract(p.x * p.y * p.z * (p.x + p.y + p.z)); }
float vnoise(vec3 x) {
  vec3 i = floor(x), f = fract(x);
  f = f * f * (3.0 - 2.0 * f);
  return mix(mix(mix(hash3(i), hash3(i + vec3(1, 0, 0)), f.x), mix(hash3(i + vec3(0, 1, 0)), hash3(i + vec3(1, 1, 0)), f.x), f.y),
             mix(mix(hash3(i + vec3(0, 0, 1)), hash3(i + vec3(1, 0, 1)), f.x), mix(hash3(i + vec3(0, 1, 1)), hash3(i + vec3(1, 1, 1)), f.x), f.y), f.z);
}
float grainAt(vec3 p) { return vnoise(p); }`;

/**
 * A field colored or plain lit material's surface finish: a fine bump from value noise at the finish's grain
 * size, layer lines along z for a printed part, both faded out once they are finer than a couple of pixels so
 * they never shimmer. vLocal is the position in the mesh's own units, uMm the millimetres in one unit.
 */
export function finishShader(shader, finish, mm) {
  const f = FINISHES[finish] || FINISHES.painted;
  shader.uniforms.uGrain = { value: f.grain };
  shader.uniforms.uGrainMm = { value: f.grainMm };
  shader.uniforms.uLayerMm = { value: f.layerMm };
  shader.uniforms.uMm = { value: mm };
  shader.vertexShader = shader.vertexShader
    .replace('#include <common>', '#include <common>\nvarying vec3 vLocal;')
    .replace('#include <begin_vertex>', '#include <begin_vertex>\nvLocal = position;');
  shader.fragmentShader = shader.fragmentShader
    .replace('#include <common>', `#include <common>\n${GRAIN}`)
    .replace('#include <normal_fragment_maps>', /* glsl */ `#include <normal_fragment_maps>
  {
    // the bump, in view space from the screen space change of the local position
    vec3 q = vLocal * uMm;
    vec3 dpx = dFdx(q), dpy = dFdy(q);
    float px = max(length(dpx), length(dpy));
    float g = uGrainMm, fade = uGrain * (1.0 - smoothstep(0.35 * g, 1.1 * g, px));
    float h = 0.0;
    if (fade > 0.001) h += fade * grainAt(q / g) * g;
    if (uLayerMm > 0.0) {
      float lf = 1.0 - smoothstep(0.25 * uLayerMm, 0.6 * uLayerMm, px);
      float t = fract(q.z / uLayerMm);
      h += lf * 0.12 * uLayerMm * (1.0 - pow(abs(2.0 * t - 1.0), 2.0));
    }
    float hx = dFdx(h), hy = dFdy(h);
    vec3 vx = dFdx(-vViewPosition), vy = dFdy(-vViewPosition);
    vec3 r1 = cross(vy, normal), r2 = cross(normal, vx);
    float det = dot(vx, r1);
    vec3 grad = sign(det) * (hx * r1 + hy * r2) / max(abs(det), 1e-12);
    // scaled into view units: the bump was in mm, the view in the mesh's units
    normal = normalize(normal - grad / uMm);
  }`);
}


// ---- the key light and its soft shadow ------------------------------------------------------------------
// One accent light on top of the studio, where the Cycles stills put theirs (blender/still.py light): up, to the
// right of the camera and toward it, five times the scene's radius from the view's target, as bright there as
// their disc light. It gives the part's own soft shadows, a rib across its base and a pin across its link, which
// image light alone cannot. It stays where the view opened, as the studio does, and its shadow covers what the
// camera sees.

const _d = new Vector3();
const KEY_I = 3;

export function keyLight(size = 1024) {
  const l = new DirectionalLight(0xffffff, KEY_I);
  l.castShadow = true;
  l.shadow.mapSize.set(size, size);
  l.shadow.radius = 4;
  l.shadow.blurSamples = 8;
  l.matrixAutoUpdate = true;
  return l;
}

/**
 * Put the key light where the Cycles stills put theirs for a camera looking along -z (z the unit vector from the
 * view's target to the camera the view opened on), R the scene's radius, its shadow box r round target. Returns a
 * key that changes only when the shadow has to be drawn again.
 */
export function aimKey(l, z, target, r, R) {
  // still.py: (0.8 z + 0.9 x + 0.9 up) with x the camera's right, up cross z
  const xl = Math.hypot(z[0], z[1]) || 1, x0 = -z[1] / xl, x1 = z[0] / xl;
  _d.set(0.8 * z[0] + 0.9 * x0, 0.8 * z[1] + 0.9 * x1, 0.8 * z[2] + 0.9).normalize();
  const D = 5 * R;
  LOOK.uKey.value.set(target.x + _d.x * D, target.y + _d.y * D, target.z + _d.z * D, D * D);
  LOOK.uKeyAxis.value.copy(_d).negate();
  l.position.copy(target).addScaledVector(_d, r * 3);
  l.target.position.copy(target);
  l.target.updateMatrixWorld();
  const cam = l.shadow.camera;
  cam.left = -r; cam.right = r; cam.top = r; cam.bottom = -r;
  cam.near = r * 0.5; cam.far = r * 6;
  cam.updateProjectionMatrix();
  l.shadow.bias = -0.0004;
  l.shadow.normalBias = r * 0.004;
  l.updateMatrixWorld();
  return `${target.x.toPrecision(5)},${target.y.toPrecision(5)},${target.z.toPrecision(5)},${r.toPrecision(5)},${_d.x.toFixed(4)},${_d.y.toFixed(4)},${_d.z.toFixed(4)}`;
}

/**
 * The depth a shadow is drawn from, moved the way the material moves its vertices: patch adds the same
 * attributes and lines to the vertex shader (and fragment, for a part cut open), with the same uniforms.
 */
export function depthMaterial(uniforms, vertex, fragment = null, key = '') {
  const m = new MeshDepthMaterial({ depthPacking: RGBADepthPacking });
  m.onBeforeCompile = (sh) => {
    Object.assign(sh.uniforms, uniforms);
    sh.vertexShader = sh.vertexShader
      .replace('#include <common>', `#include <common>\n${vertex.pars}`)
      .replace('#include <begin_vertex>', `#include <begin_vertex>\n${vertex.main}`);
    if (fragment) {
      sh.fragmentShader = sh.fragmentShader
        .replace('#include <common>', `#include <common>\n${fragment.pars}`)
        .replace('#include <clipping_planes_fragment>', `#include <clipping_planes_fragment>\n${fragment.main}`);
    }
  };
  m.customProgramCacheKey = () => `wobbl-depth-${key}`;
  return m;
}
