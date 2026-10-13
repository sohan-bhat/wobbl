// The hero part's shaders and its belt pull arrow, for three.js, which is bundled from npm. The
// vertex shader combines the two unit pulls' stress tensors for the pull's size and angle and takes
// von Mises from the result at every vertex, since von Mises does not add up across loads.
import { ConeGeometry, CylinderGeometry, Group, Mesh, MeshBasicMaterial, SphereGeometry } from 'three';

export const VERT = /* glsl */ `
attribute vec4 stressA;
attribute vec4 stressB;
attribute vec4 shear;
attribute vec4 dispA;
attribute vec3 dispB;
uniform float uF;
uniform vec2 uCS;
uniform float uExag;
uniform float uLimit;
// each channel's largest value: the attributes arrive as -1 to 1
uniform vec4 uRA;
uniform vec4 uRB;
uniform vec4 uRS;
uniform vec3 uRDA;
uniform vec3 uRDB;
varying vec3 vC;
varying float vE;
varying vec3 vN;
varying vec3 vV;
#ifdef PICK
varying float vS;
#endif
// Turbo, as a polynomial (Google, Apache 2.0), so the color needs no texture lookup
vec3 turbo(float x) {
  const vec4 kr4 = vec4(0.13572138, 4.61539260, -42.66032258, 132.13108234);
  const vec4 kg4 = vec4(0.09140261, 2.19418839, 4.84296658, -14.18503333);
  const vec4 kb4 = vec4(0.10667330, 12.64194608, -60.58204836, 110.36276771);
  const vec2 kr2 = vec2(-152.94239396, 59.28637943);
  const vec2 kg2 = vec2(4.27729857, 2.82956604);
  const vec2 kb2 = vec2(-89.90310912, 27.34824973);
  x = clamp(x, 0.0, 1.0);
  vec4 v4 = vec4(1.0, x, x * x, x * x * x);
  vec2 v2 = v4.zw * v4.z;
  return vec3(dot(v4, kr4) + dot(v2, kr2), dot(v4, kg4) + dot(v2, kg2), dot(v4, kb4) + dot(v2, kb2));
}
void main() {
  // the stress tensor for this pull, then von Mises from it
  vec4 s = uCS.x * (stressA * uRA) + uCS.y * (stressB * uRB);
  vec4 h = shear * uRS;
  vec2 t = uCS.x * h.xy + uCS.y * h.zw;
  vec3 d = s.xyz - s.yzx;
  float vm = uF * sqrt(0.5 * dot(d, d) + 3.0 * (s.w * s.w + dot(t, t)));
  float x = vm / uLimit;
  vC = x > 1.0 ? vec3(0.949, 0.949, 0.933) : turbo(x);
  vE = dispA.w;
#ifdef PICK
  vS = vm;
#endif
  vec3 p = position + (uExag * uF) * (uCS.x * dispA.xyz * uRDA + uCS.y * dispB * uRDB);
  vec4 mv = modelViewMatrix * vec4(p, 1.0);
  vN = normalize(normalMatrix * normal);
  vV = -mv.xyz;
  gl_Position = projectionMatrix * mv;
}`;

export const FRAG = /* glsl */ `
uniform float uHatch;
uniform float uPx;
varying vec3 vC;
varying float vE;
varying vec3 vN;
varying vec3 vV;
void main() {
  vec3 v = normalize(vV);
  vec3 n = normalize(vN);
  n = faceforward(n, -v, n);
  vec3 l1 = normalize(vec3(0.35, 0.55, 0.75));
  vec3 l2 = normalize(vec3(-0.7, -0.1, 0.45));
  float d = 0.34 + 0.6 * max(dot(n, l1), 0.0) + 0.2 * max(dot(n, l2), 0.0);
  float s = 0.16 * pow(max(dot(n, normalize(l1 + v)), 0.0), 28.0);
  vec3 col = vC * d + vec3(s);
  if (uHatch > 0.5 && vE > 0.5) {
    float k = mod(gl_FragCoord.x - gl_FragCoord.y, 9.0 * uPx);
    if (k < 2.5 * uPx) col = mix(col, vec3(0.055, 0.059, 0.067), 0.72);
  }
  gl_FragColor = vec4(col, 1.0);
}`;

// The probe reads von Mises under the pointer from a one pixel render: 24 bits of MPa / 4096 in
// red, green and blue, and alpha 1 in an excluded zone, a half elsewhere, 0 off the part.
export const PICK_FRAG = /* glsl */ `
varying float vS;
varying float vE;
void main() {
  float a = clamp(vS / 4096.0, 0.0, 0.99999) * 255.0;
  float r = floor(a); a = (a - r) * 255.0;
  float g = floor(a); a = (a - g) * 255.0;
  gl_FragColor = vec4(r / 255.0, g / 255.0, floor(a) / 255.0, vE > 0.5 ? 1.0 : 0.5);
}`;

/** The belt pull: drawn solid, plus a faint copy that shows through the part when it is behind it. */
export function arrow(color) {
  const g = new Group();
  const solid = new MeshBasicMaterial({ color });
  const ghost = new MeshBasicMaterial({ color, transparent: true, opacity: 0.35, depthTest: false, depthWrite: false });
  const parts = {};
  for (const [k, geo] of Object.entries({
    shaft: new CylinderGeometry(1.1, 1.1, 1, 12),
    head: new ConeGeometry(3.6, 10, 20),
    hub: new SphereGeometry(3.2, 16, 10),
  })) {
    const m = new Mesh(geo, solid);
    const x = new Mesh(geo, ghost);
    x.renderOrder = 2;
    m.add(x);
    parts[k] = m;
    g.add(m);
  }
  g.userData = { ...parts, materials: [solid, ghost] };
  return g;
}
