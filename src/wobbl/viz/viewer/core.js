// The viewer's core: its stylesheet, a little vector math with z up, color maps, the scene file's decoder, the
// primitive shapes and the WebGL programs. objects.js and mount.js follow it, all three inside one function scope
// (embed.py joins them), so nothing here leaks onto the page.

const CSS = `
:host{display:block}
.wv{--bg:#0e0f11;--panel:#16181b;--line:#2a2d32;--fg:#f2f2ee;--mut:#a3a7ad;--y:#ffd400;--pass:#3ccf7a;--warn:#ffb020;--fail:#ff4d4f;
position:relative;background:var(--bg);color:var(--fg);border:1px solid var(--line);font:13px/1.4 "IBM Plex Mono",ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;container-type:inline-size;-webkit-user-select:none;user-select:none}
.st{position:relative;aspect-ratio:16/10;min-height:260px;max-height:78vh;overflow:hidden;outline:none}
.st:focus-visible{box-shadow:inset 0 0 0 2px var(--y)}
canvas{position:absolute;inset:0;width:100%;height:100%;display:block;touch-action:pan-y}
.ttl{position:absolute;left:12px;right:12px;top:10px;font:600 14px/1.25 Archivo,"Helvetica Neue",Arial,sans-serif;letter-spacing:.01em;pointer-events:none;text-shadow:0 1px 2px #000;max-width:60%}
.ro{position:absolute;left:12px;top:38px;display:grid;gap:2px;pointer-events:none;z-index:1}
.ro div{background:rgba(14,15,17,.82);border-left:2px solid var(--y);padding:3px 8px;white-space:nowrap}
.ro b{font-weight:500;font-variant-numeric:tabular-nums}.ro span{color:var(--mut)}.ro .over{border-color:var(--fail)}.ro .over b{color:var(--fail)}
.lg{position:absolute;right:12px;top:10px;width:min(210px,40%);background:rgba(14,15,17,.82);padding:6px 8px;pointer-events:none}
.lg canvas{position:static;height:10px;margin:4px 0 2px}.lg .t{display:flex;justify-content:space-between;color:var(--mut);font-size:11px}
.lg .ov{display:flex;gap:6px;align-items:center;font-size:11px;color:var(--mut);margin-top:3px}.lg .ov i{width:10px;height:10px;background:#f2f2ee}
.lb{position:absolute;left:0;top:0;pointer-events:none;transform:translate(-50%,-50%)}
.lb i{position:absolute;left:-4px;top:-4px;width:8px;height:8px;border-radius:50%;background:var(--y);box-shadow:0 0 0 2px #111}
.lb span{position:absolute;left:10px;top:-10px;white-space:nowrap;background:var(--y);color:#111;padding:1px 6px;font-weight:500}
.lb.fail i{background:var(--fail)}.lb.fail span{background:var(--fail);color:#fff}.lb.note i{background:var(--fg)}.lb.note span{background:rgba(14,15,17,.88);color:var(--fg);border:1px solid var(--line)}
.tag{position:absolute;right:12px;bottom:10px;color:var(--mut);font-size:11px;pointer-events:none;text-align:right}
.tag em{font-style:normal;color:var(--y)}
.vd{position:absolute;left:50%;bottom:34px;transform:translate(-50%,8px);width:min(92%,640px);background:rgba(14,15,17,.92);border:1px solid var(--line);border-top:3px solid var(--y);padding:10px 14px;opacity:0;transition:opacity .35s,transform .35s;pointer-events:none}
.vd.on{opacity:1;transform:translate(-50%,0)}.vd b{font:800 20px/1 Archivo,"Helvetica Neue",Arial,sans-serif;letter-spacing:.04em;margin-right:10px}
.vd.pass{border-top-color:var(--pass)}.vd.pass b{color:var(--pass)}.vd.warn{border-top-color:var(--warn)}.vd.warn b{color:var(--warn)}.vd.fail{border-top-color:var(--fail)}.vd.fail b{color:var(--fail)}
.ts{position:absolute;left:50%;top:44px;transform:translateX(-50%);background:var(--y);color:#111;padding:3px 10px;font-weight:500;opacity:0;transition:opacity .25s;pointer-events:none;white-space:nowrap}
.ts.on{opacity:1}.ts.fail{background:var(--fail);color:#fff}
.pr{position:absolute;pointer-events:none;background:rgba(14,15,17,.92);border:1px solid var(--y);padding:3px 7px;transform:translate(12px,-50%);white-space:nowrap}
.msg{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;padding:24px;text-align:center;color:var(--mut)}
.bar{display:flex;align-items:center;gap:8px;padding:8px 10px;border-top:1px solid var(--line);background:var(--panel);flex-wrap:wrap}
button,select{font:inherit;color:var(--fg);background:transparent;border:1px solid var(--line);padding:4px 9px;cursor:pointer;min-height:30px}
button:hover,select:hover{border-color:var(--y)}button:focus-visible,select:focus-visible,input:focus-visible{outline:2px solid var(--y);outline-offset:1px}
button[aria-pressed=true]{background:var(--y);color:#111;border-color:var(--y)}
.pl{width:38px;font-weight:600}.trk{position:relative;flex:1 1 180px;min-width:120px;height:30px;display:flex;align-items:center}
input[type=range]{width:100%;accent-color:var(--y);margin:0;background:transparent}
.ev{position:absolute;top:1px;width:3px;height:8px;background:var(--y);transform:translateX(-50%);cursor:pointer;border:0;padding:0;min-height:0}
.ev.fail{background:var(--fail)}
.tm{min-width:15ch;font-variant-numeric:tabular-nums;white-space:nowrap}
.nt{padding:6px 10px 8px;color:var(--mut);font-size:11.5px;border-top:1px solid var(--line);display:grid;gap:2px}
@container (max-width:620px){.ro{position:static;display:flex;flex-wrap:wrap;gap:4px;padding:8px 10px 0;font-size:12px}.ttl{max-width:54%;font-size:12.5px}.st{aspect-ratio:4/3}.tm{min-width:0;flex:1 1 100%;order:9}.lg{width:42%;font-size:11px;padding:4px 6px}.lg .t,.lg .ov{font-size:10px}.vd{bottom:10px;padding:7px 10px;font-size:12px}.vd b{font-size:15px}.tag{display:none}}
@media (prefers-reduced-motion:reduce){.vd,.ts{transition:none}}
`;

// ---------- small math, z up
const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const crs = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const nrm = (a) => { const l = Math.hypot(a[0], a[1], a[2]) || 1; return [a[0] / l, a[1] / l, a[2] / l]; };
const lerp = (a, b, k) => a + (b - a) * k;
const clamp = (x, a, b) => Math.min(b, Math.max(a, x));
function mul(a, b) {
  const o = new Float32Array(16);
  for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) {
    let s = 0;
    for (let k = 0; k < 4; k++) s += a[k * 4 + j] * b[i * 4 + k];
    o[i * 4 + j] = s;
  }
  return o;
}
function persp(fov, asp, n, f) {
  const t = 1 / Math.tan((fov * Math.PI) / 360), o = new Float32Array(16);
  o[0] = t / asp; o[5] = t; o[10] = (f + n) / (n - f); o[11] = -1; o[14] = (2 * f * n) / (n - f);
  return o;
}
function look(e, c) {
  const z = nrm(sub(e, c)); let x = crs([0, 0, 1], z);
  x = Math.hypot(...x) < 1e-6 ? [1, 0, 0] : nrm(x);
  const y = crs(z, x);
  return new Float32Array([x[0], y[0], z[0], 0, x[1], y[1], z[1], 0, x[2], y[2], z[2], 0, -dot(x, e), -dot(y, e), -dot(z, e), 1]);
}
// a rigid transform from a quaternion (w, x, y, z) and a position, column major
function pose(q, p) {
  const [w, x, y, z] = q;
  return new Float32Array([1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y), 0, 2 * (x * y - w * z), 1 - 2 * (x * x + z * z),
    2 * (y * z + w * x), 0, 2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y), 0, p[0], p[1], p[2], 1]);
}
const I4 = pose([1, 0, 0, 0], [0, 0, 0]);
// the rotation taking +z onto direction d, as a quaternion
function zTo(d) {
  const v = nrm(d), c = v[2];
  if (c < -0.999999) return [0, 1, 0, 0];
  const w = Math.sqrt((1 + c) / 2), s = 1 / (2 * w);
  return [w, -v[1] * s, v[0] * s, 0];
}
function rot(q, v) { const m = pose(q, [0, 0, 0]); return [m[0] * v[0] + m[4] * v[1] + m[8] * v[2], m[1] * v[0] + m[5] * v[1] + m[9] * v[2], m[2] * v[0] + m[6] * v[1] + m[10] * v[2]]; }

// ---------- colors
const hex = (h) => { const n = parseInt(h.slice(1), 16); return [(n >> 16) / 255, ((n >> 8) & 255) / 255, (n & 255) / 255]; };
const STOPS = {
  viridis: '440154 482878 3e4989 31688e 26828e 1f9e89 35b779 6ece58 b5de2b fde725',
  inferno: '000004 1b0c41 4a0c6b 781c6d a52c60 cf4446 ed6925 fb9b06 f7d13d fcffa4',
  coolwarm: '3b4cc0 6889ee 9abbff c9d8ef edd1c2 f7a889 e26952 b40426',
  blues: '0b1a33 0f3b6e 1663a6 2f8fd1 6bb8e8 b6e0f7',
  gray: '303338 e6e6e6',
  heat: '9a9488 c9a05a e8742a f2401f',
};
function turbo(x) {
  const r = 0.13572138 + x * (4.6153926 + x * (-42.66032258 + x * (132.13108234 + x * (-152.94239396 + x * 59.28637943))));
  const g = 0.09140261 + x * (2.19418839 + x * (4.84296658 + x * (-14.18503333 + x * (4.27729857 + x * 2.82956604))));
  const b = 0.1066733 + x * (12.64194608 + x * (-60.58204836 + x * (110.36276771 + x * (-89.90310912 + x * 27.34824973))));
  return [clamp(r, 0, 1), clamp(g, 0, 1), clamp(b, 0, 1)];
}
const LUTS = {};
function lut(name) {
  if (LUTS[name]) return LUTS[name];
  const o = new Uint8Array(256 * 3), s = (STOPS[name] || '').split(' ').filter(Boolean).map((h) => hex('#' + h));
  for (let i = 0; i < 256; i++) {
    const x = i / 255;
    let c;
    if (!s.length) c = turbo(x);
    else { const f = x * (s.length - 1), a = Math.min(Math.floor(f), s.length - 2), k = f - a; c = [0, 1, 2].map((j) => lerp(s[a][j], s[a + 1][j], k)); }
    o.set(c.map((v) => Math.round(v * 255)), i * 3);
  }
  return (LUTS[name] = o);
}
const OVER = [242, 242, 238], NONE = [85, 88, 94];
const CPK = { H: 'f2f2f2', C: '555a60', N: '3b6fe0', O: 'e8423b', F: '7fd17f', P: 'ff8c1a', S: 'f2d230', Cl: '33d14a', Na: 'a35cf2', K: '8f40d4', Li: 'cc80ff', Mg: '8aff00', Al: 'bfa6a6', Si: 'd9b380', Fe: 'e06633', Cu: 'c88033', Zn: '7d80b0', Ti: 'bfc2c7', Ni: '50d050', Ca: '3dff00', B: 'ffb5b5', Ar: '80d1e3', He: 'd9ffff' };
const RAD = { H: 0.31, C: 0.76, N: 0.71, O: 0.66, F: 0.57, S: 1.05, P: 1.07, Cl: 1.02, Na: 1.66, K: 2.03, Li: 1.28, Fe: 1.32, Al: 1.21, Si: 1.11, Cu: 1.32, Ti: 1.6 };

// ---------- the file
async function bytesOf(src) {
  if (src instanceof Uint8Array) return src;
  if (src instanceof ArrayBuffer) return new Uint8Array(src);
  if (/^[A-Za-z0-9+/=\s]+$/.test(src) && src.length > 64) { const b = atob(src.replace(/\s+/g, '')), o = new Uint8Array(b.length); for (let i = 0; i < b.length; i++) o[i] = b.charCodeAt(i); return o; }
  const r = await fetch(src);
  if (!r.ok) throw new Error(`could not load the scene (${r.status})`);
  return new Uint8Array(await r.arrayBuffer());
}
async function decode(src) {
  let raw = await bytesOf(src);
  if (raw[0] === 0x1f && raw[1] === 0x8b) raw = new Uint8Array(await new Response(new Blob([raw]).stream().pipeThrough(new DecompressionStream('gzip'))).arrayBuffer());
  if (String.fromCharCode(...raw.subarray(0, 4)) !== 'WBSC') throw new Error('this is not a Wobbl scene');
  const n = new DataView(raw.buffer, raw.byteOffset).getUint32(8, true), h = JSON.parse(new TextDecoder().decode(raw.subarray(12, 12 + n)));
  let at = 12 + n; at += (8 - (at % 8)) % 8;
  const blob = raw.slice(at), T = { u8: Uint8Array, u16: Uint16Array, u32: Uint32Array, f32: Float32Array };
  h.arr = (d) => {
    if (!d) return null;
    const b = h.buffers[d.b], q = new T[b.t](blob.buffer.slice(b.o, b.o + b.n));
    if (!d.lo) return q;
    const M = b.t === 'u8' ? 255 : 65535, top = d.nan ? M - 1 : M, c = d.c, o = new Float32Array(q.length);
    for (let i = 0; i < q.length; i++) { const k = i % c; o[i] = d.nan && q[i] === M ? NaN : d.lo[k] + (q[i] / top) * (d.hi[k] - d.lo[k]); }
    return o;
  };
  return h;
}

// ---------- geometry: every primitive is a profile turned round z
function lathe(prof, seg, flat) {
  const P = [], N = [], I = [], np = prof.length;
  const cols = flat ? seg * 2 : seg + 1;
  for (let c = 0; c < cols; c++) {
    const j = flat ? (c >> 1) + (c & 1) : c, a = (j / seg) * Math.PI * 2, an = flat ? (((c >> 1) + 0.5) / seg) * Math.PI * 2 : a;
    for (const [r, z, nr, nz] of prof) { P.push(r * Math.cos(a), r * Math.sin(a), z); N.push(nr * Math.cos(an), nr * Math.sin(an), nz); }
  }
  for (let c = 0; c < cols - 1; c += flat ? 2 : 1) for (let i = 0; i < np - 1; i++) {
    const a = c * np + i, b = a + np;
    I.push(a, b, a + 1, a + 1, b, b + 1);
  }
  return { P, N, I };
}
function arc(r, z0, a0, a1, n, out) { for (let i = 0; i <= n; i++) { const a = lerp(a0, a1, i / n); out.push([r * Math.cos(a), z0 + r * Math.sin(a), Math.cos(a), Math.sin(a)]); } return out; }
const cylP = (r, h) => [[0, -h, 0, -1], [r, -h, 0, -1], [r, -h, 1, 0], [r, h, 1, 0], [r, h, 0, 1], [0, h, 0, 1]];
function prim(shape, s) {
  const H = Math.PI / 2;
  switch (shape) {
    case 'sphere': case 'ellipsoid': {
      const g = lathe(arc(1, 0, -H, H, 14, []), 28);
      const k = shape === 'sphere' ? [s[0], s[0], s[0]] : s;
      for (let i = 0; i < g.P.length; i++) { g.P[i] *= k[i % 3]; g.N[i] /= k[i % 3]; }
      return g;
    }
    case 'capsule': return lathe(arc(s[0], s[1], 0, H, 7, arc(s[0], -s[1], -H, 0, 7, [])), 24);
    case 'cylinder': return lathe(cylP(s[0], s[1]), 32);
    case 'hex': return lathe(cylP(s[0], s[1]), 6, true);
    case 'cone': { const l = Math.hypot(2 * s[1], s[0]); return lathe([[0, -s[1], 0, -1], [s[0], -s[1], 0, -1], [s[0], -s[1], (2 * s[1]) / l, s[0] / l], [0, s[1], (2 * s[1]) / l, s[0] / l]], 28); }
    case 'torus': { const p = []; for (let i = 0; i <= 16; i++) { const a = (i / 16) * Math.PI * 2; p.push([s[0] + s[1] * Math.cos(a), s[1] * Math.sin(a), Math.cos(a), Math.sin(a)]); } return lathe(p, 32); }
    case 'plane': return { P: [-s[0], -s[1], 0, s[0], -s[1], 0, s[0], s[1], 0, -s[0], s[1], 0], N: [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1], I: [0, 1, 2, 0, 2, 3] };
    default: { // box
      const P = [], N = [], I = [];
      for (let ax = 0; ax < 3; ax++) for (const sg of [-1, 1]) {
        const u = (ax + 1) % 3, v = (ax + 2) % 3, b = P.length / 3;
        for (const [du, dv] of [[-1, -1], [1, -1], [1, 1], [-1, 1]]) {
          const p = [0, 0, 0], n = [0, 0, 0];
          p[ax] = sg * s[ax]; p[u] = du * sg * s[u]; p[v] = dv * s[v]; n[ax] = sg;
          P.push(...p); N.push(...n);
        }
        I.push(b, b + 1, b + 2, b, b + 2, b + 3);
      }
      return { P, N, I };
    }
  }
}
// a tube from a to b of radius r, and an arrow from a along v
function placed(g, q, at, P, N, I, base) {
  const m = pose(q, at);
  for (let i = 0; i < g.P.length; i += 3) {
    const x = g.P[i], y = g.P[i + 1], z = g.P[i + 2], nx = g.N[i], ny = g.N[i + 1], nz = g.N[i + 2];
    P.push(m[0] * x + m[4] * y + m[8] * z + m[12], m[1] * x + m[5] * y + m[9] * z + m[13], m[2] * x + m[6] * y + m[10] * z + m[14]);
    N.push(m[0] * nx + m[4] * ny + m[8] * nz, m[1] * nx + m[5] * ny + m[9] * nz, m[2] * nx + m[6] * ny + m[10] * nz);
  }
  for (const i of g.I) I.push(i + base);
}
function tube(a, b, r, P, N, I, seg = 8) {
  const d = sub(b, a), l = Math.hypot(...d);
  if (l < 1e-9) return;
  placed(lathe([[r, 0, 1, 0], [r, l, 1, 0]], seg), zTo(d), a, P, N, I, P.length / 3);
}
function arrow(a, v, r, P, N, I) {
  const l = Math.hypot(...v);
  if (l < 1e-9) return;
  const hl = Math.min(l * 0.45, r * 7), hr = r * 2.6, k = hr / Math.hypot(hl, hr), c = hl / Math.hypot(hl, hr);
  const prof = [[0, 0, 0, -1], [r, 0, 0, -1], [r, 0, 1, 0], [r, l - hl, 1, 0], [r, l - hl, 0, -1], [hr, l - hl, 0, -1], [hr, l - hl, c, k], [0, l, c, k]];
  placed(lathe(prof, 12), zTo(v), a, P, N, I, P.length / 3);
}

// ---------- GL
const VS = `#version 300 es
in vec3 p;in vec3 n;in vec4 c;in float r;in vec4 io;in vec4 ic;
uniform mat4 M,VP;uniform float inst;
out vec3 vn,vw;out vec4 vc;out float vr;
void main(){vec3 q=inst>.5?p*io.w+io.xyz:p;vec4 w=M*vec4(q,1.);vw=w.xyz;vn=mat3(M)*n;vc=inst>.5?ic:c;vr=r;gl_Position=VP*w;}`;
const FS = `#version 300 es
precision highp float;in vec3 vn,vw;in vec4 vc;in float vr;
uniform vec3 eye,gc;uniform float th,prop,glow,alpha;out vec4 o;
void main(){if(vr>th)discard;vec3 N=normalize(vn),v=normalize(eye-vw);if(dot(N,v)<0.)N=-N;
vec3 l1=normalize(vec3(.35,.55,.75)),l2=normalize(vec3(-.7,-.1,.45));
float d=.34+.6*max(dot(N,l1),0.)+.2*max(dot(N,l2),0.),s=.16*pow(max(dot(N,normalize(l1+v)),0.),28.);
vec3 col=vc.rgb*d+s;if(prop>.5&&mod(gl_FragCoord.x+gl_FragCoord.y,12.)<6.)col*=.8;
col=mix(col,gc,glow);o=vec4(col,vc.a*alpha);}`;
const LVS = `#version 300 es
in vec3 a;in vec3 b;in float sd;in vec4 c;in float s;in float g;in float td;
uniform mat4 VP;uniform vec2 vp;uniform float w;out vec4 vc;out float vs,vg,vt;
void main(){vec4 A=VP*vec4(a,1.),B=VP*vec4(b,1.);vc=c;vs=s;vg=g;vt=td;
if(A.w<1e-4||B.w<1e-4){gl_Position=vec4(2.,2.,2.,1.);return;}
vec2 d=normalize((B.xy/B.w-A.xy/A.w)*vp+1e-6);A.xy+=vec2(-d.y,d.x)*sd*w/vp*A.w;gl_Position=A;}`;
const LFS = `#version 300 es
precision highp float;in vec4 vc;in float vs,vg,vt;uniform float tn;out vec4 o;
void main(){if(vt>tn)discard;float k=vg*pow(.5+.5*cos(6.2832*vs),6.);o=vec4(mix(vc.rgb,vec3(1.,.97,.7),k),vc.a);}`;

function program(gl, vs, fs) {
  const p = gl.createProgram();
  for (const [t, s] of [[gl.VERTEX_SHADER, vs], [gl.FRAGMENT_SHADER, fs]]) {
    const sh = gl.createShader(t);
    gl.shaderSource(sh, s); gl.compileShader(sh);
    if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(sh));
    gl.attachShader(p, sh);
  }
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
  const loc = {};
  const nA = gl.getProgramParameter(p, gl.ACTIVE_ATTRIBUTES), nU = gl.getProgramParameter(p, gl.ACTIVE_UNIFORMS);
  for (let i = 0; i < nA; i++) { const a = gl.getActiveAttrib(p, i); loc[a.name] = gl.getAttribLocation(p, a.name); }
  for (let i = 0; i < nU; i++) { const u = gl.getActiveUniform(p, i); loc[u.name] = gl.getUniformLocation(p, u.name); }
  return { p, loc };
}

// A drawable: a vertex array with its buffers, refilled whenever its data changes.
function drawable(gl, prog, spec) {
  const vao = gl.createVertexArray(), bufs = {};
  gl.bindVertexArray(vao);
  for (const [name, size, type, norm, div] of spec) {
    const l = prog.loc[name];
    if (l === undefined || l < 0) continue;
    const b = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, b);
    gl.enableVertexAttribArray(l);
    gl.vertexAttribPointer(l, size, type || gl.FLOAT, !!norm, 0, 0);
    if (div) gl.vertexAttribDivisor(l, 1);
    bufs[name] = b;
  }
  const ib = gl.createBuffer();
  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ib);
  gl.bindVertexArray(null);
  const d = {
    vao, count: 0, inst: 0, big: false,
    set(name, data) { if (!bufs[name]) return; gl.bindBuffer(gl.ARRAY_BUFFER, bufs[name]); gl.bufferData(gl.ARRAY_BUFFER, data, gl.DYNAMIC_DRAW); },
    index(data) {
      d.big = data.some((x) => x > 65535);
      gl.bindVertexArray(vao); gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ib);
      gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, d.big ? new Uint32Array(data) : new Uint16Array(data), gl.STATIC_DRAW);
      gl.bindVertexArray(null); d.count = data.length;
    },
    free() { for (const b of Object.values(bufs)) gl.deleteBuffer(b); gl.deleteBuffer(ib); gl.deleteVertexArray(vao); },
  };
  return d;
}

// One vertex per smooth group of triangles round each node, so sharp CAD edges shade sharp and fillets smooth.
function shade(pos, tri, n, flat) {
  const nt = tri.length / 3, COS = flat ? 2 : Math.cos((35 * Math.PI) / 180), un = faceN(pos, tri, true);
  const start = new Uint32Array(n + 1);
  for (let j = 0; j < tri.length; j++) start[tri[j] + 1]++;
  for (let i = 0; i < n; i++) start[i + 1] += start[i];
  const fill = start.slice(0, n), slots = new Uint32Array(tri.length);
  for (let j = 0; j < tri.length; j++) slots[fill[tri[j]]++] = j;
  const index = new Uint32Array(tri.length), vn = [];
  const par = new Int32Array(256), ids = new Int32Array(256);
  const find = (x) => { while (par[x] !== x) x = par[x] = par[par[x]]; return x; };
  for (let v = 0; v < n; v++) {
    const s0 = start[v], d = Math.min(start[v + 1] - s0, 256);
    for (let p = 0; p < d; p++) { par[p] = p; ids[p] = -1; }
    for (let p = 0; p < d; p++) for (let q = p + 1; q < d; q++) {
      const tp = (slots[s0 + p] / 3) | 0, tq = (slots[s0 + q] / 3) | 0;
      if (un[tp * 3] * un[tq * 3] + un[tp * 3 + 1] * un[tq * 3 + 1] + un[tp * 3 + 2] * un[tq * 3 + 2] >= COS) par[find(p)] = find(q);
    }
    for (let p = 0; p < d; p++) { const r = find(p); if (ids[r] < 0) { ids[r] = vn.length; vn.push(v); } index[slots[s0 + p]] = ids[r]; }
  }
  return { index, vNode: Uint32Array.from(vn), nt };
}
function faceN(pos, tri, unit) {
  const o = new Float32Array(tri.length);
  for (let t = 0; t < tri.length; t += 3) {
    const a = tri[t] * 3, b = tri[t + 1] * 3, c = tri[t + 2] * 3;
    const ux = pos[b] - pos[a], uy = pos[b + 1] - pos[a + 1], uz = pos[b + 2] - pos[a + 2], vx = pos[c] - pos[a], vy = pos[c + 1] - pos[a + 1], vz = pos[c + 2] - pos[a + 2];
    let x = uy * vz - uz * vy, y = uz * vx - ux * vz, z = ux * vy - uy * vx;
    if (unit) { const l = Math.hypot(x, y, z) || 1; x /= l; y /= l; z /= l; }
    o[t] = x; o[t + 1] = y; o[t + 2] = z;
  }
  return o;
}

const fmt = (v, d = 3) => {
  if (v === null || v === undefined || Number.isNaN(v)) return '–';
  const a = Math.abs(v);
  if (a !== 0 && (a >= 1e6 || a < 1e-3)) return v.toExponential(Math.max(d - 1, 1));
  const dec = a === 0 ? 0 : Math.max(0, d - 1 - Math.floor(Math.log10(a)));
  return v.toLocaleString('en-US', { minimumFractionDigits: Math.min(dec, 6), maximumFractionDigits: Math.min(dec, 6) });
};
const el = (tag, cls, parent, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text !== undefined) e.textContent = text; if (parent) parent.append(e); return e; };
