// The scene's objects: each one decoded into what the GPU draws, brought to any playback position and drawn.
// G holds the GL context and programs, the color fields and the scene's size; without WebGL, G.gl is null and
// the objects still follow playback so readouts and labels work.

const colorOf = (f, v, out, o) => {
  const c = Number.isNaN(v) ? NONE : f.limit !== undefined && v > f.limit ? OVER : null;
  if (c) { out[o] = c[0]; out[o + 1] = c[1]; out[o + 2] = c[2]; return; }
  const i = clamp(Math.round(((v - f.lo) / (f.hi - f.lo || 1)) * 255), 0, 255) * 3;
  out[o] = f.lut[i]; out[o + 1] = f.lut[i + 1]; out[o + 2] = f.lut[i + 2];
};
// a per frame array at playback (a, b, k), item i of n per frame
const at2 = (arr, n, a, b, k, i) => (k ? lerp(arr[a * n + i], arr[b * n + i], k) : arr[a * n + i]);
const series = (s, a, b, k) => (s.length === 1 ? s[0] : lerp(s[a], s[b], k));
const shown = (w, t) => !w || ((w[0] === null || t >= w[0] - 1e-9) && (w[1] === null || t <= w[1] + 1e-9));
const nrm4 = (q) => { const l = Math.hypot(...q) || 1; return q.map((v) => v / l); };
const SP = [['p', 3], ['n', 3], ['c', 4, 5121, true], ['r', 1]];
const SPI = [...SP, ['io', 4, 0, false, 1], ['ic', 4, 5121, true, 1]];
const LP = [['a', 3], ['b', 3], ['sd', 1], ['c', 4, 5121, true], ['s', 1], ['g', 1], ['td', 1]];
const fixed = (d, g) => { d.set('p', new Float32Array(g.P)); d.set('n', new Float32Array(g.N)); d.set('r', new Float32Array(g.P.length / 3)); d.index(g.I); };

function context(S, gl) {
  const G = { gl, fields: S.fields.map((f) => ({ ...f, lut: lut(f.colormap || 'turbo'), lo: f.range[0], hi: f.range[1] })) };
  if (gl) {
    Object.assign(G, { surf: program(gl, VS, FS), line: program(gl, LVS, LFS), sph: prim('sphere', [1]), low: lathe(arc(1, 0, -Math.PI / 2, Math.PI / 2, 5, []), 10) });
    gl.enable(gl.DEPTH_TEST);
    gl.clearColor(0.055, 0.059, 0.067, 1);
  }
  const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  const grow = (P, w) => { for (let i = 0; i < P.length; i += 3) for (let k = 0; k < 3; k++) { const v = P[i + k] + (w ? w[k] : 0); if (Number.isFinite(v)) { lo[k] = Math.min(lo[k], v); hi[k] = Math.max(hi[k], v); } } };
  const arr = (x) => (x.b !== undefined ? S.arr(x) : x);
  G.objs = S.objects.map((o) => build(S, G, o, grow, arr));
  for (let k = 0; k < 3; k++) if (!Number.isFinite(lo[k])) { lo[k] = -1; hi[k] = 1; }
  G.center = lo.map((v, k) => (v + hi[k]) / 2);
  G.radius = Math.max(Math.hypot(...sub(hi, lo)) / 2, 1e-6);
  return G;
}

function build(S, G, o, grow, arr) {
  const gl = G.gl, ob = { o, prop: !!o.prop, alpha: o.opacity ?? 1, M: I4, glow: 0, gcol: [1, 0.85, 0.2] };
  const base = hex(o.color || '#9aa0a6').map((v) => Math.round(v * 255));
  if (o.field) ob.f = G.fields[o.field.f];
  if (o.kind === 'mesh') {
    const P = S.arr(o.pos), tri = S.arr(o.tri), n = P.length / 3;
    Object.assign(ob, { P, tri, n, sh: shade(P, tri, n, o.flat), cur: P.slice(), val: new Float32Array(n).fill(NaN), base });
    if (o.field) ob.fv = S.arr(o.field.values || o.field.frames);
    if (o.deform) ob.du = S.arr(o.deform.values || o.deform.frames);
    if (o.reveal) ob.rv = S.arr(o.reveal.values);
    if (o.pose) { ob.pp = arr(o.pose.pos); ob.pq = arr(o.pose.quat); }
    grow(P, Array.isArray(ob.pp) ? ob.pp : null);
    if (gl) { ob.d = drawable(gl, G.surf, SP); ob.d.index(Array.from(ob.sh.index)); }
  } else if (o.kind === 'body') {
    ob.pp = arr(o.pos); ob.pq = arr(o.quat); ob.color = base;
    if (o.glow) ob.gcol = hex(o.color).map((v) => Math.min(1, v * 0.5 + 0.55));
    const p0 = Array.from(ob.pp.slice(0, 3)), r0 = Math.max(...o.size.slice(0, 3));
    grow([p0[0] - r0, p0[1] - r0, p0[2] - r0, p0[0] + r0, p0[1] + r0, p0[2] + r0]);
    if (ob.pp.length > 3) grow(ob.pp);
    if (gl) { const g = prim(o.shape, o.size); ob.d = drawable(gl, G.surf, SP); fixed(ob.d, g); ob.cbuf = new Uint8Array((g.P.length / 3) * 4); }
  } else if (o.kind === 'atoms' || o.kind === 'points') {
    const atoms = o.kind === 'atoms';
    ob.P = S.arr(o.pos); ob.n = atoms ? o.el.length : o.pos.shape[o.pos.shape.length - 2];
    grow(ob.P.subarray(0, ob.n * 3));
    ob.cols = new Uint8Array(ob.n * 4); ob.rad = new Float32Array(ob.n);
    for (let i = 0; i < ob.n; i++) {
      const e = atoms ? o.el[i] : '', c = e ? hex('#' + (CPK[e] || 'c0c4ca')) : base.map((v) => v / 255);
      ob.cols.set([c[0] * 255, c[1] * 255, c[2] * 255, 255], i * 4);
      ob.rad[i] = e ? (RAD[e] || 1.2) * o.radius : o.size;
    }
    if (o.field) ob.fv = S.arr(o.field.values || o.field.frames);
    if (o.bonds) ob.bonds = S.arr(o.bonds);
    if (gl) {
      ob.d = drawable(gl, G.surf, SPI); fixed(ob.d, atoms ? G.sph : G.low); ob.d.inst = ob.n;
      if (ob.bonds || o.cell) ob.b = drawable(gl, G.surf, SP);
    }
  } else if (o.kind === 'lines' || o.kind === 'ground') {
    if (o.kind === 'ground') {
      const s = o.size / 2, c = o.center, z = o.z, ps = [], runs = [];
      if (o.grid) for (let x = -s; x <= s + 1e-9; x += o.grid) { ps.push(c[0] + x, c[1] - s, z, c[0] + x, c[1] + s, z, c[0] - s, c[1] + x, z, c[0] + s, c[1] + x, z); runs.push(2, 2); }
      Object.assign(ob, { P: new Float32Array(ps), runs, width: 1, lineCol: [255, 255, 255, 22] });
      if (gl) {
        ob.d = drawable(gl, G.surf, SP); fixed(ob.d, prim('plane', [s, s]));
        ob.d.set('c', new Uint8Array([...base, 255, ...base, 255, ...base, 255, ...base, 255]));
        ob.M = pose([1, 0, 0, 0], [c[0], c[1], z - 1e-3 * s]);
      }
    } else {
      Object.assign(ob, { P: S.arr(o.pos), runs: o.runs, width: o.width, lineCol: [...base, 255] });
      grow(ob.P);
      if (o.field) ob.fv = S.arr(o.field.values);
      if (o.drawn) ob.td = S.arr(o.drawn);
    }
    if (gl && ob.runs.length) ob.l = drawable(gl, G.line, LP);
    ribbons(ob);
    if (o.kind === 'ground' && ob.l) { for (let i = 0; i < ob.cl.length; i += 4) ob.cl.set(ob.lineCol, i); ob.l.set('c', ob.cl); ob.l.set('g', ob.gl); ob.l.set('s', ob.gl); }
  } else if (o.kind === 'arrows') {
    ob.O = S.arr(o.origin); ob.V = S.arr(o.vec); ob.n = o.vec.shape[o.vec.shape.length - 2]; ob.color = base;
    grow(ob.O.subarray(0, ob.n * 3));
    if (gl) ob.d = drawable(gl, G.surf, SP);
  }
  return ob;
}

// Lines become ribbons, four vertices a segment, each knowing its line, its distance along it and when it is drawn.
function ribbons(ob) {
  let segs = 0;
  for (const r of ob.runs) segs += Math.max(r - 1, 0);
  const A = new Float32Array(segs * 12), B = new Float32Array(segs * 12), SD = new Float32Array(segs * 4), SS = new Float32Array(segs * 4), TD = new Float32Array(segs * 4), I = [];
  Object.assign(ob, { segs, segRun: new Uint32Array(segs), segPt: new Uint32Array(segs) });
  let s = 0, p = 0;
  for (const [ri, r] of ob.runs.entries()) {
    let arc0 = 0;
    for (let i = 0; i < r - 1; i++, s++) {
      const a = (p + i) * 3, b = a + 3, len = Math.hypot(ob.P[b] - ob.P[a], ob.P[b + 1] - ob.P[a + 1], ob.P[b + 2] - ob.P[a + 2]);
      ob.segRun[s] = ri; ob.segPt[s] = p + i;
      for (let v = 0; v < 4; v++) {
        const at = v < 2 ? a : b, ot = v < 2 ? b : a;
        A.set(ob.P.subarray(at, at + 3), (s * 4 + v) * 3); B.set(ob.P.subarray(ot, ot + 3), (s * 4 + v) * 3);
        SD[s * 4 + v] = [1, -1, -1, 1][v]; SS[s * 4 + v] = v < 2 ? arc0 : arc0 + len;
        TD[s * 4 + v] = ob.td ? ob.td[v < 2 ? p + i : p + i + 1] : -1e30;
      }
      arc0 += len;
      I.push(s * 4, s * 4 + 1, s * 4 + 2, s * 4, s * 4 + 2, s * 4 + 3);
    }
    p += r;
  }
  Object.assign(ob, { arc: SS, cl: new Uint8Array(segs * 16), gl: new Float32Array(segs * 4) });
  if (ob.l) { ob.l.set('a', A); ob.l.set('b', B); ob.l.set('sd', SD); ob.l.set('td', TD); ob.l.index(I); }
}

// Bring one object to playback position (a, b, k) at time t.
function frame(G, ob, a, b, k, t) {
  const o = ob.o, gl = G.gl;
  if (o.glow) ob.glow = series(o.glow, a, b, k);
  if (ob.pp && ob.pq) {
    const pp = ob.pp.length > 3 ? [0, 1, 2].map((i) => at2(ob.pp, 3, a, b, k, i)) : ob.pp;
    const q = ob.pq.length > 4 ? nrm4([0, 1, 2, 3].map((i) => at2(ob.pq, 4, a, b, k, i))) : ob.pq;
    ob.M = pose(q, pp); ob.at = pp;
  }
  if (o.kind === 'mesh') meshFrame(gl, ob, a, b, k);
  else if (o.kind === 'body') {
    if (ob.f) colorOf(ob.f, series(o.field.values, a, b, k), ob.color = [0, 0, 0], 0);
    if (gl) { for (let i = 0; i < ob.cbuf.length; i += 4) ob.cbuf.set([...ob.color, 255], i); ob.d.set('c', ob.cbuf); }
  } else if (o.kind === 'atoms' || o.kind === 'points') pointsFrame(gl, ob, a, b, k);
  else if (o.kind === 'lines') linesFrame(ob, t, a, b, k);
  else if (o.kind === 'arrows') arrowsFrame(G, ob, a, b, k);
}

function meshFrame(gl, ob, a, b, k) {
  const o = ob.o, n = ob.n, P = ob.P, cur = ob.cur;
  if (ob.du) {
    const ex = o.deform.exaggerate || 1, sc = o.deform.scale;
    if (sc) { const s = series(sc, a, b, k) * ex; for (let i = 0; i < n * 3; i++) cur[i] = P[i] + ob.du[i] * s; }
    else for (let i = 0; i < n * 3; i++) cur[i] = P[i] + at2(ob.du, n * 3, a, b, k, i) * ex;
  }
  if (ob.fv) {
    const sc = o.field.scale;
    if (sc) { const s = series(sc, a, b, k); for (let i = 0; i < n; i++) ob.val[i] = ob.fv[i] * s; }
    else if (o.field.frames) for (let i = 0; i < n; i++) ob.val[i] = at2(ob.fv, n, a, b, k, i);
    else ob.val.set(ob.fv);
  }
  ob.th = o.reveal ? series(o.reveal.at, a, b, k) : 1e30;
  if (!gl) return;
  const sh = ob.sh, nv = sh.vNode.length;
  if (!ob.pbuf || ob.du) {
    const fnm = faceN(cur, ob.tri, false), pb = ob.pbuf || (ob.pbuf = new Float32Array(nv * 3)), nb = new Float32Array(nv * 3);
    for (let v = 0; v < nv; v++) pb.set(cur.subarray(sh.vNode[v] * 3, sh.vNode[v] * 3 + 3), v * 3);
    for (let j = 0; j < sh.index.length; j++) { const v = sh.index[j] * 3, f = ((j / 3) | 0) * 3; nb[v] += fnm[f]; nb[v + 1] += fnm[f + 1]; nb[v + 2] += fnm[f + 2]; }
    ob.d.set('p', pb); ob.d.set('n', nb);
  }
  if (!ob.cbuf || ob.fv) {
    const cb = ob.cbuf || (ob.cbuf = new Uint8Array(nv * 4)), tmp = [0, 0, 0];
    for (let v = 0; v < nv; v++) {
      if (ob.f) { colorOf(ob.f, ob.val[sh.vNode[v]], tmp, 0); cb.set(tmp, v * 4); } else cb.set(ob.base, v * 4);
      cb[v * 4 + 3] = 255;
    }
    ob.d.set('c', cb);
  }
  if (!ob.rbuf) { ob.rbuf = new Float32Array(nv); if (ob.rv) for (let v = 0; v < nv; v++) ob.rbuf[v] = ob.rv[sh.vNode[v]]; ob.d.set('r', ob.rbuf); }
}

function pointsFrame(gl, ob, a, b, k) {
  const n = ob.n, P = ob.P, many = P.length > n * 3, io = new Float32Array(n * 4), tmp = [0, 0, 0];
  ob.cur = ob.cur || new Float32Array(n * 3);
  for (let i = 0; i < n * 3; i++) ob.cur[i] = many ? at2(P, n * 3, a, b, k, i) : P[i];
  for (let i = 0; i < n; i++) {
    io.set(ob.cur.subarray(i * 3, i * 3 + 3), i * 4); io[i * 4 + 3] = Number.isNaN(ob.cur[i * 3]) ? 0 : ob.rad[i];
    if (ob.f) { colorOf(ob.f, ob.o.field.frames ? at2(ob.fv, n, a, b, k, i) : ob.fv[i], tmp, 0); ob.cols.set(tmp, i * 4); }
  }
  if (!gl) return;
  ob.d.set('io', io); ob.d.set('ic', ob.cols);
  if (!ob.b) return;
  const Pp = [], Nn = [], Ii = [], c = ob.cur, o = ob.o, pt = (i) => Array.from(c.subarray(i * 3, i * 3 + 3));
  if (ob.bonds) for (let j = 0; j < ob.bonds.length; j += 2) tube(pt(ob.bonds[j]), pt(ob.bonds[j + 1]), 0.11 * (o.radius / 0.35), Pp, Nn, Ii, 8);
  if (o.cell) {
    const { origin: q, a: A, b: B, c: C } = o.cell, corner = (u) => [0, 1, 2].map((x) => q[x] + u[0] * A[x] + u[1] * B[x] + u[2] * C[x]);
    for (const u of [[0, 0, 0], [1, 1, 0], [1, 0, 1], [0, 1, 1]]) for (let x = 0; x < 3; x++) { const v = [...u]; v[x] = 1 - v[x]; tube(corner(u), corner(v), 0.03, Pp, Nn, Ii, 6); }
  }
  fixed(ob.b, { P: Pp, N: Nn, I: Ii });
  ob.b.set('c', new Uint8Array((Pp.length / 3) * 4).fill(150));
}

function linesFrame(ob, t, a, b, k) {
  const o = ob.o, tmp = [0, 0, 0];
  let speeds = null, newest = -Infinity;
  if (o.flow) { const sp = o.flow.speed; speeds = o.flow.line_speed || (Array.isArray(sp[0]) ? sp[a].map((v, i) => (k ? lerp(v, sp[b][i], k) : v)) : [sp[0]]); }
  if (o.at) for (const v of o.at) if (v <= t + 1e-9) newest = Math.max(newest, v);
  for (let s = 0; s < ob.segs; s++) {
    const r = ob.segRun[s];
    let c = ob.lineCol, al = 255;
    if (o.at) { const at = o.at[r]; al = at > t + 1e-9 ? 0 : !o.fade || at >= newest ? 255 : 90; }
    for (let v = 0; v < 4; v++) {
      if (ob.f) { colorOf(ob.f, ob.fv[ob.segPt[s] + (v < 2 ? 0 : 1)], tmp, 0); c = tmp; }
      ob.cl.set([c[0], c[1], c[2], Math.min(al, c[3] ?? 255)], (s * 4 + v) * 4);
      if (speeds) ob.gl[s * 4 + v] = Math.abs(speeds[speeds.length > 1 ? r : 0]) > 1e-12 ? 1 : 0;
    }
  }
  if (speeds) { ob.phase = ob.phase || new Float32Array(ob.runs.length); ob.speeds = speeds; }
  if (ob.l) { ob.l.set('c', ob.cl); ob.l.set('g', ob.gl); }
  ob.tn = t;
}

// Glows move along lines that carry flow, by each line's speed for dt seconds of playback.
function flowStep(G, dt) {
  for (const ob of G.objs) {
    if (!ob.speeds || !ob.l || !ob.vis) continue;
    const sp = ob.o.flow.spacing, ph = ob.phase, s4 = new Float32Array(ob.segs * 4);
    for (let r = 0; r < ph.length; r++) ph[r] += dt * ob.speeds[ob.speeds.length > 1 ? r : 0];
    for (let s = 0; s < ob.segs; s++) for (let v = 0; v < 4; v++) s4[s * 4 + v] = (ob.arc[s * 4 + v] - ph[ob.segRun[s]]) / sp;
    ob.l.set('s', s4);
  }
}

function arrowsFrame(G, ob, a, b, k) {
  const o = ob.o, n = ob.n, mV = ob.V.length > n * 3, mO = ob.O.length > n * 3, s = o.scale ? series(o.scale, a, b, k) : 1;
  const P = [], N = [], I = [];
  ob.tips = [];
  for (let i = 0; i < n; i++) {
    const og = [0, 1, 2].map((j) => (mO ? at2(ob.O, n * 3, a, b, k, i * 3 + j) : ob.O[i * 3 + j]));
    const v = [0, 1, 2].map((j) => (mV ? at2(ob.V, n * 3, a, b, k, i * 3 + j) : ob.V[i * 3 + j]) * s);
    arrow(og, v, o.radius || Math.max(Math.hypot(...v) * 0.035, G.radius * 0.006), P, N, I);
    ob.tips.push([og[0] + v[0], og[1] + v[1], og[2] + v[2]]);
  }
  if (!G.gl) return;
  fixed(ob.d, { P, N, I });
  const c = new Uint8Array((P.length / 3) * 4);
  for (let i = 0; i < c.length; i += 4) c.set([...ob.color, 255], i);
  ob.d.set('c', c);
}

// One frame on screen: opaque surfaces, then see through ones, then lines and glows over them.
function render(G, VPm, eye, w, h) {
  const gl = G.gl, L = G.surf.loc, ty = (d) => (d.big ? gl.UNSIGNED_INT : gl.UNSIGNED_SHORT);
  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
  gl.useProgram(G.surf.p);
  gl.uniformMatrix4fv(L.VP, false, VPm); gl.uniform3fv(L.eye, eye);
  for (const pass of [0, 1]) {
    if (pass) { gl.enable(gl.BLEND); gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA); gl.depthMask(false); }
    for (const ob of G.objs) {
      if (!ob.vis || !ob.d || (ob.alpha < 1) !== !!pass) continue;
      gl.uniformMatrix4fv(L.M, false, ob.M); gl.uniform1f(L.inst, ob.d.inst ? 1 : 0); gl.uniform1f(L.th, ob.th ?? 1e30);
      gl.uniform1f(L.prop, ob.prop && ob.o.kind !== 'ground' ? 1 : 0); gl.uniform1f(L.glow, ob.glow * 0.85); gl.uniform3fv(L.gc, ob.gcol); gl.uniform1f(L.alpha, ob.alpha);
      gl.bindVertexArray(ob.d.vao);
      if (ob.d.inst) gl.drawElementsInstanced(gl.TRIANGLES, ob.d.count, ty(ob.d), 0, ob.d.inst); else if (ob.d.count) gl.drawElements(gl.TRIANGLES, ob.d.count, ty(ob.d), 0);
      if (ob.b && ob.b.count) { gl.uniform1f(L.inst, 0); gl.uniformMatrix4fv(L.M, false, I4); gl.bindVertexArray(ob.b.vao); gl.drawElements(gl.TRIANGLES, ob.b.count, ty(ob.b), 0); }
    }
  }
  const Q = G.line.loc;
  gl.useProgram(G.line.p);
  gl.uniformMatrix4fv(Q.VP, false, VPm); gl.uniform2f(Q.vp, w, h);
  for (const ob of G.objs) {
    if (!ob.vis || !ob.l) continue;
    gl.uniform1f(Q.w, ob.width); gl.uniform1f(Q.tn, ob.tn ?? 1e30);
    gl.bindVertexArray(ob.l.vao);
    gl.drawElements(gl.TRIANGLES, ob.l.count, ty(ob.l), 0);
  }
  gl.disable(gl.BLEND); gl.depthMask(true);
  gl.bindVertexArray(null);
}
