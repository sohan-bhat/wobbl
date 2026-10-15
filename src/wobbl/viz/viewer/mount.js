// The viewer on the page: the stage, the camera and its controls, playback, readouts, labels, events, the verdict
// and the probe. mount() is the one entry point; a page can also mark an element with data-wobbl-scene.

async function mount(host, opt = {}) {
  const root = host.shadowRoot || host.attachShadow({ mode: 'open' });
  root.innerHTML = '';
  el('style', '', root, CSS);
  const wv = el('div', 'wv', root), st = el('div', 'st', wv), ro = el('div', 'ro', wv), bar = el('div', 'bar', wv), nt = el('div', 'nt', wv);
  st.tabIndex = 0;
  const say = (m) => { st.querySelector('.msg')?.remove(); if (m) el('div', 'msg', st, m); };
  say('Loading the simulation…');
  if (!('DecompressionStream' in globalThis)) { say('This browser cannot unpack the simulation. A current Chrome, Edge, Firefox or Safari can.'); return null; }
  let S;
  try { S = await decode(opt.src); } catch (e) { say(`The simulation did not load. ${e.message}.`); return null; }
  say('');
  const tl = S.timeline, F = tl.frames, T = tl.t, reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;
  el('div', 'ttl', st, S.title || '');
  const lg = el('div', 'lg', st), labLayer = el('div', '', st), tag = el('div', 'tag', st);
  const vd = el('div', 'vd', st), ts = el('div', 'ts', st), pr = el('div', 'pr', st);
  pr.hidden = true;
  if (S.verdict) { vd.classList.add(S.verdict.tone); el('b', '', vd, S.verdict.word); el('span', '', vd, S.verdict.line); }
  st.setAttribute('role', 'group');
  st.setAttribute('aria-label', `${S.title || 'Simulation'}. Interactive 3D view. Drag to turn, right drag or shift drag to pan, pinch or ctrl and scroll to zoom, space to play or pause, the arrow keys to step, 0 to reset the view.`);

  const canvas = el('canvas', '', null);
  st.prepend(canvas);
  let gl = null;
  try { gl = canvas.getContext('webgl2', { antialias: true, alpha: false, preserveDrawingBuffer: !!opt.still }); } catch { gl = null; }
  const G = context(S, gl), objs = G.objs, radius = G.radius;

  // ---------- camera: orbiting a target with z up
  const cam = S.camera || {}, fov = cam.fov || 30;
  const home = () => {
    const d = cam.dir || nrm([0.62, -1.05, 0.62]), dist = cam.dist || (radius / Math.sin((fov * Math.PI) / 360)) * 1.05;
    return { t: cam.target ? [...cam.target] : [...G.center], az: Math.atan2(d[1], d[0]), el: Math.asin(clamp(d[2], -1, 1)), dist, follow: cam.follow ?? null };
  };
  let view = home(), followOff = [0, 0, 0], VPm = I4, eye = [0, 0, 1], vw = 1, vh = 1;
  const reset = () => { view = home(); followOff = [0, 0, 0]; update(pos); redraw(); };
  const eyeOf = () => { const c = Math.cos(view.el); return [view.t[0] + view.dist * c * Math.cos(view.az), view.t[1] + view.dist * c * Math.sin(view.az), view.t[2] + view.dist * Math.sin(view.el)]; };

  // ---------- playback
  // opt.end opens on the last frame, where a field run shows its full result rather than an unloaded part
  let pos = reduce || opt.end ? F - 1 : 0, playing = false, speed = 1, last = 0, holdUntil = 0;
  const ui = buildUI();
  function update(p) {
    const a = Math.floor(clamp(p, 0, F - 1)), b = Math.min(a + 1, F - 1), k = tl.between === 'step' ? 0 : p - a, t = k ? lerp(T[a], T[b], k) : T[a];
    for (const ob of objs) {
      ob.vis = shown(ob.o.when, t) && !(ob.prop && !ui.props);
      if (ob.vis) frame(G, ob, a, b, k, t);
    }
    // a label can name the timeline itself, as "{load}" on a strength load arrow, and a readout of that name wins
    if (tl.label) ui.vals[tl.label] = `${fmt(t, 4)}${tl.unit ? ' ' + tl.unit : ''}`;
    for (const [i, r] of S.readouts.entries()) {
      const v = series(r.values, a, b, k), e = ui.ro[i];
      e.v.textContent = fmt(v, r.digits);
      const lim = r.limits ? series(r.limits, a, b, k) : r.limit;
      e.box.classList.toggle('over', lim !== undefined && v > lim + 1e-12);
      ui.vals[r.name] = `${fmt(v, r.digits)}${r.unit ? ' ' + r.unit : ''}`;
    }
    ui.time.textContent = `${tl.label || 'frame'} ${fmt(t, 4)}${tl.unit ? ' ' + tl.unit : ''}`;
    const blended = tl.between === 'interpolated' && k > 1e-3;
    tag.innerHTML = F > 1 ? `frame ${a + 1} of ${F}${blended ? ` · <em>interpolated toward ${b + 1}</em>` : ''}` : '';
    ui.scrub.value = String(p);
    vd.classList.toggle('on', p >= F - 1 - 1e-6 && !!S.verdict);
    ui.t = t;
    const fo = view.follow !== null ? objs[view.follow] : null;
    if (fo && fo.at) view.t = [0, 1, 2].map((i) => fo.at[i] + followOff[i]);
  }

  // ---------- drawing, on demand and only while on screen
  let raf = 0, visible = true;
  const draw = () => {
    raf = 0;
    if (!gl) return;
    const w = st.clientWidth, h = st.clientHeight, px = Math.min(devicePixelRatio || 1, 2);
    if (!w || !h) return;
    if (canvas.width !== Math.round(w * px) || canvas.height !== Math.round(h * px)) { canvas.width = Math.round(w * px); canvas.height = Math.round(h * px); }
    vw = w; vh = h;
    gl.viewport(0, 0, canvas.width, canvas.height);
    eye = eyeOf();
    const far = view.dist * 4 + radius * 8, near = Math.max(view.dist * 0.004, far * 2e-6);
    VPm = mul(persp(fov, w / h, near, far), look(eye, view.t));
    render(G, VPm, eye, w, h);
    labels();
  };
  const redraw = () => { if (!raf && visible) raf = requestAnimationFrame(draw); };

  // a 3D point to CSS pixels in the stage, and whether it is in front of the camera
  function toScreen(p) {
    const m = VPm, x = m[0] * p[0] + m[4] * p[1] + m[8] * p[2] + m[12], y = m[1] * p[0] + m[5] * p[1] + m[9] * p[2] + m[13], w = m[3] * p[0] + m[7] * p[1] + m[11] * p[2] + m[15];
    return [((x / w + 1) / 2) * vw, ((1 - y / w) / 2) * vh, w > 0];
  }
  function labels() {
    const items = (S.labels || []).map((l) => ({ ...l, p: l.follow !== undefined && objs[l.follow]?.at ? [0, 1, 2].map((i) => l.at[i] + objs[l.follow].at[i]) : l.at }));
    for (const ob of objs) if (ob.o.kind === 'arrows' && ob.o.labels && ob.vis && ob.tips) ob.o.labels.forEach((t, i) => t && items.push({ text: t, p: ob.tips[i], kind: 'note' }));
    while (ui.labs.length < items.length) ui.labs.push(el('div', 'lb', labLayer));
    ui.labs.forEach((e, i) => {
      const it = items[i];
      if (!it || !shown(it.when, ui.t) || (it.follow !== undefined && !objs[it.follow]?.vis)) { e.hidden = true; return; }
      const [x, y, front] = toScreen(it.p);
      e.hidden = !front || x < -20 || y < -20 || x > vw + 20 || y > vh + 20;
      if (e.hidden) return;
      const text = it.text.replace(/\{([^}]+)\}/g, (m, k) => ui.vals[k] ?? m);
      if (e.dataset.t !== text) { e.dataset.t = text; e.innerHTML = '<i></i><span></span>'; e.lastChild.textContent = text; }
      e.className = `lb ${it.kind || 'peak'}`;
      e.style.transform = `translate(${x}px,${y}px)`;
      const sp = e.lastChild;
      sp.style.left = x > vw - sp.offsetWidth - 24 ? `${-sp.offsetWidth - 10}px` : '10px';
    });
  }

  // ---------- controls
  function buildUI() {
    const u = { ro: [], vals: {}, labs: [], props: true, t: T[0] };
    for (const r of S.readouts) {
      const box = el('div', '', ro); el('span', '', box, `${r.name} `); const v = el('b', '', box); if (r.unit) el('span', '', box, ` ${r.unit}`);
      u.ro.push({ box, v });
    }
    if (!S.readouts.length) ro.hidden = true;
    const f = G.fields[0];
    if (f) {
      el('div', '', lg, `${f.name}${f.unit ? ` (${f.unit})` : ''}`);
      const c = el('canvas', '', lg), x = (c.width = 200, c.height = 1, c.getContext('2d')), img = x.createImageData(200, 1);
      for (let i = 0; i < 200; i++) { const j = Math.round((i / 199) * 255) * 3; img.data.set([f.lut[j], f.lut[j + 1], f.lut[j + 2], 255], i * 4); }
      x.putImageData(img, 0, 0);
      const t = el('div', 't', lg); el('span', '', t, fmt(f.lo, f.digits)); el('span', '', t, fmt(f.hi, f.digits));
      if (f.limit !== undefined) { const ov = el('div', 'ov', lg); el('i', '', ov); el('span', '', ov, `over ${f.limit_label || 'the limit'} (${fmt(f.limit, f.digits)})`); }
      if (G.fields.length > 1) el('div', 'ov', lg, `and ${G.fields.slice(1).map((g) => g.name).join(', ')}`);
    } else lg.hidden = true;
    const btn = (text, title, fn, cls) => { const b = el('button', cls || '', bar, text); b.type = 'button'; b.title = title; b.setAttribute('aria-label', title); b.onclick = fn; return b; };
    u.play = btn('▶', 'Play', () => (playing ? pause() : play()), 'pl');
    const trk = el('div', 'trk', bar);
    u.scrub = el('input', '', trk);
    Object.assign(u.scrub, { type: 'range', min: '0', max: String(Math.max(F - 1, 0)), step: tl.between === 'exact' ? 'any' : '1' });
    u.scrub.setAttribute('aria-label', 'Playback position');
    u.scrub.oninput = () => { const v = +u.scrub.value; pause(); seek(v); };
    const tPos = (t) => { for (let i = 0; i < F - 1; i++) if (t <= T[i + 1]) return i + (t - T[i]) / ((T[i + 1] - T[i]) || 1); return F - 1; };
    for (const e of S.events || []) {
      const m = el('button', `ev ${e.kind === 'fail' ? 'fail' : ''}`, trk), p = clamp(tPos(e.t), 0, F - 1), x = p / Math.max(F - 1, 1);
      m.type = 'button';
      m.style.left = `calc(${x * 100}% + ${(0.5 - x) * 14}px)`;
      m.title = `${e.name}${e.detail ? `, ${e.detail}` : ''} at ${fmt(e.t, 4)} ${tl.unit || ''}`; m.setAttribute('aria-label', m.title);
      m.onclick = () => { pause(); seek(p); toast(e); };
      e.p = p;
    }
    u.time = el('span', 'tm', bar);
    const sp = el('select', '', bar);
    sp.title = 'Playback speed'; sp.setAttribute('aria-label', 'Playback speed');
    for (const s of [0.25, 0.5, 1, 2, 4]) { const o = el('option', '', sp, `${s}x`); o.value = String(s); o.selected = s === 1; }
    sp.onchange = () => { speed = +sp.value; };
    if (objs.some((o) => o.prop)) { const b = btn('Setup', 'Show or hide the setup props', () => { u.props = !u.props; b.setAttribute('aria-pressed', String(u.props)); update(pos); redraw(); }); b.setAttribute('aria-pressed', 'true'); }
    for (const pv of cam.presets || []) {
      btn(pv.name || 'View', `View ${pv.name || ''}`.trim(), () => {
        view = { ...home(), follow: pv.follow ?? null };
        if (pv.target) view.t = [...pv.target];
        if (pv.dist) view.dist = pv.dist;
        if (pv.dir) { view.az = Math.atan2(pv.dir[1], pv.dir[0]); view.el = Math.asin(pv.dir[2]); }
        followOff = [0, 0, 0]; update(pos); redraw();
      });
    }
    btn('Reset', 'Reset the view', reset);
    if (F < 2) { u.play.hidden = true; trk.hidden = true; sp.hidden = true; }
    const lines = [...(S.notes || [])];
    if (tl.between === 'interpolated' && F > 1) lines.push('Between saved frames the view blends the two nearest ones and says so. Paused, it rests on a saved frame.');
    if (tl.note) lines.push(tl.note);
    if (S.props_note && objs.some((o) => o.prop)) lines.push(S.props_note + '.');
    if (S.source) lines.push(S.source);
    for (const l of lines) el('div', '', nt, l);
    if (!lines.length) nt.hidden = true;
    return u;
  }
  let toastTimer = 0;
  function toast(e) {
    ts.textContent = `${e.name}${e.detail ? `, ${e.detail}` : ''}`;
    ts.className = `ts on ${e.kind === 'fail' ? 'fail' : ''}`;
    clearTimeout(toastTimer); toastTimer = setTimeout(() => ts.classList.remove('on'), 2600);
  }
  function seek(p) { pos = clamp(p, 0, F - 1); update(pos); redraw(); }
  const label = (t) => { ui.play.textContent = t ? '❚❚' : '▶'; ui.play.title = t ? 'Pause' : 'Play'; ui.play.setAttribute('aria-label', ui.play.title); };
  function play() {
    if (F < 2) return;
    if (pos >= F - 1) pos = 0;
    playing = true; last = 0; holdUntil = 0; label(true);
    if (!tickRaf) tick();
  }
  function pause() {
    playing = false; label(false);
    if (tl.between !== 'exact') pos = Math.round(pos);
    update(pos); redraw();
  }
  const flowing = () => !reduce && objs.some((o) => o.speeds && o.vis);
  let tickRaf = 0;
  function tick(now) {
    tickRaf = 0;
    if (!playing && !flowing()) return;
    now = now || performance.now();
    const dt = last ? clamp((now - last) / 1000, 0, 0.25) : 0;
    last = now;
    if (playing && visible) {
      if (holdUntil) {
        if (now >= holdUntil) { holdUntil = 0; if (tl.loop) pos = 0; else { pause(); return; } }
      } else {
        const before = pos;
        pos = Math.min(F - 1, pos + (dt * speed * (F - 1)) / (tl.seconds || 6));
        for (const e of S.events || []) if (e.p > before + 1e-9 && e.p <= pos + 1e-9) toast(e);
        if (pos >= F - 1) holdUntil = now + 3200 / Math.max(speed, 0.5);
      }
      update(pos);
    }
    if (!reduce) flowStep(G, dt);
    draw();
    if (visible) tickRaf = requestAnimationFrame(tick);
  }

  // ---------- pointer, wheel and keys
  const ptrs = new Map();
  let span = 0, moved = false, down = null;
  const pan = (dx, dy) => {
    const k = (2 * view.dist * Math.tan((fov * Math.PI) / 360)) / (vh || 1), se = Math.sin(view.el);
    const right = [-Math.sin(view.az), Math.cos(view.az), 0], up = [-se * Math.cos(view.az), -se * Math.sin(view.az), Math.cos(view.el)];
    for (let i = 0; i < 3; i++) { const d = -dx * k * right[i] + dy * k * up[i]; view.t[i] += d; followOff[i] += d; }
  };
  const pair = () => { const [a, b] = [...ptrs.values()]; return Math.hypot(a.x - b.x, a.y - b.y); };
  const on = (t, fn, o) => canvas.addEventListener(t, fn, o);
  on('pointerdown', (e) => {
    canvas.setPointerCapture(e.pointerId);
    ptrs.set(e.pointerId, { x: e.clientX, y: e.clientY, x0: e.clientX, y0: e.clientY, b: e.button, sh: e.shiftKey, touch: e.pointerType !== 'mouse' });
    if (ptrs.size === 2) span = pair();
    moved = false; down = { x: e.clientX, y: e.clientY };
  });
  on('pointermove', (e) => {
    const p = ptrs.get(e.pointerId);
    if (!p) return;
    const dx = e.clientX - p.x, dy = e.clientY - p.y;
    // one finger: up and down scroll the page until the finger has clearly gone sideways
    if (!moved && p.touch && ptrs.size === 1) { const sx = e.clientX - p.x0, sy = e.clientY - p.y0; if (Math.abs(sx) < 8 || Math.abs(sx) < Math.abs(sy)) return; }
    p.x = e.clientX; p.y = e.clientY;
    if (Math.hypot(e.clientX - down.x, e.clientY - down.y) > 4) moved = true;
    if (ptrs.size === 1) {
      if (p.b === 2 || p.b === 1 || p.sh) pan(dx, dy);
      else { const k = (2 * Math.PI * 0.8) / (vh || 1); view.az -= dx * k; view.el = clamp(view.el + dy * k, -1.55, 1.55); }
    } else if (ptrs.size === 2) {
      const s = pair();
      if (span > 0 && s > 0) view.dist = clamp(view.dist * (span / s), radius * 0.02, radius * 60);
      span = s; pan(dx / 2, dy / 2);
    }
    redraw();
  }, { passive: true });
  const up = (e) => {
    ptrs.delete(e.pointerId);
    if (ptrs.size === 2) span = pair();
    if (!moved && down && e.type === 'pointerup' && e.button === 0) probe(e);
  };
  on('pointerup', up);
  on('pointercancel', up);
  on('contextmenu', (e) => e.preventDefault());
  on('dblclick', reset);
  // a plain wheel scrolls the page unless the view has focus; ctrl or cmd with the wheel, or a pinch, zooms
  on('wheel', (e) => {
    if (!e.ctrlKey && !e.metaKey && root.activeElement !== st) return;
    e.preventDefault();
    view.dist = clamp(view.dist * Math.exp(e.deltaY * (e.ctrlKey ? 0.01 : 0.0015)), radius * 0.02, radius * 60);
    redraw();
  }, { passive: false });
  st.addEventListener('keydown', (e) => {
    const k = e.key;
    if (e.target !== st) return;
    if (k === ' ' || k === 'k') playing ? pause() : play();
    else if (k === 'ArrowRight' || k === 'ArrowLeft') { pause(); seek(Math.round(pos) + (k === 'ArrowRight' ? 1 : -1)); }
    else if (k === 'ArrowUp' || k === 'ArrowDown') { view.el = clamp(view.el + (k === 'ArrowUp' ? 0.08 : -0.08), -1.55, 1.55); redraw(); }
    else if (k === '+' || k === '=' || k === '-' || k === '_') { view.dist *= k === '-' || k === '_' ? 1.12 : 1 / 1.12; redraw(); }
    else if (k === '0') reset();
    else if (k === 'Home' || k === 'End') { pause(); seek(k === 'Home' ? 0 : F - 1); }
    else return;
    e.preventDefault();
  });

  // The probe: the field value at the vertex nearest where a click lands on a colored mesh.
  function probe(e) {
    const r = canvas.getBoundingClientRect(), nx = ((e.clientX - r.left) / r.width) * 2 - 1, ny = 1 - ((e.clientY - r.top) / r.height) * 2;
    const tf = Math.tan((fov * Math.PI) / 360), z = nrm(sub(eye, view.t)), x = nrm(crs([0, 0, 1], z)), y = crs(z, x);
    const d = nrm([0, 1, 2].map((i) => nx * tf * (vw / vh) * x[i] + ny * tf * y[i] - z[i]));
    let best = null;
    for (const ob of objs) {
      if (!ob.vis || ob.o.kind !== 'mesh' || !ob.f) continue;
      const P = ob.cur, t = ob.tri;
      for (let j = 0; j < t.length; j += 3) {
        const a = t[j] * 3, b = t[j + 1] * 3, q = t[j + 2] * 3;
        const e1 = [P[b] - P[a], P[b + 1] - P[a + 1], P[b + 2] - P[a + 2]], e2 = [P[q] - P[a], P[q + 1] - P[a + 1], P[q + 2] - P[a + 2]];
        const h = crs(d, e2), det = dot(e1, h);
        if (Math.abs(det) < 1e-12) continue;
        const s = [eye[0] - P[a], eye[1] - P[a + 1], eye[2] - P[a + 2]], u = dot(s, h) / det;
        if (u < 0 || u > 1) continue;
        const qq = crs(s, e1), v = dot(d, qq) / det;
        if (v < 0 || u + v > 1) continue;
        const dist = dot(e2, qq) / det;
        if (dist > 0 && (!best || dist < best.dist)) best = { dist, ob, i: t[j + (u > v && u > 1 - u - v ? 1 : v > 1 - u - v ? 2 : 0)] };
      }
    }
    if (!best || (best.ob.rv && best.ob.rv[best.i] > best.ob.th)) { pr.hidden = true; return; }
    const f = best.ob.f;
    pr.textContent = `${fmt(best.ob.val[best.i], f.digits + 1)} ${f.unit} ${f.name}`;
    pr.style.left = `${e.clientX - r.left}px`; pr.style.top = `${e.clientY - r.top}px`; pr.hidden = false;
    clearTimeout(pr.t); pr.t = setTimeout(() => (pr.hidden = true), 4000);
  }

  // ---------- lifetime
  const io = new IntersectionObserver(([e]) => {
    visible = e.isIntersecting;
    if (visible) { redraw(); if ((playing || flowing()) && !tickRaf) tick(); }
  });
  io.observe(st);
  const rz = new ResizeObserver(() => redraw());
  rz.observe(st);
  if (!gl) say('This browser cannot draw WebGL2 here, so the 3D view is off. The readouts and the timeline still follow the run.');
  update(pos);
  redraw();
  if (opt.autoplay !== false && !reduce && F > 1) play();
  else if (flowing()) tick();
  const api = {
    play, pause, seek, frames: F, scene: S, get position() { return pos; },
    dispose() {
      playing = false; cancelAnimationFrame(raf); cancelAnimationFrame(tickRaf); io.disconnect(); rz.disconnect();
      for (const o of objs) { o.d?.free(); o.l?.free(); o.b?.free(); }
      gl?.getExtension('WEBGL_lose_context')?.loseContext(); root.innerHTML = '';
    },
  };
  host.wobblView = api;
  return api;
}

W.mount = mount;
W.decode = decode;
W.scan = () => {
  for (const e of document.querySelectorAll('[data-wobbl-scene]')) {
    if (e.wobblMounted) continue;
    e.wobblMounted = true;
    const s = e.dataset.wobblScene, src = s.startsWith('#') ? document.querySelector(s)?.textContent || '' : s;
    mount(e, { src, autoplay: e.dataset.autoplay !== 'false' }).catch((err) => console.error(err));
  }
};
if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', W.scan); else W.scan();
