// The viewer on the page: the stage, the camera and its controls, the slider over the scene's input, the live
// number and verdict, readouts, labels, events and the probe. Nothing moves on its own. mount() is the one entry point; a page can also mark an element with data-wobbl-scene.

async function mount(host, opt = {}) {
  const root = host.shadowRoot || host.attachShadow({ mode: 'open' });
  root.innerHTML = '';
  el('style', '', root, CSS);
  const wv = el('div', 'wv', root), st = el('div', 'st', wv), ro = el('div', 'ro', wv), pc = el('div', 'pc', wv), bar = el('div', 'bar', wv), nt = el('div', 'nt', wv);
  st.tabIndex = 0;
  const say = (m) => { st.querySelector('.msg')?.remove(); if (m) el('div', 'msg', st, m); };
  say('Loading the simulation…');
  if (!('DecompressionStream' in globalThis)) { say('This browser cannot unpack the simulation. A current Chrome, Edge, Firefox or Safari can.'); return null; }
  let S;
  try { S = await decode(opt.src); } catch (e) { say(`The simulation did not load. ${e.message}.`); return null; }
  say('');
  const tl = S.timeline, F = tl.frames, T = tl.t, input = tl.kind === 'input';
  const name = tl.label ? tl.label[0].toUpperCase() + tl.label.slice(1) : 'Frame';
  el('div', 'ttl', st, S.title || '');
  const labLayer = el('div', '', st), ts = el('div', 'ts', st), pr = el('div', 'pr', st);
  pr.hidden = true;
  st.setAttribute('role', 'group');
  st.setAttribute('aria-label', `${S.title || 'Simulation'}. Interactive 3D view. Drag to turn, right drag or shift drag to pan, pinch or ctrl and scroll to zoom, the arrow keys to move the ${name.toLowerCase()}, 0 to reset the view.`);

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

    // ---------- the input: the slider over the timeline, opened at the asked value or else the last frame
  const tPos = (t) => { for (let i = 0; i < F - 1; i++) if (t <= T[i + 1]) return clamp(i + (t - T[i]) / ((T[i + 1] - T[i]) || 1), i, i + 1); return F - 1; };
  let pos = tl.design !== undefined ? tPos(tl.design) : F - 1;
  if (tl.between !== 'exact') pos = Math.round(pos);
  const pos0 = pos, f0 = G.fields[0], ui = buildUI();
  function update(p) {
    const a = Math.floor(clamp(p, 0, F - 1)), b = Math.min(a + 1, F - 1), k = tl.between === 'step' ? 0 : p - a, t = k ? lerp(T[a], T[b], k) : T[a];
    for (const ob of objs) {
      ob.vis = shown(ob.o.when, t) && !(ob.prop && !ui.props);
      if (ob.vis) frame(G, ob, a, b, k, t);
    }
    flowStep(G, 0);
    const tv = `${fmt(t, 4)}${tl.unit ? ' ' + tl.unit : ''}`;
    // a label can name the timeline itself, as "{load}" on a strength load arrow, and a readout of that name wins
    if (tl.label) ui.vals[tl.label] = tv;
    for (const [i, r] of S.readouts.entries()) {
      const v = series(r.values, a, b, k), e = ui.ro[i], lim = r.limits ? series(r.limits, a, b, k) : r.limit;
      if (e) { e.v.textContent = fmt(v, r.digits); e.box.classList.toggle('over', lim !== undefined && v > lim + 1e-12); }
      ui.vals[r.name] = `${fmt(v, r.digits)}${r.unit ? ' ' + r.unit : ''}`;
    }
    ui.out.textContent = tv;
    ui.scrub.value = String(p);
    live(a, b, k, p, tv);
    ui.t = t;
    const fo = view.follow !== null ? objs[view.follow] : null;
    if (fo && fo.at) view.t = [0, 1, 2].map((i) => fo.at[i] + followOff[i]);
  }

  // The highest value of the main color field on what is drawn now.
  function peak(a, b, k) {
    let m = -Infinity;
    for (const ob of objs) {
      if (!ob.vis || !ob.o.field || ob.o.field.f !== 0) continue;
      if (ob.o.kind === 'mesh') { for (let i = 0; i < ob.n; i++) if (ob.val[i] > m && !(ob.rv && ob.rv[i] > ob.th)) m = ob.val[i]; }
      else if (ob.o.kind === 'body') m = Math.max(m, series(ob.o.field.values, a, b, k));
      else if (ob.fv) { const n = ob.n || ob.fv.length, many = !!ob.o.field.frames; for (let i = 0; i < n; i++) { const v = many ? at2(ob.fv, n, a, b, k, i) : ob.fv[i]; if (v > m) m = v; } }
    }
    return m;
  }
  // The big number, the verdict and the peak marker on the legend. On an input the verdict follows the worst
  // of the main field's peak and every readout that has a limit: over its limit it fails, over its target (0.8
  // of the limit unless the scene gives one) it is marginal. A run over time keeps its own verdict, and a scene
  // with no limit shows its verdict at the value the question asked.
  function live(a, b, k, p, tv) {
    const pk = f0 ? peak(a, b, k) : -Infinity, r = S.readouts[ui.big];
    const v = r ? series(r.values, a, b, k) : pk;
    ui.num.textContent = Number.isFinite(v) ? fmt(v, r ? r.digits : f0.digits) : '–';
    let word = '', tone = '', worst = -Infinity, warn = false;
    const weigh = (x, L, m) => { if (!(L > 0) || !Number.isFinite(x)) return; worst = Math.max(worst, x / L); warn ||= x > m; };
    if (ui.lim !== null) weigh(pk, ui.lim, ui.tgt);
    for (const q of S.readouts) if (q.limit !== undefined || q.limits) { const L = q.limits ? series(q.limits, a, b, k) : q.limit; weigh(series(q.values, a, b, k), L, L * 0.8); }
    if (S.verdict && (!input || (worst === -Infinity && Math.abs(p - pos0) < 1e-6))) { word = S.verdict.word; tone = { pass: 'holds', warn: 'marginal', fail: 'fails' }[S.verdict.tone] || 'info'; }
    else if (worst > -Infinity) { tone = worst > 1 + 1e-9 ? 'fails' : warn ? 'marginal' : 'holds'; word = { holds: 'OK', marginal: 'Marginal', fails: 'Fails' }[tone]; }
    ui.vd.textContent = word; ui.vd.dataset.v = tone; ui.vd.style.visibility = word ? '' : 'hidden';
    ui.scrub.setAttribute('aria-valuetext', `${tv}${r || f0 ? `, ${ui.numName} ${ui.num.textContent}${ui.numUnit ? ' ' + ui.numUnit : ''}` : ''}${word ? `, ${word.toLowerCase()}` : ''}`);
    if (ui.pk && Number.isFinite(pk)) {
      ui.pk.hidden = false;
      ui.pk.style.left = ui.lim !== null && pk > ui.lim ? 'calc(100% + 6px)' : `${clamp((pk - f0.lo) / ((ui.top - f0.lo) || 1), 0, 1) * 100}%`;
    } else if (ui.pk) ui.pk.hidden = true;
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

  // ---------- controls: the slider and its legend beside the live number and verdict, as on the landing page
  function buildUI() {
    const u = { ro: [], vals: {}, labs: [], props: true, t: T[0], lim: null, tgt: null, top: 0, pk: null };
    // the big number is the first readout that is not the input itself
    u.big = S.readouts.findIndex((r) => r.name !== tl.label && !r.values.every((v, i) => Math.abs(v - (T[i] ?? NaN)) <= 1e-4 * Math.max(1, Math.abs(T[i]))));
    for (const [i, r] of S.readouts.entries()) {
      if (i === u.big) { u.ro.push(null); continue; }
      const box = el('div', '', ro); el('span', '', box, `${r.name} `); const v = el('b', '', box); if (r.unit) el('span', '', box, ` ${r.unit}`);
      u.ro.push({ box, v });
    }
    if (!ro.children.length) ro.hidden = true;
    const ctl = el('div', 'ctl', pc), lab = el('label', '', ctl);
    el('span', '', lab, `${name}`);
    u.out = el('output', '', lab);
    const trk = el('div', 'trk', ctl);
    u.scrub = el('input', '', trk);
    Object.assign(u.scrub, { type: 'range', min: '0', max: String(Math.max(F - 1, 0)), step: tl.between === 'exact' ? 'any' : '1' });
    u.scrub.setAttribute('aria-label', name);
    u.scrub.oninput = () => seek(+u.scrub.value);
    for (const e of S.events || []) {
      const m = el('button', `ev ${e.kind === 'fail' ? 'fail' : ''}`, trk), p = clamp(tPos(e.t), 0, F - 1), x = p / Math.max(F - 1, 1);
      m.type = 'button';
      m.style.left = `calc(${x * 100}% + ${(0.5 - x) * 20}px)`;
      m.title = `${e.name}${e.detail ? `, ${e.detail}` : ''} at ${fmt(e.t, 4)} ${tl.unit || ''}`; m.setAttribute('aria-label', m.title);
      m.onclick = () => { seek(p); toast(e); };
    }
    if (F < 2) { lab.hidden = true; trk.hidden = true; }
    const f = f0, lg = el('div', 'lg', ctl);
    if (f) {
      if (f.limit !== undefined) { u.lim = f.limit; u.tgt = f.target ?? f.limit * 0.8; }
      u.top = u.lim ?? f.hi;
      el('div', 'lk', lg, `${f.name}${f.unit ? ` (${f.unit})` : ''}`);
      const bar2 = el('div', `lb2${u.lim !== null ? ' over' : ''}`, lg), c = el('canvas', '', bar2), x = (c.width = 200, c.height = 1, c.getContext('2d')), img = x.createImageData(200, 1);
      for (let i = 0; i < 200; i++) { const v = f.lo + (i / 199) * (u.top - f.lo), j = clamp(Math.round(((v - f.lo) / (f.hi - f.lo || 1)) * 255), 0, 255) * 3; img.data.set([f.lut[j], f.lut[j + 1], f.lut[j + 2], 255], i * 4); }
      x.putImageData(img, 0, 0);
      u.pk = el('span', 'pk', bar2);
      const t = el('div', 't', lg), d = f.digits, un = f.unit ? ` ${f.unit}` : '';
      el('span', '', t, fmt(f.lo, d));
      if (u.tgt !== null && u.tgt > f.lo && u.tgt < u.top) {
        const at = `${((u.tgt - f.lo) / (u.top - f.lo)) * 100}%`;
        el('i', 'tg', bar2).style.left = at;
        u.tgl = el('span', 'tgl', t, `${fmt(u.tgt, d)} ${f.target !== undefined ? 'target' : 'marginal'}`);
        u.tgl.style.left = at;
      }
      u.lml = el('span', '', t, u.lim !== null ? `${fmt(u.lim, d)}${un} ${f.limit_label || 'limit'}` : `${fmt(f.hi, d)}${un}`);
      if (G.fields.length > 1) el('div', 'lk', lg, `and ${G.fields.slice(1).map((g) => g.name).join(', ')}`);
    } else lg.hidden = true;
    const lv = el('div', 'live', pc), big = S.readouts[u.big];
    u.numName = big ? big.name : f ? `peak ${f.name}` : '';
    u.numUnit = big ? big.unit : f ? f.unit : '';
    el('p', 'lk', lv, u.numName);
    const n = el('p', 'num', lv); u.num = el('span', '', n); if (u.numUnit) el('span', 'u', n, ` ${u.numUnit}`);
    u.vd = el('p', 'vw', lv);
    u.vd.setAttribute('aria-live', 'polite');
    if (!big && !f) lv.hidden = true;
    const btn = (text, title, fn, cls, parent = bar) => { const b = el('button', cls || '', parent, text); b.type = 'button'; b.title = title; b.setAttribute('aria-label', title); b.onclick = fn; return b; };
    // the bending toggle, as on the landing page, when a part is drawn deformed larger than it is
    const ex = Math.max(...objs.map((o) => (o.o.deform ? o.o.deform.exaggerate || 1 : 1)));
    if (ex > 1) {
      const b = btn('', 'Show the bending larger than it is', () => { G.unbent = !G.unbent; b.setAttribute('aria-pressed', String(!G.unbent)); update(pos); redraw(); }, 'bd', st);
      el('i', '', b); el('span', '', b, `Bending ×${+ex.toPrecision(3)}`);
      b.setAttribute('aria-pressed', 'true');
    }
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
    const lines = [...(S.notes || [])];
    if (tl.note) lines.push(tl.note);
    if (S.props_note && objs.some((o) => o.prop)) lines.push(S.props_note + '.');
    if (S.source) lines.push(S.source);
    for (const l of lines) el('div', '', nt, l);
    if (!lines.length) nt.hidden = true;
    return u;
  }
  // The target's tick label drops a line when it would run into the end labels.
  function fitTicks() {
    if (!ui.tgl) return;
    ui.tgl.classList.remove('dn');
    const a = ui.tgl.getBoundingClientRect(), b = ui.lml.getBoundingClientRect(), c = ui.tgl.parentElement.firstChild.getBoundingClientRect();
    if (a.right + 6 > b.left || a.left - 6 < c.right) ui.tgl.classList.add('dn');
  }
  let toastTimer = 0;
  function toast(e) {
    ts.textContent = `${e.name}${e.detail ? `, ${e.detail}` : ''}`;
    ts.className = `ts on ${e.kind === 'fail' ? 'fail' : ''}`;
    clearTimeout(toastTimer); toastTimer = setTimeout(() => ts.classList.remove('on'), 2600);
  }
  function seek(p) { pos = clamp(p, 0, F - 1); update(pos); redraw(); }

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
    if (!moved && down && e.type === 'pointerup' && e.button === 0) probe(e, true);
  };
  // a mouse over a colored part reads the value under it, as the landing page's part does
  let hov = null, hovRaf = 0;
  on('pointermove', (e) => {
    if (e.pointerType !== 'mouse' || e.buttons) return;
    hov = e;
    if (!hovRaf) hovRaf = requestAnimationFrame(() => { hovRaf = 0; if (hov) probe(hov, false); });
  }, { passive: true });
  on('pointerleave', (e) => { if (e.pointerType === 'mouse') { hov = null; pr.hidden = true; } });
  on('pointerdown', (e) => { if (e.pointerType === 'mouse') { hov = null; pr.hidden = true; } });
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
    if (k === 'ArrowRight' || k === 'ArrowLeft') seek(Math.round(pos) + (k === 'ArrowRight' ? 1 : -1));
    else if (k === 'ArrowUp' || k === 'ArrowDown') { view.el = clamp(view.el + (k === 'ArrowUp' ? 0.08 : -0.08), -1.55, 1.55); redraw(); }
    else if (k === '+' || k === '=' || k === '-' || k === '_') { view.dist *= k === '-' || k === '_' ? 1.12 : 1 / 1.12; redraw(); }
    else if (k === '0') reset();
    else if (k === 'Home' || k === 'End') seek(k === 'Home' ? 0 : F - 1);
    else return;
    e.preventDefault();
  });

  // The probe: the field value at the vertex nearest the pointer on a colored mesh, on a leader line from that point.
  function probe(e, tap) {
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
    const f = best.ob.f, v = best.ob.val[best.i], px = e.clientX - r.left;
    pr.innerHTML = '<i></i>';
    const box = el('div', '', pr);
    el('b', '', box, `${fmt(v, f.digits + 1)}${f.unit ? ' ' + f.unit : ''}`); el('span', '', box, f.name);
    if (f.limit !== undefined && f.limit > 0 && Number.isFinite(v)) el('span', 'x', box, `${Math.round((v / f.limit) * 100)}% of ${f.limit_label || 'the limit'}`);
    pr.classList.toggle('l', px > vw - 200);
    pr.style.left = `${px}px`; pr.style.top = `${e.clientY - r.top}px`; pr.hidden = false;
    clearTimeout(pr.t);
    if (tap) pr.t = setTimeout(() => (pr.hidden = true), 4000);
  }

  // ---------- lifetime
  const io = new IntersectionObserver(([e]) => {
    visible = e.isIntersecting;
    if (visible) redraw();
  });
  io.observe(st);
  const rz = new ResizeObserver(() => { redraw(); fitTicks(); });
  rz.observe(st);
  if (!gl) say('This browser cannot draw WebGL2 here, so the 3D view is off. The readouts still follow the slider.');
  update(pos);
  redraw();
  const api = {
    seek, frames: F, scene: S, get position() { return pos; },
    dispose() {
      cancelAnimationFrame(raf); cancelAnimationFrame(hovRaf); clearTimeout(pr.t); clearTimeout(toastTimer); io.disconnect(); rz.disconnect();
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
    mount(e, { src }).catch((err) => console.error(err));
  }
};
if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', W.scan); else W.scan();
