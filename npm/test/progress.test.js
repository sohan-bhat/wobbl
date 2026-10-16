'use strict';

// The install display, drawn into a fake terminal.

const test = require('node:test');
const assert = require('node:assert/strict');
const { Display, UvTracker, seconds, sized } = require('../lib/progress');

function screen(isTTY) {
  const s = { isTTY, columns: 70, text: '', write: (t) => (s.text += t) };
  return s;
}

function clock() {
  let now = 0;
  const c = () => now;
  c.add = (ms) => (now += ms);
  return c;
}

test('formats time, bars and sizes', () => {
  assert.equal(seconds(400), '0s');
  assert.equal(seconds(14200), '14s');
  assert.equal(seconds(125000), '2m 05s');
  const line = sized(50 * 1048576, 200 * 1048576);
  assert.match(line, / 25%  50\.0 of 200 MiB$/);
});

test('a terminal sees the stage redraw in place, then one done line', () => {
  const s = screen(true);
  const c = clock();
  const d = new Display(s, { now: c });
  d.start('Downloading uv 0.12.17');
  d.detail(sized(1048576, 4 * 1048576));
  d.draw();
  assert.match(s.text, /\r\x1b\[2K {2}\S Downloading uv 0\.12\.17 {2}\S+ +25% {2}1\.0 of 4\.0 MiB {2}0s/);
  c.add(3000);
  d.done('Got uv 0.12.17');
  assert.ok(s.text.endsWith('\r\x1b[2K  ✓ Got uv 0.12.17, done in 3s\n') || s.text.endsWith('\r\x1b[2K  ok Got uv 0.12.17, done in 3s\n'));
  assert.equal(d.timer, null, 'no timer is left running');
});

test('a long line is cut to the terminal width so it never wraps', () => {
  const s = screen(true);
  const d = new Display(s, { now: clock() });
  d.start('x'.repeat(200));
  d.draw();
  d.fail();
  const drawn = s.text.split('\r\x1b[2K').filter(Boolean);
  for (const line of drawn) assert.ok(line.length <= 69, line);
});

test('a pipe sees only the done lines', () => {
  const s = screen(false);
  const d = new Display(s, { now: clock() });
  d.start('Installing wobbl');
  d.detail('resolving');
  d.done('Installed wobbl 0.1.2');
  assert.equal(s.text, process.platform === 'win32' ? '  ok Installed wobbl 0.1.2, done in 0s\n' : '  ✓ Installed wobbl 0.1.2, done in 0s\n');
});

test('follows uv through Python, downloads and the install', () => {
  const s = screen(false);
  const d = new Display(s, { now: clock() });
  const uv = new UvTracker(d, 'Installing wobbl');
  uv.feed('Downloading cpython-3.14.7-macos-x86_64-none (download) (25.4MiB)\n');
  assert.equal(d.step.label, 'Downloading Python 3.14.7');
  uv.feed(' Downloaded cpython-3.14.7-macos-x86_64-none (download)\nResolved 61 packages in 866ms\nDownloading numpy (6.5MiB)\nDownloading gm');
  uv.feed('sh (38.3MiB)\nDownloading vtk (1.0GiB)\n Downloaded numpy\n');
  assert.equal(uv.packages, 61);
  assert.equal(uv.total, Math.round(6.5 * 1048576) + Math.round(38.3 * 1048576) + 1073741824);
  assert.equal(uv.got, Math.round(6.5 * 1048576));
  assert.match(d.step.detail, /0%  6\.5 of 1069 MiB/);
  uv.feed(' Downloaded gmsh\n Downloaded vtk\nPrepared 61 packages in 6.91s\n');
  assert.equal(d.step.detail, 'installing 61 packages');
  uv.feed('Installed 61 packages in 848ms\n + numpy==2.5.4\n + wobbl==0.1.2');
  uv.finish((done) => `Installed wobbl ${done.version}`);
  assert.equal(uv.version, '0.1.2');
  const done = s.text.split('\n').filter(Boolean).map((l) => l.replace(/^ {2}\S+ /, ''));
  assert.deepEqual(done, ['Got Python 3.14.7, done in 0s', 'Installed wobbl 0.1.2, done in 0s']);
});

test('beside npm\'s spinner the stage is drawn one line above the cursor', () => {
  const s = screen(true);
  const d = new Display(s, { now: clock(), shared: true });
  d.start('Installing wobbl');
  d.draw();
  d.done('Installed wobbl 0.1.2');
  d.say('Wobbl 0.1.2 is ready\n');
  const writes = s.text.split(/(?=\r\x1b\[2K|\x1b\[1A)/);
  assert.match(writes[0], /^\r\x1b\[2K {2}\S Installing wobbl {2}0s\n$/);
  assert.ok(writes.slice(1, -1).every((w) => w.startsWith('\x1b[1A') || w.startsWith('\r\x1b[2K')));
  assert.ok(s.text.includes('\x1b[1A\r\x1b[2K  '), 'redraws move up to the stage line');
  assert.match(s.text, /Installed wobbl 0\.1\.2, done in 0s\n\r\x1b\[2KWobbl 0\.1\.2 is ready\n$/);
});
