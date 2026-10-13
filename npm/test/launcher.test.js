'use strict';

// End to end runs of bin/wobbl.js against fake uv, pipx and wobbl commands.

const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('fs');
const path = require('path');
const { IS_WIN, LAUNCHER, VERSION, launch, launchAsync, sandbox, writeShim } = require('./support');

const UV_INSTALL = ['tool', 'install', '--python', '>=3.10', `wobbl==${VERSION}`];

function firstOnPath(call) {
  return call.path.split(path.delimiter)[0];
}

test('runs an installed wobbl of the same version and installs nothing', (t) => {
  const box = sandbox(t);
  box.shim('wobbl', 'wobbl', VERSION);
  box.shim('uv', 'uv');
  const r = launch(box, ['doctor']);
  assert.equal(r.status, 0, r.stderr);
  assert.equal(r.stderr, '');
  assert.deepEqual(box.calls('wobbl').map((c) => c.args), [['doctor']]);
  assert.deepEqual(box.calls('uv'), []);
});

test('leaves a wobbl of another version alone and installs the matching one with uv', (t) => {
  const box = sandbox(t);
  box.shim('wobbl', 'wobbl', '0.0.1');
  box.shim('uv', 'uv');
  const r = launch(box, ['--help']);
  assert.equal(r.status, 0, r.stderr);
  assert.deepEqual(box.calls('uv').map((c) => c.args), [UV_INSTALL, ['tool', 'dir', '--bin']]);
  const ran = box.calls('wobbl');
  assert.equal(ran.length, 1);
  assert.equal(ran[0].version, VERSION);
  assert.deepEqual(ran[0].args, ['--help']);
});

test('installs with uv, says where, and runs wobbl with its folder first on PATH', (t) => {
  const box = sandbox(t);
  box.shim('uv', 'uv');
  const r = launch(box, ['setup']);
  assert.equal(r.status, 0, r.stderr);
  assert.equal(r.stdout, '', 'stdout stays clean for MCP');
  const [ran] = box.calls('wobbl');
  assert.deepEqual(ran.args, ['setup']);
  // wobbl setup registers shutil.which("wobbl"), which has to be the lasting install
  assert.equal(firstOnPath(ran), box.toolBin);
  assert.match(r.stderr, new RegExp(`Installed wobbl ${VERSION.replace(/\./g, '\\.')} with uv`));
  assert.match(r.stderr, /is not on your PATH yet/);
  assert.match(r.stderr, /uv tool update-shell/);
});

test('says nothing about PATH when the tool folder is already on it', (t) => {
  const box = sandbox(t);
  box.shim('uv', 'uv');
  const r = launch(box, ['doctor'], { PATH: [box.bin, box.toolBin].join(path.delimiter) });
  assert.equal(r.status, 0, r.stderr);
  assert.doesNotMatch(r.stderr, /PATH/);
});

test('falls back to pipx when there is no uv', (t) => {
  const box = sandbox(t);
  box.shim('pipx', 'pipx');
  const r = launch(box, ['doctor']);
  assert.equal(r.status, 0, r.stderr);
  const calls = box.calls('pipx').map((c) => c.args);
  assert.deepEqual(calls[0], ['install', '--force', `wobbl==${VERSION}`]);
  assert.deepEqual(box.calls('wobbl').map((c) => c.args), [['doctor']]);
  assert.match(r.stderr, /with pipx/);
  assert.match(r.stderr, /pipx ensurepath/);
});

const PIPX_LINKS = IS_WIN && 'pipx copies wobbl.exe on Windows instead of linking it';

test('uses pipx when pipx already manages an older wobbl, even with uv around', { skip: PIPX_LINKS }, (t) => {
  const box = sandbox(t);
  const venvBin = path.join(box.home, '.local', 'pipx', 'venvs', 'wobbl', 'bin');
  fs.symlinkSync(writeShim(venvBin, 'wobbl', ['wobbl', '0.0.1']), path.join(box.bin, 'wobbl'));
  box.shim('uv', 'uv');
  box.shim('pipx', 'pipx');
  const r = launch(box, ['doctor'], { PIPX_BIN_DIR: path.join(box.root, 'pipx-bin') });
  assert.equal(r.status, 0, r.stderr);
  assert.deepEqual(box.calls('uv'), []);
  assert.equal(box.calls('pipx')[0].args[0], 'install');
});

test('a second run finds the first install even though its folder is not on PATH', (t) => {
  const box = sandbox(t);
  box.shim('uv', 'uv');
  assert.equal(launch(box, ['one']).status, 0);
  const r = launch(box, ['two']);
  assert.equal(r.status, 0, r.stderr);
  assert.equal(r.stderr, '');
  assert.equal(box.calls('uv').filter((c) => c.args[1] === 'install').length, 1);
  assert.deepEqual(box.calls('wobbl').map((c) => c.args), [['one'], ['two']]);
});

test('never mistakes another copy of this launcher for the real wobbl', (t) => {
  const box = sandbox(t);
  writeShim(box.bin, 'wobbl', [], LAUNCHER); // like a global npm install of the launcher
  box.shim('uv', 'uv');
  const r = launch(box, ['doctor']);
  assert.equal(r.status, 0, r.stderr);
  const [ran] = box.calls('wobbl');
  assert.equal(ran.version, VERSION);
  assert.equal(firstOnPath(ran), box.toolBin);
});

test('replaces a wobbl command that another installer left where uv puts commands', (t) => {
  const box = sandbox(t);
  box.shim('uv', 'uv');
  const r = launch(box, ['doctor'], { FAKE_UV_CONFLICT: '1' });
  assert.equal(r.status, 0, r.stderr);
  const installs = box.calls('uv').filter((c) => c.args[1] === 'install').map((c) => c.args);
  assert.deepEqual(installs, [UV_INSTALL, ['tool', 'install', '--force', ...UV_INSTALL.slice(2)]]);
  assert.match(r.stderr, /Another installer already put a wobbl command/);
});

test('passes every argument through untouched', (t) => {
  const box = sandbox(t);
  box.shim('wobbl', 'wobbl', VERSION);
  const args = ['run', 'my part.step', 'will it hold 200 N?', '--load=50 N down @ top', '-q', 'folder\\sub dir\\'];
  if (!IS_WIN) args.push('', 'say "hi"', '$HOME', 'a;b|c&d', 'ünïcödé', '*.step');
  const r = launch(box, args);
  assert.equal(r.status, 0, r.stderr);
  assert.deepEqual(box.calls('wobbl')[0].args, args);
});

test('forwards the exit code', (t) => {
  const box = sandbox(t);
  box.shim('wobbl', 'wobbl', VERSION);
  assert.equal(launch(box, ['run'], { FAKE_WOBBL_EXIT: '3' }).status, 3);
  assert.equal(launch(box, ['run'], { FAKE_WOBBL_EXIT: '0' }).status, 0);
  assert.equal(launch(box, ['run'], { FAKE_WOBBL_EXIT: '130' }).status, 130);
});

test('passes a signal on to wobbl and ends with that signal', { skip: IS_WIN && 'POSIX signals' }, async (t) => {
  const box = sandbox(t);
  box.shim('wobbl', 'wobbl', VERSION);
  const ready = path.join(box.root, 'ready');
  const { child, done } = launchAsync(box, ['mcp'], { FAKE_WOBBL_WAIT: ready });
  for (let i = 0; i < 400 && !fs.existsSync(ready); i++) await new Promise((r) => setTimeout(r, 25));
  child.kill('SIGTERM');
  const r = await done;
  assert.equal(r.signal, 'SIGTERM', r.stderr);
  assert.ok(box.calls('wobbl').some((c) => c.signal === 'SIGTERM'));
});

test('explains a version that is not on PyPI', (t) => {
  const box = sandbox(t);
  box.shim('uv', 'uv');
  const fail = '  × No solution found when resolving tool dependencies\n  ╰─▶ Because there is no version of wobbl==9.9.9';
  const r = launch(box, ['doctor'], { FAKE_UV_FAIL: fail });
  assert.equal(r.status, 1);
  assert.match(r.stderr, /is not on PyPI yet/);
  assert.deepEqual(box.calls('wobbl'), []);
});
