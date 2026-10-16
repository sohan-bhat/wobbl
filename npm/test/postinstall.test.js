'use strict';

// The postinstall that npm i wobbl and npm i -g wobbl run, against fake uv and wobbl.
// It must always exit 0, whatever happens.

const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('fs');
const path = require('path');
const { spawn } = require('child_process');
const { skipReason } = require('../lib/postinstall');
const { IS_WIN, VERSION, sandbox } = require('./support');

const SCRIPT = path.join(__dirname, '..', 'lib', 'postinstall.js');
const PACKAGE_ROOT = path.join(__dirname, '..');

// Like npm runs it for a dependency. A new session has no terminal to open, so everything
// it says lands on stderr where the test can read it.
function postinstall(box, extra = {}) {
  const env = { INIT_CWD: box.root, npm_lifecycle_event: 'postinstall', ...box.env, ...extra };
  const child = spawn(process.execPath, [SCRIPT], { env, detached: !IS_WIN, stdio: ['ignore', 'pipe', 'pipe'] });
  let stdout = '';
  let stderr = '';
  child.stdout.on('data', (d) => (stdout += d));
  child.stderr.on('data', (d) => (stderr += d));
  return new Promise((resolve) => child.on('close', (status) => resolve({ status, stdout, stderr })));
}

test('skips CI, the opt out variable and the launcher source folder', () => {
  assert.match(skipReason({ CI: 'true' }), /CI/);
  assert.match(skipReason({ CI: '1' }), /CI/);
  assert.equal(skipReason({ CI: 'false', INIT_CWD: '/somewhere/else' }), null);
  assert.equal(skipReason({ CI: '0' }), null);
  assert.match(skipReason({ WOBBL_SKIP_POSTINSTALL: '1' }), /WOBBL_SKIP_POSTINSTALL/);
  assert.match(skipReason({ INIT_CWD: PACKAGE_ROOT }), /source folder/);
  assert.equal(skipReason({ INIT_CWD: PACKAGE_ROOT, npm_config_global: 'true' }), null);
});

test('in CI it installs nothing and exits 0', async (t) => {
  const box = sandbox(t);
  box.shim('uv', 'uv');
  const r = await postinstall(box, { CI: 'true' });
  assert.equal(r.status, 0);
  assert.match(r.stderr, /skipped installing the Python CLI because this looks like CI/);
  assert.deepEqual(box.calls(), []);
});

test('installs the real wobbl with uv and prints how to put its folder on PATH', async (t) => {
  const box = sandbox(t);
  box.shim('uv', 'uv');
  const r = await postinstall(box, { SHELL: '/bin/zsh' });
  assert.equal(r.status, 0, r.stderr);
  assert.equal(r.stdout, '');
  assert.ok(fs.existsSync(path.join(box.toolBin, IS_WIN ? 'wobbl.cmd' : 'wobbl')));
  assert.match(r.stderr, new RegExp(`Wobbl ${VERSION.replace(/\./g, '\\.')} is ready`));
  assert.match(r.stderr, /is not on your PATH yet/);
  assert.match(r.stderr, /uv tool update-shell/);
  if (!IS_WIN) assert.match(r.stderr, /add this line to ~\/\.zshrc yourself\n\n {4}export PATH="\$HOME\/\.local\/bin:\$PATH"/);
  assert.match(r.stderr, /Until then the full command works/);
  assert.equal(fs.existsSync(path.join(box.home, '.zshrc')), false, 'no shell profile is touched');
});

test('says only what to run next when the tool folder is already on PATH', async (t) => {
  const box = sandbox(t);
  box.shim('uv', 'uv');
  const r = await postinstall(box, { PATH: [box.bin, box.toolBin].join(path.delimiter) });
  assert.equal(r.status, 0, r.stderr);
  assert.doesNotMatch(r.stderr, /PATH/);
  assert.match(r.stderr, /Run wobbl setup next/);
});

test('a global install with npm bin on PATH needs no PATH note', async (t) => {
  const box = sandbox(t);
  box.shim('uv', 'uv');
  const prefix = path.join(box.root, 'prefix');
  const globalBin = IS_WIN ? prefix : path.join(prefix, 'bin');
  fs.mkdirSync(globalBin, { recursive: true });
  const r = await postinstall(box, {
    npm_config_global: 'true',
    npm_config_prefix: prefix,
    PATH: [box.bin, globalBin].join(path.delimiter),
  });
  assert.equal(r.status, 0, r.stderr);
  assert.doesNotMatch(r.stderr, /PATH/);
  assert.match(r.stderr, /Run wobbl setup next/);
});

test('finds a wobbl that is already installed and installs nothing', async (t) => {
  const box = sandbox(t);
  box.shim('wobbl', 'wobbl', VERSION);
  box.shim('uv', 'uv');
  const r = await postinstall(box);
  assert.equal(r.status, 0, r.stderr);
  assert.match(r.stderr, /Wobbl is already installed at/);
  assert.deepEqual(box.calls('uv'), []);
});

test('a failed install is a note and exit code 0', async (t) => {
  const box = sandbox(t);
  box.shim('uv', 'uv');
  const r = await postinstall(box, { FAKE_UV_FAIL: 'error: the disk is full' });
  assert.equal(r.status, 0);
  assert.match(r.stderr, /error: the disk is full/);
  assert.match(r.stderr, /npm installed the wobbl launcher anyway/);
  assert.match(r.stderr, /npx wobbl --version/);
});

test('no uv, no pipx and no network is a note and exit code 0', async (t) => {
  const box = sandbox(t);
  const r = await postinstall(box, { WOBBL_UV_DOWNLOAD_URL: 'http://127.0.0.1:9/nothing' });
  assert.equal(r.status, 0);
  assert.match(r.stderr, /npm installed the wobbl launcher anyway/);
  assert.equal(fs.existsSync(path.join(box.home, '.wobbl', 'uv', IS_WIN ? 'uv.exe' : 'uv')), false);
});
