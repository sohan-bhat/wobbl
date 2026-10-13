'use strict';

const fs = require('fs');
const os = require('os');
const path = require('path');
const { spawn, spawnSync } = require('child_process');
const { writeShim } = require('./fake');

const LAUNCHER = path.join(__dirname, '..', 'bin', 'wobbl.js');
const VERSION = require('../package.json').version;
const IS_WIN = process.platform === 'win32';

// What Windows needs to start node and cmd.exe at all. Nothing else leaks in from the
// machine running the tests, so a real uv, pipx or wobbl can never be picked up.
const KEEP = ['SystemRoot', 'ComSpec', 'PATHEXT', 'TEMP', 'TMP', 'TMPDIR', 'windir'];

// A throwaway home and PATH folder for one test, removed when the test ends.
function sandbox(t) {
  const root = fs.realpathSync(fs.mkdtempSync(path.join(os.tmpdir(), 'wobbl-npm-')));
  t.after(() => fs.rmSync(root, { recursive: true, force: true }));
  const home = path.join(root, 'home');
  const bin = path.join(root, 'bin');
  fs.mkdirSync(home);
  fs.mkdirSync(bin);
  const env = {};
  for (const key of KEEP) if (process.env[key]) env[key] = process.env[key];
  Object.assign(env, { PATH: bin, HOME: home, USERPROFILE: home, FAKE_LOG: path.join(root, 'calls.jsonl') });
  return {
    root,
    home,
    bin,
    env,
    toolBin: path.join(home, '.local', 'bin'),
    shim: (name, ...args) => writeShim(bin, name, args),
    calls(kind) {
      const text = fs.existsSync(env.FAKE_LOG) ? fs.readFileSync(env.FAKE_LOG, 'utf8') : '';
      const all = text.split('\n').filter(Boolean).map((line) => JSON.parse(line));
      return kind ? all.filter((c) => c.kind === kind) : all;
    },
  };
}

function launch(box, args, extra = {}) {
  return spawnSync(process.execPath, [LAUNCHER, ...args], {
    env: { ...box.env, ...extra },
    encoding: 'utf8',
    timeout: 120000,
  });
}

// The same, without blocking the test process, for tests that serve files or send signals.
function launchAsync(box, args, extra = {}) {
  const child = spawn(process.execPath, [LAUNCHER, ...args], { env: { ...box.env, ...extra } });
  let stdout = '';
  let stderr = '';
  child.stdout.on('data', (d) => (stdout += d));
  child.stderr.on('data', (d) => (stderr += d));
  const done = new Promise((resolve) => {
    child.on('close', (status, signal) => resolve({ status, signal, stdout, stderr }));
  });
  return { child, done };
}

module.exports = { IS_WIN, LAUNCHER, VERSION, launch, launchAsync, sandbox, writeShim };
