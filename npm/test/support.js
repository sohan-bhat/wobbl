'use strict';

const fs = require('fs');
const os = require('os');
const path = require('path');
const zlib = require('zlib');
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

// A .tar.gz the way uv's releases are laid out, with a pax header in front of the binary.
function tarGz(files) {
  const blocks = [];
  const header = (name, size, type) => {
    const h = Buffer.alloc(512);
    h.write(name, 0, 100);
    h.write('0000755\0', 100);
    h.write('0000000\0', 108);
    h.write('0000000\0', 116);
    h.write(`${size.toString(8).padStart(11, '0')}\0`, 124);
    h.write('00000000000\0', 136);
    h.fill(' ', 148, 156);
    h.write(type, 156);
    h.write('ustar\0', 257);
    h.write('00', 263);
    let sum = 0;
    for (const b of h) sum += b;
    h.write(`${sum.toString(8).padStart(6, '0')}\0 `, 148);
    return h;
  };
  const pad = (n) => Buffer.alloc((512 - (n % 512)) % 512);
  for (const [name, content] of Object.entries(files)) {
    const body = Buffer.from(content);
    const record = ` path=${name}\n`; // a pax record counts its own length digits
    let length = record.length + 1;
    while (String(length).length + record.length !== length) length = String(length).length + record.length;
    const pax = `${length}${record}`;
    blocks.push(header('PaxHeader', pax.length, 'x'), Buffer.from(pax), pad(pax.length));
    blocks.push(header(name.slice(0, 99), body.length, '0'), body, pad(body.length));
  }
  blocks.push(Buffer.alloc(1024));
  return zlib.gzipSync(Buffer.concat(blocks));
}

// A deflated .zip the way uv's Windows releases are laid out.
function zip(files) {
  const locals = [];
  const central = [];
  let offset = 0;
  for (const [name, content] of Object.entries(files)) {
    const raw = Buffer.from(content);
    const packed = zlib.deflateRawSync(raw);
    const n = Buffer.from(name);
    const local = Buffer.alloc(30);
    local.writeUInt32LE(0x04034b50, 0);
    local.writeUInt16LE(20, 4);
    local.writeUInt16LE(8, 8);
    local.writeUInt32LE(packed.length, 18);
    local.writeUInt32LE(raw.length, 22);
    local.writeUInt16LE(n.length, 26);
    const entry = Buffer.alloc(46);
    entry.writeUInt32LE(0x02014b50, 0);
    entry.writeUInt16LE(20, 4);
    entry.writeUInt16LE(20, 6);
    entry.writeUInt16LE(8, 10);
    entry.writeUInt32LE(packed.length, 20);
    entry.writeUInt32LE(raw.length, 24);
    entry.writeUInt16LE(n.length, 28);
    entry.writeUInt32LE(offset, 42);
    locals.push(local, n, packed);
    central.push(entry, n);
    offset += local.length + n.length + packed.length;
  }
  const dir = Buffer.concat(central);
  const end = Buffer.alloc(22);
  end.writeUInt32LE(0x06054b50, 0);
  end.writeUInt16LE(central.length / 2, 8);
  end.writeUInt16LE(central.length / 2, 10);
  end.writeUInt32LE(dir.length, 12);
  end.writeUInt32LE(offset, 16);
  return Buffer.concat([...locals, dir, end]);
}

module.exports = { IS_WIN, LAUNCHER, VERSION, launch, launchAsync, sandbox, tarGz, writeShim, zip };
