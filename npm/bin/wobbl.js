#!/usr/bin/env node
'use strict';

// npx wobbl runs the real Wobbl, which is a Python package, and installs it on first use.
//
//   1. A wobbl of this version or newer is already installed. Run it. An older one never
//      runs, so a stale npm cache can not hand anyone an old CLI.
//   2. Otherwise install the newest wobbl that is at least this version for good with
//      `uv tool install`, or `pipx install` when there
//      is no uv. An npx cache is not enough, because `wobbl setup` registers the installed
//      command with Claude Code, Codex, Cursor and Gemini CLI, and it has to stay put.
//   3. With neither uv nor pipx around, download a pinned uv into ~/.wobbl/uv, check it
//      against the SHA256 checksums of its GitHub release, and install with that.
//
// Everything this script says goes to stderr, so `npx wobbl mcp` keeps stdout clean. The
// installers' own output goes to ~/.wobbl/logs/install.log, and is shown only on failure.

const { spawn, spawnSync } = require('child_process');
const crypto = require('crypto');
const fs = require('fs');
const os = require('os');
const path = require('path');
const zlib = require('zlib');
const { Display, UvTracker, seconds, sized } = require('../lib/progress');

const VERSION = require('../package.json').version;
const PYTHON_SPEC = '>=3.10'; // keep in step with requires-python in pyproject.toml
const UV_VERSION = '0.12.17';
const UV_RELEASES = 'https://github.com/astral-sh/uv/releases/download';
const UV_DOCS = 'https://docs.astral.sh/uv/getting-started/installation/';
const ISSUES = 'https://github.com/sohan-bhat/wobbl/issues';
const IS_WIN = process.platform === 'win32';

// Another copy of this launcher on PATH (a global npm install, an npx shim) answers a
// probe with this instead of running, so it is never mistaken for the real wobbl.
const PROBE_ENV = 'WOBBL_NPM_PROBE';
const PROBE_REPLY = 'wobbl-npm-launcher';

// A problem with a friendly explanation. main() prints its lines and exits with 1.
class Stop extends Error {
  constructor(lines) {
    super(lines.join('\n'));
    this.lines = lines;
  }
}

// Where messages and the progress display go. The npm postinstall points this at the
// terminal, since npm hides what install scripts write to stderr.
let out = process.stderr;
let display = new Display(out);

// `shared` when npm draws its own spinner on the same terminal.
function useOutput(stream, { shared = false } = {}) {
  out = stream;
  display = new Display(stream, { shared });
}

function say(...lines) {
  display.say(`${lines.join('\n')}\n`);
}

function homeOf(env) {
  return (IS_WIN ? env.USERPROFILE : env.HOME) || os.homedir();
}

// -- finding commands -------------------------------------------------------

function pathKey(env) {
  return Object.keys(env).find((k) => k.toUpperCase() === 'PATH') || 'PATH';
}

function pathDirs(env) {
  return (env[pathKey(env)] || '')
    .split(path.delimiter)
    .map((d) => d.trim().replace(/^"(.*)"$/, '$1'))
    .filter(Boolean);
}

function sameFile(a, b) {
  const norm = (p) => path.resolve(p).replace(/[\\/]+$/, '');
  return IS_WIN ? norm(a).toLowerCase() === norm(b).toLowerCase() : norm(a) === norm(b);
}

function realpath(file) {
  try {
    return fs.realpathSync(file);
  } catch {
    return path.resolve(file);
  }
}

// On Windows a command only runs with one of the PATHEXT extensions, like wobbl.exe or
// uv.cmd. Elsewhere the bare name has to be an executable file.
function pathExts(env) {
  if (!IS_WIN) return [''];
  return (env.PATHEXT || '.COM;.EXE;.BAT;.CMD').split(';').filter(Boolean).map((e) => e.toLowerCase());
}

function scan(name, dirs, env) {
  const found = [];
  for (const dir of dirs) {
    for (const ext of pathExts(env)) {
      const file = path.join(dir, name + ext);
      try {
        if (!fs.statSync(file).isFile()) continue;
        if (!IS_WIN) fs.accessSync(file, fs.constants.X_OK);
        found.push(file);
      } catch {
        // not there, or not executable
      }
    }
  }
  return found;
}

// Every `name` command on PATH, in PATH order. Windows asks `where $PATH:name`, which
// leaves out the current folder that a plain `where` would search first.
function onPath(name, env) {
  if (IS_WIN) {
    const root = env.SystemRoot || env.SYSTEMROOT || 'C:\\Windows';
    const r = spawnSync(path.join(root, 'System32', 'where.exe'), [`$PATH:${name}`], {
      env,
      encoding: 'utf8',
      windowsHide: true,
    });
    if (!r.error && r.status === 0) {
      const exts = pathExts(env);
      const found = r.stdout
        .split(/\r?\n/)
        .map((line) => line.trim())
        .filter((f) => f && exts.includes(path.extname(f).toLowerCase()) && fs.existsSync(f));
      if (found.length) return found;
    }
  }
  return scan(name, pathDirs(env), env);
}

// uv and pipx put commands in these folders. Looking here finds an earlier install even
// when the folder is not on PATH yet.
function toolBinDirs(env) {
  return [
    env.UV_TOOL_BIN_DIR,
    env.PIPX_BIN_DIR,
    env.XDG_BIN_HOME,
    env.XDG_DATA_HOME && path.join(env.XDG_DATA_HOME, '..', 'bin'),
    path.join(homeOf(env), '.local', 'bin'),
  ].filter(Boolean);
}

// -- running commands -------------------------------------------------------

// .cmd and .bat files only run through cmd.exe, so their arguments are quoted and escaped
// the way cmd.exe and then the C runtime read them back. Everything else starts directly.
const CMD_META = /([()\][%!^"`<>&|;, *?])/g;

function cmdArg(arg) {
  const quoted = String(arg)
    .replace(/(\\*)"/g, '$1$1\\"')
    .replace(/(\\*)$/, '$1$1');
  return `"${quoted}"`.replace(CMD_META, '^$1');
}

function command(file, args, env) {
  if (IS_WIN && /\.(cmd|bat)$/i.test(file)) {
    const line = [file.replace(CMD_META, '^$1'), ...args.map(cmdArg)].join(' ');
    const shell = env.ComSpec || env.COMSPEC || 'cmd.exe';
    return [shell, ['/d', '/s', '/c', `"${line}"`], { windowsVerbatimArguments: true }];
  }
  return [file, args, {}];
}

function runSync(file, args, env, timeout = 60000) {
  const [f, a, opts] = command(file, args, env);
  return spawnSync(f, a, { ...opts, env, encoding: 'utf8', windowsHide: true, timeout });
}

// Run an installer. Its output goes to the install log and to `onOutput`, which drives the
// progress display, and a copy is kept to explain failures.
function runInstaller(file, args, env, onOutput = () => {}) {
  log(env, `$ ${file} ${args.join(' ')}`);
  return new Promise((resolve) => {
    const [f, a, opts] = command(file, args, env);
    let out = '';
    const child = spawn(f, a, { ...opts, env, stdio: ['ignore', 'pipe', 'pipe'], windowsHide: true });
    const tee = (chunk) => {
      const text = chunk.toString();
      out += text;
      log(env, text, '');
      onOutput(text);
    };
    child.stdout.on('data', tee);
    child.stderr.on('data', tee);
    child.on('error', (e) => resolve({ code: null, out: `${out}\n${e.message}` }));
    child.on('close', (code) => resolve({ code, out }));
  });
}

// -- versions ---------------------------------------------------------------

// npm writes 0.2.0-rc.1 where Python writes 0.2.0rc1. Both become the PEP 440 form, and
// 0.2 equals 0.2.0, so the two sides of the release compare equal.
function canonical(version) {
  const s = String(version)
    .trim()
    .toLowerCase()
    .replace(/^v/, '')
    .split('+')[0]
    .replace(/[-_.]?(alpha|a|beta|b|preview|pre|rc|c)[-_.]?(\d*)/, (_, tag, n) => {
      const short = { alpha: 'a', a: 'a', beta: 'b', b: 'b' }[tag] || 'rc';
      return short + (n || '0');
    })
    .replace(/[-_.]?(post|rev|r)(?![a-z])[-_.]?(\d*)/, (_, tag, n) => `.post${n || '0'}`)
    .replace(/[-_.]?dev[-_.]?(\d*)/, (_, n) => `.dev${n || '0'}`);
  const m = /^(\d+(?:\.\d+)*)(.*)$/.exec(s);
  if (!m) return s;
  const release = m[1].split('.').map(Number);
  while (release.length > 1 && release[release.length - 1] === 0) release.pop();
  return release.join('.') + m[2];
}

// Sort keys for a canonical version, enough of PEP 440 for wobbl's own releases. A dev
// release comes before its pre releases, those before the release, and post releases after.
function versionKey(version) {
  const m = /^(\d+(?:\.\d+)*)(?:(a|b|rc)(\d+))?(?:\.post(\d+))?(?:\.dev(\d+))?$/.exec(canonical(version));
  if (!m) return null;
  const pre = m[2] ? { a: 0, b: 1, rc: 2 }[m[2]] : m[5] !== undefined && m[4] === undefined ? -1 : 3;
  return {
    release: m[1].split('.').map(Number),
    rest: [pre, Number(m[3] || 0), m[4] === undefined ? -1 : Number(m[4]), m[5] === undefined ? Infinity : Number(m[5])],
  };
}

// True when `have` is `want` or newer. A version that can not be read never counts.
function atLeast(have, want) {
  const a = versionKey(have);
  const b = versionKey(want);
  if (!a || !b) return false;
  for (let i = 0; i < Math.max(a.release.length, b.release.length); i++) {
    const x = a.release[i] || 0;
    const y = b.release[i] || 0;
    if (x !== y) return x > y;
  }
  for (let i = 0; i < a.rest.length; i++) {
    if (a.rest[i] !== b.rest[i]) return a.rest[i] > b.rest[i];
  }
  return true;
}

// What the installers are asked for. Never older than this launcher, and the newest PyPI has.
const SPEC = `wobbl>=${canonical(VERSION)}`;

// The version a wobbl command reports, 'launcher' for a copy of this script, or null when
// it does not run.
function probe(file, env) {
  const r = runSync(file, ['--version'], { ...env, [PROBE_ENV]: '1' });
  if (r.error || r.status !== 0) return null;
  if (r.stdout.includes(PROBE_REPLY)) return 'launcher';
  const m = /(\d+(?:\.\d+)+[0-9A-Za-z.+_-]*)/.exec(r.stdout);
  return m ? m[1] : null;
}

// The first wobbl of this launcher's version or newer, looking in firstDirs, then PATH,
// then the usual uv and pipx folders. `others` lists the older wobbl commands it passed.
function pickWobbl(env, firstDirs = []) {
  const key = (file) => (IS_WIN ? realpath(file).toLowerCase() : realpath(file));
  const seen = new Set([key(__filename)]);
  const others = [];
  const candidates = [...scan('wobbl', firstDirs, env), ...onPath('wobbl', env), ...scan('wobbl', toolBinDirs(env), env)];
  for (const file of candidates) {
    if (seen.has(key(file))) continue;
    seen.add(key(file));
    const version = probe(file, env);
    if (!version || version === 'launcher') continue;
    if (atLeast(version, VERSION)) return { match: file, version, others };
    others.push({ file, real: realpath(file), version });
  }
  return { match: null, others };
}

// -- bootstrapping uv -------------------------------------------------------

function wobblHome(env) {
  const base = env.WOBBL_HOME ? env.WOBBL_HOME.replace(/^~(?=$|[\\/])/, homeOf(env)) : path.join(homeOf(env), '.wobbl');
  return path.resolve(base);
}

function uvDir(env) {
  return path.join(wobblHome(env), 'uv');
}

// -- the install log --------------------------------------------------------

function logFile(env) {
  return path.join(wobblHome(env), 'logs', 'install.log');
}

// Begin a fresh log for one install.
function startLog(env) {
  try {
    fs.mkdirSync(path.dirname(logFile(env)), { recursive: true });
    const head = `wobbl npm launcher ${VERSION} on ${process.platform} ${process.arch}, node ${process.version}, ${new Date().toISOString()}`;
    fs.writeFileSync(logFile(env), `${head}\n`);
  } catch {
    // no log is no reason to stop the install
  }
}

function log(env, text, end = '\n') {
  try {
    fs.mkdirSync(path.dirname(logFile(env)), { recursive: true });
    fs.appendFileSync(logFile(env), text + end);
  } catch {
    // no log is no reason to stop the install
  }
}

// The end of the log, for printing beside an explanation.
function logTail(env, count = 60) {
  try {
    const lines = fs.readFileSync(logFile(env), 'utf8').replace(/\s+$/, '').split(/\r?\n/);
    return lines.length > count ? ['...', ...lines.slice(-count)] : lines;
  } catch {
    return [];
  }
}

function existingUv(env) {
  const exe = path.join(uvDir(env), IS_WIN ? 'uv.exe' : 'uv');
  return fs.existsSync(exe) ? exe : null;
}

function hasGlibc() {
  try {
    return Boolean(process.report.getReport().header.glibcVersionRuntime);
  } catch {
    return true;
  }
}

// The name of uv's release build for a machine, or null when uv publishes none for it.
function uvTarget(platform = process.platform, arch = process.arch, glibc = platform === 'linux' && hasGlibc()) {
  const cpu = { x64: 'x86_64', arm64: 'aarch64', ia32: 'i686' }[arch];
  if (platform === 'darwin' && (arch === 'x64' || arch === 'arm64')) return `${cpu}-apple-darwin`;
  if (platform === 'win32' && cpu) return `${cpu}-pc-windows-msvc`;
  if (platform !== 'linux') return null;
  const libc = glibc ? 'gnu' : 'musl';
  if (cpu) return `${cpu}-unknown-linux-${libc}`;
  if (arch === 'arm') return `armv7-unknown-linux-${libc}eabihf`;
  if (arch === 'riscv64') return `riscv64gc-unknown-linux-${libc}`;
  if (arch === 'ppc64' && glibc) return 'powerpc64le-unknown-linux-gnu';
  if (arch === 's390x' && glibc) return 's390x-unknown-linux-gnu';
  return null;
}

// `onProgress(got, total)` hears about each chunk when the server says how big the file is.
function download(url, onProgress, redirects = 5) {
  return new Promise((resolve, reject) => {
    const client = url.startsWith('http:') ? require('http') : require('https');
    const req = client.get(url, { headers: { 'user-agent': `wobbl-npm/${VERSION}` } }, (res) => {
      const { statusCode: status, headers } = res;
      if (status >= 300 && status < 400 && headers.location && redirects > 0) {
        res.resume();
        const next = new URL(headers.location, url).href;
        if (url.startsWith('https:') && !next.startsWith('https:')) reject(new Error(`refused to leave https for ${next}`));
        else resolve(download(next, onProgress, redirects - 1));
        return;
      }
      if (status !== 200) {
        res.resume();
        reject(Object.assign(new Error(`HTTP ${status} for ${url}`), { status }));
        return;
      }
      const chunks = [];
      const total = Number(headers['content-length']) || 0;
      let got = 0;
      res.on('data', (c) => {
        chunks.push(c);
        got += c.length;
        if (total && onProgress) onProgress(got, total);
      });
      res.on('end', () => resolve(Buffer.concat(chunks)));
      res.on('error', reject);
    });
    req.setTimeout(30000, () => req.destroy(Object.assign(new Error(`no answer from ${url}`), { code: 'ETIMEDOUT' })));
    req.on('error', reject);
  });
}

// The expected SHA256 of one asset in a release's sha256.sum, which lists `<hash> *<file>`.
function checksumFor(sums, asset) {
  for (const line of sums.split(/\r?\n/)) {
    const m = /^([0-9a-f]{64})\s+\*?(\S+)$/i.exec(line.trim());
    if (m && m[2] === asset) return m[1].toLowerCase();
  }
  return null;
}

// One file, found by base name, out of a .tar.gz with ustar, pax or GNU long name headers.
function fromTarGz(archive, name) {
  const tar = zlib.gunzipSync(archive);
  let longName = null;
  for (let off = 0; off + 512 <= tar.length; ) {
    const h = tar.subarray(off, off + 512);
    if (h.every((b) => b === 0)) break;
    const field = (a, b) => h.toString('utf8', a, b).replace(/\0[\s\S]*$/, '');
    const size = parseInt(field(124, 136).trim() || '0', 8);
    const type = String.fromCharCode(h[156] || 48);
    const body = tar.subarray(off + 512, off + 512 + size);
    off += 512 + Math.ceil(size / 512) * 512;
    if (type === 'L') {
      longName = body.toString('utf8').replace(/\0[\s\S]*$/, '');
    } else if (type === 'x') {
      const m = /(?:^|\n)\d+ path=([^\n]*)\n/.exec(body.toString('utf8'));
      if (m) longName = m[1];
    } else if (type !== 'g') {
      const prefix = h.toString('latin1', 257, 263) === 'ustar\0' ? field(345, 500) : '';
      const full = longName || (prefix ? `${prefix}/${field(0, 100)}` : field(0, 100));
      longName = null;
      if ((type === '0' || type === '7') && path.posix.basename(full) === name) return Buffer.from(body);
    }
  }
  return null;
}

// One file, found by base name, out of a .zip, read through its central directory.
function fromZip(archive, name) {
  let end = archive.length - 22;
  while (end >= Math.max(0, archive.length - 65557) && archive.readUInt32LE(end) !== 0x06054b50) end--;
  if (end < 0 || archive.readUInt32LE(end) !== 0x06054b50) return null;
  let p = archive.readUInt32LE(end + 16);
  for (let n = archive.readUInt16LE(end + 10); n > 0; n--) {
    if (archive.readUInt32LE(p) !== 0x02014b50) return null;
    const method = archive.readUInt16LE(p + 10);
    const size = archive.readUInt32LE(p + 20);
    const nameLength = archive.readUInt16LE(p + 28);
    const local = archive.readUInt32LE(p + 42);
    const entry = archive.toString('utf8', p + 46, p + 46 + nameLength).replace(/\\/g, '/');
    p += 46 + nameLength + archive.readUInt16LE(p + 30) + archive.readUInt16LE(p + 32);
    if (path.posix.basename(entry) !== name) continue;
    const start = local + 30 + archive.readUInt16LE(local + 26) + archive.readUInt16LE(local + 28);
    const data = archive.subarray(start, start + size);
    if (method === 0) return Buffer.from(data);
    if (method === 8) return zlib.inflateRawSync(data);
    return null;
  }
  return null;
}

async function downloadFailure(error, host, deps) {
  const yourself = ['Install uv or pipx yourself, then run the same command again.', '', `    ${UV_DOCS}`];
  if (error.status) {
    return new Stop([`${host} answered with HTTP ${error.status} when Wobbl asked for uv ${UV_VERSION}.`, ...yourself]);
  }
  if ((await networkState(deps)) === 'offline') {
    return new Stop([
      'Wobbl could not reach the internet, so it could not download uv.',
      'Check your connection, then run the same command again.',
    ]);
  }
  return new Stop([
    `Wobbl could not download uv from ${host} (${error.code || error.message}).`,
    `${host} may be blocked on this network, or the network needs a proxy, which this download does not use.`,
    ...yourself,
  ]);
}

// Download uv into ~/.wobbl/uv and check its SHA256 before anything runs it. The binary
// is written under a temporary name and renamed into place, so a broken download never runs.
async function bootstrapUv(env, deps = {}) {
  const show = deps.display || (deps.say ? new Display(null) : display);
  const target = uvTarget(deps.platform, deps.arch, deps.glibc);
  if (!target) {
    throw new Stop([
      'Wobbl needs uv or pipx to install itself, and neither is on this computer.',
      `uv has no ready made download for ${deps.platform || process.platform} on ${deps.arch || process.arch}, so please install uv or pipx yourself, then run the same command again.`,
      '',
      `    ${UV_DOCS}`,
    ]);
  }
  const windows = target.includes('windows');
  const asset = `uv-${target}${windows ? '.zip' : '.tar.gz'}`;
  const exeName = windows ? 'uv.exe' : 'uv';
  const base = `${(env.WOBBL_UV_DOWNLOAD_URL || UV_RELEASES).replace(/\/+$/, '')}/${UV_VERSION}`;
  const host = new URL(base).host;
  log(env, `No uv or pipx found. Downloading uv ${UV_VERSION} for ${target} from ${base}/${asset}`);
  show.start(`Downloading uv ${UV_VERSION}`);
  show.detail(`from ${host}`);
  const get = deps.download || download;
  let archive;
  let sums;
  try {
    [archive, sums] = await Promise.all([
      get(`${base}/${asset}`, (got, total) => show.detail(sized(got, total))),
      get(`${base}/sha256.sum`),
    ]);
  } catch (e) {
    show.fail();
    throw await downloadFailure(e, host, deps);
  }
  const want = checksumFor(sums.toString('utf8'), asset);
  if (want !== crypto.createHash('sha256').update(archive).digest('hex')) {
    show.fail();
    throw new Stop([
      want
        ? 'The uv download did not match its published SHA256 checksum, so Wobbl threw it away without running it.'
        : `The uv ${UV_VERSION} checksum file does not list ${asset}, so Wobbl threw the download away without running it.`,
      'This usually means the download was damaged on the way. Run the same command again.',
    ]);
  }
  const binary = windows ? fromZip(archive, exeName) : fromTarGz(archive, exeName);
  if (!binary) {
    show.fail();
    throw new Stop([
      `The uv ${UV_VERSION} download has no ${exeName} in it.`,
      'Install uv yourself, then run the same command again.',
      '',
      `    ${UV_DOCS}`,
    ]);
  }
  const dir = uvDir(env);
  const exe = path.join(dir, exeName);
  const part = `${exe}.${process.pid}.part`;
  fs.mkdirSync(dir, { recursive: true });
  fs.writeFileSync(part, binary, { mode: 0o755 });
  try {
    fs.renameSync(part, exe);
  } catch (e) {
    // Another launcher running at the same moment got there first, and Windows will not
    // replace a uv.exe that is running. That copy passed the same check, so use it.
    fs.rmSync(part, { force: true });
    if (!fs.existsSync(exe)) throw e;
  }
  show.done(`Got uv ${UV_VERSION}, checked against its SHA256`);
  log(env, `Saved uv at ${exe}. Its SHA256 matched. It is only used to install wobbl, and PATH and shell profiles were not changed.`);
  return exe;
}

// -- installing -------------------------------------------------------------

// uv comes first because it fetches a Python 3.10 or newer by itself. pipx is the fallback,
// and also the choice when it already manages an older wobbl, so there is one copy.
function chooseInstaller({ uv, pipx, pipxOwnsWobbl }) {
  if (pipx && pipxOwnsWobbl) return 'pipx';
  if (uv) return 'uv';
  if (pipx) return 'pipx';
  return 'bootstrap';
}

const PATTERNS = {
  network: /dns error|failed to lookup|name or service not known|nodename nor servname|getaddrinfo|name resolution|connection refused|connection reset|tcp connect|network is unreachable|timed out|error sending request|failed to fetch|could not fetch|newconnectionerror|max retries exceeded|proxyerror|sslerror|certificate verify/i,
  python: /no interpreter found|no python|python downloads are not allowed|python-downloads|require a different python|requires-python/i,
  missing: /not found in the package registry|there is no version of wobbl|no versions of wobbl|only wobbl\S* (?:is|are) available|no matching distribution found|could not find a version that satisfies/i,
  conflict: /already exists.*--force/i,
  oldPython: /require a different python|requires-python|bad interpreter/i,
};

function reach(url) {
  return new Promise((resolve) => {
    const req = require('https').request(url, { method: 'HEAD', timeout: 6000 }, (res) => {
      res.resume();
      resolve(true);
    });
    req.on('timeout', () => req.destroy(new Error('timed out')));
    req.on('error', () => resolve(false));
    req.end();
  });
}

// 'offline' when nothing answers, 'pypi' when only PyPI is out of reach, else 'online'.
async function networkState(deps) {
  const check = deps.reach || reach;
  const [pypi, github] = await Promise.all([check('https://pypi.org/simple/'), check('https://github.com/')]);
  if (pypi) return 'online';
  return github ? 'pypi' : 'offline';
}

// Turn an installer's failed output into an explanation and a way forward.
async function failure(tool, out, deps = {}) {
  const retry = 'Check your connection, then run the same command again.';
  if (PATTERNS.network.test(out)) {
    const state = await networkState(deps);
    if (state === 'offline') {
      return new Stop([
        'Wobbl could not reach the internet, so it could not download itself.',
        retry,
        'If you go online through a proxy, set HTTPS_PROXY so uv and pip can use it.',
      ]);
    }
    if (state === 'pypi') {
      return new Stop([
        'Wobbl could not reach PyPI (pypi.org), where the wobbl package lives, even though the internet works.',
        'pypi.org may be blocked on this network or briefly down. Try again in a minute.',
        'If your company has its own package index, point uv at it with UV_INDEX_URL, or pipx with PIP_INDEX_URL.',
      ]);
    }
    return new Stop([
      `${tool} ran into a network problem while downloading wobbl ${VERSION}. Its output above has the details.`,
      retry,
      'If you go online through a proxy, set HTTPS_PROXY so uv and pip can use it.',
    ]);
  }
  if (PATTERNS.python.test(out)) {
    return new Stop([
      `wobbl ${VERSION} needs Python 3.10 or newer, and ${tool} could not find or download one.`,
      'Install Python 3.10 or newer, or let uv fetch it by unsetting UV_PYTHON_DOWNLOADS, then run the same command again.',
    ]);
  }
  if (PATTERNS.missing.test(out)) {
    return new Stop([
      `wobbl ${VERSION} or newer is not on PyPI yet, so there is nothing to install.`,
      'If it came out in the last few minutes, wait a moment and run the same command again.',
    ]);
  }
  const manual = tool === 'pipx' ? 'pipx install' : 'uv tool install';
  return new Stop([
    `${tool} could not install wobbl ${VERSION}. Its output above says why.`,
    'You can also install it yourself and run wobbl directly',
    '',
    `    ${manual} "${SPEC}"`,
  ]);
}

function shown(file, env) {
  const bare = path.basename(file, path.extname(file));
  if (onPath(bare, env).some((f) => sameFile(f, file))) return bare;
  return /\s/.test(file) ? `"${file}"` : file;
}

function binDirFrom(file, args, fallback, env) {
  const r = runSync(file, args, env);
  const dir = !r.error && r.status === 0 ? r.stdout.trim().split(/\r?\n/).pop() : '';
  return dir || fallback;
}

// A failed install's explanation, marked so the log is printed above it.
async function installFailed(tool, out, deps) {
  display.fail();
  const stop = await failure(tool, out, deps);
  stop.showLog = true;
  return stop;
}

async function installWithPipx(pipx, env, deps) {
  display.start('Installing wobbl with pipx');
  display.detail('pipx shows no progress, this takes a minute or two');
  const r = await runInstaller(pipx, ['install', '--force', SPEC], env);
  if (r.code === 0) {
    const fallback = env.PIPX_BIN_DIR || path.join(homeOf(env), '.local', 'bin');
    const bin = binDirFrom(pipx, ['environment', '--value', 'PIPX_BIN_DIR'], fallback, env);
    display.done('Installed wobbl with pipx');
    return { tool: 'pipx', bin, fix: `${shown(pipx, env)} ensurepath` };
  }
  display.fail();
  if (PATTERNS.oldPython.test(r.out)) return null;
  throw await installFailed('pipx', r.out, deps);
}

function uvDone(tracker) {
  const what = tracker.version ? `wobbl ${tracker.version}` : 'wobbl';
  return tracker.packages ? `Installed ${what} and its ${tracker.packages - 1} dependencies` : `Installed ${what}`;
}

async function installWithUv(uv, env, deps) {
  const args = ['tool', 'install', '--python', PYTHON_SPEC, SPEC];
  let tracker = new UvTracker(display, 'Installing wobbl');
  let r = await runInstaller(uv, args, env, (text) => tracker.feed(text));
  if (r.code !== 0 && PATTERNS.conflict.test(r.out)) {
    display.fail();
    log(env, 'Another installer left a wobbl command where uv keeps commands. Replacing it.');
    tracker = new UvTracker(display, 'Replacing an older wobbl command');
    r = await runInstaller(uv, ['tool', 'install', '--force', ...args.slice(2)], env, (text) => tracker.feed(text));
  }
  if (r.code !== 0) throw await installFailed('uv', r.out, deps);
  tracker.finish(uvDone);
  const fallback = env.UV_TOOL_BIN_DIR || env.XDG_BIN_HOME || path.join(homeOf(env), '.local', 'bin');
  const bin = binDirFrom(uv, ['tool', 'dir', '--bin'], fallback, env);
  return { tool: 'uv', bin, fix: `${shown(uv, env)} tool update-shell` };
}

async function install(env, others, deps = {}) {
  startLog(env);
  say('Setting up Wobbl. This happens once.');
  const uv = onPath('uv', env)[0] || existingUv(env);
  const pipx = onPath('pipx', env)[0];
  const pipxOwnsWobbl = others.some((w) => /[\\/]pipx[\\/]venvs[\\/]/i.test(w.real));
  const how = chooseInstaller({ uv, pipx, pipxOwnsWobbl });
  if (how === 'pipx') {
    const done = await installWithPipx(pipx, env, deps);
    if (done) return done;
    say('  pipx has no Python 3.10 or newer here, so Wobbl will use uv instead, which brings its own Python.');
  }
  return installWithUv(uv || (await bootstrapUv(env, deps)), env, deps);
}

function profileFile(env) {
  const shell = path.basename(env.SHELL || '');
  if (shell === 'zsh') return '~/.zshrc';
  if (shell === 'bash') return process.platform === 'darwin' ? '~/.bash_profile' : '~/.bashrc';
  return '~/.profile';
}

// wobbl setup registers the full path of the wobbl command, so agents find it either way.
// PATH only matters for typing `wobbl` in a terminal. This prints the fix and changes nothing.
function pathHint(bin, fix, env) {
  if (pathDirs(env).some((d) => sameFile(d, bin))) return [];
  const lines = [
    '',
    `${bin} is not on your PATH yet. Your agents find Wobbl anyway, because wobbl setup saves the full path.`,
  ];
  if (fix) lines.push('To type wobbl in new terminals as well, run this once', '', `    ${fix}`, '');
  const or = fix ? 'or ' : 'To type wobbl in new terminals as well, ';
  if (IS_WIN) {
    const add = `[Environment]::SetEnvironmentVariable("Path", "${bin};" + [Environment]::GetEnvironmentVariable("Path", "User"), "User")`;
    lines.push(`${or}run this in PowerShell`, '', `    ${add}`);
  } else if (path.basename(env.SHELL || '') === 'fish') {
    lines.push(`${or}run this in fish`, '', `    fish_add_path "${bin}"`);
  } else {
    const home = homeOf(env);
    const dir = bin === home || bin.startsWith(home + path.sep) ? `$HOME${bin.slice(home.length)}` : bin;
    lines.push(`${or}add this line to ${profileFile(env)} yourself`, '', `    export PATH="${dir}:$PATH"`);
  }
  return lines;
}

// -- handing over -----------------------------------------------------------

function withPathFirst(env, dir) {
  const key = pathKey(env);
  const rest = pathDirs(env).filter((d) => !sameFile(d, dir));
  return { ...env, [key]: [dir, ...rest].join(path.delimiter) };
}

// Run wobbl in place of this script. stdio is shared, signals are passed on, and its exit
// code, or the signal that ended it, becomes ours.
function execWobbl(file, args, env) {
  // wobbl setup records `shutil.which("wobbl")` as the MCP command. Under npx that would find
  // this script in npx's temporary cache first, so the real wobbl's folder leads PATH.
  const childEnv = withPathFirst(env, path.dirname(file));
  const [f, a, opts] = command(file, args, childEnv);
  // On Windows Ctrl+C reaches wobbl directly, and kill() there would end it before it can
  // clean up, so the launcher only waits. Elsewhere each signal is passed on.
  const signals = IS_WIN ? ['SIGINT', 'SIGBREAK'] : ['SIGINT', 'SIGTERM', 'SIGHUP', 'SIGQUIT', 'SIGUSR2'];
  return new Promise((resolve) => {
    const child = spawn(f, a, { ...opts, env: childEnv, stdio: 'inherit' });
    const relay = (signal) => {
      if (IS_WIN) return;
      try {
        child.kill(signal);
      } catch {
        // already gone
      }
    };
    for (const s of signals) process.on(s, relay);
    const stopRelaying = () => {
      for (const s of signals) process.removeListener(s, relay);
    };
    child.on('error', (e) => {
      stopRelaying();
      say(`Wobbl could not start ${file} (${e.code || e.message}).`);
      resolve(1);
    });
    child.on('exit', (code, signal) => {
      stopRelaying();
      if (signal) {
        // End the same way, so a shell or an agent sees the signal too.
        try {
          process.kill(process.pid, signal);
        } catch {
          // fall back to the exit code a shell would show
        }
        resolve(128 + (os.constants.signals[signal] || 0));
        return;
      }
      resolve(code === null ? 1 : code);
    });
  });
}

// Find a wobbl of this version or newer, installing one when there is none. Returns the
// command, and for a fresh install also what installed it and where.
async function ensureWobbl(env, deps = {}) {
  const found = pickWobbl(env);
  if (found.match) return found;
  const started = Date.now();
  const done = await install(env, found.others, deps);
  const after = pickWobbl(env, [done.bin]);
  if (!after.match) {
    throw new Stop([
      `${done.tool} finished, but there is no wobbl ${VERSION} or newer in ${done.bin}.`,
      `Please report this with the log at ${logFile(env)} to`,
      '',
      `    ${ISSUES}`,
    ]);
  }
  say(`Wobbl ${after.version} is ready, set up in ${seconds(Date.now() - started)}. The command is ${after.match}`);
  return { ...after, done };
}

// Print a Stop, with the end of the install log above it when an installer failed.
function explain(e, env) {
  display.fail();
  if (e.showLog) {
    const tail = logTail(env);
    if (tail.length) say('', `The installer's output, also saved in ${logFile(env)}`, '', ...tail.map((l) => `  ${l}`));
  }
  say('', ...e.lines);
}

async function main(argv, env = process.env, deps = {}) {
  if (env[PROBE_ENV] === '1') {
    process.stdout.write(`${PROBE_REPLY} ${VERSION}\n`);
    return 0;
  }
  try {
    const { match, done } = await ensureWobbl(env, deps);
    if (done) say(...pathHint(done.bin, done.fix, env), '');
    return await execWobbl(match, argv, env);
  } catch (e) {
    if (!(e instanceof Stop)) throw e;
    explain(e, env);
    return 1;
  }
}

if (require.main === module) {
  main(process.argv.slice(2)).then(
    (code) => {
      process.exitCode = code;
    },
    (e) => {
      say('', 'Wobbl hit an unexpected problem while starting. Please report it with these details at', '', `    ${ISSUES}`, '');
      say(String((e && e.stack) || e));
      process.exitCode = 1;
    },
  );
}

module.exports = {
  VERSION,
  PYTHON_SPEC,
  SPEC,
  UV_VERSION,
  Stop,
  atLeast,
  bootstrapUv,
  canonical,
  checksumFor,
  chooseInstaller,
  ensureWobbl,
  explain,
  failure,
  fromTarGz,
  fromZip,
  logFile,
  onPath,
  pathDirs,
  pathHint,
  sameFile,
  say,
  useOutput,
  uvTarget,
};
