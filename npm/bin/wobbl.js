#!/usr/bin/env node
'use strict';

// npx wobbl runs the real Wobbl, which is a Python package, and installs it on first use.
//
//   1. A wobbl of this exact version is already installed. Run it.
//   2. Otherwise install it for good with `uv tool install`, or `pipx install` when there
//      is no uv. An npx cache is not enough, because `wobbl setup` registers the installed
//      command with Claude Code, Codex, Cursor and Gemini CLI, and it has to stay put.
//
// Everything this script says goes to stderr, so `npx wobbl mcp` keeps stdout clean.

const { spawn, spawnSync } = require('child_process');
const fs = require('fs');
const os = require('os');
const path = require('path');

const VERSION = require('../package.json').version;
const PYTHON_SPEC = '>=3.10'; // keep in step with requires-python in pyproject.toml
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

function say(...lines) {
  process.stderr.write(`${lines.join('\n')}\n`);
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

// Run an installer, showing its output on stderr and keeping a copy to explain failures.
function runInstaller(file, args, env) {
  return new Promise((resolve) => {
    const [f, a, opts] = command(file, args, env);
    let out = '';
    const child = spawn(f, a, { ...opts, env, stdio: ['ignore', 'pipe', 'pipe'], windowsHide: true });
    const tee = (chunk) => {
      out += chunk;
      process.stderr.write(chunk);
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

// The version a wobbl command reports, 'launcher' for a copy of this script, or null when
// it does not run.
function probe(file, env) {
  const r = runSync(file, ['--version'], { ...env, [PROBE_ENV]: '1' });
  if (r.error || r.status !== 0) return null;
  if (r.stdout.includes(PROBE_REPLY)) return 'launcher';
  const m = /(\d+(?:\.\d+)+[0-9A-Za-z.+_-]*)/.exec(r.stdout);
  return m ? m[1] : null;
}

// The first wobbl of this launcher's version, looking in firstDirs, then PATH, then the
// usual uv and pipx folders. `others` lists the real wobbl commands that did not match.
function pickWobbl(env, firstDirs = []) {
  const want = canonical(VERSION);
  const key = (file) => (IS_WIN ? realpath(file).toLowerCase() : realpath(file));
  const seen = new Set([key(__filename)]);
  const others = [];
  const candidates = [...scan('wobbl', firstDirs, env), ...onPath('wobbl', env), ...scan('wobbl', toolBinDirs(env), env)];
  for (const file of candidates) {
    if (seen.has(key(file))) continue;
    seen.add(key(file));
    const version = probe(file, env);
    if (!version || version === 'launcher') continue;
    if (canonical(version) === want) return { match: file, others };
    others.push({ file, real: realpath(file), version });
  }
  return { match: null, others };
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
  missing: /not found in the package registry|there is no version of wobbl|no versions of wobbl|no matching distribution found|could not find a version that satisfies/i,
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
      `wobbl ${VERSION} is not on PyPI yet, so there is nothing to install.`,
      'If it came out in the last few minutes, wait a moment and run the same command again.',
    ]);
  }
  const manual = tool === 'pipx' ? 'pipx install' : 'uv tool install';
  return new Stop([
    `${tool} could not install wobbl ${VERSION}. Its output above says why.`,
    'You can also install it yourself and run wobbl directly',
    '',
    `    ${manual} wobbl==${VERSION}`,
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

async function installWithPipx(pipx, env, deps) {
  const r = await runInstaller(pipx, ['install', '--force', `wobbl==${VERSION}`], env);
  if (r.code === 0) {
    const fallback = env.PIPX_BIN_DIR || path.join(homeOf(env), '.local', 'bin');
    const bin = binDirFrom(pipx, ['environment', '--value', 'PIPX_BIN_DIR'], fallback, env);
    return { tool: 'pipx', bin, fix: `${shown(pipx, env)} ensurepath` };
  }
  if (PATTERNS.oldPython.test(r.out)) return null;
  throw await failure('pipx', r.out, deps);
}

async function installWithUv(uv, env, deps) {
  const args = ['tool', 'install', '--python', PYTHON_SPEC, `wobbl==${VERSION}`];
  let r = await runInstaller(uv, args, env);
  if (r.code !== 0 && PATTERNS.conflict.test(r.out)) {
    say('', `Another installer already put a wobbl command where uv keeps commands. Replacing it with wobbl ${VERSION}.`, '');
    r = await runInstaller(uv, ['tool', 'install', '--force', ...args.slice(2)], env);
  }
  if (r.code !== 0) throw await failure('uv', r.out, deps);
  const fallback = env.UV_TOOL_BIN_DIR || env.XDG_BIN_HOME || path.join(homeOf(env), '.local', 'bin');
  const bin = binDirFrom(uv, ['tool', 'dir', '--bin'], fallback, env);
  return { tool: 'uv', bin, fix: `${shown(uv, env)} tool update-shell` };
}

async function obtainUv() {
  throw new Stop([
    'Wobbl needs uv to install itself here, and uv is not on this computer.',
    'Install uv, then run the same command again.',
    '',
    `    ${UV_DOCS}`,
  ]);
}

async function install(env, others, deps = {}) {
  say(`Setting up wobbl ${VERSION}. This happens once for each new version.`, '');
  const uv = onPath('uv', env)[0];
  const pipx = onPath('pipx', env)[0];
  const pipxOwnsWobbl = others.some((w) => /[\\/]pipx[\\/]venvs[\\/]/i.test(w.real));
  const how = chooseInstaller({ uv, pipx, pipxOwnsWobbl });
  if (how === 'pipx') {
    const done = await installWithPipx(pipx, env, deps);
    if (done) return done;
    say(
      '',
      'pipx has no Python 3.10 or newer to work with here, and Wobbl needs one.',
      'Wobbl will use uv instead, which brings its own Python.',
      '',
    );
  }
  return installWithUv(uv || (await obtainUv()), env, deps);
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
    'To type wobbl in new terminals as well, run this once',
    '',
    `    ${fix}`,
    '',
  ];
  if (IS_WIN) {
    const add = `[Environment]::SetEnvironmentVariable("Path", "${bin};" + [Environment]::GetEnvironmentVariable("Path", "User"), "User")`;
    lines.push('or run this in PowerShell', '', `    ${add}`);
  } else if (path.basename(env.SHELL || '') === 'fish') {
    lines.push('or run this in fish', '', `    fish_add_path "${bin}"`);
  } else {
    const home = homeOf(env);
    const dir = bin === home || bin.startsWith(home + path.sep) ? `$HOME${bin.slice(home.length)}` : bin;
    lines.push(`or add this line to ${profileFile(env)} yourself`, '', `    export PATH="${dir}:$PATH"`);
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

async function main(argv, env = process.env, deps = {}) {
  if (env[PROBE_ENV] === '1') {
    process.stdout.write(`${PROBE_REPLY} ${VERSION}\n`);
    return 0;
  }
  try {
    let { match, others } = pickWobbl(env);
    if (!match) {
      const done = await install(env, others, deps);
      ({ match } = pickWobbl(env, [done.bin]));
      if (!match) {
        throw new Stop([
          `${done.tool} finished, but there is no wobbl ${VERSION} command in ${done.bin}.`,
          'Please report this with the output above at',
          '',
          `    ${ISSUES}`,
        ]);
      }
      say('', `Installed wobbl ${VERSION} with ${done.tool}. The wobbl command is ${match}.`, ...pathHint(done.bin, done.fix, env), '');
    }
    return await execWobbl(match, argv, env);
  } catch (e) {
    if (!(e instanceof Stop)) throw e;
    say('', ...e.lines);
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
  Stop,
  canonical,
  chooseInstaller,
  failure,
};
