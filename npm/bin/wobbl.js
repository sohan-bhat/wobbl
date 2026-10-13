#!/usr/bin/env node
'use strict';

// npx wobbl runs the real Wobbl, which is a Python package.
//
//   1. A wobbl of this exact version is already installed. Run it.
//
// Everything this script says goes to stderr, so `npx wobbl mcp` keeps stdout clean.

const { spawn, spawnSync } = require('child_process');
const fs = require('fs');
const os = require('os');
const path = require('path');

const VERSION = require('../package.json').version;
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

async function main(argv, env = process.env) {
  if (env[PROBE_ENV] === '1') {
    process.stdout.write(`${PROBE_REPLY} ${VERSION}\n`);
    return 0;
  }
  try {
    const { match } = pickWobbl(env);
    if (!match) {
      throw new Stop([
        `wobbl ${VERSION} is not installed yet. Install it with uv, then run the same command again.`,
        '',
        `    uv tool install wobbl==${VERSION}`,
      ]);
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
  Stop,
  canonical,
};
