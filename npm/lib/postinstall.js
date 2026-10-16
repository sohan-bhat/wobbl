#!/usr/bin/env node
'use strict';

// Runs after `npm i wobbl` and `npm i -g wobbl`. npm only links the launcher into
// node_modules/.bin, or the global bin folder, so this installs the real wobbl the same way
// the launcher would, with uv or pipx. Then a plain `wobbl` works in any terminal.
//
// It never fails the npm install. Every problem becomes a note and exit code 0, and the
// launcher tries again on its next run. It skips CI, `npm i --ignore-scripts` never runs it,
// and it never edits a shell profile. When the tool folder is missing from PATH it prints
// the exact line that adds it.
//
// npm hides what install scripts write, so the notes go straight to the terminal when there
// is one.

const fs = require('fs');
const path = require('path');
const tty = require('tty');

const PACKAGE_ROOT = path.join(__dirname, '..');

function truthy(value) {
  return Boolean(value) && !/^(0|false|no|off)$/i.test(String(value).trim());
}

// Why to skip, or null to go ahead.
function skipReason(env) {
  if (truthy(env.WOBBL_SKIP_POSTINSTALL)) return 'WOBBL_SKIP_POSTINSTALL is set';
  if (truthy(env.CI)) return 'this looks like CI';
  // npm runs a package's own postinstall when you run `npm install` inside its folder,
  // which only happens while working on the launcher itself.
  if (!truthy(env.npm_config_global) && env.INIT_CWD && path.resolve(env.INIT_CWD) === path.resolve(PACKAGE_ROOT)) {
    return 'this is the launcher source folder';
  }
  return null;
}

// The terminal npm runs in, which shows what stderr would not. Falls back to stderr.
function terminal() {
  if (process.platform !== 'win32') {
    try {
      const fd = fs.openSync('/dev/tty', 'w');
      const stream = new tty.WriteStream(fd);
      stream.on('error', () => {});
      return stream;
    } catch {
      // no terminal, as under CI or a piped install
    }
  }
  return process.stderr;
}

// True after `npm i -g wobbl` when npm's global bin folder is on PATH.
function globalBinOnPath(env, launcher) {
  if (!truthy(env.npm_config_global) || !env.npm_config_prefix) return false;
  const bin = process.platform === 'win32' ? env.npm_config_prefix : path.join(env.npm_config_prefix, 'bin');
  return launcher.pathDirs(env).some((d) => launcher.sameFile(d, bin));
}

async function main(env = process.env, deps = {}) {
  const reason = skipReason(env);
  if (reason) {
    if (reason !== 'this is the launcher source folder') {
      process.stderr.write(`wobbl skipped installing the Python CLI because ${reason}. Run npx wobbl --version to install it later.\n`);
    }
    return 0;
  }
  const launcher = require('../bin/wobbl');
  const stream = deps.stream || terminal();
  launcher.useOutput(stream, { shared: stream !== process.stderr });
  const { say } = launcher;
  try {
    const { match, done } = await launcher.ensureWobbl(env, deps);
    if (!done) say(`Wobbl is already installed at ${match}`);
    // After npm i -g, npm's own wobbl launcher already answers in every terminal.
    const hint = globalBinOnPath(env, launcher) ? [] : launcher.pathHint(path.dirname(match), done && done.fix, env);
    if (hint.length) {
      say(...hint, '', `Until then the full command works, as in ${match} setup`, '');
    } else {
      say('Run wobbl setup next to install the solvers and connect your agents.');
    }
  } catch (e) {
    try {
      if (e instanceof launcher.Stop) launcher.explain(e, env);
      else say('', `Wobbl hit an unexpected problem (${(e && e.message) || e}).`);
      say('', 'npm installed the wobbl launcher anyway. It finishes the setup the first time it runs, so try', '', '    npx wobbl --version', '');
    } catch {
      // nothing more can be said
    }
  }
  return 0;
}

if (require.main === module) {
  const end = () => process.exit(0);
  process.on('uncaughtException', end);
  process.on('unhandledRejection', end);
  main().then(end, end);
}

module.exports = { main, skipReason, truthy };
