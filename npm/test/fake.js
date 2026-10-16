'use strict';

// Stand ins for uv, pipx and wobbl. Each fake is a tiny shim on a temporary PATH that runs
// this file with its kind, so the launcher spawns them exactly like the real commands.
// Every call is appended to $FAKE_LOG as one JSON line.

const fs = require('fs');
const os = require('os');
const path = require('path');

const IS_WIN = process.platform === 'win32';

// A command called `name` in dir that runs `script` (this file by default) with args first.
function writeShim(dir, name, args, script = __filename) {
  fs.mkdirSync(dir, { recursive: true });
  const words = args.map((a) => `"${a}"`).join(' ');
  if (IS_WIN) {
    const file = path.join(dir, `${name}.cmd`);
    fs.writeFileSync(file, `@"${process.execPath}" "${script}" ${words} %*\r\n`);
    return file;
  }
  const file = path.join(dir, name);
  fs.writeFileSync(file, `#!/bin/sh\nexec "${process.execPath}" "${script}" ${words} "$@"\n`, { mode: 0o755 });
  return file;
}

function log(entry) {
  if (process.env.FAKE_LOG) fs.appendFileSync(process.env.FAKE_LOG, `${JSON.stringify(entry)}\n`);
}

const binDir = (name) => process.env[name] || path.join(os.homedir(), '.local', 'bin');

// `uv tool install ... wobbl>=X` and `pipx install ... wobbl>=X` drop a fake wobbl X in
// their bin folder, or FAKE_INSTALLS_VERSION when PyPI has a newer one, unless the test
// asked for a failure. FAKE_UV_OUTPUT is printed first, the way uv reports its progress.
function install(args, binVar, failure) {
  if (failure) {
    process.stderr.write(`${failure}\n`);
    return 2;
  }
  if (process.env.FAKE_UV_CONFLICT && binVar === 'UV_TOOL_BIN_DIR' && !args.includes('--force')) {
    process.stderr.write('error: Executable already exists: wobbl (use `--force` to overwrite)\n');
    return 2;
  }
  if (process.env.FAKE_UV_OUTPUT && binVar === 'UV_TOOL_BIN_DIR') process.stderr.write(process.env.FAKE_UV_OUTPUT);
  const spec = args.find((a) => /^wobbl(==|>=)/.test(a));
  const version = process.env.FAKE_INSTALLS_VERSION || spec.slice('wobbl>='.length);
  writeShim(binDir(binVar), 'wobbl', ['wobbl', version]);
  process.stderr.write('Installed 1 executable: wobbl\n');
  return 0;
}

function wobbl(version, args) {
  if (args[0] === '--version') {
    process.stdout.write(`wobbl, version ${version}\n`);
    return 0;
  }
  log({ kind: 'wobbl', version, args, path: process.env.PATH, file: process.argv[1] });
  if (process.env.FAKE_WOBBL_WAIT) {
    process.on('SIGTERM', () => {
      log({ kind: 'wobbl', signal: 'SIGTERM' });
      process.removeAllListeners('SIGTERM');
      process.kill(process.pid, 'SIGTERM');
    });
    fs.writeFileSync(process.env.FAKE_WOBBL_WAIT, 'ready');
    setInterval(() => {}, 1000);
    return null;
  }
  return Number(process.env.FAKE_WOBBL_EXIT || 0);
}

function print(text) {
  process.stdout.write(`${text}\n`);
  return 0;
}

function main([kind, ...rest]) {
  if (kind === 'wobbl') return wobbl(rest[0], rest.slice(1));
  log({ kind, args: rest });
  const [verb, sub] = rest;
  if (kind === 'uv') {
    if (verb === '--version') return print('uv 0.12.17');
    if (verb === 'tool' && sub === 'dir') return print(binDir('UV_TOOL_BIN_DIR'));
    if (verb === 'tool' && sub === 'install') return install(rest, 'UV_TOOL_BIN_DIR', process.env.FAKE_UV_FAIL);
  }
  if (kind === 'pipx') {
    if (verb === 'environment') return print(binDir('PIPX_BIN_DIR'));
    if (verb === 'install') return install(rest, 'PIPX_BIN_DIR', process.env.FAKE_PIPX_FAIL);
  }
  process.stderr.write(`fake ${kind} does not know ${rest.join(' ')}\n`);
  return 1;
}

if (require.main === module) {
  const code = main(process.argv.slice(2));
  if (code !== null) process.exitCode = code;
}

module.exports = { writeShim };
