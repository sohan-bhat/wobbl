'use strict';

// How the launcher chooses an installer, checks versions, and bootstraps uv, with every
// download mocked or served from a local server.

const test = require('node:test');
const assert = require('node:assert/strict');
const crypto = require('crypto');
const fs = require('fs');
const http = require('http');
const path = require('path');
const launcher = require('../bin/wobbl');
const { IS_WIN, VERSION, launchAsync, sandbox, tarGz, zip } = require('./support');

const quiet = () => {};
const sha256 = (buf) => crypto.createHash('sha256').update(buf).digest('hex');

// What a uv release serves for one target, keyed by file name.
function release(target, uvBody) {
  const windows = target.includes('windows');
  const asset = `uv-${target}${windows ? '.zip' : '.tar.gz'}`;
  const archive = windows
    ? zip({ 'uv.exe': uvBody, 'uvx.exe': 'uvx' })
    : tarGz({ [`uv-${target}/uvx`]: 'uvx', [`uv-${target}/uv`]: uvBody });
  const sums = `${'0'.repeat(64)} *source.tar.gz\n${sha256(archive)} *${asset}\n`;
  return { asset, files: { [asset]: archive, 'sha256.sum': Buffer.from(sums) } };
}

function mockDownload(files, seen = []) {
  return async (url) => {
    seen.push(url);
    const name = url.split('/').pop();
    if (!files[name]) throw Object.assign(new Error(`HTTP 404 for ${url}`), { status: 404 });
    return files[name];
  };
}

test('the npm version and the Python requirement match pyproject.toml', () => {
  const pyproject = fs.readFileSync(path.join(__dirname, '..', '..', 'pyproject.toml'), 'utf8');
  assert.equal(/^version = "([^"]+)"/m.exec(pyproject)[1], VERSION);
  assert.equal(/^requires-python = "([^"]+)"/m.exec(pyproject)[1], launcher.PYTHON_SPEC);
});

test('versions match across npm and Python spellings', () => {
  const same = [
    ['0.1.0', '0.1'],
    ['0.2.0-rc.1', '0.2.0rc1'],
    ['0.2.0-beta.2', '0.2.0b2'],
    ['1.0.0-alpha.1', '1.0.0a1'],
    ['0.3.0-dev.4', '0.3.0.dev4'],
    ['v1.2.3', '1.2.3'],
  ];
  for (const [a, b] of same) assert.equal(launcher.canonical(a), launcher.canonical(b), `${a} vs ${b}`);
  const different = [
    ['0.1.0', '0.1.1'],
    ['0.2.0', '0.2.0rc1'],
    ['0.2.0rc1', '0.2.0rc2'],
    ['1.0.0', '10.0.0'],
  ];
  for (const [a, b] of different) assert.notEqual(launcher.canonical(a), launcher.canonical(b), `${a} vs ${b}`);
});

test('chooses uv first, pipx second, and bootstraps uv when neither exists', () => {
  const choose = launcher.chooseInstaller;
  assert.equal(choose({ uv: '/bin/uv', pipx: '/bin/pipx' }), 'uv');
  assert.equal(choose({ uv: '/bin/uv', pipx: null }), 'uv');
  assert.equal(choose({ uv: null, pipx: '/bin/pipx' }), 'pipx');
  assert.equal(choose({ uv: null, pipx: null }), 'bootstrap');
  assert.equal(choose({ uv: '/bin/uv', pipx: '/bin/pipx', pipxOwnsWobbl: true }), 'pipx');
  assert.equal(choose({ uv: null, pipx: null, pipxOwnsWobbl: true }), 'bootstrap');
});

test('picks the uv build for each platform and none for the rest', () => {
  const target = launcher.uvTarget;
  assert.equal(target('darwin', 'arm64'), 'aarch64-apple-darwin');
  assert.equal(target('darwin', 'x64'), 'x86_64-apple-darwin');
  assert.equal(target('linux', 'x64', true), 'x86_64-unknown-linux-gnu');
  assert.equal(target('linux', 'arm64', false), 'aarch64-unknown-linux-musl');
  assert.equal(target('linux', 'arm', true), 'armv7-unknown-linux-gnueabihf');
  assert.equal(target('win32', 'x64'), 'x86_64-pc-windows-msvc');
  assert.equal(target('win32', 'arm64'), 'aarch64-pc-windows-msvc');
  assert.equal(target('freebsd', 'x64'), null);
  assert.equal(target('aix', 'ppc64'), null);
  assert.equal(target('linux', 'ppc64', false), null);
});

for (const target of ['x86_64-unknown-linux-gnu', 'aarch64-apple-darwin', 'x86_64-pc-windows-msvc']) {
  test(`bootstraps a checked uv for ${target}`, async (t) => {
    const box = sandbox(t);
    const { asset, files } = release(target, `uv for ${target}`);
    const seen = [];
    const [platform, arch] = target.includes('windows') ? ['win32', 'x64'] : target.includes('darwin') ? ['darwin', 'arm64'] : ['linux', 'x64'];
    const exe = await launcher.bootstrapUv(box.env, { platform, arch, glibc: true, download: mockDownload(files, seen), say: quiet });
    assert.equal(exe, path.join(box.home, '.wobbl', 'uv', target.includes('windows') ? 'uv.exe' : 'uv'));
    assert.equal(fs.readFileSync(exe, 'utf8'), `uv for ${target}`);
    if (!IS_WIN) assert.ok(fs.statSync(exe).mode & 0o100, 'uv is executable');
    const base = `https://github.com/astral-sh/uv/releases/download/${launcher.UV_VERSION}`;
    assert.deepEqual(seen.sort(), [`${base}/${asset}`, `${base}/sha256.sum`].sort());
    assert.deepEqual(fs.readdirSync(path.dirname(exe)), [path.basename(exe)]);
  });
}

test('throws a bootstrapped uv away when its checksum does not match', async (t) => {
  const box = sandbox(t);
  const target = 'x86_64-unknown-linux-gnu';
  const { asset, files } = release(target, 'uv');
  files['sha256.sum'] = Buffer.from(`${'a'.repeat(64)} *${asset}\n`);
  const deps = { platform: 'linux', arch: 'x64', glibc: true, download: mockDownload(files), say: quiet };
  await assert.rejects(launcher.bootstrapUv(box.env, deps), (e) => {
    assert.ok(e instanceof launcher.Stop);
    assert.match(e.message, /did not match its published SHA256 checksum/);
    return true;
  });
  assert.equal(fs.existsSync(path.join(box.home, '.wobbl', 'uv')), false);
  delete files['sha256.sum'];
  files['sha256.sum'] = Buffer.from(`${'b'.repeat(64)} *some-other-file.tar.gz\n`);
  await assert.rejects(launcher.bootstrapUv(box.env, deps), /checksum file does not list/);
});

test('explains an unsupported platform instead of downloading', async (t) => {
  const box = sandbox(t);
  const seen = [];
  const deps = { platform: 'freebsd', arch: 'x64', download: mockDownload({}, seen), say: quiet };
  await assert.rejects(launcher.bootstrapUv(box.env, deps), (e) => {
    assert.match(e.message, /no ready made download for freebsd on x64/);
    assert.match(e.message, /docs\.astral\.sh/);
    return true;
  });
  assert.deepEqual(seen, []);
});

test('tells offline apart from a blocked download', async (t) => {
  const box = sandbox(t);
  const fail = async () => {
    throw Object.assign(new Error('getaddrinfo ENOTFOUND github.com'), { code: 'ENOTFOUND' });
  };
  const base = { platform: 'linux', arch: 'x64', glibc: true, download: fail, say: quiet };
  await assert.rejects(launcher.bootstrapUv(box.env, { ...base, reach: async () => false }), /could not reach the internet/);
  await assert.rejects(launcher.bootstrapUv(box.env, { ...base, reach: async () => true }), /may be blocked on this network/);
});

test('turns installer failures into plain explanations', async () => {
  const offline = { reach: async () => false };
  const pypiDown = { reach: async (url) => !url.includes('pypi') };
  const online = { reach: async () => true };
  const dns = 'error: Failed to fetch: `https://pypi.org/simple/wobbl/`\n  Caused by: dns error: failed to lookup address information';
  assert.match((await launcher.failure('uv', dns, offline)).message, /could not reach the internet/);
  assert.match((await launcher.failure('uv', dns, pypiDown)).message, /could not reach PyPI \(pypi\.org\)/);
  assert.match((await launcher.failure('uv', dns, online)).message, /network problem/);
  const missing = 'Because wobbl was not found in the package registry and you require wobbl==0.1.0';
  assert.match((await launcher.failure('uv', missing, online)).message, /is not on PyPI yet/);
  const tooNew = 'Because only wobbl<=0.1.1 is available and you require wobbl>=0.1.2, we can conclude that your requirements are unsatisfiable.';
  assert.match((await launcher.failure('uv', tooNew, online)).message, /or newer is not on PyPI yet/);
  const python = 'error: No interpreter found for Python >=3.10 in managed installations\nPython downloads are not allowed';
  assert.match((await launcher.failure('uv', python, online)).message, /needs Python 3\.10 or newer/);
  const other = await launcher.failure('pipx', 'something else broke', online);
  assert.match(other.message, /pipx install "wobbl>=/);
  for (const e of [other, await launcher.failure('uv', dns, offline)]) {
    for (const line of e.lines.filter((l) => l && !l.startsWith('    '))) {
      assert.doesNotMatch(line, /[;–—→]|: /, `plain prose in ${JSON.stringify(line)}`);
    }
  }
});

// A local stand in for GitHub's release downloads, redirecting the archive like GitHub does.
async function serve(t, files) {
  const server = http.createServer((req, res) => {
    const name = req.url.split('/').pop();
    if (req.url.includes('/releases/') && files[name]) {
      res.writeHead(302, { location: `/objects/${name}` });
      res.end();
    } else if (req.url.startsWith('/objects/') && files[name]) {
      res.end(files[name]);
    } else {
      res.writeHead(404);
      res.end();
    }
  });
  await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
  t.after(() => server.close());
  return `http://127.0.0.1:${server.address().port}/releases`;
}

// A uv binary can not be faked on Windows, which wants a real uv.exe, so these run elsewhere.
const NEEDS_SH = IS_WIN && 'the fake uv inside the archive is a shell script';

function fakeUvScript() {
  return `#!/bin/sh\nexec "${process.execPath}" "${path.join(__dirname, 'fake.js')}" uv "$@"\n`;
}

test('with neither uv nor pipx, downloads uv and installs wobbl with it', { skip: NEEDS_SH }, async (t) => {
  const box = sandbox(t);
  const { files } = release(launcher.uvTarget(), fakeUvScript());
  const url = await serve(t, files);
  const r = await launchAsync(box, ['doctor'], { WOBBL_UV_DOWNLOAD_URL: url }).done;
  assert.equal(r.status, 0, r.stderr);
  const uv = path.join(box.home, '.wobbl', 'uv', 'uv');
  assert.ok(fs.existsSync(uv));
  assert.match(r.stderr, new RegExp(`Got uv ${launcher.UV_VERSION.replace(/\./g, '\\.')}, checked against its SHA256, done in \\d+s`));
  const log = fs.readFileSync(launcher.logFile(box.env), 'utf8');
  assert.ok(log.includes(`Saved uv at ${uv}`), log);
  assert.match(log, /shell profiles were not changed/);
  assert.deepEqual(box.calls('uv')[0].args, ['tool', 'install', '--refresh-package', 'wobbl', '--python', '>=3.10', launcher.SPEC]);
  assert.deepEqual(box.calls('wobbl').map((c) => c.args), [['doctor']]);
  // a later run reuses the downloaded uv instead of fetching it again
  fs.rmSync(path.join(box.toolBin, 'wobbl'));
  const again = await launchAsync(box, ['doctor'], { WOBBL_UV_DOWNLOAD_URL: 'http://127.0.0.1:9/nothing' }).done;
  assert.equal(again.status, 0, again.stderr);
  assert.doesNotMatch(again.stderr, /Downloading uv/);
});

test('when pipx runs on an old Python, switches to a bootstrapped uv', { skip: NEEDS_SH }, async (t) => {
  const box = sandbox(t);
  box.shim('pipx', 'pipx');
  const { files } = release(launcher.uvTarget(), fakeUvScript());
  const url = await serve(t, files);
  const oldPython = 'ERROR: Ignored the following versions that require a different python version: 0.1.0 Requires-Python >=3.10';
  const r = await launchAsync(box, ['doctor'], { WOBBL_UV_DOWNLOAD_URL: url, FAKE_PIPX_FAIL: oldPython }).done;
  assert.equal(r.status, 0, r.stderr);
  assert.match(r.stderr, /Wobbl will use uv instead, which brings its own Python/);
  assert.doesNotMatch(r.stderr, /Ignored the following versions/, 'pipx output stays in the log');
  assert.equal(box.calls('uv')[0].args[1], 'install');
  assert.deepEqual(box.calls('wobbl').map((c) => c.args), [['doctor']]);
});

test('a broken download never leaves a uv behind', { skip: NEEDS_SH }, async (t) => {
  const box = sandbox(t);
  const { asset, files } = release(launcher.uvTarget(), fakeUvScript());
  files[asset] = Buffer.concat([files[asset], Buffer.from('tampered')]);
  const url = await serve(t, files);
  const r = await launchAsync(box, ['doctor'], { WOBBL_UV_DOWNLOAD_URL: url }).done;
  assert.equal(r.status, 1);
  assert.match(r.stderr, /did not match its published SHA256 checksum/);
  assert.equal(fs.existsSync(path.join(box.home, '.wobbl', 'uv', 'uv')), false);
  assert.deepEqual(box.calls(), []);
});
