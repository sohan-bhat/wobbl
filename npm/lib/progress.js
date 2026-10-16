'use strict';

// The install display. Each stage is one line. On a terminal the line redraws in place
// with a spinner, a bar when the size is known, and the time so far, then settles into
// a done line. Anywhere else only the done lines are written, so logs stay short.
//
// `shared` is for a terminal that npm draws its own spinner on, at the start of the line
// the cursor is on. The running stage then lives one line above the cursor, out of its way.

const IS_WIN = process.platform === 'win32';
const FRAMES = IS_WIN ? ['|', '/', '-', '\\'] : ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];
const DONE = IS_WIN ? 'ok' : '✓';
const FULL = IS_WIN ? '#' : '█';
const EMPTY = IS_WIN ? '.' : '░';

function seconds(ms) {
  const s = Math.max(0, Math.round(ms / 1000));
  if (s < 60) return `${s}s`;
  return `${Math.floor(s / 60)}m ${String(s % 60).padStart(2, '0')}s`;
}

function mib(bytes) {
  const m = bytes / 1048576;
  return m >= 100 ? m.toFixed(0) : m.toFixed(1);
}

function bar(fraction, width = 16) {
  const filled = Math.round(Math.min(1, Math.max(0, fraction)) * width);
  return FULL.repeat(filled) + EMPTY.repeat(width - filled);
}

// `done` and `total` in bytes. Gives `[bar] 58%  120 of 208 MiB`.
function sized(done, total) {
  if (!total) return '';
  const f = done / total;
  return `${bar(f)} ${String(Math.floor(f * 100)).padStart(3)}%  ${mib(done)} of ${mib(total)} MiB`;
}

class Display {
  constructor(stream, { now = Date.now, shared = false } = {}) {
    this.stream = stream;
    this.live = Boolean(stream && stream.isTTY) && process.env.TERM !== 'dumb';
    this.shared = this.live && shared;
    this.drawn = false;
    this.now = now;
    this.step = null;
    this.timer = null;
    this.frame = 0;
  }

  write(text) {
    try {
      this.stream.write(text);
    } catch {
      // a closed terminal must never break the install
    }
  }

  // Start a stage. Any stage still running is closed first.
  start(label) {
    if (this.step) this.done();
    this.step = { label, detail: '', at: this.now() };
    this.drawn = false;
    if (this.live) {
      this.draw();
      this.timer = setInterval(() => this.draw(), 100);
      if (this.timer.unref) this.timer.unref();
    }
    return this;
  }

  detail(text) {
    if (this.step) this.step.detail = text;
  }

  label(text) {
    if (this.step) this.step.label = text;
  }

  draw() {
    if (!this.step) return;
    const { label, detail, at } = this.step;
    this.frame = (this.frame + 1) % FRAMES.length;
    const head = `  ${FRAMES[this.frame]} ${label}`;
    const time = seconds(this.now() - at);
    // The time always shows. On a narrow terminal the detail goes first, then the label is cut.
    const width = (this.stream.columns || 80) - 1;
    let line = [head, detail, time].filter(Boolean).join('  ');
    if (line.length > width) line = `${head.slice(0, Math.max(4, width - time.length - 2))}  ${time}`;
    this.put(line.slice(0, width));
  }

  // Replace the stage's line with `text`, and with `end` after it.
  put(text, end = '') {
    if (!this.shared) {
      this.write(`\r\x1b[2K${text}${end}`);
    } else if (!this.drawn) {
      this.write(`\r\x1b[2K${text}\n`);
    } else {
      this.write(`\x1b[1A\r\x1b[2K${text}\n`);
    }
    this.drawn = true;
  }

  // A plain message. On a shared terminal npm's spinner character is cleared first.
  say(text) {
    this.write(this.shared ? `\r\x1b[2K${text}` : text);
  }

  stop() {
    if (this.timer) clearInterval(this.timer);
    this.timer = null;
  }

  // Settle the running stage into its done line, with an optional new label.
  done(label) {
    if (!this.step) return;
    const took = seconds(this.now() - this.step.at);
    this.stop();
    const line = `  ${DONE} ${label || this.step.label}, done in ${took}`;
    if (this.live) this.put(line, '\n');
    else this.write(`${line}\n`);
    this.step = null;
  }

  // Clear the running stage without a done line, before an explanation is printed.
  fail() {
    this.stop();
    if (this.live && this.drawn && this.step) {
      if (this.shared) this.write('\x1b[1A\r\x1b[2K');
      else this.write('\r\x1b[2K');
    }
    this.step = null;
    this.drawn = false;
  }
}

const UNITS = { b: 1, kib: 1024, mib: 1048576, gib: 1073741824 };

function bytes(n, unit) {
  return Math.round(Number(n) * (UNITS[unit.toLowerCase()] || 1));
}

// Follows `uv tool install` as it writes to a pipe. uv names every download over about a
// MiB with its size and says when each one finishes, so the bar counts those bytes.
class UvTracker {
  constructor(display, label) {
    this.display = display;
    this.label = label;
    this.sizes = new Map();
    this.total = 0;
    this.got = 0;
    this.packages = 0;
    this.version = null;
    this.rest = '';
    display.start(label);
    display.detail('resolving');
  }

  feed(chunk) {
    const lines = (this.rest + chunk).split(/\r?\n/);
    this.rest = lines.pop();
    for (const line of lines) this.line(line);
  }

  line(raw) {
    const line = raw.trim();
    const d = this.display;
    let m;
    if ((m = /^Downloading (cpython|pypy|graalpy)-(\d+\.\d+\.\d+)\S*.*?\(([\d.]+)\s*([KMG]i?B)\)/i.exec(line))) {
      this.python = `Python ${m[2]}`;
      d.fail(); // uv fetches Python before it resolves anything, so that stage had not begun
      d.start(`Downloading ${this.python}`);
      d.detail(`${m[3]} ${m[4]}`);
    } else if (/^Downloaded (cpython|pypy|graalpy)-/i.test(line)) {
      d.done(`Got ${this.python}`);
      d.start(this.label);
      d.detail('resolving');
    } else if ((m = /^Resolved (\d+) packages?/.exec(line))) {
      this.packages = Number(m[1]);
      d.detail(`${this.packages} packages`);
    } else if ((m = /^Downloading (\S+) \(([\d.]+)\s*([KMG]i?B)\)/.exec(line))) {
      const size = bytes(m[2], m[3]);
      this.sizes.set(m[1], size);
      this.total += size;
      this.show();
    } else if ((m = /^Downloaded (\S+)/.exec(line))) {
      this.got += this.sizes.get(m[1]) || 0;
      this.show();
    } else if (/^Prepared \d+ packages?/.test(line)) {
      d.detail(this.packages ? `installing ${this.packages} packages` : 'installing packages');
    } else if ((m = /^\+ wobbl==(\S+)/.exec(line))) {
      this.version = m[1];
    }
  }

  show() {
    this.display.detail(sized(this.got, this.total));
  }

  // Settle the stage. `text` may be a function of this tracker, called once the last
  // line is read.
  finish(text) {
    if (this.rest) this.line(this.rest);
    this.rest = '';
    this.display.done(typeof text === 'function' ? text(this) : text);
  }
}

module.exports = { Display, UvTracker, bar, seconds, sized };
