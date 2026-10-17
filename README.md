<p align="center">
  <a href="https://sohan-bhat.github.io/wobbl/">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="site/public/assets/logo-dark.svg">
      <img alt="Wobbl" src="site/public/assets/logo-light.svg" width="280">
    </picture>
  </a>
</p>

<p align="center">
  <strong>Simulate anything before you build it.</strong><br>
  Find out if your part, circuit or rocket will work, before you make it.
</p>

<p align="center">
  <a href="https://github.com/sohan-bhat/wobbl/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/sohan-bhat/wobbl/actions/workflows/ci.yml/badge.svg"></a>
  <a href="LICENSE"><img alt="License: MIT" src="https://img.shields.io/github/license/sohan-bhat/wobbl"></a>
  <a href="pyproject.toml"><img alt="Python 3.10 or newer" src="https://img.shields.io/badge/python-3.10%2B-3776ab"></a>
</p>

<p align="center">
  <a href="https://sohan-bhat.github.io/wobbl/">Website</a> &nbsp;|&nbsp;
  <a href="https://sohan-bhat.github.io/wobbl/get-started/">Getting started</a> &nbsp;|&nbsp;
  <a href="https://sohan-bhat.github.io/wobbl/guides/">Guides</a> &nbsp;|&nbsp;
  <a href="https://sohan-bhat.github.io/wobbl/guides/cli/">CLI reference</a> &nbsp;|&nbsp;
  <a href="https://sohan-bhat.github.io/wobbl/examples/">Examples</a>
</p>

![A motor mount colored by stress, with the weakest spot marked](examples/cast-motor-mount/output/images/stress.png)

<p align="center"><sub>A real Wobbl result. This motor mount holds, but with less margin than it should have. The marked spot is its weakest point.</sub></p>

## What it does

You ask a question about something you're about to build. Wobbl runs the same solvers engineers
use, checks its own answer, and tells you in plain words whether it holds, where it's weakest, and
how to fix it.

Ask it things like

- Will this 3D printed bracket snap under a 5 kg shelf?
- Will my LED burn out without a resistor?
- How high does a C6 rocket go?
- How hot does a 10 W LED get on this heat sink?
- Can these servos lift my robot arm?

Point it at your own CAD file, a circuit, or just describe the thing in a sentence.

## Get started

```sh
npm i -g wobbl
wobbl setup
```

Then ask your first question.

```sh
wobbl "will a blue LED on 5V with a 47 ohm resistor fry?"
```

Prefer Python? `pipx install wobbl` or `uv tool install wobbl` work too.

## Works with your coding agent

`wobbl setup` also connects Wobbl to Claude Code, Codex, Cursor and Gemini CLI. Your agent can
then check a part or circuit on its own before telling you it will work.

## Can you trust it?

Every answer comes with a confidence level and the assumptions behind it. Wobbl is tested against
95 problems with known answers, and passes all of them. [See the results](https://sohan-bhat.github.io/wobbl/validation/).

## Learn more

- [Examples](https://sohan-bhat.github.io/wobbl/examples/) you can drag around in 3D
- [Getting started guide](https://sohan-bhat.github.io/wobbl/get-started/)
- [All commands](https://sohan-bhat.github.io/wobbl/guides/cli/)
- [What it can't do yet](https://sohan-bhat.github.io/wobbl/validation/limits/)
- [Contributing](https://sohan-bhat.github.io/wobbl/get-started/contributing/)

MIT licensed. Made by Sohan Bhat.
