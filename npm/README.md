# wobbl

Simulate anything before you build it. Wobbl meshes your part, runs a real solver,
checks its own work and tells you whether it holds, with a confidence level and a
suggested fix. It works from the terminal and from AI coding agents over MCP.

This package lets you run it with npx.

```sh
npx wobbl setup
npx wobbl part.step "will it hold 200 N?"
```

`wobbl setup` installs the solvers into `~/.wobbl` and registers the MCP server and
skill with Claude Code, Codex, Cursor and Gemini CLI.

## How it works

Wobbl itself is a Python package. On the first run this launcher installs the
matching version from PyPI with `uv tool install`, or `pipx install` if you have
pipx and no uv. That is a normal, lasting install, so the MCP server your agents
use keeps working after npx clears its cache. If you have neither tool, the
launcher downloads a pinned uv into `~/.wobbl/uv`, checks its SHA256 against the
release checksums, and uses that. It never edits your shell profile.

From then on every run goes straight to the installed `wobbl`.

If you would rather skip npm, `uv tool install wobbl` or `pipx install wobbl`
gives you exactly the same thing.

Docs and source live at [sohan-bhat.github.io/wobbl](https://sohan-bhat.github.io/wobbl/)
and [github.com/sohan-bhat/wobbl](https://github.com/sohan-bhat/wobbl). MIT licensed.
