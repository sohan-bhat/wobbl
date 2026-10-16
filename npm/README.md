# wobbl

Simulate anything before you build it. Wobbl meshes your part, runs a real solver,
checks its own work and tells you whether it holds, with a confidence level and a
suggested fix. It works from the terminal and from AI coding agents over MCP.

```sh
npm i -g wobbl
wobbl setup
wobbl part.step "will it hold 200 N?"
```

`wobbl setup` installs the solvers into `~/.wobbl` and registers the MCP server and
skill with Claude Code, Codex, Cursor and Gemini CLI. You can also skip the install
and run everything through `npx wobbl`.

## How it works

Wobbl itself is a Python package. When npm installs this package, a postinstall step
installs the newest Wobbl from PyPI that is at least this version, with
`uv tool install`, or `pipx install` if you have pipx and no uv. That is a normal,
lasting install, so a plain `wobbl` works in any terminal and the MCP server your
agents use keeps working after npx clears its cache. If you have neither tool, it
downloads a pinned uv into `~/.wobbl/uv`, checks its SHA256 against the release
checksums, and uses that.

The install shows one line per stage, and the installer's full output goes to
`~/.wobbl/logs/install.log`, which is printed only when something fails. The
postinstall never fails your npm install. On any problem it prints a note and the
launcher tries again the next time it runs. It skips CI and `--ignore-scripts`, and
`WOBBL_SKIP_POSTINSTALL=1` turns it off.

It never edits your shell profile. If the folder that uv or pipx puts commands in
is not on your PATH, it prints the exact line that adds it.

From then on every run goes straight to the installed `wobbl`.

If you would rather skip npm, `uv tool install wobbl` or `pipx install wobbl`
gives you exactly the same thing.

Docs and source live at [sohan-bhat.github.io/wobbl](https://sohan-bhat.github.io/wobbl/)
and [github.com/sohan-bhat/wobbl](https://github.com/sohan-bhat/wobbl). MIT licensed.
