# Publishing a release

Wobbl is on PyPI, and the release workflow publishes each new version when its tag is
pushed. These are the steps for cutting a release.

## PyPI

1. Create the project on PyPI and a scoped API token (Account settings, API
   tokens, scope: project `wobbl`), or set up trusted publishing for the GitHub
   repository `sohan-bhat/wobbl` with workflow `release.yml`.
2. Bump `version` in `pyproject.toml`, `__version__` in `src/wobbl/__init__.py` and
   `version` in `npm/package.json`. The npm tests and the release workflow check that
   all three match the tag.
3. Build and check:

   ```sh
   python -m pip install --upgrade build twine
   python -m build
   python -m twine check dist/*
   ```

4. Try the wheel in a clean environment:

   ```sh
   python -m venv /tmp/wobbl-try && /tmp/wobbl-try/bin/pip install dist/wobbl-*.whl
   /tmp/wobbl-try/bin/wobbl --version
   ```

5. Upload: `python -m twine upload dist/*` (username `__token__`, the token as password).
6. Tag and push: `git tag v0.1.0 && git push origin v0.1.0`, then create the GitHub release
   with the notes from the changelog.

After that, users install with `pipx install wobbl` or `uv tool install wobbl`.

## npm

`npm/` holds a small launcher so `npm i -g wobbl` and `npx wobbl ...` work. Its
postinstall, and its first run under npx, install the newest Python package that
is at least the launcher's version for good with `uv tool install` (or
`pipx install`, or a uv it downloads and checks into `~/.wobbl/uv`), then every
run goes to the installed `wobbl`. The launcher's version always equals the
Python package's version.

Publish it only after the same version is live on PyPI, since that is where the
launcher installs from.

The first publish is done by hand, once, because npm can only set up trusted
publishing for a package that already exists:

```sh
npm login
cd npm && npm publish --access public
```

Then turn on trusted publishing so tags publish it from CI:

1. On npmjs.com open the `wobbl` package, then Settings, then Trusted Publisher.
2. Pick GitHub Actions and fill in organization or user `sohan-bhat`, repository
   `wobbl`, workflow filename `release.yml` and environment `npm`.
3. Save. From the next tag on, the `npm` job in `release.yml` publishes with
   provenance after the PyPI upload.

Before anything is published, the `npm-check` job asks npm whether it will take a
trusted publish from `release.yml` in the `npm` environment. When it will not,
the run fails there with the settings to fix, and neither PyPI nor npm gets the
release, so the two never drift apart. Fix the setting and run the workflow
again. To check the setting without releasing, run the workflow by hand from
the Actions tab (Run workflow on `main`), which runs only that check.

A failed `npm publish` fails the run. The job skips a version that is already
on npm, so publishing a version by hand before tagging it is fine.

## Solver packages

Solvers are not bundled; `wobbl setup` pulls `calculix` and `ngspice` from
conda-forge with a pinned micromamba (`WOBBL_MICROMAMBA_VERSION` overrides the
pin). Nothing needs publishing for them.
