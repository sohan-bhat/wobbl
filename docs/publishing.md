# Publishing a release

Wobbl installs from git today. These are the exact steps to publish once the
owner's PyPI account is ready. Nothing here has been run.

## PyPI

1. Create the project on PyPI and a scoped API token (Account settings, API
   tokens, scope: project `wobbl`), or set up trusted publishing for the GitHub
   repository `sohan-bhat/wobbl` with workflow `release.yml`.
2. Bump `version` in `pyproject.toml`, `__version__` in `src/wobbl/__init__.py` and
   `version` in `npm/package.json`. The npm tests and the release workflow check that
   they match.
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

`npm/` holds a small launcher so `npx wobbl ...` works. On first run it installs
the matching Python package for good with `uv tool install` (or `pipx install`,
or a uv it downloads and checks into `~/.wobbl/uv`), then hands every run to the
installed `wobbl`. Its version always equals the Python package's version.

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

Until trusted publishing is set up, that job logs a warning and skips the
publish instead of failing the release. It also skips a version that is
already on npm, so publishing a version by hand before tagging it is fine.

## Solver packages

Solvers are not bundled; `wobbl setup` pulls `calculix` and `ngspice` from
conda-forge with a pinned micromamba (`WOBBL_MICROMAMBA_VERSION` overrides the
pin). Nothing needs publishing for them.
