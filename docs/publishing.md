# Publishing a release

Wobbl installs from git today. These are the exact steps to publish once the
owner's PyPI account is ready. Nothing here has been run.

## PyPI

1. Create the project on PyPI and a scoped API token (Account settings, API
   tokens, scope: project `wobbl`), or set up trusted publishing for the GitHub
   repository `sohan-bhat/wobbl` with workflow `release.yml`.
2. Bump `version` in `pyproject.toml` and `__version__` in `src/wobbl/__init__.py`.
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

## npm (optional launcher)

An npm package only makes sense as a thin launcher for people who live in
Node tooling (`npx wobbl ...` calling `uvx wobbl`). If wanted:

1. `npm login` with the owner's account and check the name: `npm view wobbl`.
2. Create `npm/package.json` with `"name": "wobbl"`, a `bin` entry pointing to a
   script that runs `uvx wobbl "$@"` (or `pipx run wobbl`), and `"os"` left open.
3. `npm publish --access public` from that folder.

## Solver packages

Solvers are not bundled; `wobbl setup` pulls `calculix` and `ngspice` from
conda-forge with a pinned micromamba (`WOBBL_MICROMAMBA_VERSION` overrides the
pin). Nothing needs publishing for them.
