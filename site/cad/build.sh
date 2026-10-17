#!/bin/sh
# Rebuilds the CAD context models in site/public/models from the scripts in this folder, every one of them or the
# ones named, and merges each example's entry into the manifest.
#   python3 -m venv .venv && .venv/bin/pip install -r site/cad/requirements.txt
#   PYTHON=.venv/bin/python sh site/cad/build.sh [desktop_arm.py ...]
# gltfpack (meshoptimizer 0.24) comes from PATH, from GLTFPACK ("node /path/to/gltfpack/cli.js"), or is fetched by npx.
set -e
here=$(cd "$(dirname "$0")" && pwd)
py=${PYTHON:-python3}
if [ -z "$GLTFPACK" ] && ! command -v gltfpack >/dev/null 2>&1; then
  export GLTFPACK="npx --yes gltfpack@0.24.0"
fi
if [ $# -gt 0 ]; then scripts="$*"; else scripts=$(cd "$here" && ls *.py); fi
for s in $scripts; do
  echo "[cad] $s"
  "$py" "$here/$(basename "$s")"
done
