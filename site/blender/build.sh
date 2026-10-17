#!/bin/sh
# Rebuilds every context model in site/public/models from the scripts here, then the gallery stills that changed.
#   BLENDER=/path/to/blender GLTFPACK="node /path/to/gltfpack/cli.js" sh site/blender/build.sh
set -e
here=$(cd "$(dirname "$0")" && pwd)
blender=${BLENDER:-blender}
# the wiper linkage and the mechanical props are built in CAD now (site/cad/build.sh)
for s in rocket knsb props; do
  "$blender" -b --factory-startup --python-exit-code 1 -P "$here/$s.py"
done
node "$here/stills.mjs"
