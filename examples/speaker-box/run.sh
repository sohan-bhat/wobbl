#!/usr/bin/env sh
# A closed loudspeaker cabinet with a cross brace, as one solid from CAD. Wobbl cuts the air out of the inside and
# finds its standing waves, the boxy notes a woofer in it would excite.
set -e
cd "$(dirname "$0")"
wobbl simulate box.step "where do standing waves form in the air inside the box below 1 kHz?" \
  --assign "box: rigid" --fmax "1000 Hz" --no-open -o output
