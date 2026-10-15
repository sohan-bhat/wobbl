#!/usr/bin/env sh
# A level 2 high power rocket on a 75 mm L class motor, flown dual deploy off an 8 ft rail in a 10 mph wind.
set -e
cd "$(dirname "$0")"
wobbl simulate rocket.toml "Will this L class rocket fly straight, reach 11,000 ft and land softly in a 10 mph wind?" -o output
