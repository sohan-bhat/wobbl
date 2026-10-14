#!/usr/bin/env sh
# A windshield wiper's four bar linkage driven by a 45 rpm wiper motor against dry glass.
set -e
cd "$(dirname "$0")"
wobbl mechanism fourbar --set ground=300 --set crank=80 --set coupler=300 --set rocker=110 \
  --set link_mass=80,250,300 --set stall_torque=25 --set no_load_speed=45rpm --set load=6 --set sweep=95 \
  --question "Does this wiper linkage sweep 95 degrees, and can a 45 rpm wiper motor drive it over dry glass?" \
  -o output
