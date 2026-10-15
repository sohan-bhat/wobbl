#!/usr/bin/env sh
# A printed desktop arm on MG996R hobby servos, with an MG90S at the wrist, moving 500 g from one spot to another.
set -e
cd "$(dirname "$0")"
wobbl simulate arm "Can these hobby servos lift a 500 g payload through this desktop arm's pick and place?" --physics torque \
  --set servo=mg996r --set wrist_servo=mg90s --set links=120,120,60 --set link_mass=60,50,30 \
  --set gripper=40g --set payload=500g --set pick=200,0,40 --set place=0,200,40 --set lift=80 --set move_time=1.2 \
  --set dwell=0.5 -o output
