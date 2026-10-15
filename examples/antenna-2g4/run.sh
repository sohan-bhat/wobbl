#!/usr/bin/env sh
# A half wave wire dipole for 2.44 GHz, swept from 1.83 to 3.05 GHz, with the range of a 10 dBm radio in an office.
set -e
cd "$(dirname "$0")"
wobbl antenna dipole --set frequency=2.44GHz --set length=58mm --set diameter=1mm --set points=11 \
  --set tx_power=10dBm --set sensitivity=-92dBm --set environment=office --set distance=25m \
  --question "Is this 58 mm dipole tuned for 2.4 GHz, and does the link reach across a 25 m office?" -o output
