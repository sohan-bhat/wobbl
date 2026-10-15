#!/usr/bin/env sh
# Everyday circuit questions answered with ngspice: three templates, a KiCad board and a netlist.
set -e
cd "$(dirname "$0")"
wobbl simulate "a blue LED on 5 V with a 47 ohm resistor" "Will a blue LED on 5 V with a 47 ohm resistor fry?" --set supply=5V --set resistor=47 --set led=blue -o led
wobbl simulate "12 V to 5 V buck converter at 2 A" "What is the ripple on this 12 V to 5 V buck at 2 A?" --set vin=12 --set vout=5 --set current=2 --set frequency=500k --set inductor=10u \
  --set capacitor=22u --set max_ripple=0.05 -o buck
wobbl simulate "a MOSFET with 8 mOhm Rds on switching 15 A at 20 kHz" "How hot does this MOSFET get switching 15 A at 20 kHz in a 50 C box?" --set current=15 --set rds_on=8m --set frequency=20k --set theta_ja=40 --set ambient=50 -o mosfet
wobbl simulate kicad-board.kicad_sch "Will anything on this board overheat with a 2 A LED strip on J2?" --set J2.load=2A -o kicad-board
wobbl simulate opamp-cable.cir "Is this buffer stable driving the cable?" -o opamp-cable
