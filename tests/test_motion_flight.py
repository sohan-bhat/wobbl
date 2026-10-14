"""Flights through the integrator: the order of events and phases, the rail, the pad, burnout, weathercocking,
parachutes and the ways a flight can be impossible."""
import math
from dataclasses import replace
from pathlib import Path

import pytest

from wobbl.motion.flight import L_DRAG, L_GRAV, L_STEER, M, VX, VZ, X, Z, fly
from wobbl.motion.spec import Chute, MotionStudy, Motor, SpecError, read_spec

ROCKET = Path(__file__).resolve().parents[1] / "examples" / "rocket-launch" / "rocket.toml"


@pytest.fixture(scope="module")
def rocket():
    return read_spec(ROCKET)


@pytest.fixture(scope="module")
def flight(rocket):
    return fly(rocket)


def test_rocket_events_and_phases_in_order(flight):
    assert [e.name for e in flight.events] == ["liftoff", "rail exit", "burnout", "apogee", "drogue deploy",
                                               "main deploy", "landing"]
    names = [s.phase.name for s in flight.segments]
    assert names[0] == "rail" and names[-1] == "main"
    order = [n for k, n in enumerate(names) if k == 0 or n != names[k - 1]]
    assert order == ["rail", "boost", "coast", "drogue", "main"]
    times = [e.t for e in flight.events]
    assert times == sorted(times) and flight.segments[-1].t1 == flight.end.t


def test_rail_exit_after_the_full_rail_along_its_tilt(flight, rocket):
    y = flight.event("rail exit").y
    assert math.hypot(y[X], y[Z]) == pytest.approx(rocket.rail_length, rel=1e-6)
    assert math.degrees(math.atan2(y[Z], y[X])) == pytest.approx(rocket.angle, abs=1e-6)
    assert math.degrees(math.atan2(y[VZ], y[VX])) == pytest.approx(rocket.angle, abs=1e-6)


def test_all_the_propellant_burns_by_burnout(flight, rocket):
    assert flight.event("burnout").t == pytest.approx(rocket.motor.burn_time)
    assert flight.event("burnout").y[M] == pytest.approx(rocket.burnout_mass, rel=1e-9)
    assert flight.end.y[M] == pytest.approx(rocket.burnout_mass, rel=1e-9)


def test_main_opens_at_its_altitude_and_lands_softly(flight, rocket):
    assert flight.event("main deploy").y[Z] == pytest.approx(150.0, abs=1e-6)
    assert -flight.end.y[VZ] < 6.1 and abs(flight.end.y[Z]) < 1e-6
    assert flight.end.y[VX] == pytest.approx(-rocket.wind, rel=1e-3)     # drifting with the wind


def test_weathercocking_turns_the_rocket_into_the_wind(rocket):
    calm = fly(replace(rocket, wind=0.0))
    windy = fly(rocket)
    assert windy.event("apogee").y[X] > calm.event("apogee").y[X] + 100      # turned into the wind
    assert windy.event("apogee").y[Z] < calm.event("apogee").y[Z]            # and lost height doing it
    assert windy.end.y[X] < windy.event("apogee").y[X] - 500                 # then drifted back under the chutes


def test_a_slow_motor_waits_on_the_pad_and_the_speed_budget_counts_it(rocket):
    slow = Motor("slow", [(0.0, 0.0), (2.0, 400.0), (4.0, 400.0), (4.1, 0.0)], 0.6, 1.2)
    st = replace(rocket, motor=slow, mass=2.0, airframe=None)
    fl = fly(st)
    lift = fl.event("liftoff")
    weight = st.liftoff_mass * 9.80665 * math.sin(math.radians(st.angle))
    assert lift.t == pytest.approx(weight / 200.0, rel=0.02)          # thrust passes the weight on its ramp
    assert lift.y[M] < st.liftoff_mass and lift.y[L_GRAV] > 0         # burned propellant sitting still
    assert lift.y[Z] == 0 and lift.y[VX] == 0
    y = fl.event("burnout").y
    ideal = slow.exhaust_speed * math.log(st.liftoff_mass / st.burnout_mass)
    assert math.hypot(y[VX], y[VZ]) == pytest.approx(ideal - y[L_GRAV] - y[L_DRAG] - y[L_STEER], rel=1e-6)


def test_a_thrown_ball_rises_then_falls(rocket):
    fl = fly(MotionStudy(name="ball", mass=0.145, diameter=0.074, speed=40.0, angle=35.0))
    assert [e.name for e in fl.events] == ["release", "apogee", "impact"]
    assert fl.event("apogee").y[VZ] == pytest.approx(0.0, abs=1e-9)
    assert 0 < fl.end.y[X] < 40.0 ** 2 * math.sin(math.radians(70)) / 9.80665       # air shortens the throw


def test_a_drop_has_no_apogee():
    fl = fly(MotionStudy(name="rock", mass=1.0, diameter=0.09, height=50.0))
    assert [e.name for e in fl.events] == ["release", "impact"]


def test_a_low_apogee_opens_both_parachutes_at_once(rocket):
    st = replace(rocket, chutes=[rocket.chutes[0], Chute("main", 1.52, 1.5, 5000.0)])
    fl = fly(st)
    apo = fl.event("apogee")
    assert fl.event("drogue deploy").t == fl.event("main deploy").t == apo.t
    assert fl.segments[-1].phase.name == "main"


def test_a_rocket_without_parachutes_falls_and_hits(rocket):
    fl = fly(replace(rocket, chutes=[]))
    assert [e.name for e in fl.events][-2:] == ["apogee", "impact"]
    assert fl.segments[-1].phase.name == "fall" and math.hypot(fl.end.y[VX], fl.end.y[VZ]) > 50


def test_impossible_flights_say_why(rocket):
    weak = Motor("weak", [(0.0, 0.0), (1.0, 20.0), (2.0, 0.0)], 0.1, 0.5)
    with pytest.raises(SpecError, match="never lifts"):
        fly(replace(rocket, motor=weak))
    short = Motor("short", [(0.0, 3000.0), (0.01, 3000.0)], 0.01, 0.5)
    with pytest.raises(SpecError, match="burns out before the rocket leaves the rail"):
        fly(replace(rocket, motor=short, rail_length=20.0))
