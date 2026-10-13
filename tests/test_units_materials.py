import pytest

from wobbl.materials import get_material, resolve_material
from wobbl.units import UnitError, fmt, parse_quantity, parse_spice_value


@pytest.mark.parametrize("text,kind,value", [
    ("200 N", "force", 200.0),
    ("50 lbf", "force", 222.4111),
    ("1/4 in", "length", 6.35),
    ("3.5 GPa", "stress", 3500.0),
    ("10 ksi", "stress", 68.9476),
    ("2 N*m", "moment", 2000.0),
    ("25 in-lb", "moment", 2824.62),
    ("3 g", "accel", 29419.95),
    ("12 W", "power", 12000.0),
    ("5 W/m2K", "film", 0.005),
    ("2700 kg/m3", "density", 2.7e-9),
    ("10 mOhm", "resistance", 0.01),
    ("2.2 MΩ", "resistance", 2.2e6),
])
def test_parse_quantity(text, kind, value):
    assert parse_quantity(text, kind).value == pytest.approx(value, rel=1e-4)


def test_temperatures():
    assert parse_quantity("70 F", "temperature").value == pytest.approx(21.111, rel=1e-4)
    assert parse_quantity("300 K", "temperature").value == pytest.approx(26.85)
    assert parse_quantity("85 °C", "temperature").value == 85


def test_imperial_flag_and_errors():
    assert parse_quantity("45 lbf", "force").imperial
    assert not parse_quantity("200 N", "force").imperial
    with pytest.raises(UnitError):
        parse_quantity("200", "force")
    with pytest.raises(UnitError):
        parse_quantity("200 parsecs", "length")


def test_spice_values():
    assert parse_spice_value("4k7") == pytest.approx(4700)
    assert parse_spice_value("10u") == pytest.approx(1e-5)
    assert parse_spice_value("2.2meg") == pytest.approx(2.2e6)
    assert parse_spice_value("100nF") == pytest.approx(1e-7)


def test_fmt():
    assert fmt(1.23456) == "1.23"
    assert fmt(0) == "0"


def test_material_lookup():
    assert get_material("6061-T6 aluminum").key == "6061-t6"
    assert get_material("printed PETG").key == "petg"
    assert get_material("stainless steel").key == "ss304"
    assert not get_material("pla").ductile
    with pytest.raises(KeyError) as e:
        get_material("pteg")
    assert "petg" in str(e.value)


def test_custom_material():
    m = resolve_material("custom:E=3000,nu=0.35,yield=40")
    assert m.E == 3000 and m.allowable == 40 and m.ductile
    m2 = resolve_material("pla:yield=40")
    assert m2.allowable == 40 and not m2.ductile


def test_internal_units():
    p = get_material("6061-t6").internal()
    assert p["density"] == pytest.approx(2.7e-9)
    assert p["conductivity"] == pytest.approx(167)
