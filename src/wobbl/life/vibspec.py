"""What a vibration question is made of: the part and its supports, the masses it carries, and what shakes it.

    mass       "45 g at 150,0,25 @ top and within 15 of 150,0,15"   a motor, rigidly on its pad, center at a point
               "0.3 kg @ top distributed"                          spread over the faces, adding no stiffness
    rpm        "9000 rpm" or "4000-9000 rpm"                        a rotor: once per turn and once per blade
    frequency  "120 Hz" or "100-150 Hz"                             any other tone or narrow band to stay clear of
    sine       "2 g z 10-500 Hz"                                    a shaker sweep, base acceleration along an axis
               "10-500 Hz"                                          the study's loads swept as harmonic forces
    psd        "gevs-qual" or "20:0.01, 80:0.04, 500:0.04, 2000:0.01"   random base acceleration, g^2/Hz
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

import numpy as np

from ..geometry.select import parse_point, parse_vector
from ..structural.spec import SpecError, StructuralStudy
from ..units import UnitError, parse_quantity

_NUM = r"[-+]?\d*\.?\d+(?:e[-+]?\d+)?"
_VEC = rf"{_NUM}\s*,\s*{_NUM}\s*,\s*{_NUM}"


@dataclass
class MassSpec:
    mass: float                  # t
    faces: str
    point: list[float] | None = None
    rigid: bool = True
    text: str = ""

    def describe(self) -> str:
        return self.text or f"{self.mass * 1e6:g} g @ {self.faces}"


@dataclass
class Excitation:
    lo: float                    # Hz
    hi: float                    # Hz, equal to lo for a single tone
    label: str

    @property
    def tone(self) -> bool:
        return self.hi <= self.lo * (1 + 1e-9)

    def describe(self) -> str:
        from ..units import fmt

        return f"{self.label} at {fmt(self.lo)} Hz" if self.tone else f"{self.label} from {fmt(self.lo)} to {fmt(self.hi)} Hz"


@dataclass
class Sine:
    accel: float                 # mm/s2 amplitude
    direction: list[float]
    lo: float
    hi: float
    text: str = ""


@dataclass
class VibrationStudy:
    base: StructuralStudy        # part, material and supports, and any loads, which a sweep shakes the part with
    masses: list[str] = field(default_factory=list)
    modes: int = 15
    rpm: list[str] = field(default_factory=list)
    blades: int = 2
    frequencies: list[str] = field(default_factory=list)
    margin: float = 0.2          # keep natural frequencies this share away from every excitation
    sine: str = ""
    psd: str = ""
    axis: str = "z"              # direction of the random input
    duration: str = ""           # time at the random or sine input, for a fatigue estimate
    damping: float | None = None
    target_sf: float | None = None
    keep: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    def excitations(self) -> list[Excitation]:
        out = []
        for r in self.rpm:
            out += parse_rpm(r, self.blades)
        out += [parse_frequency(f) for f in self.frequencies]
        return out


def parse_mass(text: str) -> MassSpec:
    t = str(text).strip()
    if "@" not in t:
        raise SpecError(f"mass '{text}' needs faces that carry it, for example '45 g at 150,0,25 @ top and within 15 "
                        "of 150,0,15'")
    head, faces = (x.strip() for x in t.rsplit("@", 1))
    low = head.lower()
    words = re.findall(r"\b(distributed|rigid)\b", f"{low} {faces.lower()}")
    rigid = "distributed" not in words
    low = re.sub(r"\b(distributed|rigid)\b", "", low).strip()
    faces = re.sub(r"\s*\b(distributed|rigid)\b\s*$", "", faces, flags=re.I).strip()
    point = None
    m = re.search(rf"\bat\s+\(?({_VEC})\)?", low)
    if m:
        point = parse_point(m.group(1)).tolist()
        low = (low[:m.start()] + low[m.end():]).strip()
    try:
        q = parse_quantity(low, "mass")
    except UnitError as e:
        raise SpecError(f"{e} (in mass '{text}', written like '45 g' or '0.3 kg')") from None
    if q.value <= 0:
        raise SpecError(f"mass '{text}' must be positive")
    if rigid and point is None:
        rigid = False   # without a center, the mass is spread over the faces
    return MassSpec(q.value, faces, point, rigid, t)


def parse_rpm(text: str, blades: int = 2) -> list[Excitation]:
    """A rotor speed or range: once per turn (any imbalance) and once per blade passing."""
    t = str(text).strip().lower().replace("rpm", "").strip()
    m = re.match(rf"^({_NUM})\s*(?:-|to|\.\.)\s*({_NUM})$", t) or re.match(rf"^({_NUM})$", t)
    if not m:
        raise SpecError(f"could not read the speed '{text}'. Write it like '9000 rpm' or '4000-9000 rpm'")
    lo = float(m.group(1))
    hi = float(m.group(2)) if m.lastindex == 2 else lo
    if lo <= 0 or hi < lo:
        raise SpecError(f"the speed '{text}' must be positive, low to high")
    out = [Excitation(lo / 60, hi / 60, "once per turn")]
    if blades and blades > 1:
        out.append(Excitation(blades * lo / 60, blades * hi / 60, f"blade pass ({blades} blades)"))
    return out


_BAND = re.compile(rf"({_NUM})\s*(k?hz)?\s*(?:(?:-|to)\s*({_NUM})\s*(k?hz)?)?$")


def _band(t: str, what: str, text: str) -> tuple[float, float, int]:
    """A frequency or band at the end of t, Hz: '120 Hz', '100-150 Hz', '20 Hz to 2 kHz'. A unit written only at the
    end applies to both numbers. Returns (low, high, where the band starts in t)."""
    m = _BAND.search(t.strip().lower())
    if not m:
        raise SpecError(f"could not read the {what} in '{text}'. Write it like '120 Hz' or '100-150 Hz'")
    last = m.group(4) or m.group(2) or "hz"
    lo = float(m.group(1)) * (1e3 if (m.group(2) or last) == "khz" else 1.0)
    hi = float(m.group(3)) * (1e3 if last == "khz" else 1.0) if m.group(3) else lo
    if lo <= 0 or hi < lo:
        raise SpecError(f"the {what} in '{text}' must be positive, low to high")
    return lo, hi, m.start()


def parse_frequency(text: str) -> Excitation:
    t = str(text).strip().lower()
    lo, hi, start = _band(t, "frequency", text)
    if t[:start].strip():
        raise SpecError(f"could not read the frequency '{text}'. Write it like '120 Hz' or '100-150 Hz'")
    return Excitation(lo, hi, "excitation")


def parse_sine(text: str) -> Sine:
    """'2 g z 10-500 Hz': amplitude, axis and band of a base acceleration sweep. A band alone, '10-500 Hz', sweeps
    the study's loads instead, as harmonic forces of their given size, with the base held still."""
    t = str(text).strip().lower()
    m = re.match(rf"^({_NUM})\s*(g|m/s2|m/s\^2|mm/s2)\s+(\S+)\s+(.+)$", t)
    if not m:
        try:
            lo, hi, start = _band(re.sub(r"^from\s+", "", t), "band", text)
        except SpecError:
            lo, hi, start = 0.0, 0.0, 1
        if start == 0 and hi > lo > 0:
            return Sine(0.0, [0.0, 0.0, 1.0], lo, hi, str(text).strip())
        raise SpecError(f"could not read the sine sweep '{text}'. Write it like '2 g z 10-500 Hz', or a band alone "
                        "like '10-500 Hz' to sweep the loads")
    accel = parse_quantity(f"{m.group(1)} {m.group(2)}", "accel").value
    try:
        d = parse_vector(m.group(3))
    except Exception:  # noqa: BLE001
        raise SpecError(f"could not read the axis '{m.group(3)}' in '{text}'. Use x, y, z or 0,0,1") from None
    rest = re.sub(r"^from\s+", "", m.group(4).strip())
    lo, hi, start = _band(rest, "band", text)
    if rest[:start].strip() or hi <= lo:
        raise SpecError(f"the sweep '{text}' needs a band from low to high, like '10-500 Hz'")
    return Sine(accel, d.tolist(), lo, hi, str(text).strip())


def default_damping(mat) -> tuple[float, str]:
    """Modal damping ratio when none is given, with the reason."""
    if mat.category == "metal":
        return 0.02, ("2% of critical, typical of a bolted metal assembly. A one piece machined part can sit well "
                      "under 1%, and the response at a resonance grows as damping falls")
    return 0.03, ("3% of critical, typical of a plastic, wood or composite part. The response at a resonance grows "
                  "as damping falls")


def default_duration(psd_text: str) -> tuple[float, str]:
    """Seconds per axis of a standard random test: GEVS qualification runs 2 minutes, acceptance 1 minute."""
    t = psd_text.strip().lower()
    if t == "gevs-qual":
        return 120.0, "2 minutes per axis, the GEVS qualification duration"
    if t == "gevs-accept":
        return 60.0, "1 minute per axis, the GEVS acceptance duration"
    return 0.0, ""


def seconds(text: str) -> float:
    if not text:
        return 0.0
    return parse_quantity(text, "time").value


def unit_vector(text: str) -> np.ndarray:
    try:
        return parse_vector(text)
    except Exception:  # noqa: BLE001
        raise SpecError(f"could not read the axis '{text}'. Use x, y, z or 0,0,1") from None
