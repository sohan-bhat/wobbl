"""Physical test readings from a CSV file, and the metrics that compare them with a model's distribution.

One reading per row. Columns, in any order, with a header row:

    name          gauge or sensor id, e.g. SG1 (rows with the same name and load are repeated readings)
    quantity      strain, displacement or temperature, or a circuit value such as i_led_A, ripple_V or v(out)
    value         the reading
    unit          microstrain, strain, mm, in, m, C, F, K, V, mV, A, mA (default: Wobbl's own units)
    uncertainty   the reading's standard uncertainty, in the same unit (optional)
    x, y, z       where the gauge sits, mm, for strain, displacement and temperature
    direction     the gauge axis or the measured direction, e.g. x, -z or 0.87,0,-0.5 (strain defaults to the
                  larger principal direction in the surface, displacement to the way that point moves)
    load          the load during the reading, in the unit of the study's load (default: the study's load)
    time          seconds into a transient or a waveform
    gauge_length  mm of strain gauge grid (default 3)

Lines that start with # are notes, and a note that says the readings are synthetic marks every result made from
them as synthetic.
"""
from __future__ import annotations

import csv
import io
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from ..units import UnitError, convert, parse_quantity


class ReadingError(ValueError):
    pass


@dataclass
class Reading:
    name: str
    quantity: str
    value: float                # Wobbl's units: strain as a ratio, mm, C, SI for circuits
    u: float | None             # standard uncertainty in the same unit, None when the file gives none
    unit: str = ""
    xyz: list[float] | None = None
    direction: list[float] | None = None
    load: float | None = None
    time: float | None = None
    gauge_length: float | None = None
    row: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


STRAIN_UNITS = {"": 1.0, "strain": 1.0, "mm/mm": 1.0, "microstrain": 1e-6, "ustrain": 1e-6, "ue": 1e-6,
                "µε": 1e-6, "με": 1e-6, "µstrain": 1e-6, "%": 1e-2}
QUANTITY_ALIASES = {"strains": "strain", "deflection": "displacement", "disp": "displacement", "temp": "temperature",
                    "t": "temperature"}


def _scale(quantity: str, unit: str) -> tuple[float, float]:
    """(scale, offset) that turn a reading in unit into Wobbl's units; the uncertainty takes only the scale."""
    u = unit.strip()
    if quantity == "strain":
        if u.lower() not in STRAIN_UNITS and u not in STRAIN_UNITS:
            raise ReadingError(f"unknown strain unit '{unit}' (use microstrain or strain)")
        return STRAIN_UNITS.get(u.lower(), STRAIN_UNITS.get(u, 1.0)), 0.0
    kind = {"displacement": "length", "temperature": "temperature"}.get(quantity)
    if kind is None:
        if not u:
            return 1.0, 0.0
        from .domains import unit_of

        target = unit_of(quantity.split(":")[-1])   # a value named with its unit, such as apogee_m or t_max_C
        if target:
            try:
                a = convert(0.0, u, target)
                return convert(1.0, u, target) - a, a
            except UnitError:
                raise ReadingError(f"cannot turn {unit} into {target}, the unit of {quantity}") from None
        # circuit values: plain SI, or V, mV, A, mA and the like
        for k in ("voltage", "current", "si_power", "resistance", "frequency"):
            try:
                return parse_quantity(f"1 {u}", k).value, 0.0
            except UnitError:
                continue
        raise ReadingError(f"unknown unit '{unit}' for {quantity}")
    if not u:
        return 1.0, 0.0
    if kind == "temperature":
        a = parse_quantity(f"0 {u}", "temperature").value
        return parse_quantity(f"1 {u}", "temperature").value - a, a
    return parse_quantity(f"1 {u}", kind).value, 0.0


def _num(row: dict, key: str) -> float | None:
    v = (row.get(key) or "").strip()
    if not v:
        return None
    try:
        return float(v)
    except ValueError:
        raise ReadingError(f"'{v}' in column {key} is not a number") from None


def read_tests(path) -> tuple[list[Reading], dict]:
    """The readings in a CSV file and its notes. Raises ReadingError with the row at fault."""
    from ..geometry.select import SelectionError, parse_vector

    text = Path(path).read_text(encoding="utf-8-sig")
    notes = [ln.lstrip("#").strip() for ln in text.splitlines() if ln.strip().startswith("#")]
    body = "\n".join(ln for ln in text.splitlines() if ln.strip() and not ln.strip().startswith("#"))
    rows = list(csv.DictReader(io.StringIO(body)))
    if not rows:
        raise ReadingError(f"{Path(path).name} has no readings")
    out = []
    for k, raw in enumerate(rows, start=1):
        row = {(a or "").strip().lower(): (b or "") for a, b in raw.items()}
        missing = [c for c in ("name", "quantity", "value") if not row.get(c, "").strip()]
        if missing:
            raise ReadingError(f"row {k} has no {' or '.join(missing)}")
        q = row["quantity"].strip().lower()
        q = QUANTITY_ALIASES.get(q, q)
        try:
            scale, offset = _scale(q, row.get("unit", ""))
            r = Reading(row["name"].strip(), q, _num(row, "value") * scale + offset, None, row.get("unit", "").strip(),
                        row=k)
            u = _num(row, "uncertainty")
            r.u = abs(u) * scale if u is not None else None
            xyz = [_num(row, c) for c in ("x", "y", "z")]
            if all(v is not None for v in xyz):
                r.xyz = xyz
            elif q in ("strain", "displacement", "temperature"):
                raise ReadingError(f"a {q} reading needs x, y and z")
            if row.get("direction", "").strip():
                r.direction = parse_vector(row["direction"]).tolist()
            r.load, r.time, r.gauge_length = _num(row, "load"), _num(row, "time"), _num(row, "gauge_length")
        except (ReadingError, SelectionError, UnitError) as e:
            raise ReadingError(f"row {k} ({row.get('name', '').strip()}): {e}") from None
        out.append(r)
    meta = {"file": Path(path).name, "notes": notes,
            "synthetic": any(re.search(r"\bsynthetic\b", n, re.I) for n in notes), "rows": len(out)}
    return out, meta


DEFAULT_U = {"strain": (5e-6, 0.02), "displacement": (0.005, 0.01), "temperature": (0.5, 0.0)}


def default_uncertainty(r: Reading) -> float:
    """A reading with no stated uncertainty gets a typical one: 5 microstrain or 2% for a bonded gauge, 0.005 mm or
    1% for a dial gauge, 0.5 C for a calibrated thermocouple and 1% for a circuit reading."""
    floor, rel = DEFAULT_U.get(r.quantity, (0.0, 0.01))
    return max(floor, rel * abs(r.value)) or 1e-12


def area_metric(model: np.ndarray, data: np.ndarray) -> float:
    """Area between the model's CDF (from samples) and the readings' empirical CDF, in the reading's units. With one
    reading it is the mean distance of the model's samples from it."""
    model = np.sort(np.asarray(model, float))
    data = np.sort(np.atleast_1d(np.asarray(data, float)))
    xs = np.union1d(model, data)
    fm = np.searchsorted(model, xs, side="right") / len(model)
    fd = np.searchsorted(data, xs, side="right") / len(data)
    return float(np.sum(np.abs(fm - fd)[:-1] * np.diff(xs)))


def u_pooled(us: np.ndarray, reps: int = 4000, seed: int = 5) -> dict:
    """Pool readings taken under different conditions through u = F(reading), each reading's place in its own model
    distribution. A model whose spread is right gives uniform u, so the area between the empirical CDF of u and the
    uniform CDF measures the mismatch on one scale. The reference is the area a perfect model would stay under
    95% of the time with this many readings."""
    us = np.sort(np.clip(np.asarray(us, float), 0.0, 1.0))
    n = len(us)

    def area(u):
        xs = np.concatenate([[0.0], u, [1.0]])
        f = np.arange(len(u) + 1) / len(u)
        # integral of |F_n(x) - x| over each step where F_n is flat at f
        lo, hi = xs[:-1], xs[1:]
        return float(np.sum(np.where((f >= hi) | (f <= lo), np.abs((f - lo) + (f - hi)) / 2 * (hi - lo),
                                     ((f - lo) ** 2 + (hi - f) ** 2) / 2)))
    rng = np.random.default_rng(seed)
    ref = np.percentile([area(np.sort(rng.random(n))) for _ in range(reps)], 95)
    return {"area": area(us), "reference_95": float(ref), "readings": n}
