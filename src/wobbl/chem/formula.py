"""Elements and formulas: atomic data, formula parsing, Hill order, molar mass and charge.

The atomic data are PubChem's periodic table, vendored in data/elements.json so they work offline. A formula is
read with parentheses, hydrates and charges, as in Ca(OH)2, CuSO4.5H2O, CuSO4·5H2O, SO4^2-, NH4+ or [Fe(CN)6]4-.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path

DATA = Path(__file__).with_name("data")


@lru_cache(maxsize=1)
def table() -> dict[str, dict]:
    """Every element by symbol, with its atomic number, mass, colour, configuration and the rest."""
    d = json.loads((DATA / "elements.json").read_text())
    out = {}
    for row in d["rows"]:
        e = dict(zip(d["columns"], row))
        out[e["symbol"]] = e
    return out


@lru_cache(maxsize=1)
def by_name() -> dict[str, str]:
    names = {e["name"].lower(): s for s, e in table().items()}
    names.update({"aluminium": "Al", "sulphur": "S", "caesium": "Cs"})
    return names


def element(symbol: str) -> dict:
    t = table()
    if symbol not in t:
        raise ValueError(f"'{symbol}' is not an element symbol")
    return t[symbol]


def symbol_of(z: int) -> str:
    for s, e in table().items():
        if e["z"] == z:
            return s
    raise ValueError(f"no element has atomic number {z}")


def mass(symbol: str) -> float:
    return float(element(symbol)["mass"])


_TOKEN = re.compile(r"([A-Z][a-z]?|\(|\)|\[|\]|\d+(?:\.\d+)?)")


class FormulaError(ValueError):
    pass


def split_charge(text: str) -> tuple[str, int]:
    """The formula and its charge. 'SO4^2-', 'SO4 2-' and 'SO42-' are all sulfate, 'NO3-' is nitrate, 'Fe3+' and
    'Fe+3' are iron(III), 'NH4+' is ammonium and 'Cl-' is chloride. With digits run into the sign, the last digit is
    the charge when there are two or more, and a single digit is the charge only on a bare element."""
    t = text.strip().replace("−", "-").replace("⁺", "+").replace("⁻", "-")
    m = re.match(r"^(.+?)\^\(?(\d*)([+-])\)?$", t) or re.match(r"^(.+?)\s+(\d*)([+-])$", t)
    if m:
        return m.group(1).strip(), (1 if m.group(3) == "+" else -1) * int(m.group(2) or 1)
    m = re.match(r"^(.+?)([+-])(\d+)$", t)
    if m and not m.group(1).endswith(("+", "-")):
        return m.group(1), (1 if m.group(2) == "+" else -1) * int(m.group(3))
    m = re.match(r"^(.*?)(\d*)([+-]+)$", t)
    if not m or not (m.group(1) or m.group(2)):
        return t, 0
    body, digits, signs = m.group(1), m.group(2), m.group(3)
    sign = 1 if signs[0] == "+" else -1
    if len(signs) > 1:                      # 'Fe+++' or 'O--'
        return body + digits, sign * len(signs)
    if digits and body.endswith("]"):          # a bracketed complex, [Fe(CN)6]4-
        return body, sign * int(digits)
    if len(digits) >= 2:
        return body + digits[:-1], sign * int(digits[-1])
    if digits and re.fullmatch(r"[A-Z][a-z]?", body):
        return body, sign * int(digits)
    return body + digits, sign


def parse(text: str) -> tuple[Counter, int]:
    """Element counts and charge of a formula. Raises FormulaError when it is not one."""
    t = text.strip()
    if not t:
        raise FormulaError("an empty formula")
    t = t.replace("·", ".").replace("•", ".").replace("*", ".")
    body, charge = split_charge(t)
    body = body.replace(" ", "")
    parts = body.split(".")
    total: Counter = Counter()
    for part in parts:
        m = re.match(r"^(\d+)(?=[A-Z(\[])", part)
        mult = 1
        if m:
            mult = int(m.group(1))
            part = part[m.end():]
        total.update({k: v * mult for k, v in _parse_group(part, text).items()})
    if not total:
        raise FormulaError(f"'{text}' has no elements")
    return total, charge


def _parse_group(s: str, original: str) -> Counter:
    toks = _TOKEN.findall(s)
    if "".join(toks) != s:
        raise FormulaError(f"'{original}' is not a chemical formula")
    stack: list[Counter] = [Counter()]
    i = 0
    known = table()
    while i < len(toks):
        tok = toks[i]
        if tok in ("(", "["):
            stack.append(Counter())
            i += 1
            continue
        if tok in (")", "]"):
            if len(stack) == 1:
                raise FormulaError(f"unbalanced brackets in '{original}'")
            grp = stack.pop()
            n = 1
            if i + 1 < len(toks) and toks[i + 1][0].isdigit():
                n = float(toks[i + 1])
                i += 1
            for k, v in grp.items():
                stack[-1][k] += v * n
            i += 1
            continue
        if tok[0].isdigit():
            raise FormulaError(f"a number with no element before it in '{original}'")
        if tok not in known:
            raise FormulaError(f"'{tok}' in '{original}' is not an element")
        n = 1
        if i + 1 < len(toks) and toks[i + 1][0].isdigit():
            n = float(toks[i + 1])
            i += 1
        stack[-1][tok] += n
        i += 1
    if len(stack) != 1:
        raise FormulaError(f"unbalanced brackets in '{original}'")
    out = Counter()
    for k, v in stack[0].items():
        out[k] = int(v) if float(v).is_integer() else v
    return out


def is_formula(text: str) -> bool:
    try:
        counts, _ = parse(text)
    except FormulaError:
        return False
    return bool(counts) and bool(re.search(r"[A-Z]", text))


def hill(counts: dict, charge: int = 0) -> str:
    """The Hill order formula: C first, then H, then the rest alphabetically, or all alphabetically without C."""
    def term(k):
        n = counts[k]
        return k + ("" if n == 1 else (str(int(n)) if float(n).is_integer() else str(n)))
    keys = sorted(counts)
    if "C" in counts:
        keys = ["C"] + (["H"] if "H" in counts else []) + [k for k in keys if k not in ("C", "H")]
    s = "".join(term(k) for k in keys if counts[k])
    if charge:
        s += ("" if abs(charge) == 1 else f"^{abs(charge)}") + ("+" if charge > 0 else "-")
    return s


def molar_mass(counts: dict) -> float:
    """g/mol."""
    return sum(mass(k) * n for k, n in counts.items())


def electrons(counts: dict, charge: int = 0) -> int:
    return int(sum(element(k)["z"] * n for k, n in counts.items()) - charge)
