"""What a free-form request means: the everyday phrases that name each physics, and how strongly a phrase counts.

People rarely say "internal ballistics". They say "a thrust test", "a static fire" or "put it on the test stand".
SYNONYMS holds those phrases once, grouped under one plain name, and an analysis or a domain takes them into its
keywords with phrases("static fire", "thrust curve"), so the same table feeds the domain's words, the kind of
thing and the physics. Matches are weighed by how many words they span, so "thrust test" outweighs "rocket" and a
named test outweighs a single word that happens to sit inside it.
"""
from __future__ import annotations

import re

# a plain name, then every way people say it. Each entry is a regex, matched on word boundaries without case.
SYNONYMS: dict[str, tuple[str, ...]] = {
    # solid motors on the ground
    "static fire": (r"static(?:ally)?[\s-]+fir(?:e|es|ed|ing)", r"static[\s-]+(?:test|burn)s?", r"test[\s-]+fir(?:e|es|ed|ing)",
                    r"hot[\s-]+fir(?:e|es|ed|ing)", r"thrust[\s-]+tests?", r"motor[\s-]+tests?", r"test[\s-]+stands?",
                    r"thrust[\s-]+stands?", r"fir(?:e|ing)\s+(?:the|a|my)\s+motor", r"ground[\s-]+tests?\s+of\s+(?:a|the|my)\s+motor"),
    "thrust curve": (r"thrust[\s-]+curves?", r"thrust\s+(?:over|against|vs\.?)\s+time", r"thrust[\s-]+profiles?",
                     r"how\s+much\s+thrust", r"peak\s+thrust", r"average\s+thrust", r"total\s+impulse",
                     r"motor\s+class"),
    "burn": (r"burn[\s-]+times?", r"burn[\s-]+rates?", r"how\s+long\s+(?:does\s+it|will\s+it|it)\s+burns?",
             r"burns?\s+for\s+how\s+long", r"burn(?:s|ed|ing)?\s+(?:out|through)", r"burnout"),
    "chamber pressure": (r"chamber[\s-]+pressures?", r"(?:motor|casing|case)\s+pressures?", r"\bkn\b",
                         r"(?:casing|case)\s+(?:burst|bursts|rupture)", r"\bcato\b", r"blow\s+(?:up|apart)"),
    # flights
    "apogee": (r"apogee", r"how\s+high", r"max(?:imum)?\s+altitude", r"peak\s+altitude", r"how\s+far\s+up",
               r"what\s+altitude", r"top\s+(?:out|speed)"),
    "range": (r"how\s+far", r"\brange\b", r"how\s+long\s+(?:until|before)\s+it\s+lands", r"where\s+(?:does|will)\s+it\s+land"),
    # tests on a part
    "drop test": (r"drop[\s-]+tests?", r"drop(?:ped|s|ping)?\s+(?:it|my|the|this|a|an)\b", r"drop(?:ped)?\s+(?:from|off|onto)",
                  r"falls?\s+(?:off|from|onto)", r"knock(?:ed)?\s+off", r"survive\s+(?:a|the)\s+(?:fall|drop)"),
    "shake test": (r"shake[\s-]+tests?", r"shaker(?:\s+table)?", r"vib(?:e|ration)[\s-]+tests?", r"shak(?:e|es|en|ing)\s+(?:it|apart|loose)",
                   r"rattle"),
    "bend test": (r"bend(?:ing)?[\s-]+tests?", r"three[\s-]+point\s+bend(?:ing)?", r"bend(?:s|ing)?\s+under",
                  r"how\s+much\s+(?:does\s+it|will\s+it)\s+(?:bend|flex|sag|deflect)", r"\bsag(?:s|ging)?\b"),
    "pull test": (r"pull[\s-]+tests?", r"tensile\s+tests?", r"pull(?:ed|ing)?\s+(?:on|off|apart)", r"yank(?:ed)?",
                  r"rip\s+(?:off|out)", r"pull[\s-]+out"),
    "load test": (r"load[\s-]+tests?", r"proof[\s-]+tests?", r"(?:will|does|can)\s+it\s+(?:hold|carry|support)",
                  r"strong\s+enough", r"how\s+much\s+(?:weight|load)"),
    "heat soak": (r"heat[\s-]+soak(?:ed|ing)?", r"thermal[\s-]+soak", r"how\s+(?:hot|warm)", r"get(?:s)?\s+(?:hot|warm)",
                  r"run(?:s)?\s+hot", r"over[\s-]?heat(?:s|ing)?", r"stay\s+(?:cool|under)"),
    "tip over": (r"tip(?:s|ping)?\s+over", r"will\s+it\s+tip", r"top(?:ple|ples|pling)", r"fall(?:s)?\s+over",
                 r"knock(?:ed)?\s+over", r"tip(?:ping)?\s+point"),
    "fry": (r"\bfry\b", r"burn\s+out", r"blow\s+(?:up|out)", r"go\s+up\s+in\s+smoke", r"let\s+(?:the\s+)?smoke\s+out",
            r"\bpop\b"),
}


def phrases(*names: str) -> tuple[str, ...]:
    """The regexes for the plain names given, ready to add to an analysis's keywords or a domain's words."""
    out = []
    for n in names:
        for rx in SYNONYMS[n]:
            out.append(rx if rx.startswith(r"\b") else rf"\b{rx}\b")
    return tuple(out)


def weight(matched: str) -> float:
    """How much a match counts: one for a word, more for each word a phrase spans past the first, so a two word
    phrase beats two loose words and a three word phrase beats a two word one."""
    words = len(re.findall(r"[a-z0-9*]+", matched.lower()))
    if words == 0 or len(matched.strip()) < 2:
        return 0.25
    return 1.0 + 1.5 * (words - 1)


def best(patterns, text: str) -> tuple[float, str, tuple[int, int]]:
    """The heaviest match of any pattern in text: its weight, the words and where they sit."""
    top = (0.0, "", (0, 0))
    if not text:
        return top
    for rx in patterns:
        for m in re.finditer(rx, text, re.I):
            w = weight(m.group(0))
            if w > top[0]:
                top = (w, m.group(0), m.span())
    return top
