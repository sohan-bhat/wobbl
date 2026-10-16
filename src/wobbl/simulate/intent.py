"""What a free-form request means: the everyday phrases that name each physics, and how strongly a phrase counts.

People rarely say "internal ballistics". They say "a thrust test", "a static fire" or "put it on the test stand".
SYNONYMS holds those phrases once, grouped under one plain name, and an analysis or a domain takes them into its
keywords with phrases("static fire", "thrust curve"), so the same table feeds the domain's words, the kind of
thing and the physics. Matches are weighed by how many words they span, so "thrust test" outweighs "rocket" and a
named test outweighs a single word that happens to sit inside it.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

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
    "chamber pressure": (r"chamber[\s-]+pressures?", r"(?:motor|casing|case)\s+pressures?", r"(?-i:\bKn\b)",
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


# words that carry no meaning of their own, so "how far" counts as one word and "thrust test" as two
STOP = {"how", "does", "do", "did", "will", "would", "can", "could", "it", "its", "the", "a", "an", "my", "this",
        "that", "of", "on", "in", "to", "for", "is", "are", "be", "get", "gets", "what", "much", "many", "from",
        "onto", "with", "as", "or", "and"}


def weight(matched: str) -> float:
    """How much a match counts: one for a word, more for each meaningful word a phrase spans past the first, so a
    two word phrase beats two loose words and a three word phrase beats a two word one."""
    words = len([w for w in re.findall(r"[a-z0-9*]+", matched.lower()) if w not in STOP]) or \
        min(1, len(re.findall(r"[a-z0-9*]+", matched.lower())))
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


# ---------------------------------------------------------------- what the user wants back, and where
FORMATS = (("mp4", r"(?:an?\s+)?(?:3d\s+)?(?:mp4|mpeg-?4)(?:\s+(?:video|file|clip|movie))?"),
           ("mp4", r"(?:an?\s+)?(?:3d\s+|short\s+|quick\s+|little\s+)?(?:video|movie|video\s+clip)s?"),
           ("gif", r"(?:an?\s+)?(?:3d\s+|animated\s+|looping\s+|little\s+)?gifs?"),
           ("gif", r"(?:an?\s+)?(?:3d\s+)?animation"),
           ("png", r"(?:an?\s+)?(?:3d\s+)?(?:png|picture|image|screenshot|snapshot|photo|jpe?g)s?"),
           ("report", r"(?:an?|the|full)\s+(?:pdf\s+|html\s+)?report|pdf"))
FOLDER = (r"(?:in|into|to|on|onto|at|under)\s+(?:my\s+|the\s+)?(downloads?|desktop|documents|pictures|videos|movies)"
          r"(?:\s+folder)?")
PATH = r"(?:save|put|write|export|store|drop)?\s*(?:it\s+|that\s+|them\s+)?(?:to|in|into|at|as)\s+" \
       r"((?:~|\.{1,2}/|/|[a-z]:\\)\S*|\S+\.(?:gif|mp4|png|html))"
FILLER = (r"\b(?:can|could|would|will)\s+you\b", r"\bplease\b",
          r"\b(?:give|show|make|send|get|create|generate|render|produce|export|save|put)\s+(?:me|us)\b",
          r"^\s*(?:give|show|make|send|create|generate|render|produce|export|save)\b",
          r"\b(?:save|export|put|store)\s+(?:it|that|them)\b",
          r"\b(?:a|an|the)\s+(?:3d\s+)?sim(?:ulation)?s?\s+(?:of|for)\b", r"^\s*simulat(?:e|ion\s+of)\b",
          r"^\s*(?:of|showing)\s+(?=(?:a|an|the|this|that|my|how|what|will)\b)")


@dataclass
class Output:
    """A request with the words about the result taken out: what is left to simulate, the file it asks for (gif,
    mp4, png or report, or empty), where it should go (a folder named in words or a path, or empty) and the words
    that said so."""
    text: str
    format: str = ""
    out: str = ""
    said: list[str] = field(default_factory=list)


def read_output(text: str) -> Output:
    """What the user asks for in the result and where it goes, as 'in downloads give me a 3d gif of ...' or
    'give me a png on my desktop', with those words removed so they never steer the physics."""
    t = " " + (text or "") + " "
    said, fmt, out = [], "", ""

    def cut(m, group=0):
        nonlocal t
        said.append(m.group(group).strip())
        t = t[:m.start()] + " " + t[m.end():]

    m = re.search(PATH, t, re.I)
    if m and re.search(r"\.(?:gif|mp4|png|html)$|^(?:~|\.{1,2}/|/|[a-z]:\\)", m.group(1), re.I):
        out = m.group(1)
        cut(m)
        if re.search(r"\.(?:gif|mp4|png|html)$", out, re.I):
            fmt = {".gif": "gif", ".mp4": "mp4", ".png": "png", ".html": "report"}[out[out.rfind("."):].lower()]
    hits = sorted((m.start(), m.end(), name) for name, rx in FORMATS
                  for m in re.finditer(rf"\b(?:as\s+|into\s+)?{rx}\b", t, re.I))
    if hits:
        fmt = fmt or hits[0][2]
        for s0, s1, _ in reversed(hits):
            said.insert(0, t[s0:s1].strip())
            t = t[:s0] + " " + t[s1:]
    m = None if out else re.search(rf"\b{FOLDER}\b", t, re.I)
    if m:
        word = m.group(1).lower()
        out = "downloads" if word.startswith("download") else "videos" if word == "movies" else word
        cut(m)
    if not (fmt or out):
        return Output((text or "").strip())
    for rx in FILLER:
        t = re.sub(rx, " ", t, flags=re.I)
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"\s+([,.;:?!])", r"\1", t).strip()
    t = re.sub(r"^(?:[,;:.]|and|then|of|for|with|about|on|in)\s+", "", t, flags=re.I)
    tail = r"(?:,|;|:|\b(?:and|then|save|export|put|send|give|show|make|me|it|please))\s*$"
    while re.search(tail, t, re.I):
        t = re.sub(tail, "", t, flags=re.I).strip()
    return Output(t, fmt, out, said)
