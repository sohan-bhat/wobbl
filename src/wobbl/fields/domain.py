"""Electric, magnetic, radio and sound fields for wobbl simulate.

On any CAD file (the part domain) the analyses solve the user's own geometry in 3D with GetDP: each body gets a
material from --assign lines or from its name. On a printed circuit board (a KiCad .kicad_pcb or Gerber files)
the board's own traces and planes give the parasitics of every net, which feed the circuit run of that board.
A device described in words (the field domain) runs one of the parametric studies, with its sizes read from
the question.
"""
from __future__ import annotations

import re
import shlex
from pathlib import Path

from ..simulate import Analysis, Domain, Option
from ..simulate import register as _register
from ..structural.domain import PART_KINDS


def register(item):
    """Every field analysis reads all its inputs and records its guesses before it calls job.ready(), so
    'wobbl simulate --plan' can show them without solving."""
    if isinstance(item, Analysis):
        item.dry = True
    return _register(item)

QUALITY = Option("quality", "fast, normal or fine", "normal")
ASSIGN = Option("assign", "What a body is, e.g. 'plunger: steel 1010', 'coil: coil 800 turns 0.6 A', "
                          "'magnet: n42 +z', 'electrode: conductor 5 kV', 'spacer: ptfe', 'box: rigid'", multiple=True)


def _quality(job) -> str:
    return "fast" if job.quick else (job.value("quality") or "normal")


# --- CAD geometry -------------------------------------------------------------------------------
def _scene(job):
    from .scene import load

    if job.path is None:
        raise ValueError(f"file not found: {job.thing}")
    return load([str(job.path)], job.out / "work" / "scene")


def _lines(job, sc, guess) -> list[str]:
    """--assign lines, or a reading of the body names and the question with every guess recorded."""
    lines = job.values("assign")
    if lines:
        return lines
    lines = guess(sc, job.focus)
    if not lines:
        raise ValueError(f"say what each body is with --assign, for example --assign '{sc.bodies[0].name}: steel 1010'. "
                         f"The file has {sc.listing()}")
    job.assume("materials from the body names and the question: " + "; ".join(lines),
               " ".join(f"--assign {shlex.quote(x)}" for x in lines))
    job.use("assign", lines)
    return lines


def _num(q: str, rx: str) -> float | None:
    m = re.search(rx, q, re.I)
    return float(m.group(1)) if m else None


def guess_magnetic(sc, question: str) -> list[str]:
    q = question.lower()
    turns = _num(q, r"(\d+(?:\.\d+)?)\s*turns?")
    amps = _num(q, r"(\d+(?:\.\d+)?)\s*a\b")
    grade = (re.search(r"\b(n\d\d|ferrite|ceramic)\b", q) or [None, "n42"])[1]
    out = []
    for b in sc.bodies:
        n = b.name.lower()
        if re.search(r"coil|winding|bobbin", n):
            out.append(f"{b.name}: coil {turns or 100:g} turns {amps or 1:g} A")
        elif re.search(r"magnet|ndfeb|neo", n):
            ext = [b.bbox[i + 3] - b.bbox[i] for i in range(3)]
            out.append(f"{b.name}: {grade} +{'xyz'[ext.index(min(ext))]}")
        elif re.search(r"steel|iron|core|plunger|armature|yoke|plate|frame|pole|keeper|shield", n):
            out.append(f"{b.name}: steel 1010")
    return out


def guess_electric(sc, question: str) -> list[str]:
    q = question.lower()
    m = re.search(r"(\d+(?:\.\d+)?)\s*(kv|v)\b", q)
    volts = f"{m.group(1)} {m.group(2)}" if m else "1 kV"
    out = []
    for b in sc.bodies:
        n = b.name.lower()
        if re.search(r"ground|gnd|base|earth|case|chassis|shield", n):
            out.append(f"{b.name}: ground")
        elif re.search(r"electrode|hv|terminal|conductor|bus|plate|wire|pin|rod|sphere|ball|tip", n):
            out.append(f"{b.name}: conductor {volts}")
        elif re.search(r"ptfe|teflon|spacer|insulator|bushing", n):
            out.append(f"{b.name}: ptfe")
        elif re.search(r"fr4|pcb|board", n):
            out.append(f"{b.name}: fr4")
    if not out and len(sc.bodies) == 2:
        out = [f"{sc.bodies[0].name}: conductor {volts}", f"{sc.bodies[1].name}: ground"]
    elif not out and len(sc.bodies) == 1:
        out = [f"{sc.bodies[0].name}: conductor {volts}"]
    return out


def guess_sound(sc, question: str) -> list[str]:
    if len(sc.bodies) == 1:
        b = sc.bodies[0]
        word = "rigid" if re.search(r"enclosure|box|cabinet|case|housing|speaker|muffler", b.name.lower() + " " +
                                    question.lower()) else "air"
        return [f"{b.name}: {word}"]
    return []


def magnetic_part(job):
    from . import assign, ms3d

    sc = _scene(job)
    a = assign.parse(sc, _lines(job, sc, guess_magnetic))
    bodies, notes = assign.magnetic(sc, a)
    targets = [b.name for t in job.values("force-on") for b in sc.select(t)]
    need = job.value("need")
    m = None if need else re.search(r"(?:pull|hold|force|lift|attract)\D{0,24}?(\d+(?:\.\d+)?)\s*n\b", job.focus, re.I)
    if m:
        need = f"{m.group(1)} N"
        job.use("need", need)
    from .params import value
    s = ms3d.MagSetup(bodies, targets, value({"n": need}, "n", 0, "force") if need else 0.0, notes)
    _situate(job, sc, {n: b.kind for n, b in bodies.items()})
    job.ready()
    o = ms3d.run(sc, s, job.out, job.question, _quality(job), log=job.log, title=Path(job.thing).name)
    _tidy(job)
    if s.need:
        job.criterion("force", "force_target_N", s.need, "min")
    return o


def electric_part(job):
    from . import assign, es3d

    sc = _scene(job)
    a = assign.parse(sc, _lines(job, sc, guess_electric))
    conds, diel, notes = assign.electric(sc, a)
    s = es3d.ElecSetup(conds, diel, job.number("margin", 2.0) or 2.0)
    _situate(job, sc, {**{n: "conductor" for n in conds}, **{n: "dielectric" for n in diel}})
    job.ready()
    o = es3d.run(sc, s, job.out, job.question, _quality(job), log=job.log, title=Path(job.thing).name)
    _tidy(job)
    o.assumptions += [n for n in notes if n not in o.assumptions]
    return o


def acoustic_part(job):
    from . import ac3d, assign

    sc = _scene(job)
    a = assign.parse(sc, _lines(job, sc, guess_sound))
    roles = assign.sound(sc, a)
    fmax = job.value("fmax")
    from .params import value
    _situate(job, sc, {**{n: "air" for n in roles.air}, **{n: "wall" for n in roles.walls},
                       **{n: "source" for n in roles.sources}})
    job.ready()
    o = ac3d.run(sc, roles, job.out, job.question, _quality(job),
                 fmax=value({"f": fmax}, "f", 0, "frequency") if fmax else 0.0, log=job.log,
                 title=Path(job.thing).name, listen=job.values("listen"))
    _tidy(job)
    return o


def _tidy(job) -> None:
    """The scene copy of the CAD in meters is only needed while solving."""
    import shutil
    shutil.rmtree(job.out / "work", ignore_errors=True)


def _situate(job, sc, roles: dict) -> None:
    job.situate(objects=[{"name": b.name, "role": roles.get(b.name, "air"),
                          "box_mm": [round(1e3 * x, 3) for x in b.bbox]} for b in sc.bodies])


register(Analysis(
    "part", "magnetic", "Magnetic forces, torque, inductance and saturation of magnets, coils and steel in any CAD",
    magnetic_part, kinds=PART_KINDS, covers=("stress",),       # "pulls 5 N" is a magnetic force, not a load
    keywords=(r"\bmagnet", r"\bsolenoid", r"\bcoils?\b", r"\bwinding", r"inductance", r"\bflux\b", r"\btesla\b",
              r"\bgauss\b", r"\bn(?:35|42|52)\b", r"\bplunger\b", r"\barmature\b", r"\bpull force\b"),
    aliases=("magnetics", "magnetostatic", "magnetostatics"),
    options=(ASSIGN, QUALITY, Option("force-on", "Body whose force and torque are wanted (default the moving steel)",
                                     multiple=True),
             Option("need", "Force wanted on that body, e.g. '8 N'")),
    example='wobbl simulate actuator.step "does the plunger pull 8 N?" --assign "coil: coil 800 turns 0.6 A" '
            '--assign "plunger: steel 1010" --assign "frame: steel 1010"'))

register(Analysis(
    "part", "electric", "Capacitance, peak field and breakdown margin between conductors in any CAD", electric_part,
    kinds=PART_KINDS,
    keywords=(r"\b\d+(?:\.\d+)?\s*kv\b", r"\bvolts?\b|\b\d+(?:\.\d+)?\s*v\b", r"capacitance", r"breakdown",
              r"\barc(?:s|ing)?\b", r"\bspark", r"\bcorona\b", r"electric field", r"dielectric", r"creepage",
              r"clearance"),
    aliases=("electrostatic", "electrostatics", "capacitance", "breakdown"),
    options=(ASSIGN, QUALITY, Option("margin", "Wanted factor below breakdown", "2")),
    example='wobbl simulate bushing.step "will 20 kV arc?" --assign "rod: conductor 20 kV" --assign "flange: ground"'))

register(Analysis(
    "part", "acoustic", "Standing wave modes of the air in or around any CAD, and the sound a vibrating surface makes",
    acoustic_part, kinds=PART_KINDS,
    keywords=(r"\bacoustic", r"\bsound\b", r"\bnoise\b", r"standing waves?", r"room modes?", r"\bcavity\b",
              r"\bdb\b|\bspl\b", r"helmholtz", r"\bspeaker\b", r"\bboom", r"\bhum\b"),
    aliases=("sound", "acoustics", "room-modes"),
    options=(ASSIGN, QUALITY, Option("fmax", "Highest frequency for modes, e.g. '500 Hz'"),
             Option("listen", "Listening point in mm for a vibrating source, e.g. '0,0,1000'", multiple=True)),
    example='wobbl simulate speaker_box.step "where does the box resonate below 500 Hz?" --assign "box: rigid"'))


# --- printed circuit boards ---------------------------------------------------------------------
def sniff_board(path: Path) -> str | None:
    suf = path.suffix.lower()
    if path.is_dir():
        names = [p.suffix.lower() for p in path.iterdir() if p.is_file()]
        if any(s == ".kicad_pcb" for s in names):
            return "kicad"
        if sum(s in GERBER for s in names) >= 2:
            return "gerber"
        return None
    if suf == ".zip":
        import zipfile
        try:
            with zipfile.ZipFile(path) as z:
                return "gerber" if sum(Path(n).suffix.lower() in GERBER for n in z.namelist()) >= 2 else None
        except (OSError, zipfile.BadZipFile):
            return None
    return "gerber" if suf in GERBER else None


GERBER = {".gbr", ".gtl", ".gbl", ".g1", ".g2", ".g3", ".g4", ".gto", ".gbo", ".gts", ".gbs", ".gm1", ".gko", ".gbrjob"}

register(Domain(
    "board", "A printed circuit board, as a KiCad .kicad_pcb file or Gerber files",
    extensions={".kicad_pcb": "kicad", **{s: "" for s in GERBER}, ".zip": ""}, sniff=sniff_board,
    words=(r"\bpcb\b", r"kicad", r"gerbers?"), default_kind="kicad", results=("board",), priority=3,
    options=(QUALITY,),
    example='wobbl simulate buck.kicad_pcb "how much does the layout ring the switch node?" --netlist buck.cir'))


def parasitics(job):
    from .pcbfield import run_parasitics

    if job.path is None:
        raise ValueError(f"file not found: {job.thing}")
    netlist = job.value("netlist")
    nets = job.values("net")
    return run_parasitics(str(job.path), job.out, job.focus, netlist=netlist, nets=nets,
                          quality=_quality(job), log=job.log, job=job)


register(Analysis(
    "board", "parasitics", "Trace impedance, inductance, capacitance and coupling of the board's own nets, fed into "
                           "its circuit", parasitics, default=True,
    keywords=(r"impedance", r"parasitic", r"inductance", r"capacitance", r"crosstalk", r"\bring(?:s|ing)?\b",
              r"overshoot", r"\btraces?\b", r"\bnets?\b", r"layout", r"signal integrity", r"\bemi\b"),
    aliases=("trace", "traces", "board", "signal-integrity", "si"),
    options=(Option("netlist", "SPICE netlist of the same board (net names as on the PCB), to run with and without "
                               "the parasitics"),
             Option("net", "Only these nets", multiple=True), QUALITY),
    legacy="wobbl board",
    example='wobbl simulate buck.kicad_pcb "how hard does the switch node ring?" --netlist buck.cir'))


# --- antennas in files ----------------------------------------------------------------------------
FREQUENCY = re.compile(r"(\d+(?:\.\d+)?)\s*([gmk]?hz)\b", re.I)
ANTENNA_WORDS = (r"\bantennas?\b", r"\bs11\b", r"return loss", r"\bvswr\b", r"\bradiat", r"\bgain\b", r"\bdipole\b",
                 r"\bmonopole\b", r"inverted[- ]?f", r"\bmeander", r"radiation pattern", r"\b\d+(?:\.\d+)?\s*[gm]hz\b")
ANTENNA_OPTS = (Option("frequency", "Working frequency, e.g. 2.4GHz (read from the question when it names one)"),
                Option("port", "The feed gap as two points in mm, x,y,z:x,y,z, when it is not the narrowest gap "
                               "between two conductors"),
                Option("budget", "Longest the solve may be predicted to take, e.g. 20min", "15 min"), QUALITY)


def _antenna_params(job, extra: tuple = ()) -> dict:
    f = job.value("frequency")
    if not f:
        m = FREQUENCY.search(job.focus)
        if not m:
            raise ValueError("say the antenna's working frequency, for example 'at 2.4 GHz' in the question or "
                             "--frequency 2.4GHz")
        f = f"{m.group(1)} {m.group(2)}"
        job.assume(f"the working frequency is {f}, read from the question", f"--frequency {m.group(1)}{m.group(2)}")
    params = {"frequency": f, "quality": _quality(job)}
    for k in ("port", "budget") + extra:
        v = job.values(k) if k == "dielectric" else job.value(k)
        if v:
            params[k] = v
    return params


def antenna_part(job):
    from .radiate import run_file

    if job.path is None:
        raise ValueError(f"file not found: {job.thing}")
    params = _antenna_params(job, ("dielectric",))
    job.ready()
    return run_file(job.path, params, job.out, job.question, log=job.log)


def antenna_board(job):
    from .radiate import run_file

    if job.path is None:
        raise ValueError(f"file not found: {job.thing}")
    params = _antenna_params(job, ("feed",))
    job.ready()
    return run_file(job.path, params, job.out, job.question, log=job.log)


register(Analysis(
    "part", "antenna", "Resonance, S11, impedance, gain and pattern of an antenna in any CAD, from a 3D full wave solve",
    antenna_part, kinds=PART_KINDS, keywords=ANTENNA_WORDS, aliases=("radiation", "s11", "rf"),
    options=ANTENNA_OPTS + (Option("dielectric", "A body that is not metal, as 'name: fr4' or 'name: er=3.5'",
                                   multiple=True),),
    example='wobbl simulate inverted_f.step "does it resonate at 2.4 GHz, and what is its S11?" --port 12,0,0:12,0.6,0'))

register(Analysis(
    "board", "antenna", "Resonance, S11, gain and pattern of a trace antenna on the board, from a 3D full wave solve",
    antenna_board, keywords=ANTENNA_WORDS, aliases=("radiation", "s11", "rf"), covers=("parasitics",),
    options=ANTENNA_OPTS + (Option("feed", "The net that is the antenna (default one named ANT, RF or FEED)"),),
    example='wobbl simulate ifa.kicad_pcb "does the trace antenna resonate at 2.4 GHz?" --feed ANT'))


# --- devices in words -------------------------------------------------------------------------
NUM = r"(\d+(?:\.\d+)?)"
READ = {
    "solenoid": [("turns", NUM + r"\s*turns?", ""), ("current", NUM + r"\s*(m?a)\b", "A"),
                 ("stroke", NUM + r"\s*mm\s*stroke|stroke\s*(?:of\s*)?" + NUM + r"\s*mm", "mm"),
                 ("need", r"(?:pull|force|hold)\w*\s*(?:of\s*)?" + NUM + r"\s*n\b|" + NUM + r"\s*n\b", "N"),
                 ("plunger_d", NUM + r"\s*mm\s*(?:diameter\s*)?plunger", "mm")],
    "electromagnet": [("turns", NUM + r"\s*turns?", ""), ("current", NUM + r"\s*(m?a)\b", "A"),
                      ("diameter", NUM + r"\s*mm\s*(?:diameter|across|wide)", "mm"),
                      ("need", NUM + r"\s*n\b", "N"), ("plate", NUM + r"\s*mm\s*(?:thick\s*)?plate", "mm")],
    "magnet": [("diameter", NUM + r"\s*mm\s*(?:diameter|across|wide|disc|disk)", "mm"),
               ("height", NUM + r"\s*mm\s*(?:thick|tall|high)", "mm"), ("grade", r"\b(n\d\d|ferrite)\b", ""),
               ("need", NUM + r"\s*n\b", "N"), ("plate", NUM + r"\s*mm\s*(?:thick\s*)?(?:steel\s*)?plate", "mm")],
    "plates": [("diameter", NUM + r"\s*mm\s*(?:across|diameter|wide)", "mm"),
               ("gap", NUM + r"\s*mm\s*gap|gap\s*(?:of\s*)?" + NUM + r"\s*mm", "mm"),
               ("voltage", NUM + r"\s*(k?v)\b", "V"), ("dielectric", r"\b(fr4|ptfe|teflon|oil|glass|polyimide)\b", "")],
    "coax": [("inner_d", NUM + r"\s*mm\s*inner", "mm"), ("outer_d", NUM + r"\s*mm\s*outer", "mm"),
             ("voltage", NUM + r"\s*(k?v)\b", "V")],
    "sphere": [("diameter", NUM + r"\s*mm\s*(?:sphere|ball|diameter)", "mm"),
               ("gap", NUM + r"\s*mm\s*gap|gap\s*(?:of\s*)?" + NUM + r"\s*mm", "mm"), ("voltage", NUM + r"\s*(k?v)\b", "V")],
    "microstrip": [("width", NUM + r"\s*mm\s*(?:wide\s*)?(?:microstrip|trace|track|line)", "mm"),
                   ("height", r"on\s*" + NUM + r"\s*mm", "mm"), ("length", NUM + r"\s*mm\s*long", "mm"),
                   ("dielectric", r"\b(fr4|ro4350b|rogers|ptfe|polyimide|alumina)\b", "")],
    "dipole": [("frequency", NUM + r"\s*([gm]hz)\b", ""), ("length", NUM + r"\s*mm\s*long", "mm"),
               ("diameter", NUM + r"\s*mm\s*(?:wire|diameter)", "mm")],
    "room": [("size", r"(\d+(?:\.\d+)?\s*x\s*\d+(?:\.\d+)?\s*x\s*\d+(?:\.\d+)?)\s*m\b", ""),
             ("fmax", r"(?:below|under|up to)\s*" + NUM + r"\s*hz", "Hz")],
}
READ["monopole"] = READ["patch"] = READ["dipole"]
READ["loop"] = [("frequency", NUM + r"\s*([gm]hz)\b", ""), ("circumference", NUM + r"\s*mm\s*circumference", "mm"),
                ("radius", NUM + r"\s*mm\s*radius", "mm"), ("diameter", NUM + r"\s*mm\s*(?:wire|diameter)", "mm")]
READ["invf"] = [("frequency", NUM + r"\s*([gm]hz)\b", ""),
                ("dielectric", r"\b(fr4|ro4350b|rogers|ptfe|polyimide|alumina)\b", "")]
READ["stripline"] = READ["coplanar"] = READ["coupled"] = READ["microstrip"]
READ["link"] = [("frequency", NUM + r"\s*([gm]hz)\b", ""), ("tx_power", r"(-?\d+(?:\.\d+)?)\s*dbm\b", "dBm"),
                ("distance", NUM + r"\s*m\b", "m")]
READ["enclosure"] = [("size", r"(\d+\s*x\s*\d+\s*x\s*\d+)\s*mm", "")]
READ["muffler"] = [("chamber_length", NUM + r"\s*mm\s*(?:long\s*)?chamber", "mm")]
READ["panel"] = [("diameter", NUM + r"\s*mm\s*(?:panel|diameter|across)", "mm"),
                 ("velocity", NUM + r"\s*m/s", "")]


def read_words(kind: str, question: str, thing: str = "") -> dict:
    """Template values the words give, with their units, as --set would give them."""
    text = f"{thing} {question}"
    out = {}
    for key, rx, unit in READ.get(kind, []):
        m = re.search(rx, text, re.I)
        if not m:
            continue
        groups = [g for g in m.groups() if g]
        val = groups[0]
        if key in ("current", "voltage", "frequency") and len(groups) > 1:
            val = f"{val} {groups[1]}"
        elif unit:
            val = f"{val} {unit}"
        out[key] = val
    return out


TEMPLATE_OF = {"magnetic": ("solenoid", "electromagnet", "magnet"), "electric": ("plates", "coax", "sphere"),
               "board": ("microstrip", "stripline", "coplanar", "coupled", "coax"),
               "antenna": ("dipole", "monopole", "patch", "loop", "invf", "link"), "sound": ("room", "enclosure", "muffler", "panel")}


def _template_entry(physics: str):
    def entry(job):
        from .tools import check_params, runner

        kinds = TEMPLATE_OF[physics]
        kind = job.value("template") or (job.kind if job.kind in kinds else kinds[0])
        params = read_words(kind, job.focus, job.thing if not Path(job.thing).exists() else "")
        for k, v in params.items():
            job.use(k, v)
        for s in job.values("set"):
            if "=" in s:
                k, v = s.split("=", 1)
                params[k.strip()] = v.strip()
        check_params(physics, kind, params)
        job.use("template", kind)
        job.ready()
        return runner(physics)(kind, params, job.out, job.question, log=job.log)
    entry.__name__ = f"{physics}_words"
    return entry


WORD_OPTS = (Option("set", "A template value with its unit, e.g. turns=800, stroke=6mm, frequency=2.4GHz", multiple=True),
             Option("template", "Which parametric device, when the words do not say"))

register(Analysis("field", "magnetic", "A solenoid, pot electromagnet or permanent magnet described by its sizes",
                  _template_entry("magnetic"), kinds=TEMPLATE_OF["magnetic"], options=WORD_OPTS, legacy="wobbl magnetic",
                  keywords=(r"\bsolenoids?\b", r"electromagnets?", r"\bmagnets?\b", r"\bplunger\b", r"\bpull\b"),
                  aliases=("magnetics",), default=True))
register(Analysis("field", "electric", "Capacitance and breakdown of plates, a coax or a sphere gap described by size",
                  _template_entry("electric"), kinds=TEMPLATE_OF["electric"], options=WORD_OPTS,
                  legacy="wobbl electric", aliases=("electrostatic",), default=True,
                  keywords=(r"capacitance", r"breakdown", r"\barc\b", r"\bkv\b", r"spark gap")))
register(Analysis("field", "trace", "Impedance and crosstalk of a microstrip, stripline, coplanar or coax line",
                  _template_entry("board"), kinds=TEMPLATE_OF["board"], options=WORD_OPTS, legacy="wobbl board",
                  aliases=("impedance", "board"), default=True,
                  keywords=(r"impedance", r"crosstalk", r"microstrip", r"stripline", r"coplanar")))
register(Analysis("field", "antenna", "Resonance, S11, gain and range of a dipole, monopole, patch, loop or trace antenna, or a link budget",
                  _template_entry("antenna"), kinds=TEMPLATE_OF["antenna"], options=WORD_OPTS, legacy="wobbl antenna",
                  aliases=("radio", "range"), default=True,
                  keywords=(r"\bantennas?\b", r"\bdipole\b", r"\bmonopole\b", r"\bpatch\b", r"\bloop\b", r"inverted[- ]?f",
                            r"\bmeander", r"\brange\b", r"s11")))
register(Analysis("field", "acoustic", "Room and enclosure modes, muffler loss and panel noise described by size",
                  _template_entry("sound"), kinds=TEMPLATE_OF["sound"], options=WORD_OPTS, legacy="wobbl sound",
                  aliases=("sound",), default=True,
                  keywords=(r"room modes?", r"\bmuffler", r"\bsound\b", r"\bnoise\b", r"standing wave")))
