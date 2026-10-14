"""The deck readers count big models in one streaming pass, follow includes and read the steps and times."""
from __future__ import annotations

import numpy as np
import pytest

from wobbl.decks import DeckError, read_deck

INP = """*HEADING
two steps
** a comment
*NODE, NSET=NALL
{nodes}*INCLUDE, INPUT=elements.inp
*NSET, NSET=FIX
1, 2
*MATERIAL, NAME=S
*ELASTIC
200000, 0.3
*STEP, NLGEOM
*STATIC
0.1, 1., 1e-5, 0.25
*BOUNDARY
FIX, 1, 3
*NODE FILE
U
*END STEP
*STEP
*HEAT TRANSFER, STEADY STATE
1., 1.
*END STEP
*STEP
*FREQUENCY
12
*END STEP
"""


def test_calculix_deck_with_include(tmp_path):
    nodes = "".join(f"{i + 1}, {i % 10}., {i // 10 % 10}., {i // 100}.\n" for i in range(1000))
    (tmp_path / "model.inp").write_text(INP.format(nodes=nodes))
    # a C3D20 record runs over two lines, ending the first with a comma
    els = "*ELEMENT, TYPE=C3D20, ELSET=EALL\n" + "".join(
        f"{e}, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,\n16, 17, 18, 19, 20\n" for e in range(1, 41))
    els += "*ELEMENT, TYPE=C3D10, ELSET=E2\n" + "".join(f"{e}, 1,2,3,4,5,6,7,8,9,10\n" for e in range(41, 61))
    (tmp_path / "elements.inp").write_text(els)
    d = read_deck(tmp_path / "model.inp")
    assert d.solver == "calculix" and d.nodes == 1000
    assert d.elements == {"C3D20": 40, "C3D10": 20}
    assert [s.procedure for s in d.steps] == ["static", "heat-steady", "frequency"]
    assert d.steps[0].nlgeom and d.steps[0].initial == 0.1 and d.steps[0].max_inc == 0.25
    assert d.steps[2].modes == 12
    assert d.bbox == ((0.0, 0.0, 0.0), (9.0, 9.0, 9.0))
    assert len(d.sample) == 1000 and len(d.includes) == 1
    assert d.bytes == (tmp_path / "model.inp").stat().st_size + (tmp_path / "elements.inp").stat().st_size


def test_sample_stays_small_and_spread(tmp_path):
    from wobbl.decks import _Sampler

    s = _Sampler(cap=100)
    for i in range(10000):
        s.add(float(i), 0.0, 0.0)
    pts = np.asarray(s.pts)
    assert 50 <= len(pts) < 100
    assert pts[:, 0].min() == 0 and pts[:, 0].max() > 9000


def test_lsdyna_keyword_deck(tmp_path):
    k = ["*KEYWORD", "*CONTROL_TERMINATION", "$#  endtim", "     0.120", "*CONTROL_TIMESTEP",
         "$#  dtinit    tssfac      isdo    tslimt     dt2ms", "       0.0       0.9         0       0.0  -1.0E-6",
         "*DATABASE_BINARY_D3PLOT", "     0.005", "*NODE"]
    k += [f"{i + 1:8d}{i * 1.0:16.4f}{0.0:16.4f}{0.0:16.4f}" for i in range(30)]
    k += ["*ELEMENT_SHELL_THICKNESS"]
    for e in range(1, 11):
        k += [f"{e:8d}       1       1       2       3       4", "       1.0       1.0       1.0       1.0"]
    k += ["*ELEMENT_SOLID"] + [f"{e:8d}       2       1       2       3       4       5       6       7       8"
                               for e in range(11, 16)]
    k += ["*INCLUDE", "more.k", "*END"]
    (tmp_path / "car.k").write_text("\n".join(k) + "\n")
    (tmp_path / "more.k").write_text("*ELEMENT_BEAM\n" + "".join(f"{e:8d}       3       1       2\n"
                                                                 for e in range(16, 20)))
    d = read_deck(tmp_path / "car.k")
    assert d.solver == "openradioss" and d.nodes == 30
    assert d.elements == {"SHELL": 10, "SOLID": 5, "BEAM": 4}
    assert d.end_time == pytest.approx(0.12) and d.time_step == pytest.approx(1e-6)
    assert d.output_interval == pytest.approx(0.005)


def test_radioss_starter_and_engine(tmp_path):
    starter = "#RADIOSS STARTER\n/NODE\n" + "".join(f"{i:10d}{float(i):20.3f}{0.0:20.3f}{0.0:20.3f}\n"
                                                    for i in range(1, 9))
    starter += "/SHELL/1\n" + "".join(f"{e:10d}{1:10d}{2:10d}{3:10d}{4:10d}\n" for e in range(1, 4))
    (tmp_path / "crash_0000.rad").write_text(starter)
    (tmp_path / "crash_0001.rad").write_text("/RUN/crash/1/\n0.08\n/ANIM/DT\n0. 0.002\n/DT/NODA/CST/0\n0.9 5e-7\n")
    d = read_deck(tmp_path / "crash_0000.rad")
    assert d.nodes == 8 and d.elements == {"SHELL": 3}
    assert d.end_time == pytest.approx(0.08) and d.output_interval == pytest.approx(0.002)
    assert d.time_step == pytest.approx(5e-7)


def test_openfoam_case(tmp_path):
    (tmp_path / "system").mkdir()
    (tmp_path / "system" / "controlDict").write_text(
        "application     simpleFoam;\nstartFrom       startTime;\nstartTime       0;\nendTime         2000;\n"
        "deltaT          1;\nwriteInterval   500;\n")
    (tmp_path / "system" / "blockMeshDict").write_text(
        "blocks\n(\n    hex (0 1 2 3 4 5 6 7) (40 20 10) simpleGrading (1 1 1)\n    hex (4 5 6 7 8 9 10 11) "
        "(10 10 10) simpleGrading (1 1 1)\n);\n")
    d = read_deck(tmp_path)
    assert d.solver == "openfoam" and d.elements == {"CELL": 9000}
    assert d.end_time == 2000 and d.output_interval == 500
    poly = tmp_path / "constant" / "polyMesh"
    poly.mkdir(parents=True)
    (poly / "owner").write_text('FoamFile\n{\n    note        "nPoints:123456  nCells:110000  nFaces:340000  '
                                'nInternalFaces:320000";\n}\n')
    assert read_deck(tmp_path).elements == {"CELL": 110000}


def test_unknown_and_missing(tmp_path):
    with pytest.raises(DeckError):
        read_deck(tmp_path / "part.step")
    (tmp_path / "a.inp").write_text("*NODE\n1,0,0,0\n*INCLUDE, INPUT=gone.inp\n")
    with pytest.raises(DeckError, match="missing"):
        read_deck(tmp_path / "a.inp")
