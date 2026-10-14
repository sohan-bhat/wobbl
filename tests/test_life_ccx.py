"""The CalculiX steps for buckling, modes and harmonic response, and the readers of what CalculiX prints."""
import numpy as np
import pytest

from wobbl.life import ccx

BUCKLE_DAT = """
                        S T E P       1


     B U C K L I N G   F A C T O R   O U T P U T

 MODE NO       BUCKLING
                FACTOR

      1   0.1186663E+02
      2   0.4738194E+02
      3   0.1065525E+03
"""

FREQ_DAT = """
                        S T E P       1


     E I G E N V A L U E   O U T P U T

 MODE NO    EIGENVALUE                       FREQUENCY
                                     REAL PART            IMAGINARY PART
                           (RAD/TIME)      (CYCLES/TIME     (RAD/TIME)

      1   0.1171043E+06   0.3422050E+03   0.5446362E+02   0.0000000E+00
      2   0.4668192E+06   0.6832417E+03   0.1087413E+03   0.0000000E+00

     P A R T I C I P A T I O N   F A C T O R S

MODE NO.   X-COMPONENT     Y-COMPONENT     Z-COMPONENT     X-ROTATION      Y-ROTATION      Z-ROTATION

      1  -0.1229189E-08   0.1866889E-06  -0.1019022E-01  -0.6114191E-01   0.2221925E+01   0.3784508E-04
      2  -0.2605742E-08  -0.1019225E-01  -0.1090798E-06   0.3057599E-01   0.3352723E-04  -0.2222432E+01

     E F F E C T I V E   M O D A L   M A S S

MODE NO.   X-COMPONENT     Y-COMPONENT     Z-COMPONENT     X-ROTATION      Y-ROTATION      Z-ROTATION

      1   0.1510906E-17   0.3485273E-13   0.1038406E-03   0.3738333E-02   0.4936949E+01   0.1432250E-08
      2   0.6789893E-17   0.1038820E-03   0.1189841E-13   0.9348912E-03   0.1124075E-08   0.4939204E+01
TOTAL     0.4405786E-14   0.1359139E-03   0.1523844E-03   0.6709796E-02   0.5084066E+01   0.5065459E+01

     T O T A L   E F F E C T I V E   M A S S

           X-COMPONENT     Y-COMPONENT     Z-COMPONENT     X-ROTATION      Y-ROTATION      Z-ROTATION

          0.1692063E-03   0.1692063E-03   0.1692063E-03   0.1015222E-01   0.5088830E+01   0.5094922E+01


     F R A C T I O N   O F   T O T A L S

           X-COMPONENT     Y-COMPONENT     Z-COMPONENT     X-ROTATION      Y-ROTATION      Z-ROTATION

          0.2603796E-10   0.8032438E+00   0.9005833E+00   0.6609192E+00   0.9990638E+00   0.9942173E+00
"""

PRINT_DAT = """
                        S T E P       1

 displacements (vx,vy,vz) for set NRESP and time  0.5000000E+02

      5156  0.100000E-01  0.000000E+00 -0.200000E+00

 displacements (vx,vy,vz) for set NRESP and time  0.5000000E+02

      5156  0.000000E+00  0.000000E+00  0.300000E+00

 displacements (vx,vy,vz) for set NRESP and time  0.6000000E+02

      5156  0.000000E+00  0.000000E+00  0.100000E+00

 displacements (vx,vy,vz) for set NRESP and time  0.6000000E+02

      5156  0.000000E+00  0.000000E+00  0.000000E+00
"""


def test_reads_buckling_factors(tmp_path):
    p = tmp_path / "b.dat"
    p.write_text(BUCKLE_DAT)
    assert ccx.read_buckling_factors(p) == pytest.approx([11.86663, 47.38194, 106.5525])


def test_reads_frequencies_participation_and_mass(tmp_path):
    p = tmp_path / "f.dat"
    p.write_text(FREQ_DAT)
    e = ccx.read_eigen(p)
    assert e["freq"] == pytest.approx([54.46362, 108.7413])
    assert e["participation"].shape == (2, 6) and e["participation"][0, 2] == pytest.approx(-0.01019022)
    assert e["effective_mass"][1, 1] == pytest.approx(1.038820e-4)
    assert e["total_mass"][:3] == pytest.approx([1.692063e-4] * 3)
    # an effective mass is the square of its participation factor for modes of unit modal mass
    assert e["participation"][0, 2] ** 2 == pytest.approx(e["effective_mass"][0, 2], rel=1e-5)


def test_reads_printed_harmonic_motion(tmp_path):
    p = tmp_path / "h.dat"
    p.write_text(PRINT_DAT)
    r = ccx.read_node_print(p)
    assert sorted(r) == [50.0, 60.0]
    nodes, re_, im_ = r[50.0]
    assert list(nodes) == [5155]
    assert re_[0] == pytest.approx([0.01, 0, -0.2]) and im_[0] == pytest.approx([0, 0, 0.3])


def test_steps_reuse_the_static_deck():
    deck = ccx.Deck("*HEADING\nmesh\n", "*BOUNDARY\nNFIX1,1,3\n", "*CLOAD\n7,3,-1.\n", 10)
    b = ccx.buckle_inp(deck, 4)
    assert "*BUCKLE\n4\n" in b and "*CLOAD\n7,3,-1." in b and "NFIX1,1,3" in b
    f = ccx.frequency_inp(deck, 12)
    assert "*FREQUENCY, STORAGE=YES\n12\n" in f and "*CLOAD" not in f
    h = ccx.harmonic_inp(deck, "*CLOAD\n7,3,1.\n", [(10, 100, 20, 3.0), (50, 50.00005, 2, 1.0)], 0.02, 12,
                         np.array([6]))
    steps = h.split("*STEP")[1:]
    assert len(steps) == 2 and all("*MODAL DAMPING\n1,12,0.02" in s for s in steps)
    # only the last step writes the whole field; both print the response node
    assert "*EL FILE" not in steps[0] and "*EL FILE\nS" in steps[1]
    assert all("*NODE PRINT, NSET=NRESP" in s for s in steps) and "*NSET, NSET=NRESP\n7\n" in h


def test_body_load_includes_attached_masses():
    deck = ccx.Deck("", "", "", 10, mass_nodes={"motor": np.array([11])}, mass_values={"motor": np.array([0.05e-3])})
    s = ccx.body_load(deck, 9806.65, [0, 0, 2])
    assert "EALL,GRAV,9806.65,0,0,1" in s
    assert "12,3,4.903325" in s     # 50 g times one g, on the attachment's reference node


def test_blocks_and_element_numbers():
    step = "*STEP\n*STATIC\n*BOUNDARY\nA,1,1\n** note\nA,2,2\n*CLOAD\n3,1,1.\n*END STEP\n"
    blocks = ccx._blocks(step)
    assert blocks[2] == "*BOUNDARY\nA,1,1\n** note\nA,2,2\n" and blocks[3].startswith("*CLOAD")
    head = "*ELEMENT, TYPE=C3D10, ELSET=EALL\n1,1,2\n41,3,4\n*MATERIAL, NAME=MAT\n*ELEMENT, TYPE=MASS, ELSET=X\n57,9\n"
    assert ccx._max_element(head) == 57
