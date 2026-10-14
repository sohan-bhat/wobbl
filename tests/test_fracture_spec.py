"""Reading crack descriptions, with no solver."""
import pytest

from wobbl.fracture.spec import parse_crack
from wobbl.structural.spec import SpecError


@pytest.mark.parametrize("text,shape,a,c,where", [
    ("through 1.27 mm at hole d=6.4", "through", 1.27, None, "hole d=6.4"),
    ("corner 0.05 in at hole d=8", "corner", 1.27, 1.27, "hole d=8"),
    ("surface a=1 c=2 at hottest", "surface", 1.0, 2.0, "hottest"),
    ("surface 1 mm a/c=0.5 at 30,0,10", "surface", 1.0, 2.0, "30,0,10"),
    ("center 2a=20 at 50,75,2 normal y along x", "center", 10.0, None, "50,75,2"),
    ("edge 2", "through", 2.0, None, "hottest"),
])
def test_crack_shapes_and_sizes(text, shape, a, c, where):
    cs = parse_crack(text)
    assert cs.shape == shape and cs.a == pytest.approx(a) and cs.where == where
    assert (cs.c is None and c is None) or cs.c == pytest.approx(c)


def test_crack_directions():
    cs = parse_crack("through 2 mm at 0,75,2 toward +x normal y")
    assert cs.toward == [1.0, 0.0, 0.0] and cs.normal == [0.0, 1.0, 0.0] and cs.where == "0,75,2"


@pytest.mark.parametrize("bad", ["crack 2 mm", "through at hole d=8", "surface a=2 c=1", "through 0 mm"])
def test_bad_cracks_say_what_is_wrong(bad):
    with pytest.raises(SpecError):
        parse_crack(bad)


def test_placement_helpers():
    import numpy as np

    from wobbl.fracture.place import principal_direction, ray_hits

    sigma, n = principal_direction([10.0, 80.0, 0.0, 0.0, 0.0, 0.0])
    assert sigma == 80.0 and abs(n[1]) == 1.0
    # a ray through a unit cube made of two triangles per face crosses it once going out
    X = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], float)
    tris = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4], [2, 3, 7], [2, 7, 6],
                     [1, 2, 6], [1, 6, 5], [0, 4, 7], [0, 7, 3]])
    hits = ray_hits(np.array([0.3, 0.6, 0.25]), np.array([0, 0, 1.0]), X, tris)
    assert hits == pytest.approx([0.75])


@pytest.mark.parametrize("text,where", [("auto", "hottest"), ("", "hottest"), ("auto at 1,2,3", "1,2,3"),
                                        ("at hole d=8", "hole d=8")])
def test_automatic_cracks(text, where):
    cs = parse_crack(text)
    assert cs.shape == "auto" and cs.where == where and cs.a == 0
    with pytest.raises(SpecError):
        parse_crack("auto 2 mm")
