# -*- coding: utf-8 -*-
r"""Tests lattice math

"""
import numpy as np
import pytest
from neutronpy.crystal import Lattice

uc_cubic = Lattice(4, 4, 4, 90, 90, 90)
uc_ortho = Lattice(4, 5, 6, 90, 90, 90)
uc_hex = Lattice(4, 4, 5, 90, 90, 120)
uc_PG = Lattice(2.461, 2.461, 6.708, 90, 90, 120)

def test_constructor():
    """Test construction of the lattice"""

    unitcell = Lattice(4,5,6,90,91,92)
    assert unitcell.a == 4
    assert unitcell.b == 5
    assert unitcell.c == 6
    assert unitcell.alpha == 90
    assert unitcell.beta == 91
    assert unitcell.gamma == 92

    abg = np.array([uc_cubic.alpha, uc_cubic.beta, uc_cubic.gamma])
    abc = np.array([uc_cubic.a, uc_cubic.b, uc_cubic.c])

    assert (np.all(uc_cubic.abc == abc))
    assert (np.all(uc_cubic.abg == abg))
    assert (np.round(uc_cubic.volume, 12) == 4 ** 3)
    assert (np.round(uc_cubic.reciprocal_volume, 12) == np.round(8 * np.pi ** 3 / (4 ** 3), 12))
    assert (np.all(np.round(uc_cubic.Bmatrix * uc_cubic.Bmatrix.T, 12) == np.round(uc_cubic.Gstar, 12)))

    A = [[uc_ortho.a,0,0],[0,uc_ortho.b,0],[0,0,uc_ortho.c]]
    assert np.allclose( uc_ortho.Amatrix, A)

def test_get_angle_between_planes():
    """Tests get angle between planes defined by two vectors
    """
    assert np.allclose(uc_cubic.get_angle_between_planes([1,0,0], [1,1,1]), np.radians(54.73561031724535))
    assert np.allclose(uc_cubic.get_angle_between_planes([1,0,0], [0,1,0]), np.pi/2)
    assert np.allclose(uc_cubic.get_angle_between_planes([1,0,0], [1,1,0]), np.pi/4)
    assert np.allclose(uc_cubic.get_angle_between_planes([1,0,0], [1,0,0]), 0)

    assert np.allclose(uc_hex.get_angle_between_planes([1,0,0], [0,1,0]), np.pi/3)
    assert np.allclose(uc_hex.get_angle_between_planes([1,0,0], [0,0,1]), np.pi/2)

    uc = Lattice(4,4,5, 90,90,120)
    hkl_totest =  [[0,0,1], [0,1,0], [1,0,0], [1,1,0]]
    assert np.allclose(uc.get_angle_between_planes([1,0,0], hkl_totest), 
                       [np.pi/2, np.pi/3, 0, np.pi/6])

    uc.gamma = 90
    hkl_totest =  [[0,0,1], [0,1,0], [1,0,0], [1,1,0]]
    assert np.allclose(uc.get_angle_between_planes([1,0,0], hkl_totest), 
                       [np.pi/2, np.pi/2, 0, np.pi/4])

def test_get_dspacing():
    """Tests d-spacing for given HKL
    """
    assert np.allclose(uc_cubic.get_dspacing([1, 1, 1]), uc_cubic.a / np.sqrt(3))

    # PG graphite
    assert np.allclose(uc_PG.get_dspacing([0,0,2]), 3.354)

def test_get_q():
    """Tests q for given HKL
    """
    assert np.allclose(uc_cubic.get_Q([1, 1, 1]), 2 * np.pi / uc_cubic.b * np.sqrt(3))


def test_get_tth():
    """Tests 2theta for given HKL
    """
    assert np.allclose(uc_cubic.get_tth([1, 1, 1], 2), np.radians(51.3178125465))


def test_metric():
    """Test A,B,G,Gstar matrices. Compared with Takin"""

    lattice = Lattice(4,5,6, 90, 90, 120)

    A = [ [4, -2.5, 0], [0, 4.330127, 0], [0, 0, 6]]
    B = [ [1.570796, 0, 0], [0.9068997, 1.451039, 0], [0, 0, 1.047198]]
    G = [[16,-10,0],[-10,25,0],[0,0,36]]
    Gstar = [[3.289868,1.315947,0],[1.315947,2.105516,0],[0,0,1.096623]]
    assert np.allclose( lattice.Amatrix , A)
    assert np.allclose( lattice.Bmatrix , B)
    assert np.allclose( lattice.G , G)
    assert np.allclose( lattice.Gstar , Gstar)

def test_setters():
    """Test that gettters/setters work properly
    """

    uc = Lattice(4,4,4, 90,90,90)
    assert uc.lattice_type == 'cubic'

    # Tetragonal
    uc.c = 7
    assert uc.lattice_type == 'tetragonal'
    A = [[4,0,0],[0,4,0],[0,0,7]]
    assert np.allclose( uc.Amatrix, A)

    assert np.allclose( uc.get_dspacing([1,0,0]), 4 )
    assert np.allclose( uc.get_dspacing([1,1,0]), 4*np.sqrt(2)/2 )
    assert np.allclose( uc.get_dspacing([0,0,1]), 7 )

    hkl_totest =  [[0,0,1], [0,1,0], [1,1,0]]
    assert np.allclose(uc.get_angle_between_planes([1,0,0], hkl_totest), 
                       [np.pi/2, np.pi/2, np.pi/4])

    # Hexagonal
    uc.gamma = 120
    assert uc.lattice_type == 'hexagonal'
    cg, sg = np.cos(2*np.pi/3), np.sin(2*np.pi/3)
    A = [[4,4*cg,0],[0,4*sg,0],[0,0,7]]
    assert np.allclose( uc.Amatrix, A)

    assert np.allclose( uc.get_dspacing([1,0,0]), 4*sg )
    assert np.allclose( uc.get_dspacing([1,1,0]), 4*np.abs(cg) )
    assert np.allclose( uc.get_dspacing([0,0,1]), 7 )

    hkl_totest =  [[0,0,1], [0,1,0], [1,1,0]]
    assert np.allclose(uc.get_angle_between_planes([1,0,0], hkl_totest), 
                       [np.pi/2, np.pi/3, np.pi/6])


def test_lattice_type():
    """Test lattice type determination
    """
    test_cell = uc_cubic
    assert (test_cell.lattice_type == 'cubic')

    test_cell = Lattice(1, 1, 2, 90, 90, 90)
    assert (test_cell.lattice_type == 'tetragonal')

    test_cell = Lattice(1, 2, 3, 90, 90, 90)
    assert (test_cell.lattice_type == 'orthorhombic')

    test_cell = Lattice(1, 2, 3, 90, 89, 90)
    assert (test_cell.lattice_type == 'monoclinic')

    test_cell = Lattice(1, 1, 1, 39, 39, 39)
    assert (test_cell.lattice_type == 'rhombohedral')

    test_cell = Lattice(1, 1, 1, 39, 39, 39)
    assert (test_cell.lattice_type == 'rhombohedral')

    test_cell = Lattice(1, 1, 2, 90, 90, 120)
    assert (test_cell.lattice_type == 'hexagonal')

    test_cell = Lattice(1, 2, 3, 30, 60, 120)
    assert (test_cell.lattice_type == 'triclinic')

    test_cell = Lattice(1, 1, 2, 90, 90, 150)
    with pytest.raises(ValueError):
        getattr(test_cell, 'lattice_type')


def test_goniometer_constants():
    """Test constants
    """
    pass


if __name__ == "__main__":
    pytest.main()
