# -*- coding: utf-8 -*-
r"""Tests sample

"""
import numpy as np
import pytest
from neutronpy.crystal import Sample


def test_constructor():
    """Test construction of the lattice"""

    sample = Sample.make_default()

    assert np.allclose( sample.orient1, [1,0,0])
    assert np.allclose( sample.orient2, [0,1,0])

    expected_U = [
        [1,0,0],
        [0,1,0],
        [0,0,1]
    ]
    assert np.allclose( sample.Umatrix, expected_U)

    test_hkl = [[1,0,0], [1,1,0], [0,1,0]]
    assert np.allclose( sample.get_angle_to_o1(test_hkl),
                        [0, -np.pi/4, -np.pi/2])
    
    assert sample.is_in_scattering_plane([0,0,1]) == False

def test_lattice_setters():
    """Test setters of the lattice"""

    # Start with cubic, switch to hexagonal
    sample = Sample.make_default()
    
    assert sample.lattice_type == 'cubic'
    sample.c = 10
    sample.gamma = 120

    assert sample.lattice_type == 'hexagonal'
    assert np.allclose( sample.orient1, [1,0,0])
    assert np.allclose( sample.orient2, [0,1,0])

    astar = 1/sample.a * 4*np.pi/3 * np.sqrt(3)
    expected_UB = [
        [astar, astar*np.cos(np.pi/3),0],
        [0,     astar*np.sin(np.pi/3),0],
        [0,     0,  2*np.pi/sample.c]
    ]
    assert np.allclose( sample.UBmatrix, expected_UB)

    test_hkl = [[1,0,0], [1,1,0], [0,1,0]]
    assert np.allclose( sample.get_angle_to_o1(test_hkl),
                        [0, -np.pi/6, -np.pi/3])
    
    assert sample.is_in_scattering_plane([0,0,1]) == False

def test_orientation_setters():
    """Test setters of the orientation"""

    # Start with cubic, switch to hexagonal
    sample = Sample.make_default()
    sample.orient1=[1,0,0]
    sample.orient2=[0,1,0]

    assert np.allclose( sample.orient1, [1,0,0])
    assert np.allclose( sample.orient2, [0,1,0])
    assert sample.is_in_scattering_plane([1,-1,0]) == True
    assert sample.is_in_scattering_plane([0,0,1]) == False

    sample.orient1 = [1,1,0]
    sample.orient2 = [0,0,1]
    assert sample.is_in_scattering_plane([1,-1,0]) == False
    assert sample.is_in_scattering_plane([0,0,1]) == True

def test_scattering_plane():
    '''Test `is_in_scattering_plane` function'''

    sample = Sample.make_default()
    sample.orient1=[1,0,0]
    sample.orient2=[0,1,0]
    sample.gamma = 120

    test_hkl = [[1,0,0], [1,1,0], [0,1,0]]
    assert np.allclose( sample.get_angle_to_o1(test_hkl),
                        [0, -np.pi/6, -np.pi/3])

if __name__ == "__main__":
    pytest.main()