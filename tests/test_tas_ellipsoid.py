# -*- coding: utf-8 -*-
r"""Testing of the resolution library - TAS

"""
from copy import deepcopy

import numpy as np
import pytest

from neutronpy import TripleAxisSpectrometer

"""
TODO
1. Why is R0 not working?
2. Test with triclinic sample.
"""

# All tests done with:
sample_dict = dict(a=5,b=5,c=5, alpha=90,beta=90,gamma=90, orient1=[1,0,0], orient2=[0,1,0])
KF = 2.662
HKLE = np.transpose([[2,1,0,5]])

def test_EIGER_t270():
    t_R0 = 3.022173e-08 
    t_RM_ppzE = np.array([
        [    606.3195,   -985.9447,           0,    65.91468 ],
        [   -985.9447,    2917.573,           0,   -216.4979 ],
        [           0,           0,    70.04894,           0 ],
        [    65.91468,   -216.4979,           0,    26.25968 ],
    ])
    t_RM_hklE = np.array([
        [    2932.972,   -2394.079,           0,    195.7548 ],
        [   -2394.079,    2631.737,           0,   -206.2942 ],
        [           0,           0,    110.6169,           0 ],
        [    195.7548,   -206.2942,           0,    26.25968 ],
    ])

    taz_filename = r"C:\ProgramData\takin_270\instruments\EIGER_t270.taz"
    TAS = TripleAxisSpectrometer.from_taz_file(taz_filename)
    TAS.fixed_wavevector = KF


    # Test Q coordinate system
    Q = TAS.sample.get_Q(HKLE[:3])
    R0, RM, RV = TAS.calc_resolution_QE(Q, HKLE[3])
    assert np.allclose(t_RM_ppzE, RM[0])

    # Test hkl coordinate system
    R0, RM, RV = TAS.calc_resolution_HKLE(HKLE, base='hkl')
    assert np.allclose(t_RM_hklE, RM[0])

def test_in20():
    t_R0 = 5.589735e-11 
    t_RM_ppzE = np.array([
        [     12041.5,    2001.368,           0,   -425.2356 ],
        [    2001.368,     7566.89,           0,    -649.261 ],
        [           0,           0,     5835.35,           0 ],
        [   -425.2356,    -649.261,           0,    69.74816 ],
    ])
    t_RM_hklE = np.array([
        [    15073.62,    4722.669,           0,   -113.0771 ],
        [    4722.669,    15890.71,           0,   -968.7262 ],
        [           0,           0,    9214.816,           0 ],
        [   -113.0771,   -968.7262,           0,    69.74816 ],
    ])

    taz_filename = r"C:\ProgramData\takin_270\instruments\in20.taz"
    TAS = TripleAxisSpectrometer.from_taz_file(taz_filename)
    TAS.fixed_wavevector = KF

    # Test Q coordinate system
    Q = TAS.sample.get_Q(HKLE[:3])
    R0, RM, RV = TAS.calc_resolution_QE(Q, HKLE[3])
    print(RM[0])
    assert np.allclose(t_RM_ppzE, RM[0])

    # Test hkl coordinate system
    R0, RM, RV = TAS.calc_resolution_HKLE(HKLE, base='hkl')
    print(np.around(RM[0],5) )
    assert np.allclose(t_RM_hklE, RM[0])

def test_tax():
    t_R0 = 5.589735e-11 
    t_RM_ppzE = np.array([
        [    3872.356,    2470.685,           0,   -209.7517 ],
        [    2470.685,    12597.71,           0,   -1044.311 ],
        [           0,           0,    218.8214,           0 ],
        [   -209.7517,   -1044.311,           0,    89.89775 ],
    ])
    t_RM_hklE = np.array([
        [    5749.446,   -3170.482,           0,    351.1324 ],
        [   -3170.482,    20259.04,           0,   -1291.652 ],
        [           0,           0,    345.5488,           0 ],
        [    351.1324,   -1291.652,           0,    89.89775 ],
    ])

    taz_filename = r"C:\ProgramData\takin_270\instruments\tax.taz"
    TAS = TripleAxisSpectrometer.from_taz_file(taz_filename)
    TAS.fixed_wavevector = KF

    # Test Q coordinate system
    Q = TAS.sample.get_Q(HKLE[:3])
    R0, RM, RV = TAS.calc_resolution_QE(Q, HKLE[3])
    assert np.allclose(t_RM_ppzE, RM[0])

    # Test hkl coordinate system
    R0, RM, RV = TAS.calc_resolution_HKLE(HKLE, base='hkl')
    assert np.allclose(t_RM_hklE, RM[0])

if __name__ == '__main__':
    pytest.main()
