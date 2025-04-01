# -*- coding: utf-8 -*-
r"""Testing of the resolution library - TAS

"""
import numpy as np
import pytest

from neutronpy.instrument import TripleAxisInstrument
from neutronpy.crystal import Sample

def test_angles():
    tas = TripleAxisInstrument.make_default()
    
    # Direct comparisons with Takin require following
    tas.sample.a = 4
    tas.sample.b = 4
    tas.sample.c = 4
    tas.a3_offset = 90
    tas.mono.dspacing = 3.355
    tas.ana.dspacing = 3.355

    assert np.allclose( tas.get_angles( np.array([1,1,0,5])),
                       [-17.68706, -35.37412, -12.5478, 44.76372, -20.59514, -41.19029])

    assert np.allclose( tas.get_angles( np.array([2,1,0,50])),
                       [-9.648178, -19.29636, 41.73374, 29.20217, -20.59514, -41.19029])


    tas.scat_senses = (1,1,1)
    tas.fixed_kf = False
    assert np.allclose( tas.get_angles( np.array([2,1,0,5])),
                       [20.59514, 41.19029, 25.50538, 92.87613, 25.6682, 51.3364])
    

# TEST to see that default constructor fills all required fields

if __name__ == "__main__":
    pytest.main()
