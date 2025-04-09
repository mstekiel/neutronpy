# -*- coding: utf-8 -*-
r"""Testing of the resolution library - TAS

"""
import pytest

import numpy as np

from neutronpy import Neutron
from neutronpy.instruments import Monochromator, Analyzer
from neutronpy.instruments.exceptions import *

rad2deg = 180/np.pi

def test_constructor():
    config = dict(dspacing=1.86, mosaic=60, name='custom',
                  mosaic_v=60, width=60, height=50, depth=2,
                  curvr_h = 20, curvr_h_opt=True,
                  curvr_v = 20, curvr_v_opt=True)
    mono = Monochromator(**config)

    for key, value in config.items():
        assert getattr(mono, key) == value



def test_setters():
    ana = Analyzer.from_name('PG(002)')
    assert ana.dspacing == 3.354

    ana.dspacing = 2
    assert ana.dspacing == 2.0


def test_tth():
    mono = Monochromator(dspacing=3.355)

    beam_i = Neutron(wavevector=[1,2,3,4])
    assert np.allclose( np.degrees(mono.get_tth(beam_i.wavelength)), 
                       [138.9082, 55.83447, 36.37533, 27.0769])



if __name__ == "__main__":
    pytest.main()
    # test_analyzer()