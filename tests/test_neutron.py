# -*- coding: utf-8 -*-
"""Testing for Energy conversions

"""
import numpy as np
import pytest
from neutronpy import Neutron


def test_neutron():
    """Test that the output string is correct
    """
    energy = Neutron(energy=25.)
    assert (np.round(energy.energy) == 25.0)
    assert (np.round(energy.wavelength, 4) == 1.8089)
    assert (np.round(energy.wavevector, 3) == 3.473)
    assert (np.round(energy.velocity, 3) == 2186.967)
    assert (np.round(energy.temperature, 3) == 290.113)
    assert (np.round(energy.frequency, 3) == 6.045)

    string_test = u"\nEnergy: {0:3.3f} meV"
    string_test += u"\nWavelength: {1:3.3f} Å"
    string_test += u"\nWavevector: {2:3.3f} 1/Å"
    string_test += u"\nVelocity: {3:3.3f} m/s"
    string_test += u"\nTemperature: {4:3.3f} K"
    string_test += u"\nFrequency: {5:3.3f} THz\n"
    string_test = string_test.format(25.0, 1.8089, 3.473, 2186.967,
                                     290.113, 6.045)

    assert (energy.values == string_test)


def test_neutron_setters():
    """Tests the neutron setters
    """
    n = Neutron(energy=25.)

    n.energy = 25
    assert (np.round(e.wavelength, 4) == 1.8089)

    n.wavevector = 3.5
    assert (np.round(e.energy, 1) == 25.4)

    n.velocity = 2180
    assert (np.round(e.energy, 1) == 24.8)

    n.temperature = 290
    assert (np.round(e.energy, 1) == 25.0)

    n.frequency = 6
    assert (np.round(e.energy, 1) == 24.8)

    n.wavelength = 1.9
    assert (np.round(e.energy, 1) == 22.7)


if __name__ == "__main__":
    pytest.main()
