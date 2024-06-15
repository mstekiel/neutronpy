# -*- coding: utf-8 -*-
"""Testing for Energy conversions

"""
import numpy as np
import pytest
from neutronpy import Neutron

def test_neutron_list():
    '''Test initializer of the neutron'''

    # defining numbers in consecutive columns are: 
    # energy, frequency, wavevector, wavelength
    energy     = np.array([   0.0,  4.1357, 14.6836, 81.8042])
    wavelength = np.array([np.inf,  4.4475,  2.3603,     1.0])
    wavevector = np.array([   0.0,   1.413,   2.662,   6.283])
    velocity   = np.array([   0.0,  889.50, 1676.06, 3956.03])
    temperature= np.array([   0.0,  47.993, 170.396, 949.298])
    frequency  = np.array([   0.0,     1.0,  3.5505, 19.7802])

    neutron = Neutron(energy=energy)

    assert all(np.round(neutron.energy, 4) == energy)
    assert all(np.round(neutron.wavelength, 4) == wavelength)
    assert all(np.round(neutron.wavevector, 3) == wavevector)
    assert all(np.round(neutron.velocity, 2) == velocity)
    assert all(np.round(neutron.temperature, 3) == temperature)
    assert all(np.round(neutron.frequency, 4) == frequency)


def test_neutron():
    """Test that the output string is correct
    """
    neutron = Neutron(energy=25.)
    assert (np.round(neutron.energy) == 25.0)
    assert (np.round(neutron.wavelength, 4) == 1.8089)
    assert (np.round(neutron.wavevector, 3) == 3.473)
    assert (np.round(neutron.velocity, 3) == 2186.967)
    assert (np.round(neutron.temperature, 3) == 290.113)
    assert (np.round(neutron.frequency, 3) == 6.045)

    string_test = u"\nEnergy: {0:3.3f} meV"
    string_test += u"\nWavelength: {1:3.3f} Å"
    string_test += u"\nWavevector: {2:3.3f} 1/Å"
    string_test += u"\nVelocity: {3:3.3f} m/s"
    string_test += u"\nTemperature: {4:3.3f} K"
    string_test += u"\nFrequency: {5:3.3f} THz\n"
    string_test = string_test.format(25.0, 1.8089, 3.473, 2186.967,
                                     290.113, 6.045)

    assert (neutron.values == string_test)


def test_neutron_setters():
    """Tests the neutron setters
    """
    n = Neutron(energy=25.)

    n.energy = 25
    assert (np.round(n.wavelength, 4) == 1.8089)

    n.wavevector = 3.5
    assert (np.round(n.energy, 1) == 25.4)

    n.velocity = 2180
    assert (np.round(n.energy, 1) == 24.8)

    n.temperature = 290
    assert (np.round(n.energy, 1) == 25.0)

    n.frequency = 6
    assert (np.round(n.energy, 1) == 24.8)

    n.wavelength = 1.9
    assert (np.round(n.energy, 1) == 22.7)


if __name__ == "__main__":
    pytest.main([__file__])
