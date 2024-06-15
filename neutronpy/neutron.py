# -*- coding: utf-8 -*-
r""" Class to calculate the energy of a neutron in various common units
"""
from typing import Any, Iterable, Union
import numpy as np
from scipy.constants import h, hbar, k, m_n

from .constants import JOULES_TO_MEV

# Neutron unit conversion prefactors (npf)
_prefactor_energy_from_frequency = h * JOULES_TO_MEV * 1.e12
_prefactor_energy_from_temperature = k* JOULES_TO_MEV
_prefactor_energy_from_wavevector = (h ** 2 / (2. * m_n * ((2. * np.pi) / 1.e10) ** 2) * JOULES_TO_MEV)
_prefactor_energy_from_velocity =  m_n / 2. * JOULES_TO_MEV
_prefactor_energy_from_wavelength = h ** 2. * 1.0e20 / (2. * m_n ) * JOULES_TO_MEV

class Neutron(object):
    u"""Class containing the most commonly used properties of a neutron beam
    given some initial input, e.g. energy, wavelength, velocity, wavevector,
    temperature, or frequency. At least one input must be supplied.

    Parameters
    ----------
    energy : float
        Neutron energy in millielectron volts (meV)
    wavelength : float
        Neutron wavelength in angstroms (Å)
    velocity : float
        Neutron velocity in meters per second (m/s)
    wavevector : float
        Neutron wavevector k in inverse angstroms (1/Å)
    temperature : float
        Neutron temperature in kelvin (K)
    frequency : float
        Neutron frequency in terahertz (THz)

    Returns
    -------
    Neutron object
        The object containing the properties of the neutron beam

    Attributes
    ----------
    energy
    wavelength
    wavevector
    velocity
    temperature
    frequency
    values
    """

    def __init__(self, energy: float=None, wavelength: float=None, velocity: float=None, wavevector: float=None, temperature: float=None, frequency: float=None):
        self._update_values(energy, wavelength, velocity, wavevector, temperature, frequency)

    def __str__(self):
        return self.values

    def __repr__(self):
        return "Energy({0})".format(self.energy)

    def __eq__(self, right):
        return abs(self.energy - right.energy) < 1e-6

    def __ne__(self, right):
        return not self.__eq__(right)

    def _update_values(self, energy: float=None, wavelength: float=None, velocity: float=None, wavevector: float=None, temperature: float=None, frequency: float=None):
        try:
            if energy is None:
                if wavelength is not None:
                    self.en = self._energy_from_wavelength(wavelength)
                elif velocity is not None:
                    self.en = self._energy_from_velocity(velocity)
                elif wavevector is not None:
                    self.en = self._energy_from_wavevector(wavevector)
                elif temperature is not None:
                    self.en = self._energy_from_temperature(temperature)
                elif frequency is not None:
                    self.en = self._energy_from_frequency(frequency)
            else:
                self.en = energy

            self.wavelen = self._wavelength_from_energy(self.en)
            self.wavevec = self._wavevector_from_energy(self.en)
            self.vel = self._velocity_from_energy(self.en)
            self.temp = self._temperature_from_energy(self.en)
            self.freq = self._frequency_form_energy(self.en)

        except AttributeError:
            raise AttributeError("""You must define at least one of the \
                                    following: energy, wavelength, velocity, \
                                    wavevector, temperature, frequency""")


    def _energy_from_frequency(self, frequency):
        return frequency * _prefactor_energy_from_frequency
    
    def _frequency_form_energy(self, energy):
        return np.divide(energy, _prefactor_energy_from_frequency)


    def _energy_from_temperature(self, temperature):
        return temperature * _prefactor_energy_from_temperature
    
    def _temperature_from_energy(self, energy):
        return np.divide(energy, _prefactor_energy_from_temperature)


    def _energy_from_wavevector(self, wavevector):
        return wavevector**2 * _prefactor_energy_from_wavevector
    
    def _wavevector_from_energy(self, energy):
        return np.sqrt( np.divide(energy, _prefactor_energy_from_wavevector) )


    def _energy_from_velocity(self, velocity):
        return velocity**2 * _prefactor_energy_from_velocity

    def _velocity_from_energy(self, energy):
        return np.sqrt( np.divide(energy, _prefactor_energy_from_velocity ) )
    

    def _energy_from_wavelength(self, wavelength):
        return np.divide(_prefactor_energy_from_wavelength, wavelength**2)

    def _wavelength_from_energy(self, energy):
        return np.sqrt( np.divide(_prefactor_energy_from_wavelength, energy) )


    @property
    def energy(self):
        r"""Energy of the neutron in meV"""
        return self.en

    @energy.setter
    def energy(self, value):
        self._update_values(energy=value)

    @property
    def wavelength(self):
        r"""Wavelength of the neutron in Å"""
        return self.wavelen

    @wavelength.setter
    def wavelength(self, value):
        self._update_values(wavelength=value)

    @property
    def wavevector(self):
        u"""Wavevector k of the neutron in 1/Å"""
        return self.wavevec

    @wavevector.setter
    def wavevector(self, value):
        self._update_values(wavevector=value)

    @property
    def temperature(self):
        r"""Temperature of the neutron in Kelvin"""
        return self.temp

    @temperature.setter
    def temperature(self, value):
        self._update_values(temperature=value)

    @property
    def frequency(self):
        r"""Frequency of the neutron in THz"""
        return self.freq

    @frequency.setter
    def frequency(self, value):
        self._update_values(frequency=value)

    @property
    def velocity(self):
        r"""Velocity of the neutron in m/s"""
        return self.vel

    @velocity.setter
    def velocity(self, value):
        self._update_values(velocity=value)

    @property
    def values(self):
        r"""Prints all of the properties of the Neutron beam

        Parameters
        ----------
        None

        Returns
        -------
        values : string
            A string containing all the properties of the neutron including
            respective units
        """
        values = [u'',
                  u'Energy: {0:3.3f} meV'.format(self.energy),
                  u'Wavelength: {0:3.3f} Å'.format(self.wavelength),
                  u'Wavevector: {0:3.3f} 1/Å'.format(self.wavevector),
                  u'Velocity: {0:3.3f} m/s'.format(self.velocity),
                  u'Temperature: {0:3.3f} K'.format(self.temperature),
                  u'Frequency: {0:3.3f} THz'.format(self.frequency),
                  u'']

        return '\n'.join(values)
