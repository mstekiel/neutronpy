# -*- coding: utf-8 -*-
r""" Class to calculate the energy of a neutron in various common units
"""
import numpy as np
from scipy.constants import h, k, m_n

from .constants import JOULES_TO_MEV

'''
DEV notes

To provide nice I/O characteristics of the class the main attributes
are implemented with `property` decorator, and their setters
are actually calling the update function that takes care of converting
all other attributes. 

Then, the original values are stored in proteced fields `_attribute`
and these should be handled only by DEV and taken care of properly.
'''
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

    Notes
    -----
    Suppoerts use of `np.ndarray`:
    >>> Neutron(energy=np.arange(1,10).reshape(3,3))
    ... Energy [meV]    : [[1 2]
                          [3 4]]
        Wavelength [A]  : [[9.04456804 6.39547539]
                           [5.22188379 4.52228402]]
        Wavevector [1/A]: [[0.69469158 0.98244226]
                           [1.20324112 1.38938317]]
        Velocity [m/s]  : [[437.3933608  618.56762294]
                           [757.5875238  874.7867216 ]]
        Temperature [K] : [[11.60451803 23.20903605]
                           [34.81355408 46.4180721 ]]
        Frequency [THz] : [[0.24179892 0.48359784]
                           [0.72539677 0.96719569]]
    """

    # Neutron unit conversion prefactors (npf)
    _prefactor_energy_from_frequency = h * JOULES_TO_MEV * 1.e12
    _prefactor_energy_from_temperature = k* JOULES_TO_MEV
    _prefactor_energy_from_wavevector = (h ** 2 / (2. * m_n * ((2. * np.pi) / 1.e10) ** 2) * JOULES_TO_MEV)
    _prefactor_energy_from_velocity =  m_n / 2. * JOULES_TO_MEV
    _prefactor_energy_from_wavelength = h ** 2. * 1.0e20 / (2. * m_n ) * JOULES_TO_MEV

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
                    self._energy = self._energy_from_wavelength(wavelength)
                elif velocity is not None:
                    self._energy = self._energy_from_velocity(velocity)
                elif wavevector is not None:
                    self._energy = self._energy_from_wavevector(wavevector)
                elif temperature is not None:
                    self._energy = self._energy_from_temperature(temperature)
                elif frequency is not None:
                    self._energy = self._energy_from_frequency(frequency)
            else:
                self._energy = energy

            self._wavelength    = self._wavelength_from_energy(self._energy)
            self._wavevector    = self._wavevector_from_energy(self._energy)
            self._velocity      = self._velocity_from_energy(self._energy)
            self._temperature   = self._temperature_from_energy(self._energy)
            self._frequency     = self._frequency_form_energy(self._energy)

        except AttributeError:
            raise AttributeError("""You must define at least one of the \
                                    following: energy, wavelength, velocity, \
                                    wavevector, temperature, frequency""")


    def _energy_from_frequency(self, frequency):
        return frequency * self._prefactor_energy_from_frequency
    
    def _frequency_form_energy(self, energy):
        return np.divide(energy, self._prefactor_energy_from_frequency)


    def _energy_from_temperature(self, temperature):
        return temperature * self._prefactor_energy_from_temperature
    
    def _temperature_from_energy(self, energy):
        return np.divide(energy, self._prefactor_energy_from_temperature)


    def _energy_from_wavevector(self, wavevector):
        return np.power(wavevector, 2) * self._prefactor_energy_from_wavevector
    
    def _wavevector_from_energy(self, energy):
        return np.sqrt( np.divide(energy, self._prefactor_energy_from_wavevector) )


    def _energy_from_velocity(self, velocity):
        return velocity**2 * self._prefactor_energy_from_velocity

    def _velocity_from_energy(self, energy):
        return np.sqrt( np.divide(energy, self._prefactor_energy_from_velocity ) )
    

    def _energy_from_wavelength(self, wavelength):
        return np.divide(self._prefactor_energy_from_wavelength, np.power(wavelength, 2))

    def _wavelength_from_energy(self, energy):
        return np.sqrt( np.divide(self._prefactor_energy_from_wavelength, energy) )


    @property
    def energy(self):
        r"""Energy of the neutron in meV"""
        return self._energy

    @energy.setter
    def energy(self, value):
        self._update_values(energy=value)

    @property
    def wavelength(self):
        r"""Wavelength of the neutron in Å"""
        return self._wavelength

    @wavelength.setter
    def wavelength(self, value):
        self._update_values(wavelength=value)

    @property
    def wavevector(self):
        u"""Wavevector k of the neutron in 1/Å"""
        return self._wavevector

    @wavevector.setter
    def wavevector(self, value):
        self._update_values(wavevector=value)

    @property
    def temperature(self):
        r"""Temperature of the neutron in Kelvin"""
        return self._temperature

    @temperature.setter
    def temperature(self, value):
        self._update_values(temperature=value)

    @property
    def frequency(self):
        r"""Frequency of the neutron in THz"""
        return self._frequency

    @frequency.setter
    def frequency(self, value):
        self._update_values(frequency=value)

    @property
    def velocity(self):
        r"""Velocity of the neutron in m/s"""
        return self._velocity

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
                  u'Energy [meV]    : {0}'.format(self.energy),
                  u'Wavelength [A]  : {0}'.format(self.wavelength),
                  u'Wavevector [1/A]: {0}'.format(self.wavevector),
                  u'Velocity [m/s]  : {0}'.format(self.velocity),
                  u'Temperature [K] : {0}'.format(self.temperature),
                  u'Frequency [THz] : {0}'.format(self.frequency),
                  u'']

        return '\n'.join(values)
