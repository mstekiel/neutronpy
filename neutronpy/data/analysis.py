# -*- coding: utf-8 -*-
import numbers
import numpy as np
from ..constants import BOLTZMANN_IN_MEV_K
from ..energy import Energy


class Analysis(object):
    r"""Class containing methods for the Data class

    Attributes
    ----------
    detailed_balance_factor

    Methods
    -------
    integrate
    position
    width
    scattering_function
    dynamic_susceptibility
    estimate_background

    """

    @property
    def detailed_balance_factor(self):
        r"""Returns the detailed balance factor (sometimes called the Bose
        factor)

        Parameters
        ----------
            None

        Returns
        -------
        dbf : ndarray
            The detailed balance factor (temperature correction)

        """

        return 1. - np.exp(-self.Q[:, 3] / BOLTZMANN_IN_MEV_K / self.temp)

    def integrate(self, background=None, **kwargs):
        r"""Returns the integrated intensity within given bounds

        Parameters
        ----------
        bounds : bool, optional
            A boolean expression representing the bounds inside which the
            calculation will be performed

        background : float or dict, optional
            Default: None

        Returns
        -------
        result : float
            The integrated intensity either over all data, or within
            specified boundaries

        """
        result = 0
        for i in range(4):
            result += np.trapz(self.intensity[self.get_bounds(kwargs)] - self.estimate_background(background),
                               np.squeeze(self.Q[self.get_bounds(kwargs), i]))

        return result

    def position(self, background=None, **kwargs):
        r"""Returns the position of a peak within the given bounds

        Parameters
        ----------
        bounds : bool, optional
            A boolean expression representing the bounds inside which the
            calculation will be performed

        background : float or dict, optional
            Default: None

        Returns
        -------
        result : tup
            The result is a tuple with position in each dimension of Q,
            (h, k, l, e)

        """
        result = ()
        for j in range(4):
            _result = 0
            for i in range(4):
                _result += np.trapz(self.Q[self.get_bounds(kwargs), j] *
                                    (self.intensity[self.get_bounds(kwargs)] - self.estimate_background(background)),
                                    np.squeeze(self.Q[self.get_bounds(kwargs), i])) / self.integrate(**kwargs)

            result += (np.squeeze(_result),)

        return result

    def width(self, background=None, fwhm=False, **kwargs):
        r"""Returns the mean-squared width of a peak within the given bounds

        Parameters
        ----------
        bounds : bool, optional
            A boolean expression representing the bounds inside which the
            calculation will be performed

        background : float or dict, optional
            Default: None

        fwhm : bool, optional
            If True, returns width in fwhm, otherwise in mean-squared width.
            Default: False

        Returns
        -------
        result : tup
            The result is a tuple with the width in each dimension of Q,
            (h, k, l, e)

        """
        result = ()
        for j in range(4):
            _result = 0
            for i in range(4):
                _result += np.trapz((self.Q[self.get_bounds(kwargs), j] - self.position(**kwargs)[j]) ** 2 *
                                    (self.intensity[self.get_bounds(kwargs)] - self.estimate_background(background)),
                                    self.Q[self.get_bounds(kwargs), i]) / self.integrate(**kwargs)

            if fwhm:
                result += (np.sqrt(np.squeeze(_result)) * 2. * np.sqrt(2. * np.log(2.)),)
            else:
                result += (np.squeeze(_result),)

        return result

    def scattering_function(self, material, ei):
        r"""Returns the neutron scattering function, i.e. the detector counts
        scaled by :math:`4 \pi / \sigma_{\mathrm{tot}} * k_i/k_f`.

        Parameters
        ----------
        material : object
            Definition of the material given by the :py:class:`.Material`
            class

        ei : float
            Incident energy in meV

        Returns
        -------
        counts : ndarray
            The detector counts scaled by the total scattering cross section
            and ki/kf
        """
        ki = Energy(energy=ei).wavevector
        kf = Energy(energy=ei - self.e).wavevector

        return 4 * np.pi / material.total_scattering_cross_section * ki / kf * self.detector

    def dynamic_susceptibility(self, material, ei):
        r"""Returns the dynamic susceptibility
        :math:`\chi^{\prime\prime}(\mathbf{Q},\hbar\omega)`

        Parameters
        ----------
        material : object
            Definition of the material given by the :py:class:`.Material`
            class

        ei : float
            Incident energy in meV

        Returns
        -------
        counts : ndarray
            The detector counts turned into the scattering function multiplied
            by the detailed balance factor
        """
        return self.scattering_function(material, ei) * self.detailed_balance_factor

    def estimate_background(self, bg_params):
        r"""Estimate the background according to ``type`` specified.

        Parameters
        ----------
        bg_params : dict
            Input dictionary has keys 'type' and 'value'. Types are
                * 'constant' : background is the constant given by 'value'
                * 'percent' : background is estimated by the bottom x%, where x
                  is value
                * 'minimum' : background is estimated as the detector counts

        Returns
        -------
        background : float or ndarray
            Value determined to be the background. Will return ndarray only if
            `'type'` is `'constant'` and `'value'` is an ndarray
        """
        if isinstance(bg_params, type(None)):
            return 0

        elif isinstance(bg_params, numbers.Number):
            return bg_params

        elif bg_params['type'] == 'constant':
            return bg_params['value']

        elif bg_params['type'] == 'percent':
            inten = self.intensity[self.intensity >= 0.]
            Npts = int(inten.size * (bg_params['value'] / 100.))
            min_vals = inten[np.argsort(inten)[:Npts]]
            background = np.average(min_vals)
            return background

        elif bg_params['type'] == 'minimum':
            return min(self.intensity)

        else:
            return 0

    def get_bounds(self, kwargs):
        r"""Generates a to_fit tuple if bounds is present in kwargs

        Parameters
        ----------
        kwargs : dict

        Returns
        -------
        to_fit : tuple
            Tuple of indices
        """
        if 'bounds' in kwargs:
            to_fit = np.where(kwargs['bounds'])
        else:
            to_fit = np.where(self.Q[:, 0])

        return to_fit
