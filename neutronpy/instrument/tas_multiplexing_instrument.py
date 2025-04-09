# -*- coding: utf-8 -*-
r"""Various multiplexing TAS
"""
from collections import namedtuple
from typing import Any, Callable
import numpy as np
from scipy.spatial.transform.rotation import Rotation as sp_Rotation

from .tas_instrument import TripleAxisInstrument

from ..crystal import Sample
from ..neutron import Neutron
from .exceptions import ScatteringTriangleNotClosed, InstrumentError
from .general import GeneralInstrument
from .analyzer import Analyzer
from .monochromator import Monochromator
from .plot import PlotInstrument

from ..loggers import setup_TAS_logger

ScatteringSenses = namedtuple('ScatteringSenses', 'mono, sample, ana')

# DEV NOTES
# [lass23] J.Lass et al 2023 https://doi.org/10.1063/5.0128226
# [groitl26] F.Groitl et al 2016 http://dx.doi.org/10.1063/1.4943208
# 1. Weak points of the constructor are that only specified fields are multiplexing.
#    Maybe there is a smart way to allow the user to construct each analyzer row
#    with varying parameters.
#    How about inheriting TAS to MTAS, with empty fields that are supposed to be 
#    varying for each analyzer row? 
#    I really like how this could allow to still represent main components of the instrument
#    Also, each analyzer row can contain `self.sample` etc, which will point to one sample
#    and it will also allow to change the properties easily
# 2. Actually the situation seems to more complex for instrument like CAMEA. 
#    They use the multibinning technique to disentangle the data into smaller energy bins
#    utilizing the prismatic concept of energy determination. Owing to that, they actually
#    utilize only one Ei (dismiss energy interlacing) to cover the energy range 
#    with pretty small bin size.
#    1. Treat the instrument as working in constant ki mode, and allow each analyzer
#       to cover only small energy range. 
#       The range is chosen as mid points between analyzer fixed Ef. 
#       NO: The energy range is taken on the example of ana4
#       from [lass23] as E0+-dE, E0=4.035 +dE=0.142 -dE=0.134. dE/E=0.14/4.035=3.5%
class CAMEA_MTAS(GeneralInstrument, PlotInstrument):
    u"""An object that represents the CAMEA@PSI spectrometer with
    multiplexing detector bank. It contains experimental configuration, 
    including a sample and main components if the instrument.

    Implementation
    --------------
    `MultiplexingTripleAxisInstrument` is in fact a list of N `TripleAxisInstrument`,
    where N is the number of fixed-kf analyzers rows. The fact tha rea

    Parameters
    ----------
    beam : `Neutron`, optional
        Properties of the fixed beam. `Neutron` instance.

    fx: int, optional
        Flag encoding thether the instrument works in constant-ki `fx=1`,
        or constant-kf `fx=1` mode.

    sample : obj, optional
        Sample lattice constants, parameters, mosaic, and orientation
        (reciprocal-space orienting vectors). Default: A crystal with
        a,b,c = 6,7,8 and alpha,beta,gamma = 90,90,90 and orientation
        vectors u=[1 0 0] and v=[0 1 0].

    scat_senses: namedtuple[int,int,int], optional
        Scattering senses list of `Monochromator` (mono), `Sample` (sample),
        and `Analyzer' (ana).

    hcol : list(4), optional
        Horizontal Soller collimations in minutes of arc starting from the
        neutron guide. Default: [40 40 40 40]

    vcol : list(4), optional
        Vertical Soller collimations in minutes of arc starting from the
        neutron guide. Default: [120 120 120 120]

    arms: list(4), optional
        Distances of the spectrometer arms, i.e. distances between 
        source-monochromator-sample-analyzer-detector.


    Attributes
    ----------
    method
    moncor
    mono
    ana
    hcol
    vcol
    arms
    efixed
    sample
    orient1
    orient2
    infin
    beam
    detector
    monitor
    Smooth
    guide
    description_string

    Methods
    -------
    calc_resolution
    calc_resolution_in_Q_coords
    calc_projections
    get_angles_and_Q
    get_lattice
    get_resolution_params
    get_resolution
    plot_projections
    plot_ellipsoid
    plot_instrument
    resolution_convolution
    resolution_convolution_SMA
    plot_slice

    Notes
    -----
    There is usually a lot of confusion about the rotation directions, senses, coordinate systems etc.
    Here, it is assumed:
    1. Local reference system for each component is such that `x || -ki`, 
       `z` is upwards from scattering plane, `y` completes the right-hand system.
    1. All motors are right handed. TODO allowing LHS motors?
    2. Positive scattering sense is when `kf` has to be rotated in positive direction
       for scattering condition.

    """

    def __init__(self, Ei: float, sample: Sample, a3_offset: float):
        # CAMEA specifications from [lass23] and [groitl16]
        name = 'CAMEA'
        final_energies = [3.200, 3.382, 3.574, 3.787, 4.035, 4.312, 4.631, 4.987]   # in meV
        sample_ana_distances = [93.0, 99.4, 105.6, 112.0, 118.3, 124.7, 131.2, 137.9]   # in cm
        ana_det_distances = [70]*8  # in cm
        scat_senses = (+1, -1, +1)
        focussing = dict(
            mono_h = 'optimal', 
            mono_v = 'optimal', 
            ana_h  = 'optimal', 
            ana_v  = 'flat'
        )
        hcol = [9999, 9999, 1, 9999]
        vcol = [9999, 9999, 9999, 9999]
        mono = Monochromator(name='mono_CAMEA', dspacing=3.355, mosaic=50.0, mosaic_v=50.0,
                                   height=17.2, width=26.2, depth=0.2,
                                   curvr_h=1, curvr_v=1)
        ana = Analyzer(name='ana_CAMEA', dspacing=3.355, mosaic=60.0, mosaic_v=60.0,
                                   height=5, width=5, depth=0.2,
                                   curvr_h=1, curvr_v=1)
        source_config = dict(name='Source', shape='rectangular', width=6, height=12, 
                             use_guide='False', div_v=15, div_h=15)
        detector_config = dict(name='Detector', shape="rectangular", width=2.5, height=10)
        
        # Logger
        self.logger = setup_TAS_logger()
        self.logger.debug(f'Creating `MTAS` instance: name={name}')

        # Defining properties
        self.name = name
        self.incoming_neutron = Neutron(energy=Ei)
        self.final_neutrons = Neutron(energy=final_energies)

        Ef = self.final_neutrons.energy
        dEf = (Ef[1:]-Ef[:-1])/2
        self.Ef_bins = np.concatenate(( [Ef[0]-dEf[0]], Ef[:-1]+dEf, [Ef[-1]+dEf[-1]] ))

        self.sample = sample

        # Each analyzer row is in fact a `TripleAxisInstrument` in constant-ki mode
        analyzer_rows = []
        for n in range(len(final_energies)):
            # Each ana row makes a TAS with different:
            # arms, ana dimensions
            primary_arms = [200, 160, 130]
            ana_row_arms = primary_arms + [sample_ana_distances[n], ana_det_distances[n]]
            ana_row = TripleAxisInstrument(fixed_kf=False, fixed_wavevector=self.incoming_neutron.wavevector,
                                                name=name+f'_row{n}', scat_senses=scat_senses,
                                                a3_offset=a3_offset, kf_vert=True,
                                                arms=ana_row_arms, hcol=hcol,
                                                vcol=vcol, focussing=focussing, sample=sample,
                                                mono=mono, ana=ana, 
                                                source=source_config, detector=detector_config)
            analyzer_rows.append(ana_row)

        self.analyzer_rows = analyzer_rows

    
    def __repr__(self):
        keys_ordered = []
        ret = f"MTAS< {self.name}, Ei={self.incoming_neutron.energy}\n"

        for analyzer_row in self.analyzer_rows:
            ret += analyzer_row.__repr__() + '\n'
        return ret + '>'


    ###############################################################################
    # Properties
    # Implemented as needed 

    @property
    def analyzer_rows(self) -> list[TripleAxisInstrument]:
        '''List of `TripleAxisInstrument` which is treated as each analyzer row
        of the `MultiplexingTripleAxisInstrument'''
        return self._analyzer_rows
    
    @analyzer_rows.setter
    def analyzer_rows(self, new_analyzers: list[TripleAxisInstrument]):
        if not isinstance(new_analyzers, list):
            raise ValueError("New analyzers of `MTAS` must be a list of `TAS`")
        if not isinstance(new_analyzers[0], TripleAxisInstrument):
            raise ValueError("New analyzers of `MTAS` must be a list of `TAS`")
        
        self._analyzer_rows = new_analyzers


    @property
    def functional_Etransfer(self) -> tuple[float, float]:
        """Range in which the CAMEA cna determine energy transfer."""
        return (self.incoming_neutron.energy-self.Ef_bins[-1], self.incoming_neutron.energy-self.Ef_bins[0])


    @property
    def sample(self) -> Sample:
        """Sample beaing measured at CAMEA"""
        return self._sample
    
    @sample.setter
    def sample(self, new: Sample):
        if not isinstance(new, Sample):
            raise ValueError('New sample must be of `Sample` type.')
        
        self._sample = new
    
    ########################################################################################
    # Methods
    
    # Main functionality
    # Following DEV NOTES point 2.1
    def resolution_convolution(self, 
                               hkle: np.ndarray, 
                               Sqw_fast: Callable[[np.ndarray, Any], np.ndarray], 
                               Sqw_slow: Callable[[np.ndarray, Any], np.ndarray], 
                               Sqw_pars: Any, 
                               method = str, 
                               Nsamples=1000, seed=None):
        """
        Numerical convolution of the supplied spectral weight `S(Q,E)=S_{slow}(Q,E)*S_{fast}(Q,E)`
        with the instrumental parameters from `self=TripleAXisSpectrometer` on the points
        in `hkle`.

        Parameters
        ----------
        hkle: array_like (...,4)
            Array of points in 4D space defining the momentum and energy transfer at which to 
            determine the convolved intensity. Although, the momentum transfer must lie in the scattering plane.
        Sqw_fast: function signature: Sqw(hkle, params)
            Fast part of the spectral weight. The function accepts `hkle`, with the same signature
            as above. `params` are any parameters used by the function.
        Sqw_slow: funciton signature: Sqw(h,k,l,E, params)
            Slow part of the spectral weight. Same signature and specs as above `Sqw_fast`.
        Sqw_params: Any
            Parameters passed to the spectral weight functions.
        method: `pop`, `eck`
            Algorithm for resolution convolution.
        Nsamples:
            Number of points used to probe the space defined by resolution ellipsoid.
        seed:
            For the random number generator that makes the sampling points.

        Returns
        -------
        Sqw: array_like shape=shape(hkle)
            Spectral weight convolved with instrumental resolution at requested points.


        Notes
        -----
        The heaviest calculation by far is calling the spectral weight on large number of points.
        Thus, to speed up the process the spectral weight is split into fast and slow varying parts,
        and only the fast part is being convolved. This is equivalent to assuming that the slow
        varying part does not change its value significantly within the region of the resolution ellipsoid.

        """
        # Check if Erange is ok
        Emin, Emax = self.functional_Etransfer
        ind_invalidE = (hkle[...,3]<Emin) | (hkle[...,3]>Emax)
        if np.sum(ind_invalidE):
            raise ScatteringTriangleError(f"Functional Erange is {self.functional_Etransfer}. Can't reach requested energy:\n{hkle[ind_invalidE, ...]}")

        # Associate each energy from `hkle` into specific analyzer row with the digitize()
        Ef = self.incoming_neutron.energy - hkle[...,3]
        ana_ids = np.digitize(Ef, self.Ef_bins)-1

        Sqw = np.full(np.shape(hkle)[:-1], np.nan)

        for n, ana_row in enumerate(self.analyzer_rows):
            ind_ana = ( ana_ids == n)
            Sqw_ana = ana_row.resolution_convolution(hkle=hkle[ind_ana,...], 
                                                     Sqw_fast=Sqw_fast, Sqw_slow=Sqw_slow, 
                                                     Sqw_pars=Sqw_pars,
                                                     method=method, Nsamples=Nsamples, seed=seed)
            
            Sqw[ind_ana] = Sqw_ana

        return Sqw

    def resolution_convolution_single_mode(self,
                                           hkle: np.ndarray, 
                                           Sqw_fast: Callable[[np.ndarray, Any], np.ndarray], 
                                           Sqw_slow: Callable[[np.ndarray, Any], np.ndarray], 
                                           Sqw_pars: Any, 
                                           method = str, 
                                           Nsamples=1000, seed=None):
        
        return
