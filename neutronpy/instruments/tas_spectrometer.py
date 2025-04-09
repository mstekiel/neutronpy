# -*- coding: utf-8 -*-
r"""Define an instrument for resolution calculations

"""
from collections import namedtuple
from typing import Any, Callable, TYPE_CHECKING
import numpy as np
from scipy.spatial.transform.rotation import Rotation as sp_Rotation

from neutronpy.fileio.instrument import TAS_loader

from ..crystal import Sample
from ..neutron import Neutron
from .exceptions import warn, ScatteringTriangleNotClosed, VectorNotInScatteringPlane, MonochromatorError, AnalyzerError
from .neutron_spectrometer import NeutronSpectrometer
from .components import Analyzer, Monochromator, ComponentTemplate
from .plot import PlotInstrument

from .TW_respy.pop import calc as TW_calc_pop
from .TW_respy.eck import calc as TW_calc_eck

from ..loggers import setup_TAS_logger

ScatteringSenses = namedtuple('ScatteringSenses', 'mono, sample, ana')

# DEV NOTES TODO
# - [X] Since np.linalg is usually operating on the last dimension, I will od the same.
#       This changes from `h=hkle[0,...] ` to h=hkle[...,0].
# - [X] Main functionalities operate based on methods with the `hkle` argument.
#       I need to ensure this array has proper shape, the hkl vectors are
#       in the scattering plane and maybe some other checks.
#       How about writing a decorator for the function, that will performe required checks?
#       RESULT: the validators are methods of other classes. Using them requires pusing those
#       as classmethods which will not work for `is_in scattering_plane` as it needs
#       instance of `Sample.` Validate the arguments in the function call :(
# - [X] I want to be able to call `calc_resolution_HKLE(hkle)` hkle.shape=(...,N).
#       So where should I unpack the `...` dimensions? -> in `calc_resolution_HKLE`.
# - [X] Apparently `np.einsum` can be slow for arrays with high dimensions and
#       long list of arrays involved in summation. There are optimization options 
#       for it that should be looked into:
#        - https://numpy.org/doc/stable/reference/generated/numpy.einsum_path.html#numpy.einsum_path
#        - just use tensordot?
#       SOLUTION: The speed doesn't matter for resolution calulation, as much
#       as it matters for evaluatin spectral weight within the resolution ellipsoid.
#       Clarity of code is more important here, keep it as it is.
#   [ ] Change the policy of resolutions calculations. Instead of throwing errors when 
#       the `hkl` is not in the scattering plane, return NaNs.

class TripleAxisSpectrometer(NeutronSpectrometer, PlotInstrument):
    u"""An object that represents a Triple Axis Spectrometer (TAS) instrument
    experimental configuration, including a sample.

    Parameters
    ----------
    fixed_neutron : `Neutron`, optional
        Properties of the fixed beam. `Neutron` instance.

    fixed_kf: int, optional
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

    def __init__(self, fixed_kf: bool, fixed_wavevector: float, name: str,
                 scat_senses: tuple[int,int,int], a3_offset: float, kf_vert: bool,
                 arms: list[int], hcol: list[int], vcol: list[int], 
                 focussing: dict[str,str], sample: Sample, 
                 mono: Monochromator, ana: Analyzer,
                 source: ComponentTemplate, detector: ComponentTemplate):

        self.logger = setup_TAS_logger()
        self.logger.debug(f'Creating `TAS` instance: name={name}')

        # Operational mode
        self.fixed_kf = fixed_kf
        self.fixed_wavevector = fixed_wavevector

        self.kf_vert = kf_vert   # for CAMEA/BAMBUS liek secondary spectrometer

        # Main components of the instrument
        self.source = source
        self.mono = mono
        self.sample = sample
        self.ana = ana
        self.detector = detector

        # Defining properties
        self.name = name
        self.arms = np.array(arms)
        self.scat_senses = scat_senses
        self.a3_offset = a3_offset
        self.hcol = np.array(hcol)
        self.vcol = np.array(vcol)
        self.focussing = focussing

    @classmethod
    def make_default(cls):
        """Construct `TripleAxisSpectrometer` instance with default values.
        
        >>> focussing_default = dict(mono_h = 'flat', mono_v = 'optimal', 
        >>>                          ana_h  = 'flat', ana_v  = 'flat')
        >>> source_config = dict(name='Source', shape='rectangular', width=6, height=12, 
        >>>                      use_guide='False', div_v=15, div_h=15)
        >>> detector_config = dict(name='Detector', shape="rectangular", width=2.5, height=10)
        >>> 
        >>> TripleAxisInstrument(
        >>>    fixed_kf=True, fixed_wavevector=2.662, name="TAS_default",
        >>>    scat_senses=(-1,+1,-1), a3_offset=0, kf_vert=False,
        >>>    arms=[10,200,115,85,0], hcol=[40, 40, 40, 40], vcol=[120, 120, 120, 120], 
        >>>    focussing=focussing_default,
        >>>    sample=Sample.make_default(), 
        >>>    mono=Monochromator.make_default(), ana=Analyzer.make_default(),
        >>>    source=source_config, detector=detector_config
        >>>    )

        
        """
        focussing_default = dict(
            mono_h = 'flat', 
            mono_v = 'optimal', 
            ana_h  = 'flat', 
            ana_v  = 'flat'
        )

        source_config = dict(name='Source', shape='rectangular', width=6, height=12, 
                             use_guide='False', div_v=15, div_h=15)
        detector_config = dict(name='Detector', shape="rectangular", width=2.5, height=10)
        # monitor_config = dict(name='Monitor', width=6, height=12)
        # Monitor_default =  ComponentTemplate(**monitor_config)

        return cls(fixed_kf=True, fixed_wavevector=2.662, name="TAS_default",
                 scat_senses=(-1,+1,-1), a3_offset=0, kf_vert=False,
                 arms=[10,200,115,85,0], hcol=[40, 40, 40, 40], vcol=[120, 120, 120, 120], 
                 focussing=focussing_default,
                 sample=Sample.make_default(), 
                 mono=Monochromator.make_default(), ana=Analyzer.make_default(),
                 source=source_config, detector=detector_config
                 )

    @classmethod
    def from_taz_file(cls, filename):
        """Create `TripleAxisInstrument` instance from the Takin configuration file .taz.
        Assume constant-kf mode.
        Fill sample lattice_parameters and orientation with default values.
        Also sample dimensions have to be updated, as Takin has them already in w_q, w_qperp.

        Parameters
        ----------
        filename: str
            Path and name of the .taz file

        Returns
        -------
            `TripleAxisInstrument` instance.

        Notes
        -----
        In principle Takin's taz configuration file does not store
        information about the spectrometer, but about all instrumental
        parameters required to calculate the resollution ellipsoid
        at the specifi energy nad momentum transfer assuming certain ki/kf.
        As such, 
        mode (constant-ki/ocnstant-kf) in which the instrument is operating.
        This information is filled automatically, and can be edjusted. 
        """
        config = TAS_loader.from_taz(filename)
        return cls(**config)

    
    def __repr__(self):
        keys_ordered = []
        ret = f"Instrument< {self.name}, const-{'kf'}={self.fixed_wavevector}, a3_offset={self.a3_offset}\n"
        ret += f"\t scat_senses={self.scat_senses}, kf_vert={self.kf_vert}, arms={self.arms},\n"
        ret += f"\t hcol={self.hcol}, vcol={self.vcol},\n"
        ret += f"\t focussing={self.focussing},\n"
        ret += f"\t sample={self.sample},\n"
        ret += f"\t mono={self.mono},\n"
        ret += f"\t ana={self.ana},\n"
        ret += f"\t source={self.source},\n"
        ret += f"\t detector={self.detector}>"
        return ret

    ###############################################################################
    # TODO clean up the properties
    @property
    def mono(self) -> Monochromator:
        u"""`Monochromator` of the `TripleAxisInsturment`."""
        return self._mono

    @mono.setter
    def mono(self, new_mono):
        """Allowed to set with `Monochromator` instance or
        cofig dictionary which is passed to `Monochromator.__init__
        """
        self.logger.debug(f'Setting new `Monochromator`')
        if isinstance(new_mono, Monochromator):
            self._mono = new_mono
        elif isinstance(new_mono, dict):
            self._mono = Monochromator(**new_mono)
        else:
            raise ValueError(f'New monochromator must ba a valid instance of {Monochromator.__name__!r}')
        

    @property
    def ana(self) -> Analyzer:
        u"""`Analyzer` of the `TripleAxisInsturment`."""
        return self._ana

    @ana.setter
    def ana(self, new_ana):
        """Allowed to set with `Analyzer` instance or
        cofig dictionary which is passed to `Analyzer.__init__
        """
        if isinstance(new_ana, Analyzer):
            self._ana = new_ana
        elif isinstance(new_ana, dict):
            self._ana = Analyzer(**new_ana)
        else:
            raise ValueError(f'New analyzer must ba a valid instance of {Monochromator.__name__!r}')

    @property
    def hcol(self):
        r""" The horizontal Soller collimations in minutes of arc (FWHM beam
        divergence) between main TAS elements.

        [source_mono, mono_sample, sample_ana, ana_det]"""
        return self._hcol

    @hcol.setter
    def hcol(self, value):
        self._hcol = value

    @property
    def vcol(self):
        """The vertical Soller collimations in minutes of arc (FWHM beam
        divergence) between main TAS elements.

        [source_mono, mono_sample, sample_ana, ana_det]
        """
        return self._vcol

    @vcol.setter
    def vcol(self, value):
        self._vcol = value

    @property
    def arms(self):
        """Lengths of the spectrometer arms, distances between the axes.
        The 5th element is only needed if ``moncor=1``.

        [source_mono, mono_sample, sample_ana, ana_det, det_monitor] in [cm]
        """
        return self._arms

    @arms.setter
    def arms(self, value):
        self._arms = value

    @property
    def a3_offset(self):
        """Offset in A3 = sample rotation = sample theta [deg]"""
        return self._a3_offset
    
    @a3_offset.setter
    def a3_offset(self, value: float):
        self._a3_offset = value

    @property
    def scat_senses(self):
        """Scattering senses list of `Monochromator`, `Sample`, and `Analyzer'."""

        return self._scat_senses

    @scat_senses.setter
    def scat_senses(self, values: tuple[int,int,int]):
        if not set(values).issubset({-1,1}):
            raise KeyError(f'{values!r} are unrecognised scattering senses. They have to be `+-1`')

        self._scat_senses = ScatteringSenses._make(values)
        
    
    @property
    def fixed_kf(self):
        """Does the spectrometer work in fixed-kf mode?
        Otherwise it is in fixed-ki mode."""
        return self._fixed_kf
    
    @fixed_kf.setter
    def fixed_kf(self, value: bool):
        """Is the spectrometer working in `True`: constant-kf or `False`: constant-ki mode."""
        self._fixed_kf = value

    @property
    def fx(self):
        """Flag encodind the constant-ki mode `fx=1`,
        or constant-kf mode `fx=2`."""

        return int(self._fixed_kf) + 1

    
    @property
    def fixed_wavevector(self):
        """Wavevector [1/A] of the fixed beam"""
        return self.fixed_neutron.wavevector
    
    @fixed_wavevector.setter
    def fixed_wavevector(self, value):
        if value <= 0:
            raise ValueError(f'Requested value for new wavevector must be larger than zero. new_kf={value} <=0')

        self._fixed_neutron = Neutron(wavevector=value)

    @property
    def fixed_neutron(self):
        """`Neutron` representation of the fixed neutron beam, `ki` or `kf`.
        Check the `self.fixed_kf` flag to see which one it represents."""
        return self._fixed_neutron
    
    @property
    def kf_vert(self) -> bool:
        """Analyzer scatters beam vertically?"""
        return self._kf_vert
    
    @kf_vert.setter
    def kf_vert(self, value: bool):
        if not isinstance(value, bool):
            raise ValueError(f"{value!r} must be a `bool` type for `kf_vert")
        
        self._kf_vert = value

    @property
    def source(self):
        r"""Source of `TripleAxisInstrument`."""
        return self._source

    @source.setter
    def source(self, new_obj):
        """Allowed to set with `ComponentTemplate` instance or
        cofig dictionary which is passed to `ComponentTemplate.__init__
        """
        if isinstance(new_obj, ComponentTemplate):
            self._source = new_obj
        elif isinstance(new_obj, dict):
            self._source = ComponentTemplate(**new_obj)
        else:
            raise ValueError(f'New source must ba a valid instance of {ComponentTemplate.__name__!r}')
        

    @property
    def detector(self):
        """Detector of the `TripleAxisInstrument`."""
        return self._detector

    @detector.setter
    def detector(self, new_obj):
        """Allowed to set with `ComponentTemplate` instance or
        cofig dictionary which is passed to `ComponentTemplate.__init__
        """
        if isinstance(new_obj, ComponentTemplate):
            self._detector = new_obj
        elif isinstance(new_obj, dict):
            self._detector = ComponentTemplate(**new_obj)
        else:
            raise ValueError(f'New source must ba a valid instance of {ComponentTemplate.__name__!r}')

    
    ########################################################################################
    # Methods

    # MS: Hard replacement of the old function for quick testing
    def calc_resolution_QE(self, Q: np.ndarray, E: np.ndarray, method: str) -> tuple[float, np.ndarray, float]:
        """
        New resolution function as an interface to Tobi Weber python implementation.

        Parameters
        ----------
        Q : array_like 
            Momentum transfer amplitude [1/A]. 
        E : array_like
           Energy transfer.
        method: {'eck', 'pop'}
            Algorithm used to determine resolution parameters

        Returns
        -------
        R0: np.ndarray (Q.shape)
            Normalization factor. Doesn't correspond to the one in Takin.
        RE: np.ndarray (Q.shape+(4,4))
            Resolution ellipsoid variance matrix in [Q_para, Q_perp, Qz, E] coordinates.
        RV: np.ndarray (Q.shape)
            Volume of the resolution ellipsoid in [meV A^(-3)].

        In case of NaN values in `Q` or `E` the corresponding values in `R0`, `RE`, and `RV` are also NaN.

        Notes
        -----
        Calls Tobi Weber's python implementation of the algorithm.        
        [T. Weber mail 26.08.2024] Resolution ellipsoid is represented in terms of variances in Takin and res_py.
        """

        self.logger.debug('Calculating resolution in Q coordinates')

        if not Q.shape == E.shape:
            raise IndexError(f"Shape of the `Q` {Q.shape} and `E` {E.shape} matrices must be the same")

        cm2A = 1e8
        min2rad = 1./ 60. / 180.*np.pi

        # Prepare containers for the results
        R0 = np.full(Q.shape, np.nan)
        RE = np.full(Q.shape+(4,4), np.nan)
        RV = np.full(Q.shape, np.nan)

        for ind in np.ndindex(Q.shape):
            # Calculate angles and energies
            w = E[ind]
            q = Q[ind]
            ei = self.fixed_neutron.energy
            ef = self.fixed_neutron.energy

            if np.isnan([w,q]).any():
                self.logger.debug(f"Skipping index {ind} as it is NaN")
                continue

            if self.fixed_kf:
                ei = ei + w
            else:
                ef = ef - w

            neutron_i = Neutron(energy=ei)
            neutron_f = Neutron(energy=ef)
            ki = neutron_i.wavevector
            kf = neutron_f.wavevector

            # Go through mono focussing flags and determine if they are optimal
            mono_tth = self.mono.get_tth(neutron_i.wavelength)
            ana_tth = self.mono.get_tth(neutron_f.wavelength)
            if self.focussing['mono_h'] == "optimal":
                self.mono.set_opt_curvature(lenBefore=self.arms[0], lenAfter=self.arms[1], tth=mono_tth, vertical=False)
            if self.focussing['mono_v'] == "optimal":
                self.mono.set_opt_curvature(lenBefore=self.arms[0], lenAfter=self.arms[1], tth=mono_tth, vertical=True)
            if self.focussing['ana_h'] == "optimal":
                self.ana.set_opt_curvature(lenBefore=self.arms[2], lenAfter=self.arms[3], tth=ana_tth, vertical=False)
            if self.focussing['ana_v'] == "optimal":
                self.ana.set_opt_curvature(lenBefore=self.arms[2], lenAfter=self.arms[3], tth=ana_tth, vertical=True)

            # TODO Update sample dimension, i.e. update witdh wrt Q|| Q_perp

            self.logger.debug( 'Calcualting resolution at\n'+
                                f'\t\t ki={ki:.4f}, kf={kf:.4f}, q={q:.4f}, e={w:.4f}\n'+
                                f'\t\t Curvatures: mono_h={self.mono.curvr_h:.4f} cm, mono_v={self.mono.curvr_v:.4f} cm, ana_h={self.ana.curvr_h:.4f} cm, ana_v={self.ana.curvr_v:.4f} cm')           
            exp_parameters = {
                # options
                "verbose" : False,

                # scattering triangle
                "ki" : ki,
                "kf" : kf,
                "E"  : w,
                "Q"  : q,

                # d spacings
                "mono_xtal_d" : self.mono.dspacing,
                "ana_xtal_d"  : self.ana.dspacing,

                # scattering senses
                "mono_sense"   : self.scat_senses.mono,
                "sample_sense" : self.scat_senses.sample,
                "ana_sense"    : self.scat_senses.ana,
                "mirror_Qperp" : False,     # MS: unchanged, is that implemented in neutronpy?

                # distances in cm
                "dist_vsrc_mono"   : self.arms[0]*cm2A,
                "dist_hsrc_mono"   : self.arms[0]*cm2A, # MS: is that self.arms[4]?
                "dist_mono_sample" : self.arms[1]*cm2A,
                "dist_sample_ana"  : self.arms[2]*cm2A,
                "dist_ana_det"     : self.arms[3]*cm2A,

                # shapes
                "src_shape"    : self.source.shape,  # "rectangular" or "circular"
                "sample_shape" : self.sample.shape,  # "cuboid" or "cylindrical"
                "det_shape"    : self.detector.shape,   # "rectangular" or "circular"

                # component sizes
                #MS: I am not sure whether TW source and RP guide is the same idea
                "src_w"    : self.source.width   * cm2A,
                "src_h"    : self.source.height  * cm2A,    
                "mono_d"   : self.mono.depth  * cm2A,
                "mono_w"   : self.mono.width * cm2A,
                "mono_h"   : self.mono.height  * cm2A,
                "sample_d" : self.sample.width2   * cm2A,   # TODO these widths have to be adjusted for each A3
                "sample_w" : self.sample.width1   * cm2A,
                "sample_h" : self.sample.height   * cm2A,
                "ana_d"    : self.ana.depth  * cm2A,
                "ana_w"    : self.ana.width  * cm2A,
                "ana_h"    : self.ana.height  * cm2A,
                "det_w"    : self.detector.width  * cm2A,
                "det_h"    : self.detector.height  * cm2A,

                # horizontal collimation
                "coll_h_pre_mono"    : self.hcol[0] * min2rad,
                "coll_h_pre_sample"  : self.hcol[1] * min2rad,
                "coll_h_post_sample" : self.hcol[2] * min2rad,
                "coll_h_post_ana"    : self.hcol[3] * min2rad,

                # vertical collimation
                "coll_v_pre_mono"    : self.vcol[0] * min2rad,
                "coll_v_pre_sample"  : self.vcol[1] * min2rad,
                "coll_v_post_sample" : self.vcol[2] * min2rad,
                "coll_v_post_ana"    : self.vcol[3] * min2rad,

                # MS why would we need user-defined curvatures?
                # let's go vanilla and remove (==None) that option to do that automatically with standard lens formula
                # horizontal focusing
                "mono_curv_h" : self.mono.curvr_h * cm2A,
                "ana_curv_h"  : self.ana.curvr_h  * cm2A,
                "mono_is_curved_h" : self.focussing["mono_h"] in ["optimal", "curved"],
                "ana_is_curved_h"  : self.focussing["ana_h"] in ["optimal", "curved"],
                "mono_is_optimally_curved_h" : self.focussing["mono_h"]=="optimal",
                "ana_is_optimally_curved_h"  : self.focussing["ana_h"]=="optimal",
                "mono_curv_h_formula" : None,
                "ana_curv_h_formula" : None,

                # vertical focusing
                "mono_curv_v" : self.mono.curvr_v * cm2A,
                "ana_curv_v"  : self.ana.curvr_v  * cm2A,
                "mono_is_curved_v" : self.focussing["mono_v"] in ["optimal", "curved"],
                "ana_is_curved_v"  : self.focussing["ana_v"] in ["optimal", "curved"],
                "mono_is_optimally_curved_v" : self.focussing["mono_v"]=="optimal",
                "ana_is_optimally_curved_v"  :  self.focussing["ana_v"]=="optimal",
                "mono_curv_v_formula" : None,
                "ana_curv_v_formula" : None,

                # guide before monochromator
                "use_guide"   : self.source.use_guide,
                "guide_div_h" : self.source.div_h * min2rad,
                "guide_div_v" : self.source.div_v * min2rad,

                # horizontal mosaics
                "mono_mosaic"   : self.mono.mosaic   * min2rad,
                "sample_mosaic" : self.sample.mosaic * min2rad,
                "ana_mosaic"    : self.ana.mosaic    * min2rad,

                # vertical mosaics
                "mono_mosaic_v"   : self.mono.mosaic_v   * min2rad,
                "sample_mosaic_v" : self.sample.mosaic_v * min2rad,
                "ana_mosaic_v"    : self.ana.mosaic_v    * min2rad,

                # MS TODO BELOW?
                # calculate R0 factor (not needed if only the ellipses are to be plotted)
                "calc_R0" : True,

                # crystal reflectivities; TODO, so far always 1
                "dmono_refl" : 1.,
                "dana_effic" : 1.,

                # off-center scattering
                # WARNING: while this is calculated, it is not yet considered in the ellipse plots
                "pos_x" : 0. * cm2A,
                "pos_y" : 0. * cm2A,
                "pos_z" : 0. * cm2A,

                # vertical scattering in kf, keep "False" for normal TAS
                "kf_vert" : self.kf_vert,
            }

            if method=='eck':
                res = TW_calc_eck(exp_parameters)
            elif method=='pop':
                res = TW_calc_pop(exp_parameters)
            else:
                raise ValueError(f'{method!r} is a unrecognised method of resolution calculation. Try `eck` or `pop`.')

            # TODO officialize these corrections
            # 1. kf/ki into the prefactor
            cor1 = kf/ki

            R0[ind], RE[ind], RV[ind] = cor1*res['r0'], res['reso'], res['res_vol']

        return R0, RE, RV

    def calc_resolution_HKLE(self, hkle: np.ndarray, base: str, method: str="eck") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        r"""For a four-dimensional momentum(in `r.l.u`)-energy(in `meV`) transfer vector `hkle`,
        given the experimental conditions encoded in `self` instance determine
        the resolution ellipsoid utilizing the given `method`.

        Parameters
        ----------
        hkle : array_like (...,4)
            Array of the scattering vector and energy transfer at which the
            calculation should be performed in lattice coordinates.

        base: 'hkl', 'orient12', 'Q'
            Coordinate frame in which the resolution ellipsoid is calculated.
            `Q`: in the Q_parallel, Q_perp, Qz base, which refer to the scattering vector `hkle[:3]`
            `orient12`: in the base of orientation vectors
            `hkl`: in the crystal coordinate system, i.e. rescaled by lattice parameters.

        method: 'pop', 'eck'
            Algorithm used to determine the instrumental resolution.

        Returns
        -------
        R0 : (...)
            Normalization factor, doesn't correspond to the one in Takin.
        RE : (...,4,4)
            Resolution elipsoid in coordinates defined by `base`.
        RV : (...)
            Volume of the resolution elipsoid in [meV A^3].

        In case of any problems, such as the scattering triangle not closed, vector not in scattering plane,
        NaN values are returned.
            
        Notes
        -----
        Wrapper on top of the `calc_resolution_QE` function with change of base from
        [Q_par, Q_perp, Qz, E] to [h,k,l,E] for the input arguments.

        To deal with multidimensional `hkle` array there is a `for` loop going through it.
        Then, to change the basis the same `for` loop is called. 
        The question arises if this is efficient, and whether simply bounding the signature
        `hkl.shape==(4,)` is a better solution.
        However, with the main goal being resolution convolution, the heaviest 
        calculation is calling the `S(Q,w)` function within the region given
        by the ellipsoid. This shifts the bottleneck, so until a better argument is found 
        I keep it wit hthe for loops.
        """       
        if hkle.shape[-1] != 4:
            raise IndexError(f'{hkle.shape!r} is an invalid shape for the `hkle` array. Should be `(...,4)`')

        hkl = hkle[...,:3]
        E = hkle[...,3]
        Q = self.sample.get_Q(hkl)

        # Check that requested `hkle` can be reached and scattering triangle is closed
        An = self.get_angles(hkle)
        # Sum over An angles, to see if any of them is nan
        problems = np.sum( np.isnan(An), axis=-1)   
        Q[problems>0] = np.nan

        # MAIN CALCULATION
        self.logger.debug(f"Calculating resolution at Q points:", Q)
        R0, RE, RV = self.calc_resolution_QE(Q, E, method=method)

        # Apply requested base transformation
        if base=='orient12':
            # Resolution ellipsoid is in `Q` coordinates. The resolution ellipsoid
            # needs to be rotated in the `xy_lab` plane by an angle that requested `hkl`
            # makes with the first orientation vector.
            for ind in np.ndindex(E.shape):
                T = np.eye(4,4)
                Rz = sp_Rotation.from_euler('Z', self.sample.get_angle_to_o1(hkl[ind])).as_matrix()
                T[:3,:3] = Rz
                
                self.logger.debug(f"Transforming resolution ellipsoid from `Q` to `orient12` with matrix:\n {T}")
                RE[ind] = T.T @ RE[ind] @ T
        if base=='hkl':
            for ind in np.ndindex(E.shape):
                T = np.eye(4,4)

                # Same as above
                Rz = sp_Rotation.from_euler('Z', self.sample.get_angle_to_o1(hkl[ind])).as_matrix()
                # But now we also need to rotate and reshape it wrt the crystal coordinate system.
                UB = self.sample.UBmatrix

                T[:3,:3] = Rz @ UB
                self.logger.debug(f"Transforming resolution ellipsoid from `Q` to `hkl` with matrix:\n {T}")
                RE[ind] = T.T @ RE[ind] @ T

        return R0, RE, RV
    
    def get_angles(self, hkle: np.ndarray) -> np.ndarray:
        r"""Returns the Triple Axis Spectrometer angles for chosen momentum and energy transfer.
        In case of problems, warning is issued and according values are set to NaN.

        Parameters
        ----------
        hkle : arraylike (...,4)
            Array of the scattering vector and energy transfer at which the
            calculation should be performed.

        Returns
        -------
        A : arraylike (...,6)
            The angles A1, A2, A3, A4, A5, A6 [deg]. In case of any error, nan values are returned.

        Warnings
        --------
        VectorNotInScatteringPlane, ScatteringTriangleNotClosed, MonochromatorError, AnalyzerError

        Notes
        -----
        1. A1, A2 are a function of ki
        2. A5, A6 are a function of kf
        3. A4 is from the scattering triangle Q = kf-ki
        4. A3
          1. Assuming `orient1` defines A3=0
          2. Look where `hkl` is, as an angle from `orient1`
          3. Put `hkl` to the position of A3 where Q=kf-ki
        """

        # Extract some variables for easier use
        E = hkle[...,3]
        hkl = hkle[...,:3]

        # Kinematic equations
        kfix = self.fixed_neutron.wavevector
        delta_Q = Neutron(energy=np.abs(E)).wavevector
        kisq = kfix**2 + int(self.fixed_kf) * delta_Q**2
        kfsq = kfix**2 + (int(self.fixed_kf) - 1) * delta_Q**2

        # Can we the requested energy with ki and kf?
        ki_neg = kisq<0
        if np.any(ki_neg):
            warn( MonochromatorError(f'Requested energy too low to close the triangle') )

        kf_neg = kfsq<0
        if np.any(kf_neg):
            warn( AnalyzerError(f'Requested energy too low to close the triangle') )
        
        ki = np.sqrt(kisq)
        kf = np.sqrt(kfsq)


        # A1, A2 from ki
        A2 = self.scat_senses.mono * self.mono.get_tth(wavelength=2*np.pi/ki)
        A1 = 0.5*A2

        # A5, A6 from kf
        A6 = self.scat_senses.ana * self.ana.get_tth(wavelength=2*np.pi/kf)
        A5 = 0.5*A6

        # A4 from the scattering triangle
        Q = self.sample.get_Q(hkl)
        triangle_cos = (ki**2 + kf**2 - Q**2) / (2*ki*kf)

        # Check if the scattering triangle can be closed
        invalid_st_1 = np.abs(triangle_cos) > 1
        invalid_st_2 = (triangle_cos == np.nan)
        if np.sum(invalid_st_1):
            # entries_nan[invalid_st_1] = [False, False, True, True, False, False]
            warn( ScatteringTriangleNotClosed(f"Can't close the scattering triangle at requested hkle: \n{hkle[invalid_st_1]}") )
        if np.sum(invalid_st_2):
            # entries_nan[invalid_st_2] = [False, False, True, True, False, False]
            warn( ScatteringTriangleNotClosed(f"Can't close the scattering triangle at requested hkle: \n{hkle[invalid_st_2]}") )
        
        # Here nan values are gonna be returned in case of problems.
        A4 = self.scat_senses.sample * np.arccos(triangle_cos)

        # A3 is more complicated
        # `Q_angle` is the angle between momentum transfer vector and ki that closes
        # the scattering triangle. This is where the requested `hkl` will need to be rotated.
        # Scattering sense is already encoded in A4 and propagates through sine
        Q_angle = np.arctan2( -kf*np.sin(A4), ki-kf*np.cos(A4))
        hkl_angle = self.sample.get_angle_between_Qs(self.orient1, hkl)
        # Now these two angle have to be added. We assume RHS theta motor.
        A3 = hkl_angle + Q_angle + np.radians(self.a3_offset)


        not_in_scattering_plane =  ~self.sample.is_in_scattering_plane(hkl)
        if np.any( not_in_scattering_plane ):
            A3[not_in_scattering_plane] = np.nan
            warn( VectorNotInScatteringPlane(f'Vector not in scattering plane: \n{hkle[not_in_scattering_plane]}') )

        return np.degrees([A1, A2, A3, A4, A5, A6]).T

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
            determine the convolved intensity.
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
        # DEV NOTES
        # - [X] the methods of calculating the resolution ellipsoid have different deifnitions for them.
        #       Eckold-Sobolev say `resolution ellipsoid is a surface in Q-E where the probability
        #       takes value of 0.5`, i.e. resolution elipsoid is defined by its FWHM.
        #       On the other hand, Popovici writes `M is the resolution matrix, the inverse 
        #       of which is the covariance matrix. the covariance matrix is defined by
        #       variances and cross corellations, thus fundamentally different from FWHM.
        #       I need to find some with the unifying solution, or believe the solution is unified by Tobi already.
        #       [T. Weber mail 26.08.2024] Yes it in terms of variances in Takin and res_py


        R0, RE, RV = self.calc_resolution_HKLE(hkle, 'hkl', method)
        Sqw = np.zeros(hkle[...,0].shape)

        rnd_generator = np.random.default_rng(seed=seed)
        # Going through each point in the grid
        for ind in np.ndindex(hkle[...,0].shape):
            hkle0 = hkle[ind]
            covariance = np.linalg.inv(RE[ind])
            ellipsoid_samples = rnd_generator.multivariate_normal(mean=hkle0, cov=covariance, size=Nsamples)

            # Average the value of Sqw over the resolution ellipsoid region
            # This is the heaviest calculations
            Sqw[ind] = np.average(Sqw_fast(ellipsoid_samples, Sqw_pars))

        # After getting convolved values there are the prefactor and resolution ellipsoid volume factor to correct,
        # and finally the addition of the slow-varying part of Sqw
        Sqw *= Sqw_slow(hkle, Sqw_pars) * R0 * RV

        return Sqw
    
    def resolution_convolution_single_mode(self,
                                           hkle: np.ndarray, 
                                           Sqw_fast: Callable[[np.ndarray, Any], np.ndarray], 
                                           Sqw_slow: Callable[[np.ndarray, Any], np.ndarray], 
                                           Sqw_pars: Any, 
                                           method = str, 
                                           Nsamples=1000, seed=None):
        
        return