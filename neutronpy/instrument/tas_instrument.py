# -*- coding: utf-8 -*-
r"""Define an instrument for resolution calculations

"""
from collections import namedtuple
from typing import Any, Callable
import numpy as np
from scipy.spatial.transform.rotation import Rotation as sp_Rotation

from neutronpy.fileio.instrument import TAS_loader

from ..crystal import Sample
from ..neutron import Neutron
from .exceptions import ScatteringTriangleError, InstrumentError
from .general import GeneralInstrument
from .analyzer import Analyzer
from .monochromator import Monochromator
from .plot import PlotInstrument
from .tools import _CleanArgs, ComponentTemplate, _modvec, _scalar, _star, _voigt

from .TW_respy.pop import calc as TW_calc_pop
from .TW_respy.eck import calc as TW_calc_eck

from ..loggers import setup_TAS_logger

ScatteringSenses = namedtuple('ScatteringSenses', 'mono, sample, ana')

# DEV NOTES TODO
# - [ ] Since np.linalg is usually operating on the last dimension, I will od the same.
#       This changes from `h=hkle[0,...] ` to h=hkle[...,0].
# - [X] Main functionalities operate based on methods with the `hkle` argument.
#       I need to ensure this array has proper shape, the hkl vectors are
#       in the scattering plane and maybe some other checks.
#       How about writing a decorator for the function, that will performe required checks?
#       RESULT: the validators are methods of other classes. Using them requires pusing those
#       as classmethods which will not work for `is_in scattering_plane` as it needs
#       instance of `Sample.` Validate the arguments in the function call :(
# - [ ] I want to be able to call `calc_resolution(hkle)` hkle.shape=(...,N).
#       So where should I unpack the `...` dimensions?
# - [ ] Apparently `np.einsum` can be slow for arrays with high dimensions and
#       long list of arrays involved in summation. There are optimization options 
#       for it that should be looked into:
#        - https://numpy.org/doc/stable/reference/generated/numpy.einsum_path.html#numpy.einsum_path
#        - just use tensordot?

class TripleAxisInstrument(GeneralInstrument, PlotInstrument):
    u"""An object that represents a Triple Axis Spectrometer (TAS) instrument
    experimental configuration, including a sample.

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

    def __init__(self, fixed_kf: bool, fixed_wavevector: float, name: str,
                 scat_senses: tuple[int,int,int], a3_offset: float,
                 arms: list[int], hcol: list[int], vcol: list[int], 
                 focussing: dict[str,str], sample: Sample, 
                 mono: Monochromator, ana: Analyzer,
                 source: ComponentTemplate, detector: ComponentTemplate):

        self.logger = setup_TAS_logger()
        self.logger.debug(f'Creating `TAS` instance: name={name}')

        # Operational mode
        self.fixed_kf = fixed_kf
        self.fixed_wavevector = fixed_wavevector

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
        """Construct `TripleAxisSpectrometer` instance with default values."""
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
                 scat_senses=(-1,+1,-1), a3_offset=0,
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
        ret += f"\t scat_senses={self.scat_senses}, arms={self.arms},\n"
        ret += f"\t hcol={self.hcol}, vcol={self.vcol},\n"
        ret += f"\t focussing={self.focussing},\n"
        ret += f"\t sample={self.sample},\n"
        ret += f"\t mono={self.mono},\n"
        ret += f"\t ana={self.ana},\n"
        ret += f"\t source={self.source},\n"
        ret += f"\t detector={self.detector}\n"
        ret += ">"
        return ret

    def __eq__(self, right):
        self_parent_keys = sorted(list(self.__dict__.keys()))
        right_parent_keys = sorted(list(right.__dict__.keys()))

        if not np.all(self_parent_keys == right_parent_keys):
            return False

        for key, value in self.__dict__.items():
            right_parent_val = getattr(right, key)
            if not np.all(value == right_parent_val):
                print(value, right_parent_val)
                return False

        return True

    def __ne__(self, right):
        return not self.__eq__(right)

    ###############################################################################
    # TODO clean up the properties
    @property
    def mono(self) -> Monochromator:
        u"""`Monochromator` of the `TrieplAxisInsturment`."""
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
        "Analyzer of the spectrometer"
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
    def moncor(self):
        """Selects the type of normalization used to calculate ``R0``
        If ``moncor=1`` or left undefined, ``R0`` is calculated in
        normalization to monitor counts (Section II C 2). 1/k\ :sub:`i` monitor
        efficiency correction is included automatically. To normalize ``R0`` to
        source flux (Section II C 1), use ``moncor=0``.
        """
        return self._moncar

    @moncor.setter
    def moncor(self, value):
        self._moncar = value

    @property
    def hcol(self):
        r""" The horizontal Soller collimations in minutes of arc (FWHM beam
        divergence) starting from the in-pile collimator. In case of a
        horizontally-focusing analyzer ``hcol[2]`` is the angular size of the
        analyzer, as seen from the sample position. If the beam divergence is
        limited by a neutron guide, the corresponding element of :attr:`hcol`
        is the negative of the guide’s *m*-value. For example, for a 58-Ni
        guide ( *m* = 1.2 ) before the monochromator, ``hcol[0]`` should be
        -1.2.
        """
        return self._hcol

    @hcol.setter
    def hcol(self, value):
        self._hcol = value

    @property
    def vcol(self):
        """The vertical Soller collimations in minutes of arc (FWHM beam
        divergence) starting from the in-pile collimator. If the beam
        divergence is limited by a neutron guide, the corresponding element of
        :attr:`vcol` is the negative of the guide’s *m*-value. For example, for
        a 58-Ni guide ( *m* = 1.2 ) before the monochromator, ``vcol[0]``
        should be -1.2.
        """
        return self._vcol

    @vcol.setter
    def vcol(self, value):
        self._vcol = value

    @property
    def arms(self):
        """distances between the source and monochromator, monochromator
        and sample, sample and analyzer, analyzer and detector, and
        monochromator and monitor, respectively. The 5th element is only needed
        if ``moncor=1``
        """
        return self._arms

    @arms.setter
    def arms(self, value):
        self._arms = value


    @property
    def sample(self) -> Sample:
        """Sample measured at the spectrometer."""
        return self._sample

    @sample.setter
    def sample(self, new_sample):
        """Allowed to set with `Sample` instance or
        cofig dictionary which is passed to `Sample.__init__`.
        """
        if isinstance(new_sample, Sample):
            self._sample = new_sample
        elif isinstance(new_sample, dict):
            self._sample = Sample(**new_sample)
        else:
            raise ValueError(f'New sample must ba a valid instance of {Sample.__name__!r}, or dictionary.')
        

    @property
    def orient1(self):
        """Miller indexes of the first reciprocal-space orienting vector for
        the S coordinate system, as explained in Section II G.
        """
        return self._sample.orient1

    @orient1.setter
    def orient1(self, value):
        self._sample.orient1 = np.array(value)

    @property
    def orient2(self):
        """Miller indexes of the second reciprocal-space orienting vector
        for the S coordinate system, as explained in Section II G.
        """
        return self._sample.orient2

    @orient2.setter
    def orient2(self, value):
        self._sample.orient2 = np.array(value)

    @property
    def a3_offset(self):
        """Offset in A3 = sample rotation = sample theta [rad.]"""
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
    def infin(self):
        """a flag set to -1 or left unassigned if the final energy is fixed, or
        set to +1 in a fixed-incident setup.
        """
        # fx is 1 2 ki kf
        return -2*self.fx + 3
    
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
        """Properties of the fixed neutron beam, ki/kf, check the `self.fixed_kf` flag."""
        return self._fixed_neutron

    @property
    def source(self):
        r"""A structure that describes the source
        """
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
        """A structure that describes the detector
        """
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
        

    @property
    def Smooth(self):
        u"""Defines the smoothing parameters as explained in Section II H. Leave this
        field unassigned if you don’t want this correction done.

        * ``Smooth.E`` is the smoothing FWHM in energy (meV). A small number
          means “no smoothing along this direction”.

        * ``Smooth.X`` is the smoothing FWHM along the first orienting vector
          (x0 axis) in Å\ :sup:`-1`.

        * ``Smooth.Y`` is the smoothing FWHM along the y axis in Å\ :sup:`-1`.

        * ``Smooth.Z`` is the smoothing FWHM along the vertical direction in
          Å\ :sup:`-1`.

        """
        return self._Smooth

    @Smooth.setter
    def Smooth(self, value):
        self._Smooth = value

    # Depracated
    @property
    def efixed(self):
        return self.fixed_neutron.energy
    
    ########################################################################################
    # Methods

    # TODO remove
    def get_lattice(self):
        r"""Extracts lattice parameters from EXP and returns the direct and
        reciprocal lattice parameters in the form used by _scalar.m, _star.m,
        etc.

        Returns
        -------
        [lattice, rlattice] : [class, class]
            Returns the direct and reciprocal lattice sample classes

        Notes
        -----
        Translated from ResLib 3.4c, originally authored by A. Zheludev,
        1999-2007, Oak Ridge National Laboratory

        """
        lattice = Sample(self.sample.a,
                         self.sample.b,
                         self.sample.c,
                         np.deg2rad(self.sample.alpha),
                         np.deg2rad(self.sample.beta),
                         np.deg2rad(self.sample.gamma))
        rlattice = _star(lattice)[-1]

        return [lattice, rlattice]
    
    # TODO remove
    def _StandardSystem(self):
        """Reimplementation of the depracated version."""

        # U: (kx,ky,kz)_sample -> (kx,ky,kz)_lab
        # Need a reverse transform then.
        # I think theses `xyz` vectors are supposed to be orienting 
        # vectors in nominal lab system, so we also need transposition.
        x, y, z = np.linalg.inv(self.sample.Umatrix).T

        return x, y, z

    def _StandardSystem_depr(self):
        r"""DEPRACATED
        Returns rotation matrices to calculate resolution in the sample view
        instead of the instrument view

        Attributes
        ----------
        EXP : class
            Instrument class

        Returns
        -------
        [x, y, z, lattice, rlattice] : [array, array, array, class, class]
            Returns the rotation matrices and real and reciprocal lattice
            sample classes

        Notes
        -----
        Translated from ResLib 3.4c, originally authored by A. Zheludev,
        1999-2007, Oak Ridge National Laboratory

        """
        [lattice, rlattice] = self.get_lattice()

        orient1 = self.orient1
        orient2 = self.orient2

        modx = _modvec(orient1, rlattice)

        # MS: x is normalized first orientation vector
        x = orient1 / modx  # MS: this mixes the orient1_hkl and its length in xyz

        proj = _scalar(orient2, x, rlattice)

        y = orient2 - x * proj

        mody = _modvec(y, rlattice)

        if len(np.where(mody <= 0)[0]) > 0: # MS ??? mody is length of y vector. It is scalar and always positive
            raise ScatteringTriangleError('Orienting vectors are colinear')

        y /= mody

        # MS: this is np.cross(x, y)
        z = np.array([ x[1] * y[2] - y[1] * x[2],
                       x[2] * y[0] - y[2] * x[0],
                      -x[1] * y[0] + y[1] * x[0]], dtype=np.float64)

        proj = _scalar(z, x, rlattice)

        z -= x * proj

        proj = _scalar(z, y, rlattice)

        z -= y * proj

        modz = _modvec(z, rlattice)

        z /= modz

        return [x, y, z, lattice, rlattice]

    # MS: Hard replacement of the old function for quick testing
    def calc_resolution_QE(self, Q: np.ndarray, E: np.ndarray) -> tuple[float, np.ndarray, float]:
        """
        New resolution function as an interface to Tobi Weber python implementation.

        Parameters
        ----------
        Q : array_like 
            Momentum transfer amplitude [1/A]. 
        E : array_like
           Energy transfer.

        Notes
        -----
        Calls Tobi Weber's python implementation of the algorithm.
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
            self.logger.debug(f"index {ind}")

            # Calculate angles and energies
            w = E[ind]
            q = Q[ind]
            ei = self.efixed
            ef = self.efixed

            # TODO change to kfixed
            infin = self.infin
            if infin > 0:
                ef = self.efixed - w
            else:
                ei = self.efixed + w

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

                # resolution method, "eck", "pop", or "cn"
                "reso_method" : "pop",

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
                "kf_vert" : False,
            }

            res = TW_calc_eck(exp_parameters)

            # TODO officialize these corrections
            # 1. kf/ki into the prefactor
            cor1 = kf/ki

            R0[ind], RE[ind], RV[ind] = cor1*res['r0'], res['reso'], res['res_vol']

        return R0, RE, RV

    def calc_resolution(self, hkle: np.ndarray, base: str='Q', algorithm: str="eck") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        r"""For a four-dimensional momentum(in `r.l.u`)-energy(in `meV`) transfer vector `hkle`,
        given the experimental conditions encoded in `self` instance determine
        measurement resolution utilizing the `algorithm` method.

        Parameters
        ----------
        hkle : array_like (...,4)
            Array of the scattering vector and energy transfer at which the
            calculation should be performed.

        base: 'hkl', 'orient12', 'Q'
            Coordinate frame in which the resolution ellipsoid is calculated.
            `hkl`: in the crystal coordinate system
            `orient12`: in the base of orientation vectors
            `Q`: in the Q_parallel, Q_perp, Qz base, which refer to the scattering vector `hkle[:3]`

        algorithm: 'pop', 'eck'
            Algorithm used to determine the instrumental resolution.

        Returns
        -------
        R0 : (...)
            Normalization factor, doesn't correspond to the one in Takin.
        RE : (...,4,4)
            Resolution elipsoid in coordinates defined by `base`.
        RV : (...)
            Volume of the resolution elipsoid in [meV A^3].


        Notes
        -----
        Wrapper on top of the `calc_resolution_QE` function with change of base from
        [Q_par, Q_perp, Qz, E] to [h,k,l,E].

        To deal with multidimensional `hkle` array there is a `for` loop goiing through it.
        Then, to change the basis the same `for` loop is called. 
        The question arises if this is efficient, and whether simply bounding the signature
        `hkl.shape==(4,)` is a better solution.
        That would be supported by the idea that the main goal is resolution convolution.
        This requires determining resolution ellipsoids for each measured point and calling
        `S(Q,w)` many time within the region of the ellipsoid. The latter must be fast,
        and implemented on `np.array` but maybe not the resolution calculation at all.
        """
        # Check that hkl is in the scattering plane
        in_scattering_plane = self.sample.is_in_scattering_plane(hkle[...,:3])
        if np.sum( ~in_scattering_plane):
            raise InstrumentError(f'Vector not in scattering plane:\n{hkle[...,:3][in_scattering_plane]}')
        
        # Check that requested `hkle` can be reached and scattering triangle is closed
        try:
            _ = self.get_angles(hkle)
        except Exception as e:
            raise e
        
        if hkle.shape[-1] != 4:
            raise IndexError(f'{hkle.shape!r} is an invalid shape for the `hkle` array. Should be `(...,4)`')

        hkl = hkle[...,:3]
        E = hkle[...,3]
        Q = self.sample.get_Q(hkl)

        R0, RE, RV = self.calc_resolution_QE(Q, E)

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

        # This smoothing originates from ResLib sect. II H, pp13, Eq 15 and 16.
        # In principle it applies to a situation when the experimental data was smoothed to 
        # reduce the number of points on which the resolution is calculated.
        # I don't see how this applies here yet
        # e = np.identity(4)
        # for i in range(length):
        #     if hasattr(self, 'Smooth'):
        #         if self.Smooth.X:
        #             mul = np.diag([1 / (self.Smooth.X ** 2 / 8 / np.log(2)),
        #                            1 / (self.Smooth.Y ** 2 / 8 / np.log(2)),
        #                            1 / (self.Smooth.E ** 2 / 8 / np.log(2)),
        #                            1 / (self.Smooth.Z ** 2 / 8 / np.log(2))])
        #             R0[i] = R0[i] / np.sqrt(np.linalg.det(np.matrix(e) / np.matrix(RMS[i]))) * np.sqrt(
        #                 np.linalg.det(np.matrix(e) / np.matrix(mul) + np.matrix(e) / np.matrix(RMS[i])))
        #             RMS[i] = np.matrix(e) / (
        #                 np.matrix(e) / np.matrix(mul) + np.matrix(e) / np.matrix(RMS[i]))

        return R0, RE, RV
    
    def get_angles(self, hkle: np.ndarray) -> np.ndarray:
        r"""Returns the Triple Axis Spectrometer angles for chosen momentum and energy transfer.

        Parameters
        ----------
        hkle : arraylike (...,4)
            Array of the scattering vector and energy transfer at which the
            calculation should be performed.

        Returns
        -------
        A : arraylike (...,6)
            The angles A1, A2, A3, A4, A5, A6 [deg].


        Notes
        -----
        1. Checks:
          1. requested hkl in scattering plane
          2. Scattering triangle can be closed
        2. A1, A2 are a function of ki
        3. A5, A6 are a function of kf
        4. A4 is from the scattering triangle Q = kf-ki
        5. A3
          1. Assuming `orient1` defines A3=0
          2. Look where `hkl` is, as an angle from `orient1`
          3. Put `hkl` to the position of A3 where Q=kf-ki
        """

        # Extract some variables for easier use
        E = hkle[...,3]
        hkl = hkle[...,:3]

        self.logger.debug(self.sample.is_in_scattering_plane(hkl))

        if np.sum( ~self.sample.is_in_scattering_plane(hkl)):
            raise InstrumentError(f'Vector not in scattering plane')

        # Kinematic equations
        kfix = self.fixed_neutron.wavevector
        delta_Q = self.fixed_neutron._wavevector_from_energy(energy=E)   # Use the beam just to call the conversion function
        kisq = kfix**2 + int(self.fx - 1) * delta_Q**2
        kfsq = kfix**2 - int(2 - self.fx) * delta_Q**2
        if np.sum(kisq < 0):
            raise ScatteringTriangleError(f'Requested energy too low to close the triangle')
        if np.sum(kfsq < 0):
            raise ScatteringTriangleError(f'Requested energy too high to close the triangle')
        
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
        triangle_cos = (ki ** 2 + kf ** 2 - Q ** 2) / (2 * ki * kf)

        # Check if the scattering triangle can be closed
        invalid_st_1 = np.abs(triangle_cos) > 1
        invalid_st_2 = (triangle_cos == np.nan)
        if np.sum(invalid_st_1):
            raise ScatteringTriangleError(f"Can't close the scattering triangle at requested hkle: \n{hkle[invalid_st_1]}")
        if np.sum(invalid_st_2):
            raise ScatteringTriangleError(f"Can't close the scattering triangle at requested hkle: \n{hkle[invalid_st_2]}")
        
        A4 = self.scat_senses.sample * np.arccos(triangle_cos)

        # A3 is more complicated
        # `Q_angle` is the angle between momentum transfer vector and ki that closes
        # the scattering triangle. This is where the requested `hkl` will need to be rotated.
        # Scattering sense is already encoded in A4 and propagates through sine
        Q_angle = np.arctan2( -kf*np.sin(A4), ki-kf*np.cos(A4))
        hkl_angle = self.sample.get_angle_to_o1(hkl)
        # Now these two angle have to be added. We assum RHS theta motor.
        A3 = hkl_angle + Q_angle + np.radians(self.a3_offset)

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
        #       I need to some with the unifying solution, or believe the solution is unified by Tobi already.
        #       [T. Weber mail 26.08.2024] Yes it in terms of variances in Takin and res_py


        R0, RE, RV = self.calc_resolution(hkle, 'hkl', method)
        Sqw = np.zeros(hkle[...,0].shape)

        rnd_generator = np.random.default_rng(seed=seed)
        for ind in np.ndindex(hkle[...,0].shape):
            hkle0 = hkle[ind]
            covariance = np.linalg.inv(RE[ind])
            ellipsoid_samples = rnd_generator.multivariate_normal(mean=hkle0, cov=covariance, size=Nsamples)

            # Average the value of Sqw over the resolutio nellipsoid region
            Sqw[ind] = np.average(Sqw_fast(ellipsoid_samples, Sqw_pars))

        # After getting convolved values there are the prefactor and resolution ellipsoid volume factor to correct,
        # and finally the addition of the slow-varying part of Sqw
        Sqw *= Sqw_slow(hkle, Sqw_pars) * R0 * RV

        return Sqw

    # DEV notes
    # this funciton is horrifying. Too many and repeating formulas.
    # FIX THE ESIGNATURE OF THE sqw FUNCITON!!!
    def resolution_convolution_old(self, sqw, pref, nargout, hkle, METHOD='fix', ACCURACY=None, p=None, seed=None):
        r"""Numerically calculate the convolution of a user-defined
        cross-section function with the resolution function for a
        3-axis neutron scattering experiment.

        Parameters
        ----------
        sqw : func, signature: `sqw(H1, K1, L1, W1, p)`
            User-supplied "fast" model cross section.

        pref : func
            User-supplied "slow" cross section prefactor and background
            function.

        nargout : int
            Number of arguments returned by the pref function

        hkle : tup
            Tuple of H, K, L, and W, specifying the wave vector and energy
            transfers at which the convolution is to be calculated (i.e.
            define $\mathbf{Q}_0$). H, K, and L are given in reciprocal
            lattice units and W in meV.

        EXP : obj
            Instrument object containing all information on experimental setup.

        METHOD : str
            Specifies which 4D-integration method to use. 'fix' (Default):
            sample the cross section on a fixed grid of points uniformly
            distributed $\phi$-space. 2*ACCURACY[0]+1 points are sampled
            along $\phi_1$, $\phi_2$, and $\phi_3$, and 2*ACCURACY[1]+1
            along $\phi_4$ (vertical direction). 'mc': 4D Monte Carlo
            integration. The cross section is sampled in 1000*ACCURACY
            randomly chosen points, uniformly distributed in $\phi$-space.

        ACCURACY : array(2) or int
            Determines the number of sampling points in the integration.

        p : list
            A parameter that is passed on, without change to sqw and pref.

        Returns
        -------
        conv : array
            Calculated value of the cross section, folded with the resolution
            function at the given $\mathbf{Q}_0$

        Notes
        -----
        Translated from ResLib 3.4c, originally authored by A. Zheludev,
        1999-2007, Oak Ridge National Laboratory

        """
        R0, RMS, RV = self.calc_resolution(hkle, base='hkl')

        H, K, L, W = hkle.T
        [length, H, K, L, W] = _CleanArgs(H, K, L, W)
        [xvec, yvec, zvec] = self._StandardSystem()[:3]

        Mxx = RMS[:, 0, 0]
        Mxy = RMS[:, 0, 1]
        Mxw = RMS[:, 0, 3]
        Myy = RMS[:, 1, 1]
        Myw = RMS[:, 1, 3]
        Mzz = RMS[:, 2, 2]
        Mww = RMS[:, 3, 3]

        Mxx -= Mxw ** 2. / Mww
        Mxy -= Mxw * Myw / Mww
        Myy -= Myw ** 2. / Mww
        MMxx = Mxx - Mxy ** 2. / Myy

        detM = MMxx * Myy * Mzz * Mww

        tqz = 1. / np.sqrt(Mzz)
        tqx = 1. / np.sqrt(MMxx)
        tqyy = 1. / np.sqrt(Myy)
        tqyx = -Mxy / Myy / np.sqrt(MMxx)
        tqww = 1. / np.sqrt(Mww)
        tqwy = -Myw / Mww / np.sqrt(Myy)
        tqwx = -(Mxw / Mww - Myw / Mww * Mxy / Myy) / np.sqrt(MMxx)

        inte = sqw(hkle, p)
        [modes, points] = inte.shape

        if pref is None:
            prefactor = np.ones((modes, points))
            bgr = 0
        else:
            if nargout == 2:
                [prefactor, bgr] = pref(hkle, p)
            elif nargout == 1:
                prefactor = pref(hkle, p)
                bgr = 0
            else:
                raise ValueError('Invalid number or output arguments in prefactor function')


        # See ResLib manual eq 20 and description around it.
        # DEVNOTES
        # 1. So ResLib uses this tangent mapping, since the uniform mapping there will result in Lorentzian
        #    mapping in QE space, which is good approximation of the Gaussian. 
        #    But with the possibilities of numpy, can't we directly probe the gaussian distribution somehow?
        if METHOD == 'fix':
            if ACCURACY is None:
                ACCURACY = np.array([7, 0])
            else:
                assert len(ACCURACY)==2
            M = ACCURACY
            step1 = np.pi / (2 * M[0] + 1)
            step2 = np.pi / (2 * M[1] + 1)
            dd1 = np.linspace(-np.pi / 2 + step1 / 2, np.pi / 2 - step1 / 2, (2 * M[0] + 1))
            dd2 = np.linspace(-np.pi / 2 + step2 / 2, np.pi / 2 - step2 / 2, (2 * M[1] + 1))
            convs = np.zeros((modes, length))
            conv = np.zeros(length)
            [cw, cx, cy] = np.meshgrid(dd1, dd1, dd1, indexing='ij')
            tx = np.tan(cx)
            ty = np.tan(cy)
            tw = np.tan(cw)
            tz = np.tan(dd2)
            norm = np.exp(-0.5 * (tx ** 2 + ty ** 2)) * (1 + tx ** 2) * (1 + ty ** 2) * np.exp(-0.5 * (tw ** 2)) * (
                1 + tw ** 2)
            normz = np.exp(-0.5 * (tz ** 2)) * (1 + tz ** 2)

            for iz in range(len(tz)):
                for i in range(length):
                    dQ1 = tqx[i] * tx
                    dQ2 = tqyy[i] * ty + tqyx[i] * tx
                    dW = tqwx[i] * tx + tqwy[i] * ty + tqww[i] * tw
                    dQ4 = tqz[i] * tz[iz]
                    H1 = H[i] + dQ1 * xvec[0] + dQ2 * yvec[0] + dQ4 * zvec[0]
                    K1 = K[i] + dQ1 * xvec[1] + dQ2 * yvec[1] + dQ4 * zvec[1]
                    L1 = L[i] + dQ1 * xvec[2] + dQ2 * yvec[2] + dQ4 * zvec[2]
                    W1 = W[i] + dW
                    hkle1 = np.transpose([H1,K1,L1,W1])
                    inte = sqw(hkle1, p)
                    for j in range(modes):
                        add = inte[j, :] * norm * normz[iz]
                        convs[j, i] = convs[j, i] + np.sum(add)

                    conv[i] = np.sum(convs[:, i] * prefactor[:, i])

            conv = conv * step1 ** 3 * step2 / np.sqrt(detM)
            if M[1] == 0:
                conv *= 0.79788
            if M[0] == 0:
                conv *= 0.79788 ** 3

        elif METHOD == 'mc':
            if isinstance(ACCURACY, (list, np.ndarray, tuple)):
                if len(ACCURACY) == 1:
                    ACCURACY = ACCURACY[0]
                else:
                    raise ValueError('ACCURACY must be an int when using Monte Carlo method: {0}'.format(ACCURACY))
            if ACCURACY is None:
                ACCURACY = 10
            M = ACCURACY
            convs = np.zeros((modes, length))
            conv = np.zeros(length)
            for i in range(length):
                for MonteCarlo in range(M):
                    if seed is not None:
                        np.random.seed(seed)
                    r = np.random.randn(4, 1000) * np.pi - np.pi / 2
                    cx = r[0, :]
                    cy = r[1, :]
                    cz = r[2, :]
                    cw = r[3, :]
                    tx = np.tan(cx)
                    ty = np.tan(cy)
                    tz = np.tan(cz)
                    tw = np.tan(cw)
                    norm = np.exp(-0.5 * (tx ** 2 + ty ** 2 + tz ** 2 + tw ** 2)) * (1 + tx ** 2) * (1 + ty ** 2) * (
                        1 + tz ** 2) * (1 + tw ** 2)
                    dQ1 = tqx[i] * tx
                    dQ2 = tqyy[i] * ty + tqyx[i] * tx
                    dW = tqwx[i] * tx + tqwy[i] * ty + tqww[i] * tw
                    dQ4 = tqz[i] * tz
                    H1 = H[i] + dQ1 * xvec[0] + dQ2 * yvec[0] + dQ4 * zvec[0]
                    K1 = K[i] + dQ1 * xvec[1] + dQ2 * yvec[1] + dQ4 * zvec[1]
                    L1 = L[i] + dQ1 * xvec[2] + dQ2 * yvec[2] + dQ4 * zvec[2]
                    W1 = W[i] + dW
                    hkle1 = np.transpose([H1,K1,L1,W1])
                    inte = sqw(hkle1, p)
                    for j in range(modes):
                        add = inte[j, :] * norm
                        convs[j, i] = convs[j, i] + np.sum(add)

                    conv[i] = np.sum(convs[:, i] * prefactor[:, i])

            conv = conv / M / 1000 * np.pi ** 4. / np.sqrt(detM)

        else:
            raise ValueError('Unknown METHOD: {0}. Valid options are: "fix",  "mc"'.format(METHOD))

        conv *= R0
        conv += bgr

        return conv

    def resolution_convolution_SMA(self, sqw, pref, nargout, hkle, METHOD='fix', ACCURACY=None, p=None, seed=None):
        r"""Numerically calculate the convolution of a user-defined single-mode
        cross-section function with the resolution function for a 3-axis
        neutron scattering experiment.

        Parameters
        ----------
        sqw : func
            User-supplied "fast" model cross section.

        pref : func
            User-supplied "slow" cross section prefactor and background
            function.

        nargout : int
            Number of arguments returned by the pref function

        hkle : tup
            Tuple of H, K, L, and W, specifying the wave vector and energy
            transfers at which the convolution is to be calculated (i.e.
            define $\mathbf{Q}_0$). H, K, and L are given in reciprocal
            lattice units and W in meV.

        EXP : obj
            Instrument object containing all information on experimental setup.

        METHOD : str
            Specifies which 3D-integration method to use. 'fix' (Default):
            sample the cross section on a fixed grid of points uniformly
            distributed $\phi$-space. 2*ACCURACY[0]+1 points are sampled
            along $\phi_1$, and $\phi_2$, and 2*ACCURACY[1]+1 along $\phi_3$
            (vertical direction). 'mc': 3D Monte Carlo integration. The cross
            section is sampled in 1000*ACCURACY randomly chosen points,
            uniformly distributed in $\phi$-space.

        ACCURACY : array(2) or int
            Determines the number of sampling points in the integration.

        p : list
            A parameter that is passed on, without change to sqw and pref.

        Returns
        -------
        conv : array
            Calculated value of the cross section, folded with the resolution
            function at the given $\mathbf{Q}_0$

        Notes
        -----
        Translated from ResLib 3.4c, originally authored by A. Zheludev,
        1999-2007, Oak Ridge National Laboratory

        """
        R0, RMS, RV = self.calc_resolution(np.transpose(hkle))

        H, K, L, W = hkle
        [length, H, K, L, W] = _CleanArgs(H, K, L, W)
        [xvec, yvec, zvec] = self._StandardSystem()[:3]

        Mww = RMS[:, 3, 3]
        Mxw = RMS[:, 0, 3]
        Myw = RMS[:, 1, 3]

        GammaFactor = np.sqrt(Mww / 2)
        OmegaFactorx = Mxw / np.sqrt(2 * Mww)
        OmegaFactory = Myw / np.sqrt(2 * Mww)

        Mzz = RMS[:, 2, 2]
        Mxx = RMS[:, 0, 0]
        Mxx -= Mxw ** 2 / Mww
        Myy = RMS[:, 1, 1]
        Myy -= Myw ** 2 / Mww
        Mxy = RMS[:, 0, 1]
        Mxy -= Mxw * Myw / Mww

        detxy = np.sqrt(Mxx * Myy - Mxy ** 2)
        detz = np.sqrt(Mzz)

        tqz = 1. / detz
        tqy = np.sqrt(Mxx) / detxy
        tqxx = 1. / np.sqrt(Mxx)
        tqxy = Mxy / np.sqrt(Mxx) / detxy

        [disp, inte] = sqw(H, K, L, p)[:2]
        [modes, points] = disp.shape

        if pref is None:
            prefactor = np.ones(modes, points)
            bgr = 0
        else:
            if nargout == 2:
                [prefactor, bgr] = pref(H, K, L, W, self, p)
            elif nargout == 1:
                prefactor = pref(H, K, L, W, self, p)
                bgr = 0
            else:
                raise ValueError('Invalid number or output arguments in prefactor function')

        if METHOD == 'mc':
            if isinstance(ACCURACY, (list, np.ndarray, tuple)):
                if len(ACCURACY) == 1:
                    ACCURACY = ACCURACY[0]
                else:
                    raise ValueError("ACCURACY (type: {0}) must be an 'int' when using Monte Carlo method".format(type(ACCURACY)))
            if ACCURACY is None:
                ACCURACY = 10
            M = ACCURACY
            convs = np.zeros((modes, length))
            conv = np.zeros(length)
            for i in range(length):
                for MonteCarlo in range(M):
                    if seed is not None:
                        np.random.seed(seed)
                    r = np.random.randn(3, 1000) * np.pi - np.pi / 2
                    cx = r[0, :]
                    cy = r[1, :]
                    cz = r[2, :]
                    tx = np.tan(cx)
                    ty = np.tan(cy)
                    tz = np.tan(cz)
                    norm = np.exp(-0.5 * (tx ** 2 + ty ** 2 + tz ** 2)) * (1 + tx ** 2) * (1 + ty ** 2) * (1 + tz ** 2)
                    dQ1 = tqxx[i] * tx - tqxy[i] * ty
                    dQ2 = tqy[i] * ty
                    dQ4 = tqz[i] * tz
                    H1 = H[i] + dQ1 * xvec[0] + dQ2 * yvec[0] + dQ4 * zvec[0]
                    K1 = K[i] + dQ1 * xvec[1] + dQ2 * yvec[1] + dQ4 * zvec[1]
                    L1 = L[i] + dQ1 * xvec[2] + dQ2 * yvec[2] + dQ4 * zvec[2]
                    [disp, inte, WL] = sqw(H1, K1, L1, p)
                    [modes, points] = disp.shape
                    for j in range(modes):
                        Gamma = WL[j, :] * GammaFactor[i]
                        Omega = GammaFactor[i] * (disp[j, :] - W[i]) + OmegaFactorx[i] * dQ1 + OmegaFactory[i] * dQ2
                        add = inte[j, :] * _voigt(Omega, Gamma) * norm / detxy[i] / detz[i]
                        convs[j, i] = convs[j, i] + np.sum(add)

                conv[i] = np.sum(convs[:, i] * prefactor[:, i])

            conv = conv / M / 1000. * np.pi ** 3

        elif METHOD == 'fix':
            if ACCURACY is None:
                ACCURACY = [7, 0]
            M = ACCURACY
            step1 = np.pi / (2 * M[0] + 1)
            step2 = np.pi / (2 * M[1] + 1)
            dd1 = np.linspace(-np.pi / 2 + step1 / 2, np.pi / 2 - step1 / 2, (2 * M[0] + 1))
            dd2 = np.linspace(-np.pi / 2 + step2 / 2, np.pi / 2 - step2 / 2, (2 * M[1] + 1))
            convs = np.zeros((modes, length))
            conv = np.zeros(length)
            [cy, cx] = np.meshgrid(dd1, dd1, indexing='ij')
            tx = np.tan(cx.flatten())
            ty = np.tan(cy.flatten())
            tz = np.tan(dd2)
            norm = np.exp(-0.5 * (tx ** 2 + ty ** 2)) * (1 + tx ** 2) * (1 + ty ** 2)
            normz = np.exp(-0.5 * (tz ** 2)) * (1 + tz ** 2)
            for iz in range(tz.size):
                for i in range(length):
                    dQ1 = tqxx[i] * tx - tqxy[i] * ty
                    dQ2 = tqy[i] * ty
                    dQ4 = tqz[i] * tz[iz]
                    H1 = H[i] + dQ1 * xvec[0] + dQ2 * yvec[0] + dQ4 * zvec[0]
                    K1 = K[i] + dQ1 * xvec[1] + dQ2 * yvec[1] + dQ4 * zvec[1]
                    L1 = L[i] + dQ1 * xvec[2] + dQ2 * yvec[2] + dQ4 * zvec[2]
                    [disp, inte, WL] = sqw(H1, K1, L1, p)
                    [modes, points] = disp.shape
                    for j in range(modes):
                        Gamma = WL[j, :] * GammaFactor[i]
                        Omega = GammaFactor[i] * (disp[j, :] - W[i]) + OmegaFactorx[i] * dQ1 + OmegaFactory[i] * dQ2
                        add = inte[j, :] * _voigt(Omega, Gamma) * norm * normz[iz] / detxy[i] / detz[i]
                        convs[j, i] = convs[j, i] + np.sum(add)

                    conv[i] = np.sum(convs[:, i] * prefactor[:, i])

            conv = conv * step1 ** 2 * step2

            if M[1] == 0:
                conv *= 0.79788
            if M[0] == 0:
                conv *= 0.79788 ** 2
        else:
            raise ValueError('Unknown METHOD: {0}. Valid options are: "fix" or "mc".'.format(METHOD))

        conv = conv * R0
        conv = conv + bgr

        return conv
