# -*- coding: utf-8 -*-
import numpy as np

from ...constants import load_dspacings

class Monochromator(object):
    u"""Class containing monochromator information.

    Parameters
    ----------
    dspacing : float
        Interplanar spacing of the crystal [A].

    mosaic : int
        Mosaic of the crystal [arcmin].

    mosaic_v : int, optional
        Vertical mosaic of the crystal [arcmin].

    height : float, optional
        Height of the crystal [cm].

    width : float, optional
        Width of the crystal [cm].

    depth : float, optional
        Depth of the crystal in [cm]

    curvr_h : float, optional
        Horizontal curvature radius of the monochromator [cm].

    curvr_v : float, optional
        Vertical curvature radius of the monochromator [cm].
    """

    def __init__(self, dspacing: float, mosaic: float, name: str,
                 mosaic_v: float, width: float, height: float, depth: float,
                 curvr_h: float, curvr_v: float):
        # Main properties
        self.dspacing = dspacing
        self.mosaic = mosaic
        self.name = name

        # Then secondary properties
        self.mosaic_v = mosaic_v

        self.width  = width
        self.height = height
        self.depth  = depth

        # Curvatures
        self.curvr_h = curvr_h
        self.curvr_v = curvr_v

    @classmethod
    def make_default(cls):
        "Construct a `Monochromator` instance with some default values."
        return cls(dspacing=3.354, mosaic=30, name='custom',
                   mosaic_v=30, width=30, height=30, depth=0.2,
                   curvr_h=0, curvr_v=0)

    @classmethod
    def from_name(cls, name: str, **kwargs):
        """Construct `Monochromator` based on the `name` which includes the material
        and face of the crystal. Comes with default values, whic are not changed
        unless supplid in `kwargs`.

        >>> Monochromator.from_name('PG(002)')
        ... <Monochromator dspacing=3.354, mosaic=30, mosaic_v=30, height=30, width=30, depth=0.2, curvr_h=0, curvr_v=0>

        >>> Monochromator.from_name('Si(111)', mosaic=10)
        ... <Monochromator dspacing=3.13501, mosaic=10, mosaic_v=30, height=30, width=30, depth=0.2, curvr_h=0, curvr_v=0>

        >>> config = dict(mosaic=27, height=15, width=30, curvr_v=30)
        >>> Monochromator.from_name('PG(004)', **config)
        ... <Monochromator dspacing=1.677, mosaic=27, mosaic_v=30, height=15, width=30, depth=0.2, curvr_h=0, curvr_v=30>

        Parameters
        ----------
        name: str
            Typical names are `PG(002)`, `PG(004)`, `Si(111)`, but there are more implemented,
            see `constants.dspacings()` for full list.

        kwargs: dict
            Other keyword arguments passed to the `Monochromator.__init__` function,
            see `Monochromator` fields.
        """

        db_dspacings = load_dspacings()
        implemented_crystals = db_dspacings.keys()

        if name not in implemented_crystals:
            raise KeyError(f'{name!r} is not implemeted as {cls.__name__} crystal name.'+'\n'+
                           f'Try {implemented_crystals!r}')
        
        if name not in kwargs.keys():
            kwargs['name'] = name

        new_mono = cls.make_default()
        for key, value in kwargs:
            setattr(new_mono, key, value)
        
        return cls(dspacing=db_dspacings[name], **kwargs)
    
    def __repr__(self):
        printing_keys = ['name', 'dspacing', 'mosaic', 
                         'mosaic_v', 'height', 'width', 'depth', 
                         'curvr_h', 'curvr_v']
        kwargs = ['{0}={1}'.format(key, getattr(self, key))
                            for key in printing_keys if getattr(self, key, None) is not None]
        return "<Monochromator {0}>".format(', '.join(kwargs))

    
    ######################################################################
    @property
    def name(self):
        '''Name of the monochromator.'''
        return self._name
    
    @name.setter
    def name(self, value: str):
        self._name = value
    
    ######################################################################
    @property
    def dspacing(self):
        '''Interplanar spacing of the monochromator crystals in angstroem.'''
        return self._dspacing
    
    @dspacing.setter
    def dspacing(self, value: float):
        if value < 0:
            raise ValueError(f"{self.__class__.__name__!r} dspacing must be a positive number. Is: {value}")
        
        self._dspacing = value
        self._tau = 2*np.pi / value

    ######################################################################
    @property
    def tau(self):
        '''Wavevector equivalent of the interplanar spacing of the monochromator in 1/angstroem.'''
        return self._tau

    @tau.setter
    def tau(self, value: float):
        if value < 0:
            raise ValueError(f"{self.__class__.__name__!r} tau must be a positive number. Is: {value}")
        
        self._tau = value
        self._dspacing = 2*np.pi / value

    ######################################################################
    @property
    def mosaic(self):
        '''Mosaicity of the crystal [arcmin].'''
        return self._mosaic

    @mosaic.setter
    def mosaic(self, value: float):
        if value<0:
            raise ValueError(f'{value!r} invalid for {self.__class__.__name__!r}, mosaic has to be positive.')
        
        self._mosaic = value    
        
    ######################################################################
    @property
    def mosaic_v(self):
        '''Mosaicity of the crystal in certical direction [arcmin].'''
        return self._mosaic_v

    @mosaic_v.setter
    def mosaic_v(self, value: float):
        if value<0:
            raise ValueError(f'{value!r} invalid for {self.__class__.__name__!r}, mosaic_v has to be positive.')
        
        self._mosaic_v = value

    ######################################################################
    @property
    def width(self):
        '''Width of the crystalperpendicular to the face [cm].'''
        return self._width

    @width.setter
    def width(self, value: float):
        if value<0:
            raise ValueError(f'{value!r} invalid for {self.__class__.__name__!r}, width has to be positive.')
        
        self._width = value

    ######################################################################
    @property
    def height(self):
        '''Height of the crystal [cm].'''
        return self._height

    @height.setter
    def height(self, value: float):
        if value<0:
            raise ValueError(f'{value!r} invalid for {self.__class__.__name__!r}, height has to be positive.')
        
        self._height = value

    ######################################################################
    @property
    def depth(self):
        '''Depth of the crystal [cm].'''
        return self._depth

    @depth.setter
    def depth(self, value: float):
        if value<0:
            raise ValueError(f'{value!r} invalid for {self.__class__.__name__!r}, depth has to be positive.')
        
        self._depth = value

    ######################################################################
    @property
    def curvr_h(self):
        '''Curvature radius for the horizontal focussing [cm]. Value 0 means flat.'''
        return self._curvr_h

    @curvr_h.setter
    def curvr_h(self, value: float):
        if value < 0:
            raise ValueError(f'{value!r} invalid for {self.__class__.__name__!r}, curvr_h has to be positive.')
        
        self._curvr_h = value

    ######################################################################
    @property
    def curvr_v(self):
        '''Curvature radius for the horizontal focussing [cm]. Value 0 means flat.'''
        return self._curvr_v

    @curvr_v.setter
    def curvr_v(self, value: float):
        if value < 0:
            raise ValueError(f'{value!r} invalid for {self.__class__.__name__!r}, curvr_v has to be positive.')
        
        self._curvr_v = value

    ######################################################################
    def focal_length(self, lenBefore: float, lenAfter: float) -> float:
        '''Thin lens equation'''
        f_inv = 1./lenBefore + 1./lenAfter
        return 1. / f_inv


    def set_opt_curvature(self, lenBefore: float, lenAfter: float, tth: float, vertical: bool) -> float:
        '''Determine and set the curvature radius for optimal focussing.

        Parameters:
        -----------
        lenBefore: float
            Distance between source and monochromator [same as lenAfter].
        lenAfter: float
            Distance between monochromator and the target[same as lenAfter].
        tth: float
            Scattering angle of the monochromaor two-theta [rad].
        vertical: bool
            Vertical focusing curvature (True) or horizontal (False).

        Notes:
        ------
            - (Shirane 2002) p. 66
            - or nicos/nicos-core.git/tree/nicos/devices/tas/mono.py in nicos
            - or Monochromator_curved.comp in McStas'''
        
        f = self.focal_length(lenBefore, lenAfter)
        s = np.abs(np.sin(0.5*tth))

        if vertical:
            curvr = 2. * f*s
            self.curvr_v = curvr
        else:
            curvr = 2. * f/s
            self.curvr_h = curvr

        return curvr
    
    def get_tth(self, wavelength: float) -> float:
        '''Determine scattering angle two-theta for elastic scattering
        of incoming beam with `wavelength` [A].'''

        return 2*np.arcsin(0.5*wavelength/self.dspacing)
