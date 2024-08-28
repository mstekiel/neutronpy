# -*- coding: utf-8 -*-
r"""Sample class for e.g. Instrument class

"""
import numpy as np

from neutronpy.instrument.exceptions import InstrumentError

from .lattice import Lattice


class Sample(Lattice):
    """Private class containing sample information.

    Parameters
    ----------
    a : float
        Unit cell length in angstroms

    b : float
        Unit cell length in angstroms

    c : float
        Unit cell length in angstroms

    alpha : float
        Angle between b and c in degrees

    beta : float
        Angle between a and c in degrees

    gamma : float
        Angle between a and b in degrees

    orient1 : array_like
        First orientation vector

    orient2 : array_like
        Second orientation vector

    mosaic : float
        Horizontal sample mosaic (FWHM) in arc minutes

    vmosaic : float
        Vertical sample mosaic (FWHM) in arc minutes

    width1 : float
        Sample dimension along the first orienting vector in cm.

    width2 : float
        Sample dimension along the second orienting vector in cm.

    height : float
        Sample height in cm.

    shape : str
        Sample shape type. Accepts 'cuboid' or 'cylindrical'.

        
    Attributes
    ----------
    a
    b
    c
    alpha
    beta
    gamma
    orient1
    orient2
    mosaic
    vmosaic
    direct
    width1
    width2
    height
    shape
    abg_rad
    lattice_type
    volume
    reciprocal_volume
    G
    Gstar
    Bmatrix
    Umatrix
    UBmatrix

    Methods
    -------
    get_d_spacing
    get_q
    get_two_theta
    get_angle_between_planes
    get_phi

    Conventions
    -----------
    1. `Sample` object is position in the `_lab` cartesian coordinate frame. The convention is:
          - `x` is looking at the incoming beam, i.e. `e_x = - e_ki`
          - 'z' is vertical
          - `y` satisfies previous conditions to make RHS cartesian system.
    2. Since `Sample` inherits from `Lattice` it also inherits the coordinate system of the `Lattice`,
       with conventions in `Lattice.__doc__`. These coordinate systems are now suffixed `_crystal`.
    3. `Sample` has two orienting vectors that define the scattering plane `xy`. They refer to the Cartesian frame:
         - `orient1` along `x_lab` direction
         - `orient2` in the `xy_lab` plane, with `orient2.y_lab > 0`. 
           In other words, the vector product `cross(orient1, orient2) || z_lab`,
           or, right-handed rotation from `orient1` to `orient2` is around `z_lab` direction.
    
        
    Transformation notes
    --------------------
    A: ndarray((3,3))

        Transforms a real lattice point into an orthonormal coordinates system. Upper triangle matrix.
        `[u,v,w] -> [x,y,z]_{crystal}` (Angstroems)

    B: ndarray((3,3))

        Transforms a reciprocal lattice point into an orthonormal coordinates system.
        `(h,k,l) -> [kx,ky,kz]_{crystal}` (1/Angstroem)

    U: ndarray((3,3))

        Rotation matrix that relates the orthonormal, reciprocal lattice coordinate system into the diffractometer/lab coordinates
        `[kx,ky,kz]_{crystal} -> [kx,ky,kz]_{lab}`

    UA: ndarray((3,3))

        Transforms a real lattice point into lab coordinate system.
        `(u,v,w) -> [x,y,z]_{lab} (Angstroem)`

    UB: ndarray((3,3))

        Transforms a reciprocal lattice point into lab coordinate system.
        `(h,k,l) -> [kx,ky,kz]_{lab} (1/Angstroem)`
    """

    # DEV NOTES
    # 1. To keep the functionality of updating the underlying `lattice_parameters` whenever
    #    they are updated, the orientation matrix also needs to be updated. For that
    #    override the `Lattice._update_lattice` and add `Sample._updateU`

    def __init__(self, a, b, c, alpha, beta, gamma, orient1, orient2, 
                 mosaic, mosaic_v,
                 width1, height, width2, shape):
        
        # Inside `Lattice.__init__()` there is already `_update_lattice()`.
        # Apparently, the following `super().__init__()` will be called
        # with the `_update_lattice()` of the `Sample`, not `Lattice`,
        # i.e. it requires filled fields `orient1` and `orient2`.
        self._orient1 = np.array(orient1).ravel()
        self._orient2 = np.array(orient2).ravel()
        super().__init__(a, b, c, alpha, beta, gamma)

        # This above is a bit obscure, but works. At this stage the lattice and orientation are set

        self.mosaic = mosaic
        self.mosaic_v = mosaic_v
        self.shape = shape

        self.width1 = width1
        self.width2 = width2
        self.height = height

    @classmethod
    def make_default(cls):
        """Create `Sample` instance with default values."""
        return cls(a=5, b=5, c=5, alpha=90, beta=90, gamma=90, orient1=[1,0,0], orient2=[0,1,0], 
                   mosaic=60, mosaic_v=60,
                   width1=1, height=1, width2=1, shape='cylindrical')

    def _update_lattice(self):
        """Override the master function to satisfy `Convention 1`.
        That is, the orientation matrix is changed when lattice
        is changed."""
        super()._update_lattice()
        self._update_orientation()

    def _update_orientation(self):
        '''Master function recalculating the orientation matrix.'''
        self._Umatrix = self._constructU(self._orient1, self._orient2)

    def _constructU(self, o1_hkl, o2_hkl) -> np.ndarray:
        '''
        Construct the orientation matrix U from two lattice vectors.
        
        Convention
        ----------
        Construction is based on rotation matrices, such that
        `hkl1` is put along the `x` axis and `hkl2` in the `xy` plane.
        In addition, `hkl2` is located in the `y>0` halpf of the `xy` plane.

        Notes
        -----
        When orienting vectors are represented in the crystal coordinates [kx,ky,kz]_{crystal}
        by multiplying through `self.Bmatrix`, they define (1) a rotated
        frame with matrix `o123`. Now the point is to find the rotation that
        brings them to the `xy` plane == scattering plane.
        Computationally, it is easier to go other way. First (2) we construct
        accompanying matrix `Bp` as a matrix with o1 and o2 in the scattering plane.
        Care needs to be taken (3) because o1-o2 do not have to be perpendicular.
        Then (4), we find a rotation `Up` the brings it to `o123`.
        Finally (5), the rotation we are looking for is the inverted rotation to the `Up`:
        `U = Up^(-1)`
        On equations:
        >>> Up @ Bp = o123
        >>> U = Up^(-1)
        >>> U^(-1) @ Bp = o123
        >>> Bp = U @ o123
        >>> U = Bp @ o123^(-1)
        '''

        # (1) construct o123
        o1_xyz = np.dot(self.Bmatrix, o1_hkl)
        o2_xyz = np.dot(self.Bmatrix, o2_hkl)
        o3_xyz = np.cross(o1_xyz, o2_xyz)
        o123 = np.transpose( [o1_xyz, o2_xyz, o3_xyz] )

        # (2) and (3) construct Bp and take care of potential non-orthogonality
        phi = self.get_angle_between_planes(o1_hkl, o2_hkl)
        # TODO below will it work for monoclinic and triclinic cases?
        Bp = np.array([
            [self.get_Q(o1_hkl), self.get_Q(o2_hkl)*np.cos(phi), 0],
            [0,                  self.get_Q(o2_hkl)*np.sin(phi), 0],
            [0,                  0,                              np.linalg.norm(o3_xyz)],
            ])
        
        # (4) (5) steps are just rearraging equations
        return Bp @ np.linalg.inv(o123)
    


    def __repr__(self):
        args = ', '.join([str(getattr(self, key)) for key in ['a', 'b', 'c', 'alpha', 'beta', 'gamma']])
        kwargs = ', '.join(['{0}={1}'.format(key, getattr(self, key)) for key in
                            ['orient1', 'orient2', 'mosaic', 'mosaic_v', 'width1', 'width2', 'height', 'shape'] if
                            getattr(self, key, None) is not None])
        return "Sample({0}, {1})".format(args, kwargs)


    ####################################################################################################
    # Properties with setters

    @property
    def orient1(self):
        """First orientation vector [rlu.]. 
        
        Orients the sample such that the `orient1` vectors looks
        at incoming beam: `orient1 || -ki`.
        """
        return self._orient1

    @orient1.setter
    def orient1(self, vec):
        np.ravel
        self._orient1 = np.array(vec).ravel()
        self._update_orientation()

    @property
    def orient2(self):
        r"""Second orientation vector [rlu.]. 
        
        Orients the sample such that the `orient2` and `orient1` 
        lie in the scattering plane `xy`
        """
        return self._orient2

    @orient2.setter
    def orient2(self, vec):
        self._orient2 = np.array(vec)
        self._update_orientation()

    @property
    def Umatrix(self):
        r"""Rotation matrix that relates the orthonormal, reciprocal lattice coordinate system 
        into the lab coordinates.

        `U: [kx,ky,kz]_{crystal} -> [kx,ky,kz]_{lab}`
        """
        return self._Umatrix

    @property
    def UBmatrix(self):
        r"""Transforms a reciprocal lattice point into lab coordinate system.

        `UB: (h,k,l) -> [kx,ky,kz]_{lab} (1/Angstroem)`
        """
        # Maybe better to pull this as separate field not to recalculate each time?
        return self.Umatrix @ self.Bmatrix

    ########################################################################
    # Functionalities
    
    def is_in_scattering_plane(self, hkl: np.ndarray) -> bool:
        '''Determine whether vectors `hkl` are in the scattering plane.

        Parameters
        ----------
        hkl: array_like (3,...)

        Returns
        -------
        bool (...)
        List indicating whether given vectors lie in the scattering plane.
        '''
        Q_xyz = np.einsum('ij,...j->...i', self.UBmatrix, hkl)
        return Q_xyz[...,2] < 1e-10

    def get_angle_to_o1(self, hkl: np.ndarray) -> float:
        '''Calculate the angle required to rotate the scattering vector `hkl`
        to the direction of the first orientation vector. Assumes right handed 
        rotation around the `z` axis of the lab frame.
        
        Parameters
        ----------
        hkl: array_like (...,3)
            List of scattering vectors [hkl].

        Returns
        -------
        List of rotation angles in radians.

        Raises
        ------
        `InstrumentError`
            When any `hkl` is not in the scattering plane.
        '''
        # This is quite easy, as per convention o1 is along x.
        # Represent `hkl` in lab orthonormal system
        Q_xyz = np.einsum('ij,...j->...i', self.UBmatrix, hkl)

        # Ensure `hkl` is in the scattering plane, by own call to avoid repetition
        if np.sum( Q_xyz[...,2] > 1e-10 ):
            raise InstrumentError('Requested `hkl` not in the scattering plane.')

        # Since `o1 || x` we just need the polar angle of requested `hkl`
        return -np.arctan2(Q_xyz[...,1], Q_xyz[...,0])
    
    
    def get_phiZ(self, hkl: np.ndarray):
        u"""Calculate out-of-plane angle for reciprocal lattice vectors `hkl`.

        Parameters
        ----------
        hkl: array_like (..., 3)
            Scattering vectors.

        Returns
        -------
        phi : (...)
            The out-of-plane angles.
        """
        return self.get_angle_between_planes(hkl, np.cross(self.orient1, self.orient2))