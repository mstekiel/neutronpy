# -*- coding: utf-8 -*-
r"""Handles lattice geometries to find rotations and transformations

"""
from functools import cached_property
import numpy as np


class Lattice(object):
    """Class to describe a generic lattice system defined by lattice six
    parameters, (three constants and three angles).

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

    Attributes
    ----------
    a
    b
    c
    alpha
    beta
    gamma
    abc
    abg
    abg_rad
    lattice_type
    Amatrix
    Bmatrix
    G
    Gstar
    volume
    reciprocal_volume

    Conventions
    -----------
    1. Lattice vectors are positioned in the Cartesian coordinates as:
        - a || x
        - b* || y
        - c to complete RHS
    2. The phase factors are (i) exp(i k_xyz r_xyz) and (ii) exp(i Q_hkl r_uvw).
    3. All matrices, so for both real and reciprocal space, are represented for column vectors.
    4. `B` matrix contains the 2pi factor. Consequently A.T @ B = 2pi eye(3,3). Transposition due to pt 3.
    5. Implemented methods should work with arrays assuming the last index representing the `h, k, l` coordinates.

        
    Transformation notes
    --------------------
    A: ndarray((3,3))

        Transforms a real lattice point into an orthonormal coordinates system. Upper triangle matrix.
        [u,v,w] -> [x,y,z] (Angstroems)


    B: ndarray((3,3))

        Transforms a reciprocal lattice point into an orthonormal coordinates system.
        (h,k,l) -> [kx,ky,kz] (1/Angstroem)

    """

    def __init__(self, a: float, b: float, c: float, alpha: float, beta: float, gamma: float):
        self.lattice_parameters = [a,b,c, alpha,beta,gamma]
        self._update_lattice()

    #################################################################################################
    # Core methods

    def _update_lattice(self):
        '''Master function recalculating all matrices involving the lattice parameters.'''
        a,b,c, alpha,beta,gamma = self.lattice_parameters
        # A matrix follows convention 1.
        self._Amatrix = self._constructA(a,b,c, alpha,beta,gamma)

        # B matrix based on the perpendicularity condition to A
        # To get column representation it needs to be transposed
        self._Bmatrix = 2*np.pi* np.linalg.inv(self.Amatrix).T

        # Metric tensor of real space
        self._G = self.Amatrix.T @ self.Amatrix

        # Metric tensor of reciprocal space
        self._Gstar = self.Bmatrix.T @ self.Bmatrix

    def _constructA(self, a: float, b: float, c: float, alpha: float, beta: float, gamma: float) -> np.ndarray:
        '''
        Construct the `A` matrix as crystal axes in orthonormal system, ie a||x, b in xy plane, c accordingly.

        Transforms a real lattice point into an orthonormal coordinates system. Upper triangle matrix.
        A * [u,v,w] -> [x,y,z] (Angstroems)

        Shortcut to define lattice vectors:
        >>> a, b, c = A.T
        '''
        
        alpha, beta, gamma = self.abg_rad
        bx = b*np.cos(gamma)
        by = b*np.sin(gamma)
        
        cx = c*np.cos(beta)
        cy = c*(np.cos(alpha)-np.cos(gamma)*np.cos(beta))/np.sin(gamma)
        cz  = np.sqrt(c*c-cx*cx-cy*cy)
        
        return np.array([[a,bx,cx],[0,by,cy],[0,0,cz]])

    def __repr__(self):
        return "<Lattice {0}, {1}, {2}, {3}, {4}, {5}>".format(self.a, self.b, self.c, self.alpha, self.beta, self.gamma)

    #################################################################################################
    # Defining properties
    @property
    def Amatrix(self):
        '''Crystal axes in orthonormal system, ie a||x, b in xy plane, c accordingly.
        Upper triangle matrix.

        Transforms a real lattice point into an orthonormal coordinates system.
        A * [u,v,w] -> [x,y,z] (Angstroems)


        Shortcut to define lattice vectors:
        >>> a, b, c = A.T
        '''
        return self._Amatrix

    @property
    def Bmatrix(self):
        '''Reciprocal crystal axes in orthonormal system, perpendicular to real axes.
        By definition `b* || y`.

        Transforms a reciprocal lattice point into an orthonormal coordinates system.
        B*(h,k,l) -> [kx,ky,kz]_{crystal} (1/Angstroem)

        Shortcut to define reciprocal lattice vectors:
        >>> astar, bstar, cstar = B.T
        '''
        return self._Bmatrix

    @property
    def G(self):
        '''Metric tensor of the real lattice.
        G = A @ A.T
        '''
        return self._G


    @property
    def Gstar(self):
        '''Metric tensor of the reciprocal lattice.
        Gstar = B @ B.T

        Allows to calculate products of vector in hkl base.
        '''
        return self._Gstar

    #################################################################################################
    # Properties with setters
    @property
    def a(self) -> float:
        """First lattice parameter `a` in Angstrom."""
        return self.lattice_parameters[0]

    @a.setter
    def a(self, new_a: float):
        self.lattice_parameters[0] = new_a
        self._update_lattice()

    @property
    def b(self) -> float:
        """Second lattice parameter `b` in Angstrom."""
        return self.lattice_parameters[1]

    @b.setter
    def b(self, new_b: float):
        self.lattice_parameters[1] = new_b
        self._update_lattice()

    @property
    def c(self) -> float:
        """Third lattice parameter `c` in Angstrom."""
        return self.lattice_parameters[2]
    
    @c.setter
    def c(self, new_c: float):
        self.lattice_parameters[2] = new_c
        self._update_lattice()


    @property
    def alpha(self) -> float:
        """First lattice angle `alpha` in degrees."""
        return self.lattice_parameters[3]

    @alpha.setter
    def alpha(self, new_alpha):
        self.lattice_parameters[3] = new_alpha
        self._update_lattice()

    @property
    def beta(self) -> float:
        """Second lattice angle `beta` in degrees."""
        return self.lattice_parameters[4]

    @beta.setter
    def beta(self, new_beta: float):
        self.lattice_parameters[4] = new_beta
        self._update_lattice()

    @property
    def gamma(self) -> float:
        """Third lattice angle `gamma` in degrees."""
        return self.lattice_parameters[5]

    @gamma.setter
    def gamma(self, new_gamma: float):
        self.lattice_parameters[5] = new_gamma
        self._update_lattice()

    @property
    def abc(self):
        """Lattice parameters in Angstroem"""
        return self.lattice_parameters[:3]
    
    @property
    def abg(self):
        """Lattice angles in degrees."""
        return self.lattice_parameters[3:]
    
    @property
    def abg_rad(self):
        """Lattice angles in radians."""
        return np.radians(self.lattice_parameters[3:])

    @property
    def lattice_type(self):
        """Type of lattice determined by the provided lattice constants and angles"""

        if len(np.unique(self.abc)) == 3 and len(np.unique(self.abg)) == 3:
            return 'triclinic'
        elif len(np.unique(self.abc)) == 3 and self.abg[1] != 90 and np.all(np.array(self.abg)[:3:2] == 90):
            return 'monoclinic'
        elif len(np.unique(self.abc)) == 3 and np.all(np.array(self.abg) == 90):
            return 'orthorhombic'
        elif len(np.unique(self.abc)) == 1 and len(np.unique(self.abg)) == 1 and np.all(
                        np.array(self.abg) < 120) and np.all(np.array(self.abg) != 90):
            return 'rhombohedral'
        elif len(np.unique(self.abc)) == 2 and self.abc[0] == self.abc[1] and np.all(np.array(self.abg) == 90):
            return 'tetragonal'
        elif len(np.unique(self.abc)) == 2 and self.abc[0] == self.abc[1] and np.all(np.array(self.abg)[0:2] == 90) and \
                        self.abg[2] == 120:
            return 'hexagonal'
        elif len(np.unique(self.abc)) == 1 and np.all(np.array(self.abg) == 90):
            return 'cubic'
        else:
            raise ValueError('Provided lattice constants and angles do not resolve to a valid Bravais lattice')

    @property
    def volume(self) -> float:
        """Volume of the unit cell in [A^3]."""
        return np.sqrt(np.linalg.det(self.G))

    @property
    def reciprocal_volume(self) -> float:
        """Volume of the reciprocal unit cell in [1/A^3]. What about the pi factor?"""
        return np.sqrt(np.linalg.det(self.Gstar))

    #################################################################################################
    # Functionalities
    def get_scalar_product(self, hkl1: np.ndarray, hkl2: np.ndarray):
        """Returns the scalar product between two lists of vectors.
        
        Parameters
        ----------
        hkl1 : array_like (3) or (...,3)
            Vector or array of vectors in reciprocal space.
        hkl2 : array_like (...,3)
            List of vectors in reciprocal space.

        Returns
        -------
        ret : array_like (...)
            List of calculated scalar products between vectors.

        Notes
        -----
        Takes advantage of the `Gstar=B.T @ B` matrix. Simply does:
        `Q_hkl1 @ Gstar @ Q_hkl2 == (B @ Q_hkl1).T @ (B @ Q_hkl2) == Q_xyz1 @ Q_xyz2`.
        Where the last one is in the orthonormal coordinate frame and can be 
        directly computed.
        """
        v1v2_cosine = np.einsum('...i,ij,...j->...', hkl1, self.Gstar, hkl2)

        return v1v2_cosine
    
    def get_Q(self, hkl: np.ndarray) -> np.ndarray:
        '''Returns the magnitude |Q| [1/A] of reciprocal lattice vectors `hkl`.
                
        Parameters
        ----------
        hkl : array_like (3,...)
            Reciprocal lattice vector in r.l.u. Signature: `h,k,l = hkl`

        Returns
        -------
        Q : array_like (,...)
            The magnitude of the reciprocal lattice vectors in [1/A].
            Shape follows the input signature with reduced first dimension.


        Notes
        -----
        Calculates the Q vector from the inverse metric tensor: `Q = sqrt(hkl.T @ Gstar @ hkl)`.
        Alternative method of calculating from B matrix proved to be slower: `Q = norm(B @ hkl)`
        '''
        # return np.sqrt(np.einsum('i...,ij,j...->...', hkl, self.Gstar, hkl))
        return np.sqrt(self.get_scalar_product(hkl, hkl))
    
    def get_dspacing(self,hkl: np.ndarray) -> np.ndarray:
        u"""Returns the d-spacing of a given reciprocal lattice vector.

        Parameters
        ----------
        hkl : array_like (3,...)
            Reciprocal lattice vector in r.l.u. Signature: `h,k,l = hkl`

        Returns
        -------
        d : float (,...)
            The d-spacing in A.
        """
        # DEV NOTES
        # Method with metric tensor proves to be the fastest.
        # Alternative tested was determining Q from `norm(B @ hkl)`

        return 2*np.pi / self.get_Q(hkl)

    def get_tth(self, hkl: np.ndarray, wavelength: float) -> np.ndarray:
        u"""Returns the detector angle two-theta [rad] for a given reciprocal
        lattice vector [rlu] and incident wavelength [A].

        Parameters
        ----------
        hkl : array_like (3,...)
            Reciprocal lattice vector in r.l.u. Signature: `h,k,l = hkl`

        wavelength : float
            Wavelength of the incident beam in Angstroem.

        Returns
        -------
        two_theta : array_like (,...)
            The scattering angle two-theta i nradians.
            Shape follows the input signature with reduced first dimension.

        """

        return 2*np.arcsin( wavelength*self.get_Q(hkl)/4/np.pi )
    
    
    def get_angle_between_planes(self, v1, v2) -> float:
        r"""Returns the angle :math:`\phi` [rad] between two reciprocal lattice
        vectors (or planes as defined by the vectors normal to the plane).

        Parameters
        ----------
        v1 : array_like (3)
            First reciprocal lattice vector in units r.l.u. 

        v2 : array_like (3,...)
            Second reciprocal lattice vector in units r.l.u.

        Returns
        -------
        phi : float (...)
            The angle between v1 and v2 in radians.

        Notes
        -----
        Uses the `Gstar` matrix again and the fact that `Gstar=B B.T` such that
        `v1.Gstar.v2` is the cosine between v1-v2.
        Due to rounding errors the cosine(v1,v2) is clipped to [-1,1].
        """
        v1v2_cosine = np.einsum('i,ij,...j->...', v1, self.Gstar, v2)
        v1 = self.get_Q(v1)
        v2 = self.get_Q(v2)

        return np.arccos( np.clip(v1v2_cosine / (v1*v2), -1, 1) )