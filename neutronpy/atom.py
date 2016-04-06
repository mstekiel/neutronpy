import numpy as np
import neutronpy.constants as const


class Atom(object):
    r'''Class for adding atoms to the Material class.

    Parameters
    ----------
    ion : string
        The name of the Atom, or ion if necessary
    pos : list(3)
        The position of the Atom in the chosen geometry
    dpos : list(3), optional
        Deviations from the position pos
    occupancy: float, optional
        Occupancy of the _Atom (*e.g.* if there is partial occupancy from
        doping)
    Mcell : float, optional
        The mass of the unit cell. If assigned, normalize scattering lengths to
        the square-root of the mass of the atom

    Returns
    -------
    output : object
        Atom object defining an individual atom in a unit cell of a single
        crystal

    '''
    def __init__(self, ion, pos, occupancy=1., Mcell=None, massNorm=False, Uiso=0, Uaniso=np.zeros((3, 3))):
        self.ion = ion
        self.pos = np.array(pos)
        self.occupancy = occupancy
        self.Mcell = Mcell
        self.Uiso = Uiso
        self.Uaniso = np.matrix(Uaniso)

        if massNorm is True:
            self.mass = const.periodic_table()[ion]['mass']
            self.b = (const.scattering_lengths()[ion]['Coh b'] * self.occupancy * self.Mcell / np.sqrt(self.mass))
        else:
            self.b = const.scattering_lengths()[ion]['Coh b'] / 10.

        self.coh_xs = const.scattering_lengths()[ion]['Coh xs']
        self.inc_xs = const.scattering_lengths()[ion]['Inc xs']
        self.abs_xs = const.scattering_lengths()[ion]['Abs xs']


class MagneticAtom(object):
    r'''Class for adding magnetic atoms to the Material class.

    '''
    def __init__(self, ion, pos, moment, occupancy):
        self.ion = ion
        self.pos = np.array(pos)
        self.moment = moment
        self.occupancy = occupancy
