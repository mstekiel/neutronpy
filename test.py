from enum import Enum
from logging import config
from neutronpy.instrument import Analyzer, Monochromator, CAMEA_MTAS
from neutronpy.crystal import Sample, Lattice
from neutronpy import Neutron, TripleAxisInstrument
# from neutronpy.fileio import TAS_loader
import numpy as np
from timeit import default_timer as timer


n = Neutron(energy=14.87)
print(n._prefactor_energy_from_wavevector)
print(n._prefactor_energy_from_wavelength)
print(n._prefactor_energy_from_velocity)
print(n._prefactor_energy_from_frequency)
print(n._prefactor_energy_from_temperature)

print(n)


# # See energy overlapping
# Ef = np.array([3.200, 3.382, 3.574, 3.787, 4.035, 4.312, 4.631, 4.987])
# dEf = (Ef[1:]-Ef[:-1])/2
# Ebins = np.concatenate(( [Ef[0]-dEf[0]], Ef[:-1]+dEf, [Ef[-1]+dEf[-1]] ))

# N = 6
# hkle = np.zeros((N,4))
# hkle[...,2] = 1
# hkle[...,3] = np.linspace(3,6,N)

# # print(hkle)
# # print(np.digitize(hkle[...,3], Ebins))

# CAMEA = CAMEA_MTAS(Ei=10, sample=Sample.make_default(), a3_offset=0)
# CAMEA.logger.setLevel("INFO")

# print(hkle)
# print(CAMEA.Ef_bins)
# print(CAMEA.functional_Etransfer)
# print(CAMEA.resolution_convolution(hkle, None, None, None, '', 1))