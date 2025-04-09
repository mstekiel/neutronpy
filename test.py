from enum import Enum
from logging import config
from neutronpy.instruments import CAMEA_MTAS
from neutronpy.crystal import Sample, Lattice
from neutronpy import Neutron, TripleAxisSpectrometer
# from neutronpy.fileio import TAS_loader
import numpy as np
from timeit import default_timer as timer


tas = TripleAxisSpectrometer.make_default()

hkle = np.array([
    [1,1,0,5],
    [1,1,0,550],
    [1,11,0,5],
    [1,0,1,5],
])


print(tas.get_angles(hkle))
print()
print(tas.calc_resolution_HKLE(hkle, base='Q'))

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