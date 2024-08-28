from enum import Enum
from logging import config
from neutronpy.instrument import Analyzer, Monochromator, TripleAxisInstrument
from neutronpy.crystal import Sample, Lattice
from neutronpy import Neutron
# from neutronpy.fileio import TAS_loader
import numpy as np
from timeit import default_timer as timer


toDeg = 180/np.pi
print()

taz_filename = r"C:\ProgramData\takin_270\instruments\EIGER_t270.taz"
TAS = TripleAxisInstrument.from_taz_file(taz_filename)
TAS.logger.setLevel('DEBUG')

# print('alpha', 180/np.pi*TAS.sample.get_angle_to_o1([1,1,0]))

hkle = np.array([1,0,0,5])
# hkle = np.array([[1,0,0,5], [1,1,0,5]])
# hkle = np.array([
#     [[1,0,0,5], [1,1,0,5]],
#     [[1,0,0,0], [1,1,0,0]]
#     ])
Q = TAS.sample.get_Q(hkle[...,:3])
E = hkle[...,3]
print('Q', Q)
print('E', E)

for ind in np.ndindex(Q.shape):
    print('QE', Q[ind], E[ind])

print('angles: ',TAS.get_angles(hkle))

print( TAS.calc_resolution_QE(Q, hkle[...,3])[1] )
print('RES in Q')
REhkl =  TAS.calc_resolution(hkle, base='Q')[1]
print( np.around(REhkl, 5) )
print('RES in HKL')
REhkl =  TAS.calc_resolution(hkle, base='hkl')[1]
print( np.around(REhkl, 5) )

covariance = np.linalg.inv(REhkl)
rnd_generator = np.random.default_rng(seed=42)
res_conv_samples = rnd_generator.multivariate_normal(mean=hkle, cov=covariance, size=(1000))
print( np.around(res_conv_samples, 20) )


# mt = '''
# 5749.446
# -3170.482
# 0
# 351.1324
# -3170.482
# 20259.04
# 0
# -1291.652
# 0
# 0
# 345.5488
# 0
# 351.1324
# -1291.652
# 0
# 89.89775

# '''

# res = np.reshape(mt.strip().split('\n'), (4,4))
# for row in res:
#     print('[' + ','.join([f'{r.rjust(12)}' for r in row]), '],')
