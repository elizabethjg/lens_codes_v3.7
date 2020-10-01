import sys
sys.path.append('/mnt/clemente/lensing/lens_codes_v3.7')
sys.path.append('/home/eli/lens_codes_v3.7')
from multipoles_shear import *
import time

t1 = time.time()

r  = np.logspace(np.log10(0.05),np.log10(10.),30)

multipoles = multipole_shear_parallel(r,misscentred = True,ncores=len(r))


print('//////////////////////')
print('         TIME         ')
print('----------------------')
print((time.time()-t1)/60.    )

out = np.array([multipoles['Gt0'],multipoles['Gt2'],multipoles['Gx2'],multipoles['Gt0_off'],multipoles['Gt_off'],multipoles['Gt_off_cos'],multipoles['Gx_off_sin']])

# saving mcmc out

f1=open('test.out','w')
f1.write('#    Gt0  Gt2 Gx2  Gt0_off  Gt_off  Gt_off_cos  Gx_off_sin \n')
np.savetxt(f1,out.T,fmt = ['%12.6f']*7)
f1.close()

