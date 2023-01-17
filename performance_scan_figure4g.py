import sys
import numpy as np
import circularFiltering as flt
import network_filter as nwflt
from time import time

kappa_z_array = np.array([0.01,0.03,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,2,3,4,5,6,7,8,9,10,30,100])

# choose a kappa_y according to system variable
kappa_z = kappa_z_array[int(sys.argv[1])]

# number of iterations
nIter = 10000

# where to save
filename = "data_raw/figure4g/performance_kappaz="+str(kappa_z)

# seeed the run with a value related to kappa_y
np.random.seed(int(kappa_z*10))

# model parameters
T = 20 # simulation time
dt = 0.001 # step size
t = np.arange(0,T,dt)
alpha = flt.xi_fun_inv(kappa_z * dt)
timesteps = int(T/dt)
kappa_phi = 1 # inverse diffusion constant
phi_0 = 0 # initial mean
kappa_0 = 20 # initial certainty
kappa_y = 1 # certainty of increment observations

# run the simulations and read out first and second order statistics for each time step
phi_final = np.zeros([nIter])
vonMises = np.zeros([nIter,2])
vonMises_q = np.zeros([nIter,2])
noUncert = np.zeros([nIter])
CX = np.zeros([nIter,2])
start = time()

for i in range(0, nIter): # run for nIter iterations

    # generate data
    phi, dy, z = flt.generateData(T,kappa_phi,kappa_y=kappa_y,dt=dt,phi_0=phi_0,kappa_0=kappa_0,kappa_z=alpha)

    # von Mises projection filter
    mu_VM, kappa_VM = flt.vM_Projection_Run(T,kappa_phi,dy=dy,kappa_y=kappa_y,z=z,kappa_z=alpha,
                            phi_0=phi_0,kappa_0=kappa_0,dt=dt)

    # von Mises projection filter, quadratic approximation
    mu_VMq, kappa_VMq = flt.vM_Projection_quad_Run(T,kappa_phi,dy=dy,kappa_y=kappa_y,z=z,kappa_z=alpha,
                            phi_0=phi_0,kappa_0=kappa_0,dt=dt)

    # CX-like network
    mu_CX, kappa_CX = nwflt.network_filter_Run(T,kappa_phi,dy=dy,kappa_y=kappa_y,z=z,I_ext=alpha,
                        phi_0=phi_0,kappa_0=kappa_0,dt=dt)

    # read out statistics
    phi_final[i] = phi[-1]
    vonMises[i] = np.array([mu_VM[-1],kappa_VM[-1]])
    vonMises_q[i] = np.array([mu_VMq[-1],kappa_VMq[-1]])
    CX[i] = np.array([mu_CX[-1],kappa_CX[-1]])

np.savez(filename,phi_final=phi_final,vonMises=vonMises,vonMises_q=vonMises_q,CX=CX
        ,kappa_phi=kappa_phi,kappa_y=kappa_y,kappa_z=kappa_z,T=T,dt=dt)

print('kappa_z = '+str(kappa_z)+' done \n')

end = time()
print(f'It took {end - start} seconds!')