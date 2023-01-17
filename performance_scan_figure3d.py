import sys
import numpy as np
import circularFiltering as flt
from time import time

kappa_z_array = np.array([0.01,0.03,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,2,3,4,5,6,7,8,9,10,30,100])
len_kz = len(kappa_z_array)

# translate input varibles
kappa_star = float(sys.argv[1])
beta = float(sys.argv[2])

# number of iterations to average
nIter = 10000

# model parameters
T = 20 # simulation time
dt = 0.01 # step size
t = np.arange(0,T,dt)
timesteps = int(T/dt)
kappa_phi = 1 # inverse diffusion constant
phi_0 = 0 # initial mean
kappa_0 = 20 # initial certainty
kappa_y = 1 # certainty of increment observations

# run the simulations and read out first and second order statistics for each kappa_z
stats = np.zeros([len_kz,3])
start = time()

for i in np.arange(len_kz):

    kappa_z = kappa_z_array[i]

    phi_final = np.zeros([nIter])
    network_flt = np.zeros([nIter,2])

    # where to save
    filename = "data_raw/figure3d/ntwkflt_k="+str(kappa_star)+"_beta="+str(beta)

    # adjust parameters
    alpha = flt.xi_fun_inv(kappa_z * dt)

    for j in range(0, nIter): # run for nIter iterations

        # generate data
        phi, dy, z = flt.generateData(T,kappa_phi,kappa_y=kappa_y,dt=dt,phi_0=phi_0,kappa_0=kappa_0,kappa_z=alpha)

        # network-like filter
        tau = 1
        w_even = beta + 1/tau
        w_quad = beta / kappa_star
        w_odd = kappa_y/(kappa_phi+kappa_y)
        I_ext = alpha
        mu, kappa = flt.network_filter_Run(T,w_even=w_even,w_odd=w_odd,tau=tau,w_quad=w_quad,
                        I_ext=I_ext,z=z,dy=dy,phi_0=phi_0,kappa_0=kappa_0,dt=dt)

        # read out state at end of simulation
        phi_final[j] = phi[-1]
        network_flt[j] = [mu[-1],kappa[-1]]
    
    # compute statistics
    stats[i,[0,1]] = flt.circular_mean(flt.backToCirc(network_flt[:,0]-phi_final))
    stats[i,2] = np.mean(network_flt[:,1])


    # save the result
    np.savez(filename,stats=stats)

print('k_star='+ str(kappa_star) +', beta=' +str(beta) + ' done \n')

end = time()
print(f'It took {end - start} seconds!')