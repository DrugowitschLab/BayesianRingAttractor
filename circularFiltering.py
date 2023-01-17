
"""
Modified on Tue Jan 17 2023

@author: akutschireiter

"""

import numpy as np
from scipy.stats import vonmises
from scipy.special import i0, i1
from scipy.optimize import root_scalar
from numpy.core.numeric import isscalar


##### Helper functions #####

def A_Bessel(kappa): 
    """Computes the ratio of Bessel functions."""
    r = i1(kappa)/i0(kappa)
    return r

def xi_fun_inv(dt):
    """Computes the inverse of the ratio of Bessel functions by root-finding."""
    f = lambda alpha: alpha * A_Bessel(alpha) - dt
    sol = root_scalar(f,bracket=[0.001,50],method='brentq')
    alpha = sol.root
    return alpha

def f_kappa(kappa): 
    """ Computes the precision decay function in the circKF. """
    f = A_Bessel(kappa)/(kappa-A_Bessel(kappa)-kappa*A_Bessel(kappa)**2)
    return f

def polar_to_euclidean(r,phi):
    """ Converts a polar coordinate with radius r and angle phi to Cartesian coordinates. """
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    return x, y

def euclidean_to_polar(x,y):
    """ Converts a Cartesian to polar coordinates. """
    r = np.sqrt( x**2 + y**2 )
    phi = np.arctan2(y,x)
    return phi,r

def xi(alpha):
    """ Xi function. Used to compute the Fisher information for a single observation with precision alpha. """
    dt = alpha * i1(alpha)/i0(alpha)
    return dt

def circular_mean(phi,w=None):
    """ Computes a (weighted) circular mean of the vector of angles phi.
    Input:
        phi - angular positions of the particles
        w - weights
    Output:
        phi_hat - estimated angle
        r_hat - estimated precision in [0,1] """

    x = np.cos(phi)
    y = np.sin(phi)
    X = np.average(x,weights=w)
    Y = np.average(y,weights=w)
    
    # convert average back to polar coordinates
    phi_hat = np.arctan2(Y,X)
    r_hat = np.sqrt( X**2 + Y**2 )
    
    return phi_hat, r_hat


def backToCirc(phi):
    """Makes sure the angle phi is in [-pi,pi]."""
    phi = ( (phi+np.pi) % (2*np.pi) ) - np.pi
    return phi

def circDist(x,y):
    """Computes the circular distance between angles x and y."""
    d = x - y
    if np.abs(d) > np.pi:
        if y < x:
            d = x - (y+2*np.pi)
        else:
            d = (x + 2*np.pi) - y
    return d





##### Generate the data #####

def generateData(T,kappa_phi,kappa_v=0,kappa_z=0,dt=0.01,phi_0=0,kappa_0=0):

    """ Generates artifical data according to the generative model. 
    The hidden trajectory (true HD) is a diffusion on a circle. Draws increment and HD 
    observations.
    Input:
    T - simulation length
    kappa_phi - inverse diffusion constant
    kappa_v - reliability of increment observations
    kappa_z - information rate of increment observations 
    dt - time step
    phi_0 - initial mean 
    kappa_0 - initial certainty 
    Output:
    phi - trajectory of hidden process / true HD
    dy - increment observations
    z - HD observations
    """

    # hidden state init
    phi = np.zeros(int(T/dt)) 
    if kappa_0 == 0:
        phi[0] = (phi_0 + np.pi ) 
    else:
        phi_0 = np.random.vonmises(phi_0,kappa_0)
        phi[0] = (phi_0 + np.pi )

    # generate sequence
    for i in range(1,int(T/dt)):
        phi[i] = np.random.normal(phi[i-1],1/np.sqrt(kappa_phi) * np.sqrt(dt))

    # increment observations
    dy = np.zeros(int(T/dt)) 
    if kappa_v != 0:
        for i in range(1,int(T/dt)):
            dy[i] = np.random.normal(phi[i]-phi[i-1],1/np.sqrt(kappa_v) * np.sqrt(dt)) 
    
    # correct range for hidden state
    phi = (phi % (2*np.pi) ) - np.pi # range [-pi,pi]

    # HD observations
    z = np.zeros(int(T/dt))
    if kappa_z != 0:
        z = np.random.vonmises(phi,kappa_z)
    
    return phi, dy, z




##### Filtering algorithms

###### PROJECTION FILTERS
## Von Mises Projection Filter (aka circular Kalman filter)
def vM_Projection(mu,kappa,kappa_phi,z=None,kappa_z=0,dy=None,kappa_v=0,dt=0.01):
    """" A single step of the circular Kalman filter, using Euler-Maruyama.
    
    Input:
    mu          - mean estimate before update
    kappa       - certainty estimate before update
    kappa_phi   - inverse diffusion constant of hidden state process
    z           - HD observation
    kappa_z     - reliability of single HD observation (notation different from manuscript!)
    dy          - increment observation
    kappa_v     - precision of increment observation
    dt          - time step
    
    Output:
    mu_out      - mean estimate after update
    kappa_out   - certainty estimate after update """
    

    # update (on natural parameters -> robust in discrete time)
    if kappa_z != 0:
        az,bz = polar_to_euclidean(kappa_z,z)
        a,b = polar_to_euclidean(kappa,mu)
        a = a+az
        b = b+bz
        mu, kappa = euclidean_to_polar(a,b)

    # prediction (include increment observations)
    if kappa_v != 0:
        dmu_pred = kappa_v/(kappa_phi+kappa_v) * dy
    else:
        dmu_pred = 0
    dkappa_pred = - 1/2 * 1/(kappa_phi + kappa_v) * kappa * f_kappa(kappa) * dt

    mu_out = mu + dmu_pred
    mu_out = ((mu_out + np.pi) % (2*np.pi) ) - np.pi # mu in[-pi,pi]
    kappa_out = kappa + dkappa_pred
    
    return (mu_out,kappa_out)


def vM_Projection_Run(T,kappa_phi,z=None,kappa_z=0,dy=None,kappa_v=0,
                        phi_0=0,kappa_0=10,dt=0.01):
    """ Runs the circular Kalman filter for a sequence of observations.
    Input:
    T           - time at simulation end
    kappa_phi   - inverse diffusion constant of hidden state process
    z           - direct observation sequence
    alpha       - precision of direct observations
    kappa       - certainty estimate before update
    dy          - increment observation sequence
    kappa_v     - precision of increment observations
    phi_0       - initial mean estimate
    kappa_0     - initial precision estimate
    dt          - time step
    
    Output:
    mu_out      - mean estimate after update
    kappa_out   - certainty estimate after update 
    """

    mu = np.zeros(int(T/dt))
    mu[0] = phi_0
    kappa = np.zeros(int(T/dt))
    kappa[0] = kappa_0
    if kappa_v == 0:
        dy = np.zeros(int(T/dt))
    if isscalar(kappa_z):
        kappa_z = kappa_z * np.ones(int(T/dt))

    for i in range(1,int(T/dt)):
        [mu[i],kappa[i]] = vM_Projection(mu[i-1],kappa[i-1],
                                        kappa_phi, #diffusion
                                        z=z[i],kappa_z=kappa_z[i], # direct obs
                                        dy=dy[i],kappa_v=kappa_v, #relative heading info
                                        dt=dt)
    return mu, kappa



## Quadratic decay approximation of the projection filter
def vM_Projection_quad(mu,kappa,kappa_phi,z=None,kappa_z=0,dy=None,kappa_v=0,dt=0.01):
    """" A single step of the quadrativ approximation of the circular Kalman filter, using Euler-Maruyama.
    
    Input:
    mu          - mean estimate before update
    kappa       - certainty estimate before update
    kappa_phi   - inverse diffusion constant of hidden state process
    z           - HD observation
    kappa_z     - reliability of single HD observation (notation different from manuscript!)
    dy          - increment observation
    kappa_v     - precision of increment observation
    dt          - time step
    
    Output:
    mu_out      - mean estimate after update
    kappa_out   - certainty estimate after update """

    # update (on natural parameters -> robust in discrete time)
    if kappa_z != 0:
        az,bz = polar_to_euclidean(kappa_z,z)
        a,b = polar_to_euclidean(kappa,mu)
        a = a+az
        b = b+bz
        mu, kappa = euclidean_to_polar(a,b)

    # prediction (include increment observations)
    if kappa_v != 0:
        dmu_pred = kappa_v/(kappa_phi+kappa_v) * dy
    else:
        dmu_pred = 0
    dkappa_pred = - 1/(kappa_phi + kappa_v) * ( kappa**2 - kappa ) * dt

    mu_out = mu + dmu_pred
    mu_out = ((mu_out + np.pi) % (2*np.pi) ) - np.pi # mu in[-pi,pi]
    kappa_out = kappa + dkappa_pred
    
    return (mu_out,kappa_out)

def vM_Projection_quad_Run(T,kappa_phi,z=None,kappa_z=0,dy=None,kappa_v=0,
                        phi_0=0,kappa_0=10,dt=0.01):
    """ Runs the circular Kalman filter for a sequence of observations.
    Input:
    T           - time at simulation end
    kappa_phi   - inverse diffusion constant of hidden state process
    z           - direct observation sequence
    alpha       - precision of direct observations
    kappa       - certainty estimate before update
    dy          - increment observation sequence
    kappa_v     - precision of increment observations
    phi_0       - initial mean estimate
    kappa_0     - initial precision estimate
    dt          - time step
    
    Output:
    mu          - mean estimate after update
    kappa       - certainty estimate after update 
    """

    mu = np.zeros(int(T/dt))
    mu[0] = phi_0
    kappa = np.zeros(int(T/dt))
    kappa[0] = kappa_0
    if kappa_v == 0:
        dy = np.zeros(int(T/dt))
    if isscalar(kappa_z):
        kappa_z = kappa_z * np.ones(int(T/dt))
    for i in range(1,int(T/dt)):
        [mu[i],kappa[i]] = vM_Projection_quad(mu[i-1],kappa[i-1],
                                        kappa_phi, #diffusion
                                        z=z[i],kappa_z=kappa_z[i], # direct obs
                                        dy=dy[i],kappa_v=kappa_v, #relative heading info
                                        dt=dt)
    return mu, kappa




########### Baseline filter, fixed uncertainty

def no_uncertainty_filter(mu,kappa, kappa_phi,z=None,kappa_z=0,dy=None,kappa_v=0,dt=0.01):
    """" A single step of the circular Kalman filter, where kappa is held fixed (and given as argument).
    
    Input:
    mu          - mean estimate before update
    kappa       - certainty (will not be updated)
    kappa_phi   - inverse diffusion constant of hidden state process
    z           - HD observation
    kappa_z     - reliability of single HD observation (notation different from manuscript!)
    dy          - increment observation
    kappa_v     - precision of increment observation
    dt          - time step
    
    Output:
    mu_out      - mean estimate after update """

    # update (on natural parameters -> robust in discrete time)
    if kappa_z != 0:
        az,bz = polar_to_euclidean(kappa_z,z)
        a,b = polar_to_euclidean(kappa,mu)
        a = a+az
        b = b+bz
        mu, kappa = euclidean_to_polar(a,b)

    # prediction (include increment observations)
    if kappa_v != 0:
        dmu_pred = kappa_v/(kappa_phi+kappa_v) * dy
    else:
        dmu_pred = 0

    mu_out = mu + dmu_pred
    mu_out = ((mu_out + np.pi) % (2*np.pi) ) - np.pi # mu in[-pi,pi]

    return mu_out
    

def no_uncertainty_filter_Run(T,kappa,kappa_phi,z=None,kappa_z=0,dy=None,kappa_v=0,
                        phi_0=0,dt=0.01):
    """ Runs the circular Kalman filter for a sequence of observations.
    Input:
    T           - time at simulation end
    kappa_phi   - inverse diffusion constant of hidden state process
    z           - direct observation sequence
    alpha       - precision of direct observations
    kappa       - certainty estimate before update
    dy          - increment observation sequence
    kappa_v     - precision of increment observations
    phi_0       - initial mean estimate
    dt          - time step
    
    Output:
    mu_out      - mean estimate after update
    kappa_out   - certainty estimate after update 
    """
    mu = np.zeros(int(T/dt))
    mu[0] = phi_0
    if kappa_v == 0:
        dy = np.zeros(int(T/dt))
    for i in range(1,int(T/dt)):
        mu[i] = no_uncertainty_filter(mu[i-1],kappa,
                                        kappa_phi, #diffusion
                                        z=z[i],kappa_z=kappa_z, # direct obs
                                        dy=dy[i],kappa_v=kappa_v, #relative heading info
                                        dt=dt)
    return mu


########### Network-like filter

def network_filter(mu,kappa,w_even=0,w_odd=0,tau=1,w_quad=0,z=None,I_ext=0,dy=None,dt=0.01):
    """" A single step of the quadratic approximation to the circular Kalman filter, with parameters 
    given as network parameters of a single-population network tuned to perform Bayesian filtering.
    
    Input:
    mu          - mean estimate before update
    kappa       - certainty 
    w_even      - even recurrent connectivity
    w_odd       - odd recurrent connectivity
    tau         - decay time constant
    w_quad      - quadratic weight
    I_ext       - external input
    kappa_z     - reliability of single HD observation (notation different from manuscript!)
    dy          - increment observation
    dt          - time step
    
    Output:
    mu_out      - mean estimate after update
    kappa_out   - certainty estimate after update """

    if I_ext != 0:
        az,bz = polar_to_euclidean(I_ext,z)
        a,b = polar_to_euclidean(kappa,mu)
        a = a+az
        b = b+bz
        mu, kappa = euclidean_to_polar(a,b)

    # prediction (include increment observations)
    if w_odd != 0:
        dmu_pred = w_odd * dy
    else:
        dmu_pred = 0
    dkappa_pred = (w_even - 1/tau) * kappa * dt - w_quad * kappa**2 * dt 

    kappa_out = kappa + dkappa_pred

    mu_out = backToCirc( mu + dmu_pred )
    mu_out = ((mu_out + np.pi) % (2*np.pi) ) - np.pi # mu in[-pi,pi]

    return mu_out, kappa_out
    

def network_filter_Run(T,w_even=0,w_odd=0,tau=1,w_quad=0,I_ext=0,z=None,dy=None,phi_0=0,kappa_0=10,dt=0.01):
    "Runs the network filter for time T."

    mu = np.zeros(int(T/dt))
    mu[0] = phi_0
    kappa = np.zeros(int(T/dt))
    kappa[0] = kappa_0
    if w_odd == 0:
        dy = np.zeros(int(T/dt))
    for i in range(1,int(T/dt)):
        mu[i],kappa[i] = network_filter(mu[i-1],kappa[i-1],
                                        w_even=w_even, tau=tau, #diffusion
                                        w_quad = w_quad, #quadratic decay
                                        z=z[i],I_ext=I_ext, # direct obs
                                        dy=dy[i],w_odd=w_odd, #relative heading info
                                        dt=dt)
    return mu, kappa


########### RNN, single population filter
def RNN_filter_Run(T,N=100,dt=0.01,
    w_even=0,w_odd=0,tau=1,w_quad=0,
    I_ext=0,stoch_corr=0,dy=None,phi_0=0,kappa_0=10,
    sigma_N = 0):
    """" Runs a recurrent neural network dynamics, with parameters matched to 
    approximate the circKF.
    
    Input:
    T           - simulation time
    N           - number of neurons
    dt          - time step
    w_even      - even recurrent connectivity
    w_odd       - odd recurrent connectivity
    tau         - decay time constant
    w_quad      - quadratic weight
    I_ext       - external input
    stoch_corr  - stochastic correction (additional decay due to Ito conversion)
    dy          - increment observation
    phi_0       - initial mean estimate
    kappa_0       - initial certainty estimate
    sigma_N     - neural noise
    
    Output:
    mu_out      - mean estimate after update
    kappa_out   - certainty estimate after update """

    f_act = lambda x: np.maximum(0,x)

    # vector of preferred HD
    phi_0_r = np.linspace(-np.pi,np.pi-(2*np.pi)/N,N)

    # set up even recurrent connectivity matrix
    W_even = np.zeros([N,N])
    for i in np.arange(N):
        for j in np.arange(N):
            W_even[i,j] = 2/N * (  w_even * np.cos(phi_0_r[i] - phi_0_r[j]) )

    # set up odd recurrent connectivity matrix
    W_odd = np.zeros([N,N])
    for i in np.arange(N):
        for j in np.arange(N):
            W_odd[i,j] = 2/N * (  w_odd * np.sin(phi_0_r[i] - phi_0_r[j]) )
    
    # set up all-to-all summation
    M = np.pi/N * np.ones([N,N])

    if w_odd == 0:
        dy = np.zeros(int(T/dt))

    if z is None:
        z = np.zeros(int(T/dt))
        I_ext = 0

    # add Wiener process if there is neural noise
    if sigma_N != 0:
        dW = np.sqrt(dt) * np.random.randn(int(T/dt),N)
    else:
        dW = np.zeros((int(T/dt),N))

    # init
    r = np.zeros((int(T/dt),N))
    r[0] = kappa_0 * np.cos(phi_0_r - phi_0)

    # run network filter
    for i in range(1,int(T/dt)):
        W =  W_even + W_odd * dy[i]/dt
        r[i] = (r[i-1] 
                - stoch_corr * r[i-1] * dt # stochastic correction
                - 1/tau * r[i-1] * dt # decay
                + np.dot(W,r[i-1]) * dt # angular velocity integration, recurrent stabilization
                - w_quad * np.dot(M,f_act(r[i-1])) * r[i-1] * dt # quadratic inhibition
                + I_ext * np.cos(phi_0_r-z[i]) # absolute heading info (external input)
                + sigma_N * dW[i])

    # decode stochastic variables
    A_cos =  np.array([np.cos(phi_0_r),np.sin(phi_0_r)]) 
    theta = 2/N * np.dot(A_cos,r.transpose()) # FT in Cartesian Domain
    kappa = np.sqrt(np.sum(theta**2,0))
    mu = np.arctan2(theta[1],theta[0])

    return r, mu, kappa


########### Particle filters
## Circular particle filter
def PF_effectiveDiff(x_in,w,kappa_phi,z=None,kappa_z=0,dy=None,kappa_v=0,dt=0.01):
    """" A single step in the particle filter, reliazed by propagating the particles
    through an effective diffusion (modulated by increment observations).
    
    Input:
    x_in        - initial particle positions
    w           - particle importance weights
    kappa_phi   - inverse diffusion constant of hidden state process
    z           - HD observation
    kappa_z     - precision of HD observation
    dy          - increment observation
    kappa_v     - precision of increment observation
    dt          - time step
    
    Output:
    x_out       - particle positions
    w           - particle weights
    
    """
    n = x_in.shape[0]
    kappa_eff = kappa_v + kappa_phi
    
    # particle diffusion
    x_in = x_in + np.pi # range between 0 and 2pi
    dx = np.random.normal(kappa_v/kappa_eff*dy,1/np.sqrt(kappa_eff) * np.sqrt(dt),n)
    x_out = x_in + dx
    x_out = (x_out % (2*np.pi) ) - np.pi # range between -pi and pi
    
    # compute weights
    if kappa_z != 0:       # only compute weights and resample if there is an observation
        w = w * vonmises.pdf(x_out, kappa_z,loc=z)
        w = w/np.sum(w)
        #resampling
        N_eff = 1/np.sum(w**2)
        if N_eff/n < 0.5:
            # print(N_eff)
            x_out = np.random.choice(x_out,n,p=w)
            w = 1/n * np.ones(n)
    
    return x_out, w

def PF_run(T,N,kappa_phi,z=None,kappa_z=0,dy=None,kappa_v=0,
                        phi_0=0,kappa_0=10,dt=0.01):
    """ Runs the particle filter for a sequence of observations."""

    mu = np.zeros(int(T/dt))
    mu[0] = phi_0
    r = np.zeros(int(T/dt))
    r[0] = A_Bessel(kappa_0)
    phi_PF = np.random.vonmises(phi_0,kappa_0,N)
    w = 1/N * np.ones(N)
    if kappa_v == 0:
        dy = np.zeros(int(T/dt))
    # propagate particles
    for i in range(1,int(T/dt)):
        phi_PF,w = PF_effectiveDiff(phi_PF,w,kappa_phi,z=z[i],kappa_z=kappa_z,dy=dy[i],kappa_v=kappa_v,dt=dt)
        mu[i],r[i] = circular_mean(phi_PF,w) 
        
    return mu, r





#### Plotting

# circular plotting, no weird boundary effects
def circplot(t,phi):
    """ Stiches t and phi to make unwrapped circular plot. """
    
    phi_minus = phi - 2*np.pi
    phi_plus = phi + 2*np.pi

    phi_array = np.array((phi_plus , phi , phi_minus))
    difference = np.abs(phi_array[:,1:] - phi[0:-1])
    ind_up = np.where(np.argmin(difference,axis=0)==0)[0]
    ind_down = np.where(np.argmin(difference,axis=0)==2)[0]
    ind = np.union1d(ind_up,ind_down)

    phi_stiched = np.copy(phi)
    t_stiched = np.copy(t)
    for i in np.flip(np.arange(ind.size)):
        idx = ind[i]
        if np.isin(idx,ind_up):
            phi_stiched = np.concatenate((phi_stiched[0:idx+1],[phi_plus[idx+1]],
                                          [np.nan],[phi_minus[idx]],phi_stiched[(idx+1):]))
        else:
            phi_stiched = np.concatenate((phi_stiched[0:idx+1],[phi_minus[idx+1]],
                                          [np.nan],[phi_plus[idx]],phi_stiched[(idx+1):]))
        t_stiched = np.concatenate((t_stiched[0:idx+2],[np.nan],t_stiched[idx:]))

    
    return t_stiched,phi_stiched





