
"""
Created on Mon Apr 4 2022

@author: akutschireiter
"""

import numpy as np
import circularFiltering as flt

def IdMatr(x_to,x_from,sig=0.001):
    N_to = x_to.size
    N_from = x_from.size
    IdM = np.zeros((N_to,N_from))
    for i in range(N_to):
        for j in range(N_from):
            dist = np.arccos(np.cos(x_to[i]-x_from[j]))
            IdM[i,j] = (1/(np.sqrt(2*np.pi)*sig)
                            * np.exp(-(1/2)*(dist**2/sig**2 ) ) )
    IdM = IdM/np.sum(IdM,1)[0]  # normalize
    return IdM


def create_connmatrix(N_HD,N_Del7,N_AVplus,N_AVminus):
    
    sig_AVminus = 0.3
    sig_AVplus = 0.3
    sig_Del7 = 0.1

    # preferred angles
    phi_0_HD = np.linspace(-np.pi,np.pi-(2*np.pi)/N_HD,N_HD)  # HD preferred angle
    phi_0_Del7 = np.linspace(-np.pi,np.pi-(2*np.pi)/N_Del7,N_Del7)
    phi_0_AVplus = np.linspace(-np.pi,np.pi-(2*np.pi)/N_AVplus,N_AVplus)
    phi_0_AVminus = np.linspace(-np.pi,np.pi-(2*np.pi)/N_AVminus,N_AVminus)

    ### HD population
    phi_0_HD = np.linspace(-np.pi,np.pi-(2*np.pi)/N_HD,N_HD)  # preferred angle

    W_HD_HD = np.zeros((N_HD,N_HD)) # recurrent connectivity matrix HD -> HD
    w0 = 0
    w1 = 2
    o = -0.2
    for i in range(N_HD):
        for j in range(N_HD):
            W_HD_HD[i,j] = 2/N_HD * (o + np.maximum(0,w0+w1*np.cos(phi_0_HD[i]-phi_0_HD[j]))) 
            
    W_HD_AVplus = np.zeros((N_HD,N_AVplus))
    for i in range(N_HD):
        for j in range(N_AVplus):
            W_HD_AVplus[i,j] = 2/N_AVplus * np.maximum(0,np.sin(phi_0_HD[i]-phi_0_AVplus[j]+np.pi/4))
            
    W_HD_AVminus = np.zeros((N_HD,N_AVminus))
    for i in range(N_HD):
        for j in range(N_AVminus):
            W_HD_AVminus[i,j] = 2/N_AVminus * np.maximum(0,-np.sin(phi_0_HD[i]-phi_0_AVminus[j]-np.pi/4))


    ## AVplus population
    W_AVplus_HD = IdMatr(phi_0_AVplus,phi_0_HD,sig_AVplus) 

    ## AVminus population
    W_AVminus_HD = IdMatr(phi_0_AVminus,phi_0_HD,sig_AVminus) 

    ## Delta7 population
    W_Del7_HD = np.zeros((N_Del7,N_HD)) # HD -> GI connectivity matrix
    w0 = 0.5 #(np.pi-1)/np.pi
    w1 = - 1/2
    for i in range(N_Del7):
        for j in range(N_HD):
            W_Del7_HD[i,j] = 2/N_Del7 *(w0 + w1* np.cos(phi_0_Del7[i] - phi_0_HD[j])) 
            # W_Del7_HD[i,j] = 2/N_HD *(w0 + w1* np.cos(phi_0_Del7[i] - phi_0_HD[j]))

    m0 = 0.1
    m1 = 0
    W_Del7_Del7 = np.zeros((N_Del7,N_Del7))
    for i in range(N_Del7):
        for j in range(N_Del7):
            W_Del7_Del7[i,j] = 2/N_Del7 *(m0 + m1* np.cos(phi_0_Del7[i] - phi_0_Del7[j]))


    j = np.pi/2 * (1 - 2*m0)/w0
    W_HD_Del7 = - j * IdMatr(phi_0_HD,phi_0_Del7,sig_Del7)  

    params = {
        "W_HD_HD" : W_HD_HD,
        "W_HD_AVplus" : W_HD_AVplus,
        "W_HD_AVminus" : W_HD_AVminus,
        "W_AVplus_HD" : W_AVplus_HD,
        "W_AVminus_HD" : W_AVminus_HD,
        "W_Del7_HD" : W_Del7_HD,
        "W_Del7_Del7" : W_Del7_Del7,
        "W_HD_Del7" : W_HD_Del7
    }

    return params

def network_filter_Run(T,kappa_phi,dy=None,kappa_y=0,z=None,I_ext=0,
                        phi_0=0,kappa_0=10,dt=0.001):
    # N_HD = 16
    # N_Del7 = 11
    # N_AVplus = 8
    # N_AVminus = 8

    N_HD = 100
    N_Del7 = 100
    N_AVplus = 50
    N_AVminus = 50

    # preferred angles
    phi_0_HD = np.linspace(-np.pi,np.pi-(2*np.pi)/N_HD,N_HD)  # HD preferred angle
    phi_0_Del7 = np.linspace(-np.pi,np.pi-(2*np.pi)/N_Del7,N_Del7)

    AV_offset = 0  # let's give it a lot

    params = create_connmatrix(N_HD,N_Del7,N_AVplus,N_AVminus)

    #unpack
    W_HD_HD = params['W_HD_HD']
    W_HD_AVplus = params['W_HD_AVplus']
    W_HD_AVminus = params['W_HD_AVminus']
    W_AVplus_HD = params['W_AVplus_HD']
    W_AVminus_HD = params['W_AVminus_HD']
    W_Del7_Del7 = params['W_Del7_Del7']
    W_HD_Del7 = params['W_HD_Del7']
    W_Del7_HD = params['W_Del7_HD']
    
    w0 = 0
    w1 = 2
    m0 = 0.1
    m1 = 0

    alpha_tilde = 10
    alpha = alpha_tilde + 1/2 * kappa_y/kappa_phi * 1/(kappa_phi + kappa_y)

    # compute the prefactors (strength modulator)
    w_HD_HD = alpha_tilde + 1/(kappa_phi + kappa_y) - AV_offset*kappa_y/(kappa_y+kappa_phi)
    w_HD_AVplus = np.sqrt(2) * kappa_y/(kappa_y+kappa_phi) # note sqrt(2) due to 45deg
    w_HD_AVminus = np.sqrt(2) * kappa_y/(kappa_y+kappa_phi)
    w_HD_Del7 = 1/(kappa_phi+kappa_y) 
    w_AVplus_HD = 1
    w_AVminus_HD = 1
    w_Del7_HD = 1

    # time constants
    tau_AVplus = 0.01
    tau_AVminus = 0.01
    tau_Del7 = 0.001

    # activation function of global inhibitory population
    fact = lambda x: np.maximum(0,x) #threshold linear

    # init
    r_HD = np.zeros((int(T/dt),N_HD))
    r_HD[0] = kappa_0 * np.cos(phi_0_HD - phi_0)
    r_AVplus = np.zeros((int(T/dt),N_AVplus))
    r_AVplus[0] = w_AVplus_HD * W_AVplus_HD @ r_HD[0]
    r_AVminus = np.zeros((int(T/dt),N_AVminus))
    r_AVminus[0] = w_AVminus_HD * W_AVminus_HD @ r_HD[0]
    r_Del7 = np.zeros((int(T/dt),N_Del7))
    r_Del7[0] = 2 * w0 * kappa_0/(np.pi * (1-2 * m0)) + w1 * kappa_0 /(2*(1-m1)) * np.cos(phi_0_Del7 - phi_0)


    # run
    for i in range(1,int(T/dt)): 
        r_HD[i] = (r_HD[i-1] - alpha * r_HD[i-1] * dt  # leak
                + w_HD_HD * W_HD_HD @ r_HD[i-1] * dt # recurrent connectivity
                + w_HD_AVplus * W_HD_AVplus @ r_AVplus[i-1] * dt # AV integration (+)
                + w_HD_AVminus * W_HD_AVminus @ r_AVminus[i-1] * dt # AV integration (-)
                + w_HD_Del7 * (W_HD_Del7 @ fact(r_Del7[i-1]) ) * r_HD[i-1] * dt  # quadratic interaction
                + I_ext * np.cos(phi_0_HD-z[i])) # external input
        r_AVplus[i] = ( r_AVplus[i-1] 
                + 1/tau_AVplus * (-r_AVplus[i-1] + (dy[i]/dt + AV_offset) * w_AVplus_HD * W_AVplus_HD @ r_HD[i] )*dt)
        r_AVminus[i] = ( r_AVminus[i-1] 
                + 1/tau_AVminus * (-r_AVminus[i-1] + (-dy[i]/dt + AV_offset) * w_AVminus_HD * W_AVminus_HD @ r_HD[i] )*dt)
        r_Del7[i] = ( r_Del7[i-1] 
                + 1/tau_Del7 * (-r_Del7[i-1] 
                                + w_Del7_HD * W_Del7_HD @ fact(r_HD[i]) 
                                + W_Del7_Del7 @ fact(r_Del7[i-1]) )  * dt )
        
        
    # decode stochastic variables from HD population
    A_cos =  np.array([np.cos(phi_0_HD),np.sin(phi_0_HD)])
    theta = 2/N_HD * A_cos @ r_HD.transpose() # FT in Cartesian domain
    kappa = np.sqrt(np.sum(theta**2,0)) #convert to polar coordinates
    mu = np.arctan2(theta[1],theta[0])
    return mu, kappa