import numpy as np
from scipy.special import erf
import autograd.numpy as nup

def h(pos,min_,max_,sigma=0.01,ac=0.02):
    """
    This is the function h(alpha) used in the paper by Amish on INDUS
    This function that the form 

    h(alpha_i) = \int_{amin}^{amax} \Phi(alpha-alpha_i) dr
    where 

    \phi(alpha_i) = k^-1*[e^{-alpha^{2}/(2sigma^{2})} - e^{-alphac^{2}/(2sigma^{2})}]
    where k is the normalizing constant
    
    Args:
        pos(numpy.ndarray)           : Input positions (N,dim) 
        min_(numpy.ndarray or float) : The minimum of the probe volume (dim,)
        max_(numpy.ndarray or float) : The maximum of the probe volume (dim,)
        sigma(float)                 : sigma as defined in the paper by Amish
        ac(float)                    : alpha c as defined in the paper by Amish

    output:
        a float/int or numpy array depending on the input alpha_i
        if alpha_i is float/int, then output will be int that corresponds to h(alpha_i)
        else if alpha_i is numpy array, then output will be numpy array that corresponds to h(alpha_i)
    """
    # normalizing constants
    k = np.sqrt(2*np.pi*sigma**2) * erf(ac/np.sqrt(2*sigma**2)) - 2*ac*np.exp(-ac**2/(2*sigma**2))
    k1 = 1/k*np.sqrt(np.pi*sigma**2/2)
    k2 = 1/k*np.exp(-ac**2/(2*sigma**2))

    p1 = (k1*erf((max_ - pos)/(np.sqrt(2)*sigma)) - k2*(max_ - pos) - 1/2)*\
    np.heaviside(ac - np.abs(max_ - pos), 1.0)

    p2 = (k1*erf((pos - min_)/(np.sqrt(2)*sigma)) - k2*(pos - min_) - 1/2)*\
    np.heaviside(ac - np.abs(pos - min_),1.0)

    p3 = np.heaviside(ac + 1/2*(max_ - min_) - np.abs(pos - 1/2*(max_ + min_)),1.0)
    h = p1 + p2 + p3

    return h

def phi(alpha,sigma=0.01,ac=0.02):
    """
    Calculate phi function in INDUS where h = \int phi

    Args:
        alpha(numpy.ndarray)    : The alpha variable in the phi function for INDUS
        sigma(float)            : The sigma as defined in INDUS
        ac(float)               : The alphac as defined in INDUS
    """
    k = np.sqrt(2*np.pi*sigma**2) * erf(ac/np.sqrt(2*sigma**2)) - 2*ac*np.exp(-ac**2/(2*sigma**2))
    p = 1/k*(np.exp(-alpha**2/(2*sigma**2)) - np.exp(-ac**2/(2*sigma**2)))*np.heaviside(ac - np.abs(alpha),1)
    return p

def hprime(pos,max_,min_,indicator,sigma=0.01,alphac=0.02):
    """
    Function that calculates derivatve of h -> which we call h prime

    Args:
        pos(numpy.ndarray)  : The positions of the atoms ((N,3))
        max_(numpy.ndarray) : The maximum of the probe volume ((3,))
        min_(numpy.ndarray) : The minimum of the probe volume ((3,))
        indicator(numpy.ndarray)    : The indicator function for each of the atom ((N,3))
        sigma(float)        : sigma as defined in INDUS 
        alphac(float)       : alphac as defined in INDUS 
    
    Returns:
        derivative of hprime(numpy.ndarray) : ((N,3))
    """
    deriv_x = -(phi(max_[0]-pos[:,0],sigma,alphac) - phi(min_[0]-pos[:,0],sigma,alphac))*indicator[:,1]*indicator[:,2]
    deriv_y = -(phi(max_[1]-pos[:,1],sigma,alphac) - phi(min_[1]-pos[:,1],sigma,alphac))*indicator[:,2]*indicator[:,0]
    deriv_z = -(phi(max_[2]-pos[:,2],sigma,alphac) - phi(min_[2]-pos[:,2],sigma,alphac))*indicator[:,1]*indicator[:,0]

    deriv = np.hstack((deriv_x[:,np.newaxis],deriv_y[:,np.newaxis],deriv_z[:,np.newaxis]))

    return deriv

def equilibrium_k0(LC,xmin,xmax,ymin,ymax,zmin,zmax,start_t,end_t,alpha_c=0.02,sigma=0.01,skip=1,verbose=False):
    """
    A function that approximates k0 using rule of thumb 
    This function is used specifically for a cuboidal probe volume 
    The rule of thumb provided by Nick Rego is as follows:
    (a) Assume underlying free energy U0 is gaussian with <N>0 and var<N>0. 
        U0 = k0/2(N-N0)^2 so k0~1/var(N)0
    inputs:
        LC: Liquid crystal object 
        mumin: minimum mu of the probe volume (all 6)
        start_t: the starting time of the calculation
        end_t: the ending time of the calculation
        skip: the number of frames to skip between start_t, end_t
        alpha_c: alpha_c for cut-off of INDUS function h 
        sigma: the width for the INDUS function h 

    """
    time = LC.time
    time_idx = np.linspace(0,time,len(LC)) # the time index during the simulation
    start_idx = np.searchsorted(time_idx,start_t,side='left')
    end_idx = np.searchsorted(time_idx,end_t,side='right')
    calc_time = np.arange(start_idx,end_idx,skip) 

    N_tilde_tot = np.zeros((len(calc_time),))
    N_tot = np.zeros((len(calc_time),))
    ix = 0
    if LC.bulk == True:
        u = LC.universe
        atoms = u.select_atoms("resname {}CB".format(LC.n))
    else:
        u = LC.universe
        atoms = u.select_atoms("all")
    

    for idx in calc_time:
        u.trajectory[idx]
        pos = atoms.positions  

        # satisfy x positions
        pos = pos[pos[:,0]>=xmin]
        pos = pos[pos[:,0]<=xmax]
        # satisfy y positions
        pos = pos[pos[:,1]>=ymin]
        pos = pos[pos[:,1]<=ymax]
        # satisfy z positions
        pos = pos[pos[:,2]>=zmin]
        pos = pos[pos[:,2]<=zmax]

        h_x = h(pos[:,0],xmin,xmax,sigma,alpha_c)
        h_y = h(pos[:,1],ymin,ymax,sigma,alpha_c)
        h_z = h(pos[:,2],zmin,zmax,sigma,alpha_c)

        h_ = h_x*h_y*h_z
        N_tilde = h_.sum()
        N_tilde_tot[ix] = N_tilde
        N_tot[ix] = len(pos)
        ix += 1

        if verbose:
            print("time at {} is done,Ntilde is {}".format(idx,N_tilde))

    return (N_tilde_tot,N_tot,calc_time)
 
