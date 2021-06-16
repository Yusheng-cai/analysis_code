import numpy as np
from scipy.special import erf

from .ProbeVolume import _ProbeVolume
from .indicator_func_1b import indicator_func_1b
from .indicator_func_2b import indicator_func_2b


class ProbeVolume_cylinder(_ProbeVolume):
    """
    Probe Volume for a cylindrical probeVolume

    Args:
        tpr (str)               : The name of the tpr file for the universe
        xtc (str)               : The name of the xtc/trr file for the universe
        base(np.ndarray)        : The base of the cylinder (3,)
        h (float)               : height of the cylinder
        radius (float)          : The radius of the cylinder 
        dir_ (str)              : Director of the principle axis of the cylinder (x, y or z)
        sigma(float)            : sigma for coarse-graining (in units of A)
        ac(float)               : alphac for coarse-graining (in units of A)
    """          
    def __init__(self, tpr, xtc, base, h, radius, pbc=True, dir_ = 'z', sigma=0.1, ac=0.2):
        super().__init__(tpr, xtc, sigma=sigma, ac=ac)
        self.dict_      = {"x":0, "y":1, "z":2}
        self.dir_       = self.dict_[dir_]

        # store base & head of the cylinder
        self.base_      = base
        temp            = np.copy(base)
        temp[self.dir_] += h
        self.head_      = temp
        self.center_    = 1/2*(self.base_ + self.head_) 
        self.pbc_       = pbc

        # shifted head & base of the cylinder 
        self.base_shifted = self.base_ - self.center_ 
        self.head_shifted = self.head_ - self.center_
        self.d1_func_     = indicator_func_1b(radius, sigma=sigma, ac=ac) 
        self.d2_func_     = indicator_func_2b(self.base_shifted[self.dir_], self.head_shifted[self.dir_], sigma=sigma, ac=ac) 
    
    def calculate_Indicator(self, pos, ts):
        """
        Function that calculates the indicator function for a cylindrical probe volume

        Args:
        -----
            pos(np.ndarray)     : The positions of the atoms (N,3)
            ts(int)             : The time frame at which this calculation is performed
        
        Returns:
        --------
            indicator(np.ndarray) : The indicator value for each of the functions in shape (N, )
        """
        N            = pos.shape[0]
        r            = np.zeros((N,))
        center_      = self.center_
        bb           = self.bounding_box_
        d1_func      = self.d1_func_
        d2_func      = self.d2_func_
        dir_         = self.dir_
        pbc          = self.pbc_

        if pbc:
            dr           = bb.dr_pbc(pos - center_, ts) 
        else:
            dr           = pos

        for i in range(3):
            if i != dir_:
                r    += dr[:,i]**2
        
        r            = np.sqrt(r) 

        # calculate the indicators
        hz           = d2_func.calculate(dr[:,dir_])
        hr           = d1_func.calculate(r)
        h            = hz*hr

        htheta       = np.ones((N,1))
        self.hx_     = np.hstack((hr[:,np.newaxis], htheta, hz[:,np.newaxis]))

        return h
    
    def calculate_derivative(self, pos:np.ndarray, hx:np.ndarray, ts:int):
        """
        Function that calculates the derivative of the cylindrical probeVolume with respect to the positions (x,y,z)

        Args:
        ----
            pos(numpy.ndarray)      : The positions of the atoms passed in shape (N,3)
            hx(numpy.ndarray)       : The derivatives of the atoms with respect to the cylindrical coordinates (r, theta, z) where z is the principle axis of the cylinder
        
        Returns:
        --------
            dh_dr(numpy.ndarray)    : The derivative of ProbeVolume with respect to the positions (x,y,z)
        """
        dir_        = self.dir_
        N           = pos.shape[0]
        deriv       = np.zeros((N, 3))
        r           = np.zeros((N, ))
        d1_func     = self.d1_func_
        d2_func     = self.d2_func_
        center_     = self.center_
        pbc         = self.pbc_

        if pbc:
            dr          = self.bounding_box_.dr_pbc(pos-center_, ts)
        else:
            dr          = pos

        for i in range(3):
            if i != dir_:
                r    += dr[:,i]**2
        
        r            = np.sqrt(r) 

        d1_derivative = d1_func.calculate_derivative(r)
        d2_derivative = d2_func.calculate_derivative(dr[:,dir_])
        for i in range(3):
            if i == dir_:
                deriv[:,i] = d2_derivative*hx[:,0] # hx[:,0] refers to the r position
            else:
                deriv[:,i] = d1_derivative*hx[:,-1]*dr[:,i]/r # hx[:,-1] refers to the principle axis position
         
        return deriv