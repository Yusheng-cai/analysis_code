import numpy as np
from scipy.special import erf
import MDAnalysis as mda

from .ProbeVolume import _ProbeVolume
from .Bounding_box import Bounding_box
from .indicator_func_2b import indicator_func_2b

class ProbeVolume_Box(_ProbeVolume):
    """
    Probe Volume for a orthorhombic box

    Args:
        min_(numpy.ndarray)     : a (3,) numpy ndarray that contains the minimum of the Probe Volume Box
        max_(numpy.ndarray)     : a (3,) numpy ndarray that contains the maximum of the Probe Volume Box
        sigma(float)            : sigma for coarse-graining (in units of A)
        ac(float)               : alphac for coarse-graining (in units of A)
    """          
    def __init__(self,tpr:str,xtc:str, min_:np.ndarray, max_:np.ndarray, sigma=0.1,ac=0.2):
        super().__init__(tpr, xtc, sigma,ac)          
        self.max_   = max_
        self.min_   = min_

        # shift min_ & max_
        self.center_       = 1/2*(max_ + min_)
        self.max_shifted   = max_ - self.center_
        self.min_shifted   = min_ - self.center_

        self.func_  = indicator_func_2b(self.min_shifted, self.max_shifted, sigma=sigma, ac=ac)
    
    def calculate_Indicator(self,pos:np.ndarray, ts:int):
        """
        This is the function h(alpha) used in the paper by Amish on INDUS
        This function that the form 

        h(alpha_i) = \int_{amin}^{amax} \Phi(alpha-alpha_i) dr
        where 

        \phi(alpha_i) = k^-1*[e^{-alpha^{2}/(2sigma^{2})} - e^{-alphac^{2}/(2sigma^{2})}]
        where k is the normalizing constant
        
        Args:
            pos(numpy.ndarray)           : Input positions (N,dim) 

        Returns:
            indicator(numpy.ndarray)     : The indicator array in shape (N,1)
            hx(numpy.ndarray)            : The indicator array of each DIM (N,3)
        """
        bb     = self.bounding_box_

        pos_center    = bb.dr_pbc(pos - self.center_, ts) 
        hx            = self.func_.calculate(pos_center)

        indicator       = np.prod(hx,axis=1,keepdims=True)
        self.indicator_ = indicator
        self.Ntilde_    = indicator.sum()
        self.hx_        = hx

        return indicator, hx

    def calculate_derivative(self,pos,hx, ts):
        """
        Function that calculates derivatve of h -> which we call h prime

        Args:
            pos(numpy.ndarray)  : The positions of the atoms ((N,3))
            hx(numpy.ndarray)   : The hx function for each of the atom ((N,3))
            ts(int)             : The time step at which the calculation is performed on

        Returns:
            derivative of hprime(numpy.ndarray) : ((N,3))
        """
        center_       = self.center_
        bb            = self.bounding_box
        pos_center    = bb.dr_pbc(pos - center_, ts) 

        hxprime       = self.func_.calculate_derivative(pos_center)

        deriv_x       = hxprime[:,0]*hx[:,1]*hx[:,2]
        deriv_y       = hx[:,0]*hxprime[:,1]*hx[:,2]
        deriv_z       = hx[:,0]*hx[:,1]*hxprime[:,2]

        deriv           = np.hstack((deriv_x[:,np.newaxis],deriv_y[:,np.newaxis],deriv_z[:,np.newaxis]))
        self.derivative = deriv

        return deriv
