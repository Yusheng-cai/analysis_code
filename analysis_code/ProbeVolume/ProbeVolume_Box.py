import numpy as np
from scipy.special import erf
import MDAnalysis as mda

from .ProbeVolume import _ProbeVolume
from .Bounding_box import Bounding_box

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
        super().__init__(sigma,ac)          
        self.max_   = max_
        self.min_   = min_
        self.tpr    = tpr
        self.xtc    = xtc
        self.u_     = mda.Universe(tpr,xtc)

        # shift min_ & max_
        self.center_       = 1/2*(max_ + min_)
        self.max_shifted   = max_ - self.center_
        self.min_shifted   = min_ - self.center_

        self.bounding_box  = Bounding_box(self.u_)
    
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
        sigma  = self.sigma_
        ac     = self.ac_
        max_   = self.max_shifted
        min_   = self.min_shifted
        bb     = self.bounding_box

        pos_center    = bb.dr_pbc(pos - self.center_, ts) 

        # normalizing constants
        k = np.sqrt(2*np.pi*sigma**2) * erf(ac/np.sqrt(2*sigma**2)) - 2*ac*np.exp(-ac**2/(2*sigma**2))
        k1 = 1/k*np.sqrt(np.pi*sigma**2/2)
        k2 = 1/k*np.exp(-ac**2/(2*sigma**2))

        p1 = (k1*erf((max_ - pos_center)/(np.sqrt(2)*sigma)) - k2*(max_ - pos_center) - 1/2)*\
        np.heaviside(ac - np.abs(max_ - pos_center), 1.0)

        p2 = (k1*erf((pos_center - min_)/(np.sqrt(2)*sigma)) - k2*(pos_center - min_) - 1/2)*\
        np.heaviside(ac - np.abs(pos_center - min_),1.0)

        p3 = np.heaviside(ac + 1/2*(max_ - min_) - np.abs(pos_center - 1/2*(max_ + min_)),1.0)
        hx = p1 + p2 + p3

        indicator       = np.prod(hx,axis=1,keepdims=True)
        self.indicator_ = indicator
        self.Ntilde_    = indicator.sum()
        self.hx_        = hx

        return indicator, hx

    def phi(self, r):
        """
        Calculate phi function in INDUS where h = \int phi

        Args:
            r(numpy.ndarray)    : The alpha variable in the phi function for INDUS
        """
        sigma = self.sigma_
        ac    = self.ac_

        # Calculate the phi function
        k = np.sqrt(2*np.pi*sigma**2) * erf(ac/np.sqrt(2*sigma**2)) - 2*ac*np.exp(-ac**2/(2*sigma**2))
        p = 1/k*(np.exp(-r**2/(2*sigma**2)) - np.exp(-ac**2/(2*sigma**2)))*np.heaviside(ac - np.abs(r),1)

        return p

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
        max_    = self.max_shifted
        min_    = self.min_shifted
        center_ = self.center_
        bb      = self.bounding_box
        pos_center    = bb.dr_pbc(pos - center_, ts) 

        deriv_x = -(self.phi(max_[0]-pos_center[:,0]) - self.phi(min_[0]-pos_center[:,0]))*hx[:,1]*hx[:,2]
        deriv_y = -(self.phi(max_[1]-pos_center[:,1]) - self.phi(min_[1]-pos_center[:,1]))*hx[:,2]*hx[:,0]
        deriv_z = -(self.phi(max_[2]-pos_center[:,2]) - self.phi(min_[2]-pos_center[:,2]))*hx[:,1]*hx[:,0]

        deriv           = np.hstack((deriv_x[:,np.newaxis],deriv_y[:,np.newaxis],deriv_z[:,np.newaxis]))
        self.derivative = deriv

        return deriv
