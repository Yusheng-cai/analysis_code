from .Indicator_func import _indicator_func
import numpy as np
from scipy.special import erf

class indicator_func_2b(_indicator_func):
    """
    Function that calculates indicator function for dimensions with 2 boundaries, such as a cuboidal volume, this is a 
    shifted probe volume which means that we are always assuming that the middle of the box is [0,0,0]

    Args:
    -----
        min_(numpy.ndarray) : The minimum of the dimensions with 2 boundaries (d,)
        max_(numpy.ndarray) : The maximum of the dimensions with 2 boundaries (d,)
        sigma(float)        : The width of the Gaussian as defined in the INDUS paper
        ac(float)           : The cut off of the Gaussian, usually 2 times sigma
    """
    def __init__(self, min_, max_, sigma=0.1, ac=0.2):
        super().__init__(sigma, ac)
        self.k_    = np.sqrt(2*np.pi*sigma**2) * erf(ac/np.sqrt(2*sigma**2)) - 2*ac*np.exp(-ac**2/(2*sigma**2))
        self.k1_   = 1/self.k_*np.sqrt(np.pi*sigma**2/2)
        self.k2_   = 1/self.k_*np.exp(-ac**2/(2*sigma**2))

        #assert(len(min_) == len(max_))
        self.min_  = min_
        self.max_  = max_

    def calculate(self, pos:np.ndarray):
        """
        Function that calculates the indicator for indicator function with 2 boundaries

        Args:
        -----
            pos(numpy.ndarray)      : The positions of the atoms passed in with shape(N,d)
        
        Returns:
        --------
            hx(numpy.ndarray)       : The indicator functions returned for each of the dimensions
        """
        # assert that the positions passed in is with 2 dimensions (N,d)
        # assert(len(pos.shape) == 2)
        # assert that the second dimension of the positions matches with the min_ & max_ of the object
        # assert(pos.shape[1] == len(self.min_))

        sigma = self.sigma_
        ac    = self.ac_
        k1    = self.k1_
        k2    = self.k2_
        min_  = self.min_
        max_  = self.max_

        p1 = (k1*erf((max_ - pos)/(np.sqrt(2)*sigma)) - k2*(max_ - pos) - 1/2)*\
        np.heaviside(ac - np.abs(max_ - pos), 1.0)

        p2 = (k1*erf((pos - min_)/(np.sqrt(2)*sigma)) - k2*(pos - min_) - 1/2)*\
        np.heaviside(ac - np.abs(pos - min_),1.0)

        p3 = np.heaviside(ac + 1/2*(max_ - min_) - np.abs(pos - 1/2*(max_ + min_)),1.0)
        hx = p1 + p2 + p3

        return hx
    
    def calculate_derivative(self, pos):
        """
        Function that calculates the derivative of the indicator function h^{'}(alpha)

        Args:
        -----
            pos(numpy.ndarray)          : The passed in positions of shape (N,d)
        
        Returns:
            derivaitve(numpy.ndarray)   : The derivatives of the indicator function of shape (N,d)
        """
        max_        = self.max_
        min_        = self.min_

        # Some assertions
        # assert(len(pos.shape) == 2)
        # assert(pos.shape[1] == len(max_))

        derivative  =  -(self.phi(max_-pos) - self.phi(min_-pos))

        return derivative