from abc import ABC, abstractmethod
import numpy as np
from scipy.special import erf

class _indicator_func(ABC):
    def __init__(self, sigma=0.1,ac = 0.2):
        self.sigma_ = sigma
        self.ac_    = ac

    @abstractmethod 
    def calculate(self, pos):
        pass

    @abstractmethod
    def calculate_derivative(self,pos):
        pass

    def phi(self, alpha):
        """
        Calculate phi function in INDUS where h = \int phi

        Args:
            r(numpy.ndarray)    : The alpha variable in the phi function for INDUS
        """
        sigma = self.sigma_
        ac    = self.ac_

        # Calculate the phi function
        k = np.sqrt(2*np.pi*sigma**2) * erf(ac/np.sqrt(2*sigma**2)) - 2*ac*np.exp(-ac**2/(2*sigma**2))
        p = 1/k*(np.exp(-alpha**2/(2*sigma**2)) - np.exp(-ac**2/(2*sigma**2)))*np.heaviside(ac - np.abs(alpha),1)

        return p


