from abc import ABC, abstractmethod
import MDAnalysis as mda
from .Bounding_box import Bounding_box

# Abstract base class for probe volume
class _ProbeVolume(ABC):
    """
    Probe Volume base class

    Args:
        sigma(float)            : sigma for coarse-graining (in units of A)
        ac(float)               : alphac for coarse-graining (in units of A)
    """          
    def __init__(self, tpr:str, xtc:str,sigma=0.1,ac=0.2):
        self.tpr_       = tpr
        self.xtc_       = xtc

        # Every Probe Volume object has a bounding box object which takes in a universe obj
        self.u_         = mda.Universe(self.tpr_, self.xtc_)
        self.bounding_box_  = Bounding_box(self.u_)

        # coarse graining parameters 
        self.sigma_     = sigma 
        self.ac_        = ac

        # Improve this!
        self.indicator_  = 0
        self.derivative_ = 0
        self.Ntilde_     = 0.0
        self.hx_         = 0.0
    
    def get_indicator(self):
        return self.indicator_
    
    def get_derivative(self):
        return self.derivative_
    
    def get_Ntilde(self):
        return self.Ntilde_
    
    def get_hx(self):
        return self.hx_

    @abstractmethod
    def calculate_Indicator(self, pos, ts):
        """
        Abstract function that calculates the indicator function from the probe volume
        
        Args:
            pos(numpy.ndarray)           : Input positions (N,dim) 
            ts(int)                      : The time step at which this operation is performed on

        Returns:
            indicator(numpy.ndarray)     : The indicator array in shape (N,1)
        """
        pass

    @abstractmethod
    def calculate_derivative(self,pos, hx, ts):
        """
        Function that calculates derivatve of h -> which we call h prime

        Args:
            pos(numpy.ndarray)  : The positions of the atoms ((N,3))
            ts(int)             : The time step at which this operation is performed on

        Returns:
            derivative of hprime(numpy.ndarray) : ((N,3))
        """
        pass
