import numpy as np
from scipy.special import erf

from .ProbeVolume import _ProbeVolume
from .indicator_func_1b import indicator_func_1b
from .indicator_func_2b import indicator_func_2b
from .ProbeVolume_cylinder import ProbeVolume_cylinder


class ProbeVolume_tiltedcylinder(_ProbeVolume):
    """
    Probe Volume for a cylindrical probeVolume

    Args:
        tpr (str)               : The name of the tpr file for the universe
        xtc (str)               : The name of the xtc/trr file for the universe
        base(np.ndarray)        : The base of the cylinder (3,)
        h (float)               : height of the cylinder
        radius (float)          : The radius of the cylinder 
        theta(float)            : Degree rotated in the xy plane passed in as degree
        phi(float)              : Degree rotated in the zx plane after rotation of theta passed in as degree
        sigma(float)            : sigma for coarse-graining (in units of A)
        ac(float)               : alphac for coarse-graining (in units of A)
    """          
    def __init__(self, tpr, xtc, base, h, radius, theta, phi, sigma=0.1, ac=0.2):
        super().__init__(tpr, xtc, sigma=sigma, ac=ac)

        # store base & head of the cylinder
        origin          = np.zeros((3,))
        origin[2]       = origin[2] - h/2
        # create a cylinder where base lies at origin & principle axis is in the z direction
        self.cylinder   = ProbeVolume_cylinder(tpr, xtc,origin, h, radius, pbc=False, dir_='z', sigma=sigma, ac=ac) 

        self.base_      = base
        self.theta_     = theta
        self.phi_       = phi
        self.axis_      = np.zeros((3,))

        
        self.set_Geometry() 
        temp            = np.copy(base)
        self.head_      = temp  + self.axis_*h
        self.center_    = 1/2*(self.base_ + self.head_) 
    
    def set_Geometry(self):
        """
        Function that sets the geometry for the tilted cylinder, uses the rotational matrix
        """
        deg_to_rad        = np.pi/180
        self.rot_mat      = np.zeros((3,3))
        costheta          = np.cos(self.theta_*deg_to_rad)
        sintheta          = np.sin(self.theta_*deg_to_rad)
        cosphi            = np.cos(self.phi_*deg_to_rad)
        sinphi            = np.sin(self.phi_*deg_to_rad)
        self.axis_[0]     = sinphi*costheta
        self.axis_[1]     = sinphi*sintheta
        self.axis_[2]     = cosphi

        zdim              = np.array([0,0,1.0])
        rxz               = np.cross(self.axis_, zdim)
        rdz               = np.dot(self.axis_, zdim)
        factor            = 1/(1+rdz) if np.abs(1 + rdz) > 1e-7 else 0

        self.rot_mat[0,0] = rxz[0]*rxz[0]*factor + rdz
        self.rot_mat[0,1] = rxz[0]*rxz[1]*factor - rxz[2]
        self.rot_mat[0,2] = rxz[0]*rxz[2]*factor + rxz[1]
        self.rot_mat[1,0] = rxz[1]*rxz[0]*factor + rxz[2]
        self.rot_mat[1,1] = rxz[1]*rxz[1]*factor + rdz
        self.rot_mat[1,2] = rxz[1]*rxz[2]*factor - rxz[0]
        self.rot_mat[2,0] = rxz[2]*rxz[0]*factor - rxz[1]
        self.rot_mat[2,1] = rxz[2]*rxz[1]*factor + rxz[0]
        self.rot_mat[2,2] = rxz[2]*rxz[2]*factor + rdz
    
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
        bb           = self.bounding_box_
        dr           = bb.dr_pbc(pos - self.center_, ts)
        dr_rotated   = np.matmul(self.rot_mat, dr.T)
        dr_rotated   = dr_rotated.T

        indicator, self.hx_  = self.cylinder.calculate_Indicator(dr_rotated, ts)
        indicator            = indicator[:,np.newaxis]
        self.Ntilde_         = indicator.sum()
        self.indicator_      = indicator

        return h, self.hx_
    
    def calculate_derivative(self, pos:np.ndarray, hx:np.ndarray, ts:int):
        """
        Function that calculates the derivative of the cylindrical probeVolume with respect to the positions (x,y,z)

        Args:
        ----
            pos(numpy.ndarray)      : The positions of the atoms passed in shape (N,3)
            hx(numpy.ndarray)       : The derivatives of the atoms with respect to the cylindrical coordinates (r, theta, z) where z is the principle axis of the cylinder
            ts(int)                 : The time step at which the calculation is performed upon
        
        Returns:
        --------
            dh_dr(numpy.ndarray)    : The derivative of ProbeVolume with respect to the positions (x,y,z)
        """
        bb         = self.bounding_box_
        dr         = bb.dr_pbc(pos -self.center_, ts)
        dr_rotated = np.matmul(self.rot_mat, dr.T)
        dr_rotated = dr_rotated.T

        deriv      = self.cylinder.calculate_derivative(dr_rotated, hx, ts)
        deriv      = np.matmul(self.rot_mat.T, deriv.T)
        self.derivative_ = deriv.T

        return deriv.T