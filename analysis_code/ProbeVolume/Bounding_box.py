import MDAnalysis as mda
import numpy as np

class Bounding_box:
    def __init__(self,u:mda.Universe):
        self.u_ = u
    
    def get_boxdimensions(self,ts):
        """
        A function that obtains the bounding box dimensions, assuming orthorhombic box 

        Args:   
            ts(int)         : The time step at which this calculation is performed upon
        
        Returns:
            box_dim(numpy.ndarray) : (3,) vector that represents [lx, ly, lz] of a orthorhombic box
        """
        u = self.u_
        u.trajectory[ts]
        box_dim     = u.dimensions[:3]
        return box_dim
    
    def dr_pbc(self, pos,ts):
        """
        Function that takes care of periodic boundary conditions (pbc) when calculating distances between atoms or virtual sites

        Args:
            pos(numpy.ndarray)      : Passed in position (N,3)
            ts(int)                 : Time step at which is calculation is performed on
        
        Return:
            pos_fixed(numpy.ndarray): Position which have been pbc corrected
        """
        box_dim     = self.get_boxdimensions(ts)
        lt          = pos < -box_dim/2
        gt          = pos > box_dim/2
        pos_fixed   = pos + lt*box_dim - gt*box_dim

        return pos_fixed
    
    def COMag_pbc(self, ag:mda.ResidueGroup, ts):
        """
        Function that calculates the pbc corrected COM of an atomgroup at time ts

        Args:
            ag(mda.AtomGroup) : The AtomGroup object in the MDAnalysis module
        """
        pos         = ag.positions

        # shifted by the first atom
        pos_shifted = self.dr_pbc(pos - pos[0], ts)
        M           = ag.masses

        # Calculate the COM of the shifted positions
        COM_ag      = (pos_shifted * M[:,np.newaxis]).sum(axis=0)/M.sum() + pos[0]

        return COM_ag

    # Have not been tested and fairly slow 
    def COMres_pbc(self, residues:mda.ResidueGroup, ts):
        """
        Function that calculates the pbc corrected COM of a residueGroup, this is a functionaity I was not able to find in MDAnalysis

        Args:
            res(mda.ResidueGroup)   : The ResidueGroup object which wraps the AtomGroups 
        """
        COM_res     = np.zeros((len(residues),3))
        ix          = 0

        for res in residues:
            ag          = res.atoms
            COM         = self.COMag_pbc(ag,ts)
            COM_res[ix] = COM
            ix         += 1
        
        return COM_res