import numpy as np
import MDAnalysis as mda

from analysis_code.Liquid_crystal.Liquid_crystal import Liquid_crystal


class GAFF_LC(Liquid_crystal):
    """
    A class that represents Liquid crystals that is represented by GAFF which stands for General Amber force field from the paper J. Comput. Chem. 25, 1157–1174 (2004) and modified later by other group to be more suited for Liquid crystals in paper Phys. Chem. Chem. Phys. 17, 24851–24865 (2015).

    Args:
    ----
    tpr(string): The path to the .tpr file of the Liquid crystal molecule
    xtc(string): The path to the .xtc file of the Liquid crystal molecule
    u_vec(string): The atoms at which the direction of the LC molecule is defined (default C11-C14 for the mesogen in the literature 2 shown)
    """
    def __init__(self,tpr,xtc,name, head_id, tail_id,trjconv=True, bulk=True):
        super().__init__(tpr,xtc,name, head_id, tail_id,trjconv=trjconv, bulk=bulk)
      