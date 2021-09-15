import numpy as np
import MDAnalysis as mda
from .Liquid_crystal import Liquid_crystal, Liquid_crystalPV
from ..ProbeVolume.ProbeVolume import _ProbeVolume

class nCB(Liquid_crystal):
    """
    Args:
    ----
        tpr(string): Name of the tpr file
        xtc(string): Name of the xtc file
        n(int): the n in nCB
        trjconv(bool) : Whether the .xtc & .trr file has been processed with gmx trjconv 
        bulk(bool) : Whether the calculation is for bulk phase nCB
    """
    def __init__(self,tpr:str,xtc:str,n:int,bulk:bool=True,trjconv:bool=True):
        super().__init__(tpr,xtc,"{}CB".format(n),0,1,trjconv=trjconv,bulk=bulk)
        self.n_     = n
    
        if self.n_ == 5:
            self.segment_["CN"]       = np.arange(0,2)
            self.segment_["benzene1"] = np.arange(2,8)
            self.segment_["benzene2"] = np.arange(8,14)
            self.segment_["HC_tail"]  = np.arange(15,19)

        if self.n_ == 8:
            self.segment_["CN"]       = np.arange(0,2)
            self.segment_["benzene1"] = np.arange(2,8)
            self.segment_["benzene2"] = np.arange(8,14)
            self.segment_["HC_tail"]  = np.arange(15,22)
    
class nCB_PV(Liquid_crystalPV):
    def __init__(self,tpr:str,xtc:str,n:int,pv:_ProbeVolume,bulk:bool=True,trjconv:bool=True):
        super().__init__(tpr,xtc,"{}CB".format(n),0,1,pv,trjconv=trjconv,bulk=bulk)
        self.n_     = n
    
        if self.n_ == 5:
            self.segment_["CN"]       = np.arange(0,2)
            self.segment_["benzene1"] = np.arange(2,8)
            self.segment_["benzene2"] = np.arange(8,14)
            self.segment_["HC_tail"]  = np.arange(15,19)

        if self.n_ == 8:
            self.segment_["CN"]       = np.arange(0,2)
            self.segment_["benzene1"] = np.arange(2,8)
            self.segment_["benzene2"] = np.arange(8,14)
            self.segment_["HC_tail"]  = np.arange(15,22)
  
         
  

def director_z(LC,ts,segment='whole',bins_z=100,direction='z'):
    """
    finds director as a function of z along the direction provided
   
    Args:
    -----
        LC: Liquid crystal object
        ts: the time step at which the calculation is performed 
        segment: The segment at where we want to take COM at 
        bins_z: the number of bins that we want to break the analysis into
        direction:'x','y' or 'z'
        
    returns:
        p2z matrix in shape (T, n_molecules*n_atoms)
    """
    directorz = np.zeros((bins_z,3))
   
    if direction == 'x':
        d = 0

    if direction == 'y':
        d = 1

    if direction == 'z':
        d = 2
    
    # first find Center of mass matrix at time step "ts"
    COM_mat = LC.COM(ts,segment) #(n_molecues,3)

    # take only the dth dimension number of all COM 
    COM_mat = COM_mat[:,d] #(n_molecules,)

    if LC.bulk == True:
        residues = LC.u.residues
    else:
        residues = LC.u.select_atoms("resname {}CB".format(LC.n)).residues

    COM_vec = np.linspace(COM_mat.min(),COM_mat.max(),bins_z)

    for i in range(bins_z-1):
        less = COM_vec[i]
        more = COM_vec[i+1]

        index = np.argwhere(((COM_mat >= less) & (COM_mat < more)))
        index = index.flatten()
        if index.size != 0:
            Q = np.zeros((3,3))
            I = np.eye(3)
            number = len(index) # number of molecules with COM in range (less, more)
            for idx in index:
                res = residues[idx]
                N = res.atoms[0].position
                C = res.atoms[1].position
                CN_vec = (N - C)/np.sqrt(((N-C)**2).sum())

                Q += (3*np.outer(CN_vec,CN_vec)-I)/(2*number)
            eigval,eigvec = np.linalg.eig(Q)
            order = np.argsort(-np.abs(eigval))
            eigvec = eigvec[:,order]
            directorz[i] = eigvec[:,0]

    return directorz


# find p2 as a function of z where z represents one direction in the Cartesian coordinates
def p2_z(LC,start_t,end_t,bins_z = 100, segment='whole',skip=None,direction='z',verbose=False):
    """
    finds p2 as a function of z along the direction provided
    Always uses the Qtensor formulation
    
    Args:
        LC(Liquid_crystal): The Liquid crystal object
        start_t(float)    : The time at which the evaluation starts (in ns)
        end_t(float)      : The time at which the evaluation ends (in ns)
        bins_z(int)       : The number of bins
        segment(string)   : String that represents the segment of nCB molecule
        skip(int)         : Number of time steps to skip 
        direction(string) : The direction in which the calculation is performed over
        verbose(bool)     : Whether or not to be verbose

    returns:
        p2z matrix in shape (bins_z,)
    """
    time = np.linspace(0,LC.time,len(LC)) # find the list of simulation times in ns 
    start_timeidx = np.searchsorted(time,start_t,side='left') # find the index of the starting time in list of simulation times (in frame)
    end_timeidx = np.searchsorted(time,end_t,side='right') # find the index of the ending time in list of simulation times (in frame)
    time_idx = np.arange(start_timeidx,end_timeidx,skip) 
    p2z = np.zeros((bins_z,))
    u = LC.u
   
    if direction == 'x':
        d = 0

    if direction == 'y':
        d = 1

    if direction == 'z':
        d = 2
    
    for ts in time_idx:
        if verbose:
            print("performing calculations for t={}".format(ts))
        # first find Center of mass matrix at time step "ts"
        COM_mat = LC.COM(ts,segment) #(n_molecues,3)

        # take only the dth dimension number of all COM 
        COM_mat = COM_mat[:,d] #(n_molecules,)

        # set the universe trajectory to "ts" time step
        u.trajectory[ts]
        if LC.bulk == True:
            residues = u.residues
        else:
            residues = u.select_atoms("resname {}CB".format(LC.n)).residues

        COM_vec   = np.linspace(COM_mat.min()-0.1,COM_mat.max()+0.1,bins_z)
        digitized = np.digitize(COM_mat,COM_vec,right=False)

        for i in range(1,bins_z):
            index = np.argwhere(digitized == i)
            index = index.flatten()
            if index.size != 0:
                number = len(index) # number of molecules with COM in range (less, more)
                d_mat  = np.zeros((number,3))
                for j in range(len(index)):
                    idx       = index[j]
                    res       = residues[idx]
                    N         = res.atoms[0].position
                    C         = res.atoms[1].position
                    CN_vec    = (N - C)/np.sqrt(((N-C)**2).sum())
                    
                    d_mat[j] = CN_vec
                Q = 3/(2*number)*np.matmul(d_mat.T,d_mat) - 1/2*np.eye(3)
                eigval,eigvec = np.linalg.eig(Q)
                order = np.argsort(eigval)
                eigval = eigval[order]
                p2 = -2*eigval[1]
                p2z[i] += p2
    p2z /= len(time_idx)
    return (p2z,time_idx)



def write_betafactor(u:mda.Universe,data:np.ndarray,pdb_name:str, selection=None):
    """
    write cosine theta data for all atoms into a multi frame pdb file
    """
    u.add_TopologyAttr("tempfactors")
    with mda.Writer(pdb_name, multiframe=True, bonds=None, n_atoms=len(u.atoms)) as PDB:
        for i in range(len(data)):
            index = int(data[i,0])
            print("printing frame {}".format(index))
            u.trajectory[index]

            atoms = u.atoms
            if selection:
                atoms = u.select_atoms("resname {}".format(selection))

            atoms.tempfactors = data[i,1:]

            PDB.write(atoms)


# write the data to pdb file in the beta factor
def pdb_bfactor(LC,pdb_name,data,verbose=False,sel_atoms=None,others=False): 
    """
    saves the data as beta-factor which is in shape (n,n_molecules*n_atoms) into pdb file specified by pdb_name
    
    Args:
    -----
        LC(LC object): Liquid crystal object
        pdb_name(str): the name of the pdb out file
        data(tuple): a tuple containing (actual_data, time idx)

    Return:
    ------
        saves a file with beta-factor with name pdb_name into the folder specified in LC object
    """
    ix = 0

    data_array,time_idx = data 
    LC.u.add_TopologyAttr("tempfactors")
    with mda.Writer(pdb_name, multiframe=True,bonds=None,n_atoms=LC.n_atoms) as PDB:
        u = LC.u
        for ts in time_idx: 
            u.trajectory[ts]
            if sel_atoms != None:
                uni = u.select_atoms(sel_atoms)
                uni.atoms.tempfactors = data_array[ix]
                
                PDB.write(u.atoms)
            else:
                u.atoms.tempfactors = data_array[ix]
                PDB.write(u.atoms)

            if verbose:
                print("time frame {} has been written".format(ts))
            ix += 1     

def xvg_reader(file):
    """
    function that reads xvg files

    Args:
    ----
        file(str): input file path

    Return:
    ------
        data(numpy.ndarray) that is contained within the xvg file
    """
    f = open(file)

    lines = f.readlines()

    f.close()
    # define the comment symbols which are ignored in .xvg files
    comment_symbol = ['#','@']

    # ignore the lines which starts with comment symbols and take the second number (value of interest)
    xvgdata = np.array([float(line.rstrip(" ").lstrip(" ").split()[1]) for line in lines if line[0] not in comment_symbol])

    return xvgdata
