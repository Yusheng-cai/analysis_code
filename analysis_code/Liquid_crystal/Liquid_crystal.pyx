import numpy as np
cimport numpy as np
import MDAnalysis as mda
from analysis_code.timeseries import *
import analysis_code.Liquid_crystal.INDUS_util as indus

class Liquid_crystal:
    """
    Base class for all Liquid crystals

    Args:
    ----
    itp(string): The path to the .itp file of the Liquid crystal molecule
    top(string): The path to the .tpr file of the Liquid crystal molecule
    xtc(string): The path to the .xtc file of the Liquid crystal molecule
    u_vec(string): The atoms at which the direction of the LC molecule is defined (default C11-C14 for the mesogen in the literature 2 shown)
    """
    def __init__(self,top,xtc,u_vec,bulk=True,sel=None):
        self.top = top
        self.xtc = xtc
        self.bulk = bulk
        self.sel = sel
        self.u = mda.Universe(self.top,self.xtc)

        if self.bulk == True:
            self.N = len(self.u.residues)
        else:
            self.N = len(self.u.select_atoms(self.sel).residues)

        self.atom2 = u_vec.split("-")[1]
        self.initialize()
       
    def initialize(self):
        u = self.u
        if self.bulk == True:
            res = u.select_atoms(self.sel).residues
            r = res[0]
        else:
            r = u.residuesp[0]

        for i in range(len(r.atoms)):
            atom = r.atoms[i]
            if atom.name == self.atom1:
                self.aidx1 = i
            elif atom.name == self.atom2:
                self.aidx2 = i

    def pos(self,ts):
        """
        Function that returns the position of each of the atoms in the system at time ts

        Args:
            ts(int): The time frame at which the calculation is performed upon

        Return:
            pos(numpy.ndarray): The position matrix at time ts of shape (N,3)
        """
        u = self.u
        u.trajectory[ts]

        if self.bulk == True:
            return u.select_atoms(self.sel).atoms.positions
        else:
            return u.atoms.positions

    def get_celldimension(self,ts):
        """
        Function that returns the cell dimension that each of the time frame 

        Args:
            ts(int): The time frame that the user want to obtain cell dimension at 

        Return:
            cell_dimension(numpy.ndarray): The cell dimension of the box as a (3,) numpy array
        """
        u = self.u
        u.trajectory[ts]
        cell_dimension = u.dimensions[:3]

        return cell_dimension

    def COM(self,ts,segment=None):
        """
        Function that calculates the center of mass of the Liquid crystal molecule at time ts

        Args:
            ts(int): The time frame at which the calculation is performed on 
            segment(str): The segment that the user wants to calculate center of mass for (default None)

        Return:
            COM_mat(numpy.ndarray): The center of mass matrix of shape (N,3)
        """
        u = self.u
        u.trajectory[ts]
        N = self.N
        COM_mat = np.zeros((N,3))
        
        if self.bulk == True:
            residues = u.residues
        else:
            residues = u.select_atoms(self.sel).residues
        
        ix = 0
        for res in residues:
            if segment is None:
                COM_mat[ix] = res.atoms.center_of_mass()
            else:
                num1,num2 = int(segment.split("-")[0]),int(segment.split("-")[1])
                atom_grp = res.atoms[num1:num2]
                COM_mat[ix] = atom_grp.center_of_mass()
            ix += 1

        return COM_mat
 
    def director_mat(self,ts,MOI=False):
        """
        Function that finds the director vector of all the residues in the system and put it in a matrix of shape (N,3). This can also find the director matrix using Moment of inertia tensor (the eigenvector that corresponds to the lowest eigenvalue of MOI tensor)

        Args:
            ts(int): The time frame of the simulation that this operation is performed upon
            MOI(bool): Whether or not to find director matrix using Moment of Inertia tensor

        Return:
            vec(numpy.ndarray): The director matrix of all the residues in the system
        """
        u = self.u
        u.trajectory[ts]
        N = self.N
        director_mat = np.zeros((N,3))
        aidx1 = self.aidx1
        aidx2 = self.aidx2

        if self.bulk == True:
            residues = u.residues
        else:
            residues = u.select_atoms(self.sel).residues

        ix = 0 
        for res in residues:
            if MOI:
                director_mat[ix] = res.atoms.principal_axes()[-1]
            else:
                a1 = res.atoms[aidx1].position
                a2 = res.atoms[aidx2].position

                r = (a2-a1)/np.sqrt(((a2-a1)**2).sum())
                director_mat[ix] = r
            ix += 1

        return director_mat

    def Qmatrix(self,ts,MOI=False):
        """
        Function that calculates the Qmatrix of the system at time ts.

        Args:
            ts(int): The time frame of the simulation

        Return:
            1.Qmatrix(numpy.ndarray)= The Q matrix of the liquid crystalline system
            2.eigvec(numpy.ndarray)=The director of the system at time ts
            3.p2(numpy.ndarray)=The p2 value of the system at time ts
        """
        d_mat = self.director_mat(ts,MOI=MOI)
        N = self.N
        I = np.eye(3)

        Q = 3/(2*N)*np.matmul(d_mat.T,d_mat) - 1/2*I
        eigval,eigvec = np.linalg.eig(Q)
        order = np.argsort(eigval)
        eigval = eigval[order]
        eigvec = eigvec[:,order]

        return Q,eigvec[:,-1],-2*eigval[1]
 
class nCB:
    """
    Args:
    ----
        tpr(string): Name of the tpr file
        xtc(string): Name of the xtc file
        time(int): the total simulation time (ns)
        n(int): the n in nCB
        p2(bool): a boolean that tells whether or not p2 vector needs to be computed and saved in the folder
    """
    def __init__(self,tpr,xtc,n,time,bulk=True,p2=False,trjconv=True):
        self.u = mda.Universe(tpr,xtc)
        self.n = n
        self.bulk = bulk
        self.time = time
        self.trjconv = trjconv
    
        if bulk:
            self.n_molecules = len(self.u.residues)
        else:
            partial_u = self.u.select_atoms("resname {}CB".format(n))
            self.n_molecules = len(partial_u.residues)

        if self.n == 5:
            self.segments = {"CN":np.arange(0,2),\
                    "benzene1":np.arange(2,8),\
                    "benzene2":np.arange(8,14),\
                    "HC_tail":np.arange(15,19)}
            self.n_atoms = 19 

        if self.n == 8:
            self.segments = {"CN":np.arange(0,2),\
                    "benzene1":np.arange(2,8),\
                    "benzene2":np.arange(8,14),\
                    "HC_tail":np.arange(15,22)}
            self.n_atoms = 22

    def __len__(self):
        return len(self.u.trajectory)

    def get_celldimension(self,ts):
        """
        Function that returns the cell dimension that each of the time frame 

        Args:
            ts(int): The time frame that the user want to obtain cell dimension at 

        Return:
            cell_dimension(numpy.ndarray): The cell dimension of the box as a (3,) numpy array
        """
        u = self.u
        u.trajectory[ts]

        cell_dimension = u.dimensions[:3]

        return cell_dimension

    def fix_pbc(self,pos,box_dimensions):
        """
        Function that fixes periodic boundary condition of a particle's position

        Args:
            pos(numpy.ndarray)           : The position of the particle, could be passed in shape (N,3)
            box_dimensions(numpy.ndarray): Assume a orthorhombic box, this is passed in with shape (3,) with [lx,ly,lz]
        """
        lt = pos < -box_dimensions/2
        gt = pos > box_dimensions/2

        pos = pos + lt*box_dimensions - gt*box_dimensions

        return pos

    def director_mat(self,ts,MOI=False):
        """
        Function that calculates the director vector at time ts. It can either use the CN vector of nCB molecule or the moment of inertia tensor to figure out the director.

        Args:
        ----
            ts(int): the time step of the trajectory
            MOI(bool): whether or not the find director using moment of inertia tensor

        Return:
        ------
            director_mat(numpy.ndarray):numpy array of CN vectors (N,3)
        """
        u = self.u
        u.trajectory[ts]
        # Either trjconv is true or MOI is true
        assert(self.trjconv + MOI <= 1)

        if self.bulk:
            residues = u.residues
        else:
            residues = u.select_atoms("resname {}CB".format(self.n)).residues
        
        director_mat = np.zeros((len(residues),3))
        norm_mat     = np.zeros((len(residues),1))
        box_dimensions = self.get_celldimension(ts)
        for i in range(len(residues)): 
            res = residues[i]
            if MOI:
                director_mat[i] = res.atoms.principal_axes()[-1]
            else:
                N = res.atoms[0].position
                C = res.atoms[1].position
                if not self.trjconv:
                    CN_vec = self.fix_pbc(N - C, box_dimensions)
                    norm   = np.sqrt((CN_vec**2).sum())
                    norm_mat[i] = norm
                    CN_vec = CN_vec/norm
                else:
                    CN_vec      = N - C
                    norm        = np.sqrt((CN_vec**2).sum())
                    norm_mat[i] = norm
                    CN_vec      = CN_vec/norm
                director_mat[i] = CN_vec

        return director_mat, norm_mat

    def Qmatrix(self,ts,MOI=False):
        """
        calculates the Q matrix at a particular time step of the trajectory

        Args:
            ts(int): the time step at which the Qmatrix is calculated
            MOI(bool): boolean that specifies whether or not to use moment of inertia tensor

        Return:
            Q(numpy.ndarray):\sum_{l=1}^{N} (3*u_{l}u_{l}t - I)/2N (3,3)
        """ 
        director_mat,_ = self.director_mat(ts,MOI)

        ix = 0
        n = self.n_molecules
        I = np.eye(3)

        Q = 3/(2*n)*np.dot(director_mat.T,director_mat) - 1/2*I

        return Q
    
    def director(self,Q):
        """
        Calculates the system director at time ts
        the system director corresponds to the eigenvector which corresponds to the largest
        eigenvalue of the Q matrix

        Args:
            Q(numpy.ndarray): the Q matrix of LC system (3,3)

        Return:
            1. director(numpy.ndarray): director at one time step in shape(3,1)
            2. p2(float): p2 OP at one time step
        """
        eigv,eigvec = np.linalg.eig(Q)
        order = np.argsort(eigv)    
        eigvec = eigvec[:,order]
        eigv  = eigv[order]

        return eigvec,eigv

    def COM(self,ts,segment='whole'):
        """
        Output the center of mass of each of the molecule in the system at a certain time step

        Args:
            ts(int): the time step at which this center of mass measurement is at 
            segment(str): the segment at which the center of mass is measured on

        Return:
            COM(numpy.ndarray): center of mass matrix of shape (N_molecules,3)
        """
        u = self.u
        u.trajectory[ts]

        if self.bulk == True:
            residues = u.residues
        else:
            residues = u.select_atoms("resname {}CB".format(self.n)).residues

        if segment in self.segments.keys():
            idx = self.segments[segment]
        elif segment == 'whole': 
            idx = np.arange(0,self.n_atoms)
        else:
            raise NotImplementedError("The segment {} is not currently implemented".format(segment))

        COM = np.zeros((self.n_molecules,3)) 
        ix = 0

        for res in residues:
            num = res.atoms[idx].center_of_mass()
            COM[ix] = num
            ix += 1

        return COM 

    def p2(self,ts):
        """
        Function calculating p2 order parameter for nCB molecule using Q tensor formulation
        Q matrix is calculated as following:
        Q = \sum_{l}^{N}(3u_{l}u_{l}^{T} - I)/2N
        we choose p2 to be -2*lambda_{0} where lambda_{0} is the second largest eigenvalue of Q
        u_{l} is chosen to be the normalized vector between C and N in nCB

        Return:
        ------
            P2 vector as a function of time
        """
        Q = self.Qmatrix(ts)

        # Ordered eigen pair
        _,eigv = self.director(Q)

        return eigv[1]*(-2.0)
    
    def p2tilde(self, ts, min_, max_):
        """
        Function that calculates p2tilde which is p2 in a probe volume 

        Args:   
            ts(int)             : The time step at which is calculation is performed upon
            min_(numpy.ndarray) : The minimum of the bounding box (assumed orthorhombic) [xmin, ymin, zmin] (3,)
            max_(numpy.ndarray) : The maximum of the bounding box (assumed orthorhombic) [xmax, ymax, zmax] (3,)

        Returns: 
            p2tilde(float)      : P2tilde of the specified region
        """
        # Shift the box center
        center_     = (min_ + max_)/2
        min_        = min_ - center_ 
        max_        = max_ - center_
        N           = self.n_molecules
        box_dim     = self.get_celldimension(ts)
        Ncenter_mat = np.zeros((N,3))
        Q           = np.zeros((3,3))
        u           = self.u
        u.trajectory[ts]
        if self.bulk:
            residues    = u.residues
        else:
            residues    = u.select_atoms("resname {}CB".format(self.n)).residues

        for i in range(N):
            pos = residues[i].atoms.positions 
            N_pos          = pos[0]
            Ncenter_       = N_pos - center_
            if not self.trjconv:
                Ncenter_       = self.fix_pbc(Ncenter_,box_dim) 
            Ncenter_mat[i] = Ncenter_

        hr      = indus.h(Ncenter_mat, min_, max_,sigma=0.1,ac=0.2) 
        hr      = np.prod(hr,axis=1)
        Ntilde  = hr.sum()

        u, _    = self.director_mat(ts)

        for i in range(N): 
            ui   = u[i]
            Q   += (1.5*np.outer(ui, ui) -0.5*np.eye(3))*hr[i]
        Q = Q/Ntilde 

        _, eigv = self.director(Q)
        
        return eigv[1]*(-2.0)

    def p2_cos(self,ts,n):
        """
        Function that calculates p2 using the second legendre polynomial 

        Args:
            ts(int)         : The time step at which the calculation is performed on
            n(numpy.ndarray): The director at which the calculation is performed with

        Returns:
            p2(float)       : The p2 value
        """
        n   = n/np.sqrt((n**2).sum())
        u,_ = self.director_mat(ts)
        n_molecules = self.n_molecules
        dot_product = (u*n).sum(axis=1)

        p2 = 1/n_molecules*(1.5*(dot_product)**2 - 0.5).sum()

        return p2
    
    def p2_prime(self,ts):
        """
        Derivative of p2 with respect to the head & tail atoms (N & C)

        Args:
            ts(int)        : The time step the calculation is performed on

        Return:
            head_derivative(numpy.ndarray)
            tail_derivative(numpy.ndarray)
        """
        # norm  = (N,1)
        u, norm     = self.director_mat(ts)
        Q           = self.Qmatrix(ts)
        eigvec,eigv = self.director(Q)
        N           = self.n_molecules

        v1 = eigvec[:,1]
        
        dot_product     = (u*v1).sum(axis=1,keepdims=True)
        head_derivative = -6/(N*norm)*dot_product*(v1 - u*dot_product)
        tail_derivative = -1.0*head_derivative

        return (head_derivative, tail_derivative)

    def p2globaldata_pdb(self,start_time,end_time,skip=0):
        """
        Function that writes out p2 data for global P2 for all atoms 

        Args:
            start_time(float): The starting time in real time (ns)
            end_time(float)  : The ending time in real time (ns)
            skip(int)        : The skipping time step in steps units
        """
        u = self.u
        # find the time frame indexes of the simulation
        t_idx = np.linspace(0,self.time,len(u.trajectory))
        start_idx = np.searchsorted(t_idx,start_time,side='left')
        end_idx = np.searchsorted(t_idx,end_time,side='right')
        time_idx = np.arange(start_idx,end_idx,skip)
    
        p2_data = np.zeros((len(time_idx), self.n_molecules*self.n_atoms)) 
        idx = 0
        for t in time_idx:
            Q = self.Qmatrix(t)
            _,eigv = self.director(Q)

            p2 = eigv[1]*(-2.0)
            p2ts = np.ones((self.n_molecules*self.n_atoms,))*p2
            p2_data[idx] = p2ts

            idx+=1
        
        return (p2_data,time_idx)

         
    def pcost_z(self,start_time,end_time,director,min_,max_,direction='z',segment='whole',skip=1,bins_z=100,bins_t=100,verbose=False,Broken_interface=None):
        """
        Function that calculates a heat map of p(cos(theta)) as a function of z. 

        Args:
            start_time(float): the starting time in ns
            end_time(float): the ending time in ns
            director(numpy.ndarray): Array of the director of the system  
            min_(float): Minimum number of the z bin
            max_(float): Maximum number of the z bin 
            direction(str): at which direction is the calculation being performed along ('x','y','z')
            segment(str): which segment of the LC molecule to calculate COM for 
            skip(int): number of time frames to skip 
            bins_z(int): the bins along the direction where the calculation is being performed
            bins_t(int): the bins of theta for p(cos(theta))
            verbose(bool): whether to be verbose during execution
            Broken_interface(float): Where to draw the line for broken interface

        Return:
            2d array contains p(cos(theta)) as a function of z  (bins_z-1,bins_t-1)
        """
        u = self.u
        t_binned = np.linspace(-1,1,bins_t)

        if Broken_interface == None:
            z_binned = np.linspace(min_,max_,bins_z)


        # find the time frame indexes of the simulation
        t_idx = np.linspace(0,self.time,len(u.trajectory))
        start_idx = np.searchsorted(t_idx,start_time,side='left')
        end_idx = np.searchsorted(t_idx,end_time,side='right')
        time_idx = np.arange(start_idx,end_idx,skip)

        pcost_theta_director = np.zeros((bins_z-1,bins_t-1)) # a 2-d array that holds p(cos(theta)) in shape (bins_z-1,bins_t-1)

        if direction == 'x':
            d = 0
        
        if direction == 'y':
            d = 1

        if direction == 'z':
            d = 2
        
        for ts in time_idx:
            # first find Center of mass matrix at time step "ts"
            COM_mat = self.COM(ts,segment) #(n_molecues,3)

            # take only the dth dimension number of all COM 
            COM_mat = COM_mat[:,d] #(n_molecules,)

            # find the CN vectors of all the molecules 
            CN_vec, _ = self.director_mat(ts)
            cost = (CN_vec*director).sum(axis=1)
            
            if Broken_interface == None:
                prob,_,_ = np.histogram2d(COM_mat,cost,[z_binned,t_binned])
            else:
                index = np.argwhere(COM_mat < Broken_interface) 
                COM_mat[index] += (max_ - min_)
                z_binned = np.linspace(COM_mat.min(),COM_mat.max(),bins_z)
                prob,_,_ = np.histogram2d(COM_mat,cost,[z_binned,t_binned])

            pcost_theta_director += prob
             
            if verbose:
                print("time step {} is done".format(ts))
                   
        return pcost_theta_director/len(time_idx)

    def cos_t(self,start,end,director=np.array([[0],[0],[1]]),skip=1,verbose=False):
        """
        Function that calculates the cos(theta) between the CN vector of each LC molecule
        and the director of the system which is passed in by the user.

        Args:
            start(float): The time to start calculation (in ns)
            end(float): The time to end calculation (in ns)
            director(numpy.ndarray): The director of the system (default value [0,0,1])
            skip(int): The number of time frames to skip (default value 1) 
            verbose(bool): whether or not to print messages

        Return:
            cos(theta) in shape (n,n_molecules*n_atoms)
        """
        t = self.__len__()
        time = np.linspace(0,self.time,t)

        # Find the starting the ending index of the user specified times
        start_timeidx = np.searchsorted(time,start,side='left')
        end_timeidx = np.searchsorted(time,end,side='right')
        time_idx = np.arange(start_timeidx,end_timeidx,skip)

        n = len(time_idx)
        costheta = np.zeros((n,self.n_molecules*self.n_atoms))
        ix = 0

        for ts in time_idx:  
            CN_direction,_ = self.director_mat(ts) # CN_direction (N,3) 
            b = np.dot(CN_direction,director) #of shpae (N,1)
            b = np.repeat(b,self.n_atoms,axis=1).flatten() # of shape (N*N_atoms)
            costheta[ix] = b

            if verbose:
                print("time frame {} is being calculated".format(ts))
            ix += 1

        return (costheta,time_idx)


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



# find cos(theta) between CN and the global director of LC systems
# find probability distribution of cos(theta) as a function of z 

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

# find cos(theta) as a function of r where r is the distance between the center of maasses, theta is the angle each molecule's
# CN bond forms with the director
cpdef cost_r(LC,int ts,np.ndarray COM_dist_mat,float min_,float max_,int bins=100):
    """
    calculates cos(theta) between two pairs of Liquid crystal molecules as a function of R between the Center of mass distances, this
    can only be performed on bulk liquid crystals

    LC: Liquid crystal object

    ts: the time frame at which this calculation is performed

    COM_dist_mat: a matrix with shape (n1,n2) which contains the distances between 
    group N1 and group N2. This should be a upper trigonal matrix for performace reasons

    min_: minimum distance between COM to consider

    max_: maximum distance between COM to consider

    bins: number of bins to bin the separation between min_ and max_
    """
    cdef np.ndarray bin_vec = np.linspace(min_,max_,bins)  
    cdef np.ndarray CN = LC.segments["CN"]
    cdef np.ndarray indices
    cdef int idx0,idx1
    cdef float cost_CN
    cdef np.ndarray cost_r = np.zeros((len(bin_vec)-1,))

    LC.properties["universe"].trajectory[ts]
    if LC.bulk == True:
        residues = LC.properties["universe"].residues
    else:
        residues = LC.properties["universe"].select_atoms("resname {}CB".format(LC.n)).residues
    
    for i in range(len(bin_vec)-1):
        low = bin_vec[i]
        high = bin_vec[i+1]
        indices = np.argwhere((COM_dist_mat >= low) & (COM_dist_mat < high))
        angle_sum = 0
        for idx in indices:
            idx0 = idx[0]
            idx1 = idx[1]
            res0 = residues[idx0]
            res1 = residues[idx1]
            N0 = res0.atoms[CN][0].position
            C0 = res0.atoms[CN][1].position

            CN0_vec = (N0 - C0)/np.sqrt(((N0 - C0)**2).sum())

            N1 = res1.atoms[CN][0].position
            C1 = res1.atoms[CN][1].position

            CN1_vec = (N1 - C1)/np.sqrt(((N1 - C1)**2).sum())
            cost_CN = np.dot(CN0_vec,CN1_vec)
            angle_sum += cost_CN
        if len(indices) == 0:
            cost_r[i] = 0
        else:
            cost_r[i] = angle_sum/len(indices)

    return cost_r

# find the probability distribution of cos(theta) where theta is the angle between a pair of nCB molecules
cpdef pcost_CN(LC,int ts,np.ndarray COM_dist_mat,str segment='benzene1',float distance_constraint=np.inf,int bins=100,g1=False):
    """
    find the p(cos(theta)) between CN subject to the constraint that segment is within distance_constraint

    ts: the time step for which p(cos(theta)) between CN vectors are calculated
    COM_dist_mat: the COM distance matrix (size NxN) while is it really just a upper triangular matrix without diagonal 
    segment: the segment at which the constraint is upon
    distance_constraint: the distance between segment
    bins: The number of bins to split between (-1,1)
    """
    cdef np.ndarray bin_vec = np.linspace(-1,1,bins)  
    cdef np.ndarray CN = LC.segments["CN"]
    cdef np.ndarray indices = np.argwhere((COM_dist_mat <= distance_constraint) & (COM_dist_mat > 0))
    cdef int ix=0,idx0,idx1
    cdef float cost_CN
    cdef np.ndarray cost_CN_vec = np.zeros((len(indices),))
    cdef np.ndarray digitized
    cdef np.ndarray npbin_count
    cdef np.ndarray normalized_pcost
    cdef list bin_count


    LC.properties["universe"].trajectory[ts]

    for idx in indices:
        idx0 = idx[0]
        idx1 = idx[1]
        res0 = LC.properties['universe'].residues[idx0]
        res1 = LC.properties['universe'].residues[idx1]
        N0 = res0.atoms[CN][0].position
        C0 = res0.atoms[CN][1].position

        CN0_vec = (N0 - C0)/np.sqrt(((N0 - C0)**2).sum())

        N1 = res1.atoms[CN][0].position
        C1 = res1.atoms[CN][1].position

        CN1_vec = (N1 - C1)/np.sqrt(((N1 - C1)**2).sum())
        cost_CN = np.dot(CN0_vec,CN1_vec)

        cost_CN_vec[ix] = cost_CN
        ix += 1

    cost_CN_vec = np.sort(cost_CN_vec)
    cost_CN_vec = cost_CN_vec[cost_CN_vec != np.inf]
    digitized = np.digitize(cost_CN_vec,bin_vec)
    bin_count = [(digitized == i).sum() for i in range(1,len(bin_vec))]
    npbin_count = np.array(bin_count)
     

    return (bin_vec[:-1],npbin_count) 

def density_z_atoms(LC,start_time,end_time,vol,direction='z',skip=1,bins_z=100,verbose=False,Broken_interface=None):
    """
    calculates density of LC as a function of z
    
    Args:
        LC: Liquid crystal object 
        start_time(float): the starting time of the calculation
        end_time(float): the ending time of the calculation
        vol(tuple): The volume passed in as (Lx,Ly,Lz)
        direction(str): the direction where calculations is performed along(x,y,z)
        skip(int): the number of time frames to skip (default 1)
        bins_z(int): Number of bins in the direction (default 100)
        verbose(bool): whether or not to be verbose (default False)
        Broken_interface(float): Where to draw the line for broken interface (default None)
    
    returns: 
        density as a function of z (bins_z-1,)
    """
    t_idx = np.linspace(0,LC.time,len(LC))
    start_idx = np.searchsorted(t_idx,start_time,side='left')
    end_idx = np.searchsorted(t_idx,end_time,side='right')
    time_idx = np.arange(start_idx,end_idx,skip)
    density_z = np.zeros((bins_z-1,)) 

    if direction == 'x':
        d = 0
    
    if direction == 'y':
        d = 1

    if direction == 'z':
        d = 2

    u = LC["universe"]
    Lx,Ly,Lz = vol
    if Broken_interface is not None:
        draw_line = Broken_interface


    for ts in time_idx:
        # set the time frame to be ts
        u.trajectory[ts]
        pos = u.select_atoms("resname {}CB".format(LC.n)).atoms.positions

        # take only the dth dimension number of all COM 
        pos = pos[:,d] #(n_atoms,)
    
        if Broken_interface is not None:
            bot_idx = np.argwhere(pos <= draw_line)
            pos[bot_idx] += Lz

        pos_vec = np.linspace(pos.min(),pos.max(),bins_z)

        hist,_ = np.histogram(pos,pos_vec)
        hist = hist/(Lx*Ly*(pos_vec[1]-pos_vec[0]))
        density_z = density_z + hist
 
        if verbose:
            print("time step {} is done".format(ts))        
    return density_z/len(time_idx)

def density_z_COM(LC,start_time,end_time,direction='z',segment='whole',skip=1,bins_z=100,verbose=False,Broken_interface=None):
    """
    calculates density of LC as a function of z

    LC: Liquid crystal object 
    start_time: the starting time of the calculation
    end_time: the ending time of the calculation
    direction: the direction where we can perform calculations along
    segment: the segment at which we calculate COM for in LC molecules
    skip: the number of time frames to skip 
    
    returns: 
        density as a function of z (bins_z-1,)
    """
    t_idx = np.linspace(0,LC.time,len(LC))
    start_idx = np.searchsorted(t_idx,start_time,side='left')
    end_idx = np.searchsorted(t_idx,end_time,side='right')
    time_idx = np.arange(start_idx,end_idx,skip)
    density_z = np.zeros((bins_z-1,)) 

    if direction == 'x':
        d = 0
    
    if direction == 'y':
        d = 1

    if direction == 'z':
        d = 2

    for ts in time_idx:
        # first find Center of mass matrix at time step "ts"
        COM_mat = LC.COM(ts,segment) #(n_molecues,3)

        # take only the dth dimension number of all COM 
        COM_mat = COM_mat[:,d] #(n_molecules,)

        # set the universe trajectory to "ts" time step
        LC['universe'].trajectory[ts]

        if LC.bulk == True:
            residues = LC['universe'].residues
        else:
            residues = LC['universe'].select_atoms("resname {}CB".format(LC.n)).residues

        COM_vec,Ntop = fix_interface(COM_mat,Broken_interface=Broken_interface,verbose=verbose,bins=bins_z)

        for i in range(bins_z-1):
            if Ntop != 0:
                if i >= Ntop:
                    j = i + 1
                else:
                    j = i
            else:
                j = i

            less = COM_vec[j]
            more = COM_vec[j+1]

            index = np.argwhere(((COM_mat >= less) & (COM_mat < more)))
            index = index.flatten()
            density_z[i] += len(index)
        if verbose:
            print("time step {} is done".format(ts))        
    return density_z/len(time_idx)

def fix_interface(vec,Broken_interface=None,verbose=False,bins=100):
    """
    Function that identifies broken interface and fixes it

    vec: a vector that holds the quantity that is broken up by interface (N,)
    Broken_interface:
        (a) None
        (b) a tuple that holds (Lz,draw_line)
    verbose: whether to be verbose during execution
    bins: number of bins to bin the direction z 
    
    returns:
        a new_vec of length (bins,)
    """
    vec_min,vec_max = vec.min(),vec.max()

    if Broken_interface == None:
        new_vec = np.linspace(vec_min,vec_max+1,bins)
        N_top = 0
        Nbot = bins-N_top
    else:
        Lz,draw_line = Broken_interface
        # first check if the max_COM - min_COM is larger than Lz
        if vec_max - vec_min >= Lz:
            # find COM above the line and find COM below the line
            vec_top = vec[vec>draw_line]
            vec_bot = vec[vec<=draw_line]
            N_top = len(vec_top)
            N_bot = len(vec_bot)
            N_top = round(N_top/(N_bot+N_top)*bins)
            N_bot = bins-N_top

            min_top = vec_top.min()
            max_top = vec_top.max()
            min_bot = vec_bot.min()
            max_bot = vec_bot.max()
            if verbose:
                print("min_top:{}".format(min_top))
                print("max_top:{}".format(max_top))
                print("min_bot:{}".format(min_bot))
                print("max_bot:{}".format(max_bot))

            bot_vec = np.linspace(min_bot,max_bot,N_bot)
            top_vec = np.linspace(min_top,max_top,N_top+1)
            new_vec = np.concatenate((top_vec,bot_vec))
        else:
            N_top = 0
            new_vec = np.linspace(vec_min,vec_max+1,bins)

    return new_vec,N_top

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
