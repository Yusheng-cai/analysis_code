from __future__ import absolute_import
import MDAnalysis as mda
from MDAnalysis.transformations.wrap import unwrap
import numpy as np
import time

from ..ProbeVolume.Bounding_box import Bounding_box
from ..ProbeVolume.ProbeVolume  import _ProbeVolume

class Liquid_crystal:
    """
    Base class for all Liquid crystals

    Args:
    ----
    tpr(string)  : The path to the .tpr file of the Liquid crystal molecule
    xtc(string)  : The path to the .xtc file of the Liquid crystal molecule
    name(string) : The name of the residue 
    head_id(int) : The residue index of the head atom (index in a residue)
    tail_id(int) : The residue index of the tail atom
    trjconv(bool): Whether or not the .xtc || .trr file has been processed with gmx trjconv
    bulk(bool)   : Whether or not the simulation is for a bulk system
    """
    def __init__(self,tpr:str,xtc:str,name:str,head_id:int,tail_id:int,trjconv:bool=True,bulk:bool=True):
        self.tpr_           = tpr
        self.xtc_           = xtc
        self.u_             = mda.Universe(tpr,xtc)
        self.bulk_          = bulk
        self.name_          = name
        self.head_id_       = head_id
        self.tail_id_       = tail_id
        self.trjconv_       = trjconv
        self.bb_            = Bounding_box(self.u_)

        if self.bulk_ == True:
            residues       = self.u_.residues
            self.Nresidues_= len(residues)
            self.natoms    = len(self.u_.residues[0].atoms.positions)
        else:
            residues       = self.u_.select_atoms("resname {}".format(self.name_)).residues
            self.Nresidues_= len(residues)
            self.natoms    = len(residues[0].atoms.positions)
        
        self.segment_      = {"whole":np.arange(0,self.natoms)}
        self.director_mat_ = np.zeros((self.Nresidues_,3))
        self.norm_mat_     = np.zeros((self.Nresidues_,1))

        # This is for calculation of COM & MOI, very slow for other processes 
        # if not self.trjconv_:
        #     if self.bulk_:
        #         transform      = unwrap(self.u_.atoms)
        #         self.u_.trajectory.add_transformations(transform)
        #     else:
        #         atoms          = self.u_.select_atoms("resname {}".format(self.name_)).atoms
        #         transform      = unwrap(atoms)
        #         self.u_.trajectory.add_transformations(transform)
    
    def __len__(self):
        return len(self.u_.trajectory)
    
    def get_residues(self,ts:int):
        """
        Obtain the residue groups at time steps ts

        Return:
        -------
            residue(mda.ResidueGroup)   : The correct residue that corresponds to whether or not the simulation is bulk
        """ 
        u   = self.u_
        u.trajectory[ts]
        
        if self.bulk_:
            return u.residues
        else:
            return u.select_atoms("resname {}".format(self.name_)).residues

    def pos(self,ts:int):
        """
        Function that returns the position of each of the mesogen atoms in the system at time ts

        Args:
        -----
            ts(int): The time frame at which the calculation is performed upon

        Return:
        -------
            pos(numpy.ndarray): The position matrix at time ts of shape (N,3)
        """
        u = self.u_
        u.trajectory[ts]

        if self.bulk_ == True:
            return u.select_atoms("resname {}".format(self.name_)).atoms.positions
        else:
            return u.atoms.positions
    
    def head_tail_pos(self,ts:int):
        """
        Obtain the positions of the head atom at time ts

        Args:
        -----
            ts(int)     : The time step at which this calculation is performed up
        
        Return:
        -------
            head_pos(numpy.ndarray)     : The head positions returned in a np.ndarray with shape (N,3)
        """
        head_id  = self.head_id_
        tail_id  = self.tail_id_
        head_pos = np.zeros((self.Nresidues_,3))
        tail_pos = np.zeros((self.Nresidues_,3))
        residues = self.get_residues(ts)

        ix = 0
        for res in residues:
            h = res.atoms[head_id].position
            t = res.atoms[tail_id].position
            head_pos[ix] = h
            tail_pos[ix] = t
            ix += 1
        
        return head_pos, tail_pos

    def COM(self,ts:int,segment:str="whole"):
        """
        Function that calculates the center of mass of the Liquid crystal molecule at time ts

        Args:
        -----
            ts(int): The time frame at which the calculation is performed on 
            segment(str): The segment that the user wants to calculate center of mass for (default None)

        Return:
        -------
            COM_mat(numpy.ndarray): The center of mass matrix of shape (N,3)
        """
        if segment in self.segment_.keys():
            idx = self.segment_[segment]
        else:
            raise NotImplementedError("The segment {} is not currently implemented".format(segment))

        u         = self.u_
        u.trajectory[ts]
        Nresidues = self.Nresidues_
        COM_mat   = np.zeros((Nresidues,3))        
        residues  = self.get_residues(ts) 

        ix = 0
        for res in residues:
            COM_mat[ix] = res.atoms[idx].center_of_mass()
            ix += 1

        return COM_mat
 
    def director_mat(self,ts:int,MOI:bool=False):
        """
        Function that finds the director vector of all the residues in the system and put it in a matrix of shape (N,3). This can also find the director matrix using Moment of inertia tensor (the eigenvector that corresponds to the lowest eigenvalue of MOI tensor)

        Args:
        -----
            ts(int)  : The time frame of the simulation that this operation is performed upon
            MOI(bool): Whether or not to find director matrix using Moment of Inertia tensor

        Return:
        -------
            vec(numpy.ndarray): The director matrix of all the residues in the system
        """
        u       = self.u_
        u.trajectory[ts]
        bb           = self.bb_
        residues     = self.get_residues(ts) 
        
        residues     = self.get_residues(ts)

        ix = 0 
        if MOI:
            for res in residues:
                self.director_mat_[ix] = res.atoms.principal_axes()[-1]
                ix += 1
        else:
            h,t                 = self.head_tail_pos(ts)
            self.director_mat_  = h - t

        if not MOI:
            if not self.trjconv_:
                self.director_mat_     = bb.dr_pbc(self.director_mat_,ts)
                self.norm_mat_         = np.sqrt((self.director_mat_**2).sum(axis=1,keepdims=True))
                self.director_mat_     = self.director_mat_/self.norm_mat_
            else:
                self.norm_mat_         = np.sqrt((self.director_mat_**2).sum(axis=1,keepdims=True))
                self.director_mat_     = self.director_mat_/self.norm_mat_

        return self.director_mat_, self.norm_mat_

    def Qmatrix(self,ts:int,MOI:bool=False):
        """
        Function that calculates the Qmatrix of the system at time ts.

        Args:
        -----
            ts(int)     : The time frame of the simulation
            MOI(bool)   : Whether or not to use moment of inertia tensor

        Return:
        -------
            1.Qmatrix(numpy.ndarray)= The Q matrix of the liquid crystalline system
            2.eigval(numpy.ndarray) = The eigenvalue of the system at time ts
            2.eigvec(numpy.ndarray) = The eigenvector of the system at time ts
        """
        uij, _      = self.director_mat(ts,MOI=MOI)
        Nresidues   = self.Nresidues_

        Q = 1.5/Nresidues*np.matmul(uij.T,uij) - 0.5*np.eye(3)

        eigval,eigvec = np.linalg.eig(Q)
        order = np.argsort(eigval)
        eigval = eigval[order]
        eigvec = eigvec[:,order]

        return Q,eigval,eigvec
    
    def director(self, ts:int, MOI:bool=False):
        """
        Function that gets the director of the system at time ts, director is defined as the eigenvector that corresponds to the 
        largest eigenvalue of the Q tensor

        Args:
        -----
            ts(int)    : The time step at which the calculation is performed on
            MOI(bool)  : A boolean that signifies whether or not we are calculating Q tensor with moment of inertia tensor
        
        Returns:
        --------
            director(numpy.ndarray)     : The director of the system that is defined as the eigenvector that corresponds to the largest eigenvalue of the Q tensor
        """
        _,_,eigvec           = self.Qmatrix(ts, MOI=MOI)

        return eigvec[:,-1]
        
    def p2(self,ts:int, MOI:bool=False):
        """
        Function calculating p2 order parameter for nCB molecule using Q tensor formulation
        Q matrix is calculated as following:

        Q = \sum_{l}^{N}(3u_{l}u_{l}^{T} - I)/2N

        we choose p2 to be -2*lambda_{0} where lambda_{0} is the second largest eigenvalue of Q
        u_{l} is chosen to be the normalized vector between C and N in nCB

        Return:
        -------
            p2(float)    : p2 value at time ts
        """
        _, eigval, _ = self.Qmatrix(ts, MOI=MOI)
        p2           = eigval[1]*(-2.0) 

        return p2

    def dp2_dx(self,ts:int):
        """
        Derivative of p2 with respect to the head & tail atoms

        Args:
        -----
            ts(int)        : The time step the calculation is performed on

        Return:
        -------
            head_derivative(numpy.ndarray) : A numpy.ndarray with shape (N,3) with derivatives for the head atom
            tail_derivative(numpy.ndarray) : A numpy.ndarray with shape (N,3) with derivatives for the tail atom
        """
        # self.director_mat_ & self.norm_mat_ should be updated by Qmatrix call
        _,_,eigvec        = self.Qmatrix(ts)
        Nresidues         = self.Nresidues_
        uij               = self.director_mat_
        norm_mat          = self.norm_mat_

        v1 = eigvec[:,1]
        dot_product     = (uij*v1).sum(axis=1,keepdims=True)
        head_derivative = -6/(Nresidues*norm_mat)*dot_product*(v1 - uij*dot_product)
        tail_derivative = -1.0*head_derivative

        return (head_derivative, tail_derivative)

    def p2_cos(self,ts:int,n:np.ndarray):
        """
        Function that calculates p2 using the second legendre polynomial 

        Args:
        -----
            ts(int)         : The time step at which the calculation is performed on
            n(numpy.ndarray): The director at which the calculation is performed with, doesn't need to be normalized

        Return:
        -------
            p2(float)       : The p2 value
        """
        n           = n/np.sqrt((n**2).sum())
        uij,_       = self.director_mat(ts)
        Nresidues   = self.Nresidues_
        dot_product = (uij*n).sum(axis=1,keepdims=True)

        p2 = 1/Nresidues*(1.5*(dot_product)**2 - 0.5).sum()

        return p2
    
    def dp2cos_dx(self, ts:int, n:np.ndarray):
        """
        Function that calculates the derivative of p2 with second legendre polynomial formulation dp2cos_dr

        Args:  
        -----
            ts(int)         : The time step at which the calculation is performed on
            n(numpy.ndarray): The director at which the calculation is performed with, doesn't need to be normalized

        Return:
        -------
            tuple of dp2cos_dr(numpy.ndarray) : dp2cos_dr with shape (N,3) with (head_derivative, tail_derivative)
        """
        n               = n/np.sqrt((n**2).sum())
        uij,norm_mat    = self.director_mat(ts)
        Nresidues       = self.Nresidues_
        dot_product     = (uij*n).sum(axis=1,keepdims=True)

        head_derivative = 3/(Nresidues*norm_mat)*dot_product*(n - uij * dot_product)
        tail_derivative = -1.0*head_derivative

        return (head_derivative, tail_derivative)

    def v0(self, ts:int, n:np.ndarray):
        """
        Function that calculates the OP with the director of the system, eigenvector that corresponds to the largest eigenvalue
        OP: (n\cdot)**2 - 1

        Args:
        ----
            ts(int)          : The time step at which the calculation is performed on 
            n(numpy.ndarray) : The user provided director n
        
        Return:
        -------
            v0(float)        : The OrderParameter defined above for that time step
        """
        n               = n/np.sqrt((n**2).sum())
        director        = self.director(ts, MOI=False)

        v0              = np.dot(n, director)**2 - 1

        return v0

    def dv0_dr(self, ts:int, n:np.ndarray):
        """
        Function that calculates the derivative of the OP with director of the system with respect to r

        Args:
        -----
            ts(int)          : The time step at which the calculation is performed on
        
        Returns:
        --------
            derivatives(numpy.ndarray)  : The derivatives of all the atoms for this OP (N,3)
        """
        uij, norm_mat        = self.director_mat(ts, MOI=False)
        _, eigval, eigvec    = self.Qmatrix(ts, MOI=False)
        v0                   = eigvec[:,-1] 
        e0                   = eigval[-1]
        nv0                  = (n*v0).sum()
        c                    = 3*nv0/(self.Nresidues_*norm_mat)
        derivatives          = np.zeros((self.Nresidues_,3))
        uv0     = (uij*v0).sum(axis=1, keepdims=True)

        for i in [0,1]:
            vm      = eigvec[:,i]
            nvm     = (vm*n).sum()
            diff    = e0 - eigval[i]
            uvm     = (uij*vm).sum(axis=1, keepdims=True)
            derivatives += nvm/diff*(vm*uv0 - 2*uvm*uv0*uij + v0*uvm)
        
        derivatives *= c

        return derivatives

    # TODO: 
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


class Liquid_crystalPV(Liquid_crystal):
    """
    Base class for all Liquid crystals with probe volume

    Args:
    ----
    u(mda.Universe) : mda.Universe object for the LC system
    name(string)    : The name of the residue 
    head_id(int)    : The residue index of the head atom (index in a residue)
    tail_id(int)    : The residue index of the tail atom
    trjconv(bool)   : Whether or not the .xtc || .trr file has been processed with gmx trjconv
    bulk(bool)      : Whether or not the simulation is for a bulk system
    """
    def __init__(self,tpr, xtc,name,head_id,tail_id,pv:_ProbeVolume,trjconv=True,bulk=True):
        super().__init__(tpr,xtc,name,head_id,tail_id,trjconv=trjconv,bulk=bulk)
        # a probe volume object
        self.pv_         = pv
        self.head_pos    = np.zeros((self.Nresidues_,3))

    def Qtilde(self, ts):
        """
        Function that obtains Qtilde matrix as we define in a orthorhombic probe volume

        Args:
            ts(int)             : The time step at which the calculation is performed upon

        Returns:
            Q(numpy.ndarray)         : 3x3 Qtilde tensor
            Ntilde(float)            : The coarse-grained number of atoms in the system (Just head atoms)
            indicator(numpy.ndarray) : The indicator function of each atom (N,)
            Ncenter(numpy.ndarray)   : The corrected distance from N to center of probe volume
        """
        Nresidues   = self.Nresidues_
        Qtilde      = np.zeros((3,3))
        u           = self.u_
        u.trajectory[ts]
        pv          = self.pv_

        if self.bulk_:
            residues    = u.residues
        else:
            residues    = u.select_atoms("resname {}".format(self.name_)).residues

        for i in range(Nresidues):
            pos              = residues[i].atoms.positions
            hpos             = pos[0]
            self.head_pos[i] = hpos

        indicator, _  = pv.calculate_Indicator(self.head_pos, ts)
        Ntilde        = pv.get_Ntilde()
        uij, _        = self.director_mat(ts)

        for i in range(Nresidues):
            ui        = uij[i]
            Qtilde   += (1.5*np.outer(ui, ui) -0.5*np.eye(3))*indicator[i]

        Qtilde         = Qtilde/Ntilde
        eigval,eigvec  = np.linalg.eig(Qtilde)
        order          = np.argsort(eigval)
        eigval         = eigval[order]
        eigvec         = eigvec[:,order]

        return Qtilde, eigval, eigvec

    def p2tilde(self, ts):
        """
        Function that calculates p2tilde which is p2 in a probe volume

        Args:
            ts(int)             : The time step at which is calculation is performed upon

        Returns:
            p2tilde(float)      : P2tilde of the specified region
        """
        _, eigval, _   = self.Qtilde(ts)

        return eigval[1]*(-2.0)
        
    def p2tilde_prime(self,ts):
        """
        Derivative of p2tilde with respect to the head & tail atoms

        Args:
            ts(int)             : The time step the calculation is performed on
        """
        # reset these 2 variables
        self.director_mat_ = np.zeros_like(self.director_mat_)
        self.norm_mat_     = np.zeros_like(self.norm_mat_)

        # It has already calculated Indicator & hx in pv
        _, _, eigvec   = self.Qtilde(ts)
        v1             = eigvec[:,1]

        # Qtilde step updates director_mat_ & norm_mat_
        uij, norm_mat  = self.director_mat_, self.norm_mat_
        pv             = self.pv_

        dot_product = (v1*uij).sum(axis=1,keepdims=True)
        hx          =  pv.get_hx()
        Ntilde      =  pv.get_Ntilde()
        indicator   =  pv.get_indicator()
        dh_dr       =  pv.calculate_derivative(self.head_pos,hx,ts)

        first_term  = -6/(Ntilde*norm_mat)*indicator*dot_product*(v1 - uij*dot_product)
        second_term = 3/(Ntilde**2)*(dot_product**2*indicator).sum()*dh_dr - 3/Ntilde*dh_dr*(dot_product**2)

        return first_term + second_term

    # TODO: Finish from here
    def p2_tildecos(self, ts, n):
        """
        Function that calculates p2 using the second legendre polynomial in a probe volume

        Args:
            ts(int)             : The time step at which the calculation is performed on
            n(numpy.ndarray)    : The director vector
            min_(numpy.ndarray) : The minimum of the probe volume box (3,)
            max_(numpy.ndarray) : The maximum of the probe volume box (3,)

        Returns:
            p2tilde_cos(float)  : p2tilde_cos
        """
        u, norm = self.director_mat(ts)
    
    def v0_tilde(self, ts, n):
        """
        Function that calculates the OrderParameter with the director of the system which is defined as the eigenvector that 
        corresponds to the largest eigenvalue
        
        Args:
        ----
            ts(int)             : The time step at which the calculation is performed on
            n(numpy.ndarray)    : The user defined director vector (3,)
        
        Returns:
        --------
            v0_tilde(float)
        """
        pass