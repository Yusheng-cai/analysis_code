from inspect import getblock
import numpy as np
from numpy.lib.function_base import angle
from scipy import integrate
import MDAnalysis as mda

def read_dat(file_path):
    """
    Function that reads the .dat from INDUS simulations

    Args:
        file_path(str): the path to the file 

    Return:
        an numpy array that contains the N and Ntilde from the INDUS simulation (N, Ntilde) -> where both are of shape (nobs,)
    """
    f = open(file_path)
    lines = f.readlines()
    lines = [line for line in lines if line[0]!="#"]

    lines = [[float(num) for num in line.rstrip("\n").split()] for line in lines if line[0]!="#"]
    lines = np.array(lines)

    N = lines[:,1]
    Ntilde = lines[:,2]

    return (N,Ntilde,lines)

def read_dat_gen(file_path):
    """
    Function that reads the .dat from INDUS simulations

    Args:
        file_path(str): the path to the file 

    Return:
        an numpy array that contains the N and Ntilde from the INDUS simulation (N, Ntilde) -> where both are of shape (nobs,)
    """
    f = open(file_path)
    lines = f.readlines()
    lines = [line for line in lines if line[0]!="#"]

    lines = [[float(num) for num in line.rstrip("\n").split()] for line in lines if line[0]!="#"]
    lines = np.array(lines)

    return lines



def make_bins(data,min,max,bins=101):
    """
    A function that bins some data within min and max
    
    Args:
        data(np.ndarray): the data that you want to bin, pass in numpy array (shape(N,))
        min(float): minimum of the bins
        max(float): maximum of the bins
        bins(int): number of bins to make
    
    returns:
        tuple of (bins,binned_vec)
    """
    bin_ = np.linspace(min,max,bins)

    # right = False implies bins[i-1]<=x<bins[i]
    digitized = np.digitize(data,bin_,right=False)

    binned_vec = np.array([(digitized == i).sum() for i in range(1,bins)])


    return (bin_[:-1],binned_vec)

def cov_fi(wji,Ni):
    """
    This is a function that calculates the covariance matrix of fi where
    fi = -ln(Qi/Q0).

    Args:
        wji(np.ndarray): the weight matrices of different simulations (N,k) where N=number of observations 
        in total, k=number of simulations

        Ni(np.ndarray): the number of observations in each simulation

    returns 
        covariance of fi in shape (k,k)
    """
    # obtain shape information
    N,k = wji.shape

    # create a diagonal matrix of Ni
    N = np.diag(Ni)
    
    # define identity matrix 
    I = np.eye(k)

    # Perform SVD on wji matrix
    U, s , Vt = np.linalg.svd(wji,full_matrices=False)
    s = np.diag(s)


    # Inner part of covariance
    inner = I - s.dot(Vt).dot(N).dot(Vt.T).dot(s)
    pseudo_inverse = np.linalg.pinv(inner)

    # find covariance of ln(Qi)
    theta = (Vt.T).dot(s).dot(pseudo_inverse).dot(s).dot(Vt)

    # Using the above to find the covariance of -ln(Qi/Q0) where Q0 is at the 
    # end of the array Cov_i = Theta_{-1,-1} - 2*Theta_{-1,i} + Theta_{i,i}
    
    cov = np.zeros((k,))
    for i in range(k):
        cov[i] = theta[-1,-1] - 2*theta[-1,i] + theta[i,i]


    return cov

def ss_umbrella(qstar,qavg,qvar,kappa):
    """
    Sparse sampling performed for umbrella potentials k/2(q-q*)^2 

    Args:
        qstar(numpy.ndarray): A numpy array of the q* where q is the order parameter
        qavg(numpy.ndarray): A numpy array of the mean values at every simulation <q> where q is the Order Parameter
        qvar(numpy.ndarray): A numpy array of the variance values at every simulation <(q-<q>)^2>
        kappa(float): The kappa parameter in the potential

    Returns:
        F(numpy.ndarray): The unbiased free energy calculated from sparse sampling
    """
    FvkN = np.log(2*np.pi*qvar)

    integrand=kappa*(qstar - qavg)
    FkN = np.zeros((len(integrand),))
    for i in range(2,len(integrand)+1):
        FkN[i-1] = integrate.simps(integrand[:i],qstar[:i])

    UkN = kappa/2*(qstar-qavg)**2

    F = FvkN - UkN + FkN
    F = F - F.min()
    return (FvkN,FkN,UkN),F

def readTop(fname:str):
    """
    A function that reads the topology file of gromacs
    """
    f = open(fname)

    lines = f.readlines();
    lines = [ l.rstrip("\n").lstrip() for l in lines if l.isspace()==False]
    lines = [ l for l in lines if l.lstrip()[0]!=";"]
    lines = [ l for l in lines if l.lstrip()[0]!="#"]

    bracket_lines = []
    for (i,l) in enumerate(lines):
        if l.lstrip()[0] == "[":
            bracket_lines.append(i)

    dic = {}
    for (i,l) in enumerate(bracket_lines):
        index_start = l + 1
        name = lines[l].split()[1]
        dic[name] = []
        while True:
            if (index_start >= len(lines)):
                break
            line = lines[index_start]
            if line.lstrip()[0] == "[":
                break
            dic[name].append(line)
            index_start += 1
        
    return dic

def convert8CBSAMTo8BSAM(fname:str,tpr:str, gro:str,index:np.ndarray,resname="SAM",output="output.top"):
    dic_ = readTop(fname)

    u = mda.Universe(tpr,gro)
    res = u.select_atoms("resname {}".format(resname)).residues
    atoms = res[0].atoms[index]
    charges = atoms.charges.sum()
    a = [ at for at in atoms if at.type != "U1" and at.type != "U2" and at.type != "SH"]
    peratomcharge = charges/len(a)

    MapNewIndexToOldIndex = {}
    MapOldIndexToNewIndex = {}
    for (i,ix) in enumerate(index):
        MapNewIndexToOldIndex[i]  = ix
        MapOldIndexToNewIndex[ix] = i + 1

    globalIndexToResidueIndex = {}
    for (i,a) in enumerate(atoms):
        globalIndexToResidueIndex[a.index] = i + 1


    # first parse the topology file
    atomLines = dic_["atoms"]
    index2type = {}

    # atom information
    for l in atomLines:
        l = l.lstrip().split()
        if l[0] not in index2type:
            index2type[int(l[0])] = l[1]
       
    
    # bonded dic has [ ai, aj, func, b0 ,kb ]
    bonddic_ = {}
    bondLines = dic_["bonds"]

    for l in bondLines:
        l = l.lstrip().split()

        # ai, aj
        index1 = int(l[0])
        index2 = int(l[1])

        type1 = index2type[index1]
        type2 = index2type[index2]

        str_ = "{}-{}".format(type1, type2)
        l = [l[2], l[3], l[4]]

        if str_ not in bonddic_:
            bonddic_[str_] = l
    
    # pairs dict_;
    pairsdic_ = {}
    pairsLines = dic_["pairs"]

    for l in pairsLines:
        l = l.lstrip().split()

        index1 = int(l[0])
        index2 = int(l[1])

        if index1 not in MapOldIndexToNewIndex or index2 not in MapOldIndexToNewIndex:
            continue

        newIndex1 = MapOldIndexToNewIndex[index1]
        newIndex2 = MapOldIndexToNewIndex[index2]

        str_ = "{}-{}".format(newIndex1, newIndex2)

        l2 = l[2]

        if str_ not in pairsdic_:
            pairsdic_[str_] = l2 
    
    # angles_dict
    angledic_ = {}
    angleLines = dic_["angles"]

    for l in angleLines:
        l = l.lstrip().split()

        index1 = int(l[0])
        index2 = int(l[1])
        index3 = int(l[2])
        type1 = index2type[index1]
        type2 = index2type[index2]
        type3 = index2type[index3]

        str_ = "{}-{}-{}".format(type1,type2,type3)
        str_1= "{}-{}-{}".format(type3,type2,type1)

        l = [l[3], l[4], l[5]]

        if str_ not in angledic_:
            angledic_[str_] = l
            angledic_[str_1] = l
    
    # dihedral dict
    dihedraldic_ = {}
    dihedrallines = dic_["dihedrals"]

    for l in dihedrallines:
        l = l.lstrip().split()

        index1 = int(l[0])
        index2 = int(l[1])
        index3 = int(l[2])
        index4 = int(l[3])

        type1 = index2type[index1]
        type2 = index2type[index2]
        type3 = index2type[index3]
        type4 = index2type[index4]

        str_ = "{}-{}-{}-{}".format(type1,type2,type3,type4)
        str_1 = "{}-{}-{}-{}".format(type4, type3,type2,type1)

        l = [l[4], l[5], l[6], l[7]]

        if str_ not in dihedraldic_ and str_1 not in dihedraldic_:
            if str_ != str_1:
                dihedraldic_[str_] = []
                dihedraldic_[str_1] = []
                dihedraldic_[str_].append(l)
                dihedraldic_[str_1].append(l)
            else:
                dihedraldic_[str_] = []
                dihedraldic_[str_].append(l)
        else:
            if l not in dihedraldic_[str_]:
                if str_ != str_1:
                    dihedraldic_[str_].append(l)
                    dihedraldic_[str_1].append(l)
                else:
                    dihedraldic_[str_].append(l)

    f = open(output,"w")    

    f.write(" [ atoms ]\n")
    c = 0
    for (i,a) in enumerate(atoms):
        index = i + 1
        cgnr = index
        renr = 1
        if a.type != "U1" and a.type != "U2" and a.type != "SH":
            charge = a.charge - peratomcharge
        else:
            charge = a.charge
        f.write("{}\t{}\t{}\t{}\t{}\t{}\t{:.6f}\t{:.4f}\n".format(index, a.type, renr, a.resname, a.name, cgnr, charge,\
             a.mass))
        c += charge
    print("Total charge = {}".format(c))
            
    f.write("\n")
    f.write("[ pairs ]\n")
    for ix in pairsdic_:
        ix = ix.split("-")
        ix1 = ix[0]
        ix2 = ix[1]

        f.write("{}\t{}\t{}\n".format(ix1,ix2,1))
    

    f.write("\n") 
    f.write("[ bonds ]\n")
    for (i,b) in enumerate(atoms.bonds):
        if b.atoms[0].index not in globalIndexToResidueIndex or b.atoms[1].index not in globalIndexToResidueIndex:
            print("bond excluded: atom1 = {}, atom2 = {}".format(b.atoms[0].type, b.atoms[1].type))
            continue
        id1 = globalIndexToResidueIndex[b.atoms[0].index]
        id2 = globalIndexToResidueIndex[b.atoms[1].index]
        type1 = b.atoms[0].type
        type2 = b.atoms[1].type
        name1 = b.atoms[0].name
        name2 = b.atoms[1].name

        str_ = "{}-{}".format(type1,type2)
        str_2 = "{}-{}".format(type2,type1)

        if str_ not in bonddic_:
            bond_ = bonddic_[str_2]
        else:
            bond_ = bonddic_[str_]

        f.write("{}\t{}\t{}\t{}\t{}; {} - {}\n".format(id1, id2, bond_[0],bond_[1],bond_[2], name1, name2))
    
    f.write("\n")
    f.write("[ angles ]\n")

    for (i,a) in enumerate(atoms.angles):
        i1 = a.atoms[0]
        i2 = a.atoms[1]
        i3 = a.atoms[2]

        if i1.index not in globalIndexToResidueIndex or i2.index not in globalIndexToResidueIndex or i3.index not in globalIndexToResidueIndex:
            print("angle excluded: atom1 = {}, atom2 = {}, atom3 = {}".format(i1.type, i2.type, i3.type))
            continue

        type1 = i1.type
        type2 = i2.type
        type3 = i3.type
        name1 = i1.name
        name2 = i2.name
        name3 = i3.name

        id1 = globalIndexToResidueIndex[i1.index]
        id2 = globalIndexToResidueIndex[i2.index]
        id3 = globalIndexToResidueIndex[i3.index]

        str_ = "{}-{}-{}".format(type1,type2,type3)

        l = angledic_[str_]

        f.write("\t{}\t{}\t{}\t{}\t{}\t{}; {}-{}-{}\n".format(id1,id2,id3,l[0],l[1],l[2], name1, name2, name3))
    
    f.write("\n")
    f.write(" [ dihedrals ]\n")

    for (i,d) in enumerate(atoms.dihedrals):
        i1 = d.atoms[0]
        i2 = d.atoms[1]
        i3 = d.atoms[2]
        i4 = d.atoms[3]

        if i1.index not in globalIndexToResidueIndex or i2.index not in globalIndexToResidueIndex or \
            i3.index not in globalIndexToResidueIndex or i4.index not in globalIndexToResidueIndex:
            print("dihedral excluded, atom1 = {}, atom2 = {}, atom3 = {}, atom4 = {}".format(i1.type, i2.type, i3.type, i4.type))
            continue
            
        type1 = i1.type
        type2 = i2.type
        type3 = i3.type
        type4 = i4.type
        name1 = i1.name
        name2 = i2.name
        name3 = i3.name
        name4 = i4.name

        id1 = globalIndexToResidueIndex[i1.index]
        id2 = globalIndexToResidueIndex[i2.index]
        id3 = globalIndexToResidueIndex[i3.index]
        id4 = globalIndexToResidueIndex[i4.index]

        str_ = "{}-{}-{}-{}".format(type1,type2,type3,type4)
        l = dihedraldic_[str_]

        f.write("\n")

        for li in l:
            f.write("\t{}\t{}\t{}\t{}\t{}\t{}\t{:4f}\t{}; {}-{}-{}-{}\n".format(id1,id2,id3,id4,li[0],li[1],float(li[2]),li[3],\
                name1, name2, name3, name4))
        f.write("\n")
    
    f.close()




