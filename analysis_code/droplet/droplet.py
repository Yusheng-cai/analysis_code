import numpy as np
import MDAnalysis as mda
from skimage import measure 
from scipy import optimize


class droplet:
    def __init__(self,tpr,xtc):
        self.tpr = tpr
        self.xtc = xtc
        self.u = mda.Universe(tpr,xtc)

    def densityfield2d(self,resname,x1,x2,tmin,tmax,skip=1,verbose=False):
        """
        Calculate the density field of the droplet (only 2d supported)
        
        Args:
            residue_name(str): The name of the residue for which we are calculating the density field for (This is included because that in droplet simulation, there is usually a surface present)
            x1(tuple): A tuple that contains the following information (x1min,x1max,num_x1).
            x2(tuple): A tuple that contains the following information (x2min,x2max,num_x2).
            tmin(int): The minimum time frame where we start averaging
            tmax(int): The maximum time frame where we start averaging
            skip(int): The number of time frames to skip (default 1)
            dir_(str): A string siginifying which direction that we are integrating over (default 'y')
            verbose(bool): A boolean that tells the program whether or not to be verbose (default False)

        Return:
            densityfield: 2d density field 
        """
        u = self.u
        N = len(u.trajectory)

        # extract information from the tuples
        x1min,x1max,num_x1 = x1
        x2min,x2max,num_x2 = x2

        xx1 = np.linspace(x1min,x1max,num_x1)
        xx2 = np.linspace(x2min,x2max,num_x2)
        field = np.zeros((num_x1-1,num_x2-1))
        tlist = np.arange(tmin,tmax,skip)

        for t in tlist:
            u.trajectory[t]
            residues = u.select_atoms(resname)
            pos = residues.positions

            COM = residues.atoms.center_of_mass()
            COM[-1] = 0

            pos = pos - COM
            hist2d,_,_ = np.histogram2d(pos[:,0],pos[:,2],bins=[xx1,xx2])
            field += hist2d/len(tlist)
            if verbose:
                print("{} is done".format(t))

        return field
    
    @staticmethod
    def findcontour2d(field,c):
        """
        A class that finds the contour line given the isosurface parameters c

        Args:
            field(numpy.ndarray): A numpy array which has the density field 
            c(float): The isosurface number
        """
        return measure.find_contours(field,c)

    @staticmethod
<<<<<<< HEAD
    def fitcircle2d(contour):
=======
    def fitcircle(contour:np.ndarray):
>>>>>>> 2faeb95d57b680f0d1ac8c1f70fbc32d6f64d60b
        """
        Fit a circle to the contour

        Args:
            contour(numpy.ndarray): A 2d-array with [x,y] coordinates of the contour

        Return: 
            (xc,yc,Ri): The center (xc,yc) and radius Ri of the fitted circle
        """
        assert len(contour.shape) == 2, "Please pass in a 2d numpy array"

        center_estimate = contour.mean(axis=0)

        center, _ = optimize.leastsq(func,center_estimate,args=(contour))

        Ri_arr = calc_R(contour,center)
        Ri = Ri_arr.mean() 

        return (center,Ri)

    @staticmethod 
    def fitsphere3d(contour):
        """
        Args:
            pos(numpy.ndarray): A 2d-array with [x,y,z] coordinates of the contour

        Return:
            (xc,yc,zc,R): The center (xc,yc,zc) and radius Ri of the fitter sphere
        """
        x = contour[:,0]
        y = contour[:,1]
        z = contour[:,2]

        xm = x.mean()
        ym = y.mean()
        zm = z.mean()

        center_estimate = xm,ym,zm
        center, _ = optimize.leastsq(func3d,center_estimate,args=(x,y,z))

        xc,yc,zc = center
        Ri_arr = calc_R3d(x,y,z,xc,yc,zc)
        Ri = Ri_arr.mean() 

        return (xc,yc,zc,Ri)


def calc_R(pos:np.ndarray,c:np.ndarray):
    """
    A function that calculates distance of (xc,yc) and all the x,y points that is fitted to a circle

    Args:
        x(numpy.ndarray): A vector of floats that represents the x-axis of the points on the circle
        y(numpy.ndarray): A vecotr of floats that represents the y-axis of the points on the circle
        xc(float): x axis of the circle center
        yc(float): y axis of the circle center

    Return:
        R(numpy.ndarray): An array of radius
    """
    assert pos.shape[1] == len(c), "Shape of passed in position does not match the center position"
    return np.sqrt(((pos - c)**2).sum(axis=1))

<<<<<<< HEAD
def calc_R3d(x,y,z,xc,yc,zc):
    """
    A function that calculates distance of (xc,yc,zc) and all the x,y points that is fitted to a circle

    Args:
        x(numpy.ndarray): A vector of floats that represents the x-axis of the points on the circle
        y(numpy.ndarray): A vecotr of floats that represents the y-axis of the points on the circle
        z(numpy.ndarray): A vector of floats that represents the z-axis of the points on the circle
        xc(float): x axis of the circle center
        yc(float): y axis of the circle center
        zc(float): z axis of the circle center

    Return:
        R(numpy.ndarray): An array of radius
    """
    return np.sqrt((x-xc)**2+(y-yc)**2+(z-zc)**2)

def func(c,x,y):
=======
def func(c:np.ndarray,pos:np.ndarray):
>>>>>>> 2faeb95d57b680f0d1ac8c1f70fbc32d6f64d60b
    """
    Calculate the algebraic distance between the data points and the mean circle centered at c=(xc,yc)

    Args:
        x(numpy.ndarray): A vector of floats that represents the x-axis of the points on the circle
        y(numpy.ndarray): A vecotr of floats that represents the y-axis of the points on the circle
        c(tuple): tuple of (xc,yc)

    Return:
        dr(numpy.ndarray): A vector of values that represents the distance between the data points and the mean circle centered at c=(xc,yc) 
    """
    Ri = calc_R(pos,c)
    return Ri - Ri.mean() 

def func3d(c,x,y,z):
    """
    Calculate the algebraic distance between the data points and the mean circle centered at c=(xc,yc)

    Args:
        x(numpy.ndarray): A vector of floats that represents the x-axis of the points on the circle
        y(numpy.ndarray): A vecotr of floats that represents the y-axis of the points on the circle
        c(tuple): tuple of (xc,yc)

    Return:
        dr(numpy.ndarray): A vector of values that represents the distance between the data points and the mean circle centered at c=(xc,yc) 
    """
    Ri = calc_R3d(x,y,z,*c)
    return Ri - Ri.mean() 
