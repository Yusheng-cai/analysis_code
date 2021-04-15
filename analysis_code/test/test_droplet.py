import sys
import matplotlib.pyplot as plt
import os
import numpy as np
sys.path.insert(0,"../droplet/")
from droplet import droplet

if __name__ == "__main__":
    d = droplet(os.getcwd() + '/droplet_data/eqrun0p05.tpr',\
            os.getcwd() + '/droplet_data/traj_compp5_pbc.xtc')
    u = d.u
    N = len(u.trajectory)
    c = 2.0

    x1 = (-50,50,101)
    x2 = (15,55,51)
    f = d.densityfield2d("resname RM1",x1,x2,150,N,verbose=True)
    contour = droplet.findcontour(f,2.0)[0]
    xc,yc,Ri = droplet.fitcircle(contour)

    plt.scatter(contour[:,0],contour[:,1])
    theta = np.linspace(0,2*np.pi,100)
    xcircle = Ri*np.cos(theta) + xc
    ycircle = Ri*np.sin(theta) + yc
    print(xc)
    print(yc)
    plt.plot(xcircle,ycircle)

    plt.savefig("droplet_test.png")
