import sys
import numpy as np
sys.path.insert(0,"../Liquid_crystal")
from analysis_code.Liquid_crystal.nCB import nCB, nCB_PV
from analysis_code.ProbeVolume.ProbeVolume_Box import ProbeVolume_Box

min_  = np.array([10.0,20.0,0.0])
max_  = np.array([20.0,40.0,30.0])
n  = np.array([1,2,3])

def make_ref():
    LC      = nCB("5cb_traj/run.tpr","5cb_traj/traj_comp.xtc",5,trjconv=False)
    pv_     = ProbeVolume_Box("5cb_traj/run.tpr","5cb_traj/traj_comp.xtc",min_,max_)
    LC_tilde= nCB_PV("5cb_traj/run.tpr","5cb_traj/traj_comp.xtc",5,pv_,trjconv=False)

    p2_list = np.zeros((len(LC),))
    p2tilde_list = np.zeros((len(LC),))
    p2cos_list = np.zeros((len(LC),))

    for i in range(len(LC)):
        p2_list[i] = LC.p2(i)
        p2tilde_list[i] = LC_tilde.p2tilde(i)
        p2cos_list[i] = LC.p2_cos(i,n)

        print("p2 at {} is {:.5f}".format(i, p2_list[i]))
        print("p2tilde at {} is {:.5f}".format(i, p2tilde_list[i]))
        print("p2cos at {} is {:.5f}".format(i, p2cos_list[i]))

    np.save("5cb_traj/p2_ref.npy",p2_list)
    np.save("5cb_traj/p2tilde_ref.npy",p2tilde_list)
    np.save("5cb_traj/p2cos_ref.npy",p2cos_list)

def test_p2():
    LC      = nCB("5cb_traj/run.tpr","5cb_traj/traj_comp.xtc",5,trjconv=False)
    pv_     = ProbeVolume_Box("5cb_traj/run.tpr","5cb_traj/traj_comp.xtc",min_,max_)
    LC_tilde= nCB_PV("5cb_traj/run.tpr","5cb_traj/traj_comp.xtc",5,pv_,trjconv=False)

    p2_list = np.zeros((len(LC),))
    p2tilde_list = np.zeros((len(LC),))
    p2cos_list = np.zeros((len(LC),))

    for i in range(len(LC)):
        p2_list[i] = LC.p2(i)
        p2tilde_list[i] = LC_tilde.p2tilde(i)
        p2cos_list[i] = LC.p2_cos(i,n)

    p2_ref = np.load("5cb_traj/p2_ref.npy")
    p2tilde_ref = np.load("5cb_traj/p2tilde_ref.npy")
    p2cos_ref   = np.load("5cb_traj/p2cos_ref.npy")

    err1 = np.max(np.abs(p2_ref - p2_list)/np.abs(p2_ref))
    err2 = np.max(np.abs(p2tilde_ref - p2tilde_list)/np.abs(p2tilde_ref))
    err3 = np.max(np.abs(p2cos_ref - p2cos_list)/np.abs(p2cos_list))

    assert(err1 < 1e-10)
    assert(err2 < 1e-10)
    assert(err3 < 1e-10)

if __name__ == '__main__':
    #make_ref()
    test_p2()