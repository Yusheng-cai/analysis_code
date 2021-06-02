import sys
import matplotlib.pyplot as plt
import numpy as np
sys.path.insert(0,"../Liquid_crystal")
from Liquid_crystal import *


def make_p2_ref():
    LC = nCB("5cb_traj/run.tpr","5cb_traj/traj_comp.xtc", 5, 1, trjconv=False)
    p2_list = np.zeros((len(LC),))

    for i in range(len(LC)):
        p2_list[i] = LC.p2(i)

    np.save("5cb_traj/p2_ref.npy",p2_list)

def make_p2tilde_ref():
    LC = nCB("5cb_traj/run.tpr","5cb_traj/traj_comp.xtc", 5, 1, trjconv=False)
    p2tilde_list = np.zeros((len(LC),))
    min_  = np.array([10.0,20.0,0.0])
    max_  = np.array([20.0,40.0,30.0])

    for i in range(len(LC)):
        p2tilde_list[i] = LC.p2tilde(i, min_, max_)
        print(p2tilde_list[i])

    np.save("5cb_traj/p2tilde_ref.npy",p2tilde_list)

def make_p2cos_ref():
    LC = nCB("5cb_traj/run.tpr","5cb_traj/traj_comp.xtc", 5, 1, trjconv=False)
    p2cos_list = np.zeros((len(LC),))
    n  = np.array([1,2,3])

    for i in range(len(LC)):
        p2cos_list[i] = LC.p2_cos(i,n)
        print(p2cos_list[i])

    np.save("5cb_traj/p2cos_ref.npy",p2cos_list)

def test_p2():
    LC = nCB("5cb_traj/run.tpr","5cb_traj/traj_comp.xtc", 5, 1, trjconv=False)
    p2_list = np.zeros((len(LC),))

    for i in range(len(LC)):
        p2_list[i] = LC.p2(i)

    ref = np.load("5cb_traj/p2_ref.npy")
    err = np.max(np.abs(ref-p2_list)/np.abs(ref))
    assert err < 1e-10

def test_p2tilde():
    LC = nCB("5cb_traj/run.tpr","5cb_traj/traj_comp.xtc", 5, 1, trjconv=False)
    p2tilde_list = np.zeros((len(LC),))
    min_  = np.array([10.0,20.0,0.0])
    max_  = np.array([20.0,40.0,30.0])

    for i in range(len(LC)):
        p2tilde_list[i] = LC.p2tilde(i, min_, max_)

    ref = np.load("5cb_traj/p2tilde_ref.npy")
    err = np.max(np.abs(ref-p2tilde_list)/np.abs(ref))
    assert err < 1e-10

def test_p2cos():
    LC = nCB("5cb_traj/run.tpr","5cb_traj/traj_comp.xtc", 5, 1, trjconv=False)
    p2cos_list = np.zeros((len(LC),))
    n  = np.array([1,2,3])

    for i in range(len(LC)):
        p2cos_list[i] = LC.p2_cos(i,n)

    ref = np.load("5cb_traj/p2cos_ref.npy")
    err = np.max(np.abs(ref - p2cos_list)/np.abs(ref))
    assert err < 1e-10



if __name__ == '__main__':
    # make ref
    test_p2()
    test_p2tilde()
