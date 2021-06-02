import sys
import matplotlib.pyplot as plt
import numpy as np
sys.path.insert(0,"../Liquid_crystal")
from Liquid_crystal import *


def make_p2derivative_ref():
    LC = nCB("5cb_traj/run.tpr","5cb_traj/traj_comp.xtc", 5, 1, trjconv=False)
    head_d = np.zeros((len(LC),LC.n_molecules, 3))

    for i in range(len(LC)):
        head_d[i] = LC.p2_prime(i)[0]

    np.save("5cb_traj/p2prime_ref.npy",head_d)

# I did this because I made sure that none of the derivative is zero, TAKE CARE DOING ERROR THIS WAY!!!
def test_p2derivative():
    LC = nCB("5cb_traj/run.tpr","5cb_traj/traj_comp.xtc", 5, 1, trjconv=False)
    head_d = np.zeros((len(LC),LC.n_molecules, 3))

    for i in range(len(LC)):
        head_d[i] = LC.p2_prime(i)[0]

    ref = np.load("5cb_traj/p2prime_ref.npy")
    err = np.max(np.abs(head_d - ref)/np.abs(ref))

    assert err < 1e-10

if __name__ == "__main__":
    make_p2derivative_ref()
