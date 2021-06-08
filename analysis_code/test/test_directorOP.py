import sys
import numpy as np
from analysis_code.Liquid_crystal.nCB import nCB, nCB_PV 

def make_reference():
    LC = nCB("5cb_traj/run.tpr","5cb_traj/traj_comp.xtc", 5, trjconv=False)

    director_ref = np.zeros((len(LC),3))
    OP_ref       = np.zeros((len(LC),))
    n            = np.array([1,0,0])
    dv0_ref      = np.zeros((len(LC),LC.Nresidues_,3))
    for t in range(len(LC)):
        director_ref[t] = LC.director(t, MOI=False)
        OP_ref[t]       = LC.v0(t,n)
        dv0_ref[t]      = LC.dv0_dr(t, n)

    np.save("5cb_traj/director_ref.npy", director_ref)
    np.save("5cb_traj/OP_ref.npy", OP_ref)
    np.save("5cb_traj/dv0_ref.npy", dv0_ref)

def test_director():
    LC = nCB("5cb_traj/run.tpr","5cb_traj/traj_comp.xtc", 5, trjconv=False)

    director = np.zeros((len(LC),3))
    OP       = np.zeros((len(LC),))
    n            = np.array([1,0,0])
    dv0      = np.zeros((len(LC),LC.Nresidues_,3))
    for t in range(len(LC)):
        director[t] = LC.director(t, MOI=False)
        OP[t]       = LC.v0(t,n)
        dv0[t]      = LC.dv0_dr(t, n)

    director_ref = np.load("5cb_traj/director_ref.npy")
    OP_ref       = np.load("5cb_traj/OP_ref.npy")
    dv0_ref      = np.load("5cb_traj/dv0_ref.npy")

    err1         = np.max(np.abs(director_ref - director)/np.abs(director_ref))
    err2         = np.max(np.abs(OP_ref - OP)/np.abs(OP_ref))
    err3         = np.max(np.abs(dv0_ref - dv0)/np.abs(dv0_ref))

    assert(err1 < 1e-10)
    assert(err2 < 1e-10)
    assert(err3 < 1e-10)

if __name__ == "__main__":
    make_reference()
