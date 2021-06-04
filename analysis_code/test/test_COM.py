from analysis_code.Liquid_crystal.nCB import nCB
import numpy as np

def make_ref():
    LC = nCB("5cb_traj/run.tpr","5cb_traj/traj_comp.xtc",4, trjconv=False)
    COM_mat = np.zeros((len(LC),LC.Nresidues_,3))

    for t in range(len(LC)):
        COM_mat[t] = LC.COM(t)
    
    np.save("unittests_data/COM_ref.npy",COM_mat)

def test_COM():
    LC = nCB("5cb_traj/run.tpr","5cb_traj/traj_comp.xtc",4, trjconv=False)
    COM_mat = np.zeros((len(LC),LC.Nresidues_,3))

    for t in range(len(LC)):
        COM_mat[t] = LC.COM(t)
    
    COM_ref = np.load("unittests_data/COM_ref.npy")
    err     = np.max(np.abs(COM_mat - COM_ref))

    assert err < 1e-10

if __name__ == "__main__":
    #make_ref()
    test_COM()
