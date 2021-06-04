import numpy as np
import sys

sys.path.insert(0,"../Liquid_crystal")
sys.path.insert(0,"../ProbeVolume")
from analysis_code.Liquid_crystal.nCB import nCB_PV
from analysis_code.ProbeVolume.ProbeVolume_Box import ProbeVolume_Box

def make_reference():
   min_ = np.array([10,20,0])
   max_ = np.array([20,40,30])
   pv = ProbeVolume_Box("5cb_traj/run.tpr","5cb_traj/traj_comp.xtc",min_,max_)
   LC_pv = nCB_PV("5cb_traj/run.tpr","5cb_traj/traj_comp.xtc", 5, pv,trjconv=False) 
   Ntilde_ref    = np.zeros((len(LC_pv),1))
   indicator_ref = np.zeros((len(LC_pv),LC_pv.Nresidues_,1)) 
   hx_ref        = np.zeros((len(LC_pv),LC_pv.Nresidues_,3))

   for i in range(len(LC_pv)):
       head_pos , _  = LC_pv.head_tail_pos(i)
       indicator, hx = pv.calculate_Indicator(head_pos,i)
       indicator_ref[i] = indicator
       hx_ref[i] = hx
       Ntilde_ref[i] = pv.get_Ntilde()
    
   np.save("INDUS_test/indicator_ref.npy", indicator_ref)
   np.save("INDUS_test/hx_ref.npy",hx_ref)
   np.save("INDUS_test/Ntilde_ref.npy", Ntilde_ref)

def test_INDUS():
   min_ = np.array([10,20,0])
   max_ = np.array([20,40,30])
   pv = ProbeVolume_Box("5cb_traj/run.tpr","5cb_traj/traj_comp.xtc",min_,max_)
   LC_pv = nCB_PV("5cb_traj/run.tpr","5cb_traj/traj_comp.xtc", 5, pv,trjconv=False) 
   Ntilde    = np.zeros((len(LC_pv),1))
   indicator = np.zeros((len(LC_pv),LC_pv.Nresidues_,1)) 
   hx_     = np.zeros((len(LC_pv),LC_pv.Nresidues_,3))

   for i in range(len(LC_pv)):
       head_pos , _  = LC_pv.head_tail_pos(i)
       ind, hx = pv.calculate_Indicator(head_pos,i)
       indicator[i] = ind
       hx_[i] = hx
       Ntilde[i] = pv.get_Ntilde()
    
   ans_Ntilde       = np.load("INDUS_test/Ntilde_ref.npy")
   ans_indicator    = np.load("INDUS_test/indicator_ref.npy")
   ans_hx           = np.load("INDUS_test/hx_ref.npy")

   err1 = np.max(np.abs(ans_Ntilde - Ntilde))
   err2 = np.max(np.abs(ans_indicator - indicator))
   err3 = np.max(np.abs(ans_hx - hx_))

   assert(err1 < 1e-10)
   assert(err2 < 1e-10)
   assert(err3 < 1e-10)


if __name__ == "__main__":
    #make_reference()
    test_INDUS()