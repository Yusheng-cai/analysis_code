import numpy as np
import sys

sys.path.insert(0,"../Liquid_crystal")
sys.path.insert(0,"../ProbeVolume")
from analysis_code.Liquid_crystal.nCB import nCB_PV, nCB
from analysis_code.ProbeVolume.ProbeVolume_Box import ProbeVolume_Box
from analysis_code.ProbeVolume.ProbeVolume_cylinder import ProbeVolume_cylinder
from analysis_code.ProbeVolume.ProbeVolume_tiltedcylinder import ProbeVolume_tiltedcylinder

def make_reference_cylinder():
   # for Cylinder
   base = np.array([0,0,0])
   h     = 30
   radius = 20

   pv_cylinder = ProbeVolume_cylinder("5cb_traj/run.tpr", "5cb_traj/traj_comp.xtc",base, h, radius, pbc=True, dir_='z') 

   LC            = nCB("5cb_traj/run.tpr","5cb_traj/traj_comp.xtc", 5,trjconv=False) 
   natoms        = LC.natoms
   nresidues     = LC.Nresidues_
   Ntilde_ref    = np.zeros((len(LC),1))
   indicator_ref = np.zeros((len(LC),natoms*nresidues,1)) 
   hx_ref        = np.zeros((len(LC),natoms*nresidues,3))

   for i in range(len(LC)):
        pos              = LC.pos(i)
        indicator, hx    = pv_cylinder.calculate_Indicator(pos,i)
        indicator_ref[i] = indicator
        hx_ref[i]        = hx
        Ntilde_ref[i]    = pv_cylinder.get_Ntilde()
     
   np.save("INDUS_test/indicator_cyl_ref.npy", indicator_ref)
   np.save("INDUS_test/hx_cyl_ref.npy",hx_ref)
   np.save("INDUS_test/Ntilde_cyl_ref.npy", Ntilde_ref)

def make_reference_box():
   # for Box
   min_ = np.array([10,20,0])
   max_ = np.array([20,40,30])
 
   pv    = ProbeVolume_Box("5cb_traj/run.tpr","5cb_traj/traj_comp.xtc",min_,max_)
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
    
   np.save("INDUS_test/indicator_box_ref.npy", indicator_ref)
   np.save("INDUS_test/hx_box_ref.npy",hx_ref)
   np.save("INDUS_test/Ntilde_box_ref.npy", Ntilde_ref)

def test_INDUS_box():
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
    
   ans_Ntilde       = np.load("INDUS_test/Ntilde_box_ref.npy")
   ans_indicator    = np.load("INDUS_test/indicator_box_ref.npy")
   ans_hx           = np.load("INDUS_test/hx_box_ref.npy")

   err1 = np.max(np.abs(ans_Ntilde - Ntilde))
   err2 = np.max(np.abs(ans_indicator - indicator))
   err3 = np.max(np.abs(ans_hx - hx_))
   print("err 1 is {}".format(err1))
   print("err 2 is {}".format(err2))
   print("err 3 is {}".format(err3))

   assert(err1 < 1e-10)
   assert(err2 < 1e-10)
   assert(err3 < 1e-10)


if __name__ == "__main__":
    print("hello")
    #make_reference_cylinder()
    #make_reference_box()
    test_INDUS_box()