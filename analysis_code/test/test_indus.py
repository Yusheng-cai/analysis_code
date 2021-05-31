import sys
import numpy as np
sys.path.insert(0,"../Liquid_crystal/")
from INDUS_util import *

def test_INDUS():
    pos = np.load("INDUS_test/random_pos_for_indus.npy")
    min_= np.array([-1.2,-1.2,-1.2])
    max_= -min_

    kk=h(pos,min_,max_)
    ans = np.load("INDUS_test/indus_answer.npy")
    assert(np.max(np.abs(ans-kk)) < 1e-10)

if __name__ == "__main__":
   test_INDUS()