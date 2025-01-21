import numpy as np
import sys
import os
sys.path.append('..')
import mash_rpmd
import utils
import linecache

nbds = 2

mash_ = mash_rpmd.mash_rpmd(nstates=2, nnuc=1, nbds=nbds, beta=16, mass=np.array([2000]), potype='tully_2ac', 
                            potparams=[np.array([0.1]), np.array([0.28]), np.array([0.015]), np.array([0.06]), np.array([0.05])], 
                            nucR=np.array([[-2.0],[-1.8]]), nucP=np.array([[15.0],[18.0]]),
                            mapSx=np.array([-3.45072805e-01, -3.45072805e-01]), mapSy=np.array([8.98779576e-01, 8.98779576e-01]), mapSz=np.array([-0.2704, -0.2704]),
                            spinmap_bool=True, centroid_bool=True)

mash_.potential.calc_Hel(mash_.nucR)
mash_.potential.calc_Hel_deriv(mash_.nucR)

#mash_.init_map_spin(0) #initialize electrons state to state 0

#dSy, dSz = mash_.get_timederiv_mapSyz()
#print(dSy)
#print(dSz)
#print(mash_.get_2nd_timederiv_mapSx(dSy, dSz))
#print(mash_.potential.calc_NAC())

mash_.run_dynamics(Nsteps=4000, Nprint=100, delt=0.01, intype="vv", init_time=0.0, small_dt_ratio=1)
#print(mash_.potential.Hel)
#print(mash_.potential.d_Hel)

#print(mash_.potential.get_bopes_derivs())

#print(mash_.get_timederiv_mapSx())

#print(mash_.get_timederiv_nucP())

#print(mash_.potential.get_bopes()[:,1])
