import numpy as np
import sys
import os
sys.path.append('..')
import mash_rpmd
import utils
import linecache

rng = np.random.default_rng()

nbds = 4
nnuc = 1
nstates = 2
mass=np.array([2000])

intype = 'vv'
Ntraj = 25
delt   = 0.1
Nsteps = int(100 * 42 / delt) #about 1fs=41.34au
Nprint = int(10/delt)

fmt_str = '%20.8e'

nucR_init = -10.0
nucP_init = 10.0
ga = 0.25
nucR = rng.normal(nucR_init, np.sqrt( 1/ (ga*2) ), (nbds,nnuc))
#nucP = rng.normal(nucP_init, np.sqrt( ga/2 ), (nbds,nnuc))
nucP = nucP_init * np.ones((nbds,nnuc))

mash_ = mash_rpmd.mash_rpmd(nstates=nstates, nnuc=nnuc, nbds=nbds, beta=2*mass[0]/nucP_init**2, mass=mass, potype='tully_1ac', 
                            potparams=[np.array([0.01]), np.array([1.6]), np.array([0.005]), np.array([1.0])], 
                            nucR=nucR, nucP=nucP, spinmap_bool=True, centroid_bool=True)

# Initial electronic state variables using spin angle variables
mash_.init_map_spin(0)

#Run trajectory
mash_.run_dynamics(Nsteps, Nprint, delt, intype, init_time=0.0, small_dt_ratio=1)


#mash_.potential.calc_Hel(mash_.nucR)
#mash_.potential.calc_Hel_deriv(mash_.nucR)

#mash_.init_map_spin(0) #initialize electrons state to state 0

#dSy, dSz = mash_.get_timederiv_mapSyz()
#print(dSy)
#print(dSz)
#print(mash_.get_2nd_timederiv_mapSx(dSy, dSz))
#print(mash_.potential.calc_NAC())

#mash_.run_dynamics(Nsteps=4000, Nprint=100, delt=0.01, intype="vv", init_time=0.0, small_dt_ratio=1)
#print(mash_.potential.Hel)
#print(mash_.potential.d_Hel)

#print(mash_.potential.get_bopes_derivs())

#print(mash_.get_timederiv_mapSx())

#print(mash_.get_timederiv_nucP())

#print(mash_.potential.get_bopes()[:,1])
