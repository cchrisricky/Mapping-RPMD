import numpy as np
import sys
import os
sys.path.append('/storage/home/hcoda1/8/zcao91/p-jkretchmer3-0/mapping_rpmd/')
import mash_rpmd
import utils

nbds = 4
nnuc = 1
nstates = 2
mass=np.array([2000])

intype = 'vv'
Ntraj = 5
Nsteps = 75 * 4200 #about 1fs=41.34au
Nprint = 1000
delt   = 0.01
fmt_str = '%20.8e'

rng = np.random.default_rng()

print('running trajectory 0')
#Initialize system in beta
#Using the thermal wigner distribution of the uncoupled bath to initialize the nuclear configuration
#omega_vec = np.sqrt(kvec/mass)
#tanh_vec = np.tanh( beta * omega_vec / 2 )
#nucR = np.zeros( [nbds, nnuc] )
#nucP = np.zeros( [nbds, nnuc] )

nucR_init = -15.0
nucP_init = 25.0
ga = 0.5
nucR = rng.normal(nucR_init, np.sqrt( 1/ (ga*2) ), (nbds,nnuc))
nucP = rng.normal(nucP_init, np.sqrt( ga/2 ), (nbds,nnuc))

mash_ = mash_rpmd.mash_rpmd(nstates=nstates, nnuc=nnuc, nbds=nbds, beta=6.4, mass=mass, potype='tully_2ac', 
                            potparams=[np.array([0.1]), np.array([0.28]), np.array([0.015]), np.array([0.06]), np.array([0.05])], 
                            nucR=nucR, nucP=nucP, spinmap_bool=True, functional_param=None)

# Initial electronic state variables using spin angle variables
mash_.init_map_spin(0)

#Run trajectory
mash_.run_dynamics(Nsteps, Nprint, delt, intype, init_time=0.0, small_dt_ratio=1)

#create average files
Sz_data = np.loadtxt('mapSz.dat')

pop_ave = np.copy(Sz_data[:,0:2])
pop_sum = np.sum(np.heaviside(Sz_data[:,1:], 0.5), axis = 1) / nbds
pop_ave[:,1] = pop_sum

time_pts = Sz_data.shape[0]

#open combined files
comb_out   = open('comb_output.dat', 'w')
comb_mapSz = open('comb_mapSz.dat', 'w')
comb_nucR  = open('comb_nucR.dat', 'w')

#write current run data to combined files
with open('output.dat') as f: comb_out.write( f.read() )
with open('mapSz.dat') as f: comb_mapSz.write(f.read())
with open('nucR.dat') as f: comb_nucR.write(f.read())
comb_out.write( '\n' )
comb_mapSz.write( '\n' )
comb_nucR.write( '\n' )
comb_out.flush()
comb_mapSz.flush()
comb_nucR.flush()

for traj in range(1, Ntraj):

    print('running trajectory', traj)
    #Initial nuclear configuration pulled from MC routine
    nucR = rng.normal(nucR_init, np.sqrt( 1/ (ga*2) ), (nbds,nnuc))
    nucP = rng.normal(nucP_init, np.sqrt( ga/2 ), (nbds,nnuc))    #initialize the mapping variables from the electron-only gamma sampling
    mash_ = mash_rpmd.mash_rpmd(nstates=nstates, nnuc=nnuc, nbds=nbds, beta=6.4, mass=mass, potype='tully_2ac', 
                                potparams=[np.array([0.1]), np.array([0.28]), np.array([0.015]), np.array([0.06]), np.array([0.05])], 
                                nucR=nucR, nucP=nucP, spinmap_bool=True, functional_param=None)

    # Initial electronic state variables using spin angle variables
    mash_.init_map_spin(0)

    #Run trajectory
    mash_.run_dynamics(Nsteps, Nprint, delt, intype, init_time=0.0, small_dt_ratio=1)

    #write current run data to combined files
    with open('output.dat') as f: comb_out.write( f.read() )
    with open('mapSz.dat') as f: comb_mapSz.write(f.read())
    with open('nucR.dat') as f: comb_nucR.write(f.read())
    comb_out.write( '\n' )
    comb_mapSz.write( '\n' )
    comb_nucR.write( '\n' )
    comb_out.flush()
    comb_mapSz.flush()
    comb_nucR.flush()

    #create average files
    Sz_data = np.loadtxt('mapSz.dat')
    pop_sum += np.sum(np.heaviside(Sz_data[:,1:], 0.5), axis = 1) / nbds

    pop_ave[:,1] = pop_sum / (traj+1)

    np.savetxt('pop_sum.dat', pop_sum, fmt = fmt_str )
    np.savetxt('pop_ave.dat', pop_ave, fmt = fmt_str )

comb_out.close()
comb_mapSz.close()
comb_nucR.close()
print('done')
