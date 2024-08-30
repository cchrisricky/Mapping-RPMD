import numpy as np
import utils
import sys
from scipy.linalg import expm

#Define class for Mash-rpmd
#It's a child-class of the rpmd parent class

import map_rpmd
import integrator

class mash_rpmd( map_rpmd.map_rpmd ):

    def __init__( self, nstates=1, nnuc=1, nbds=1, beta=1.0, mass=1.0, potype=None, potparams=None, mapR=None, mapP=None, nucR=None, nucP=None, spin_map=False):

        super().__init__( 'RP-MASH', nstates, nnuc, nbds, beta, mass, potype, potparams, mapR, mapP, nucR, nucP )

        if (spin_map == True):

            if (nstates != 2):
                print('ERROR: The spin mapping variable is only for 2-state systems')
                exit()

            else:
                self.spinX = np.zeros([nbds])
                self.spinY = np.zeros([nbds])
                self.spinZ = np.zeros([nbds])

        self.spin_map = spin_map

    #####################################################################

    def get_timederivs( self ):
        #subroutine to calculate the time-derivatives of all position/momenta

        #update electronic Hamiltonian matrix
        self.potential.calc_Hel( self.nucR )

        #Calculate time-derivative of nuclear position
        dnucR = self.get_timederiv_nucR()

        #Calculate time-derivative of nuclear momenta
        dnucP = self.get_timederiv_nucP()

        #Calculate time-derivative of mapping position and momentum
        dmapR = self.get_timederiv_mapR()
        dmapP = self.get_timederiv_mapP()

        return dnucR, dnucP, dmapR, dmapP


    #####################################################################

    def get_timederiv_nucR( self ):
        pass

    #####################################################################

    def get_timederiv_nucP( self ):
        pass

    #####################################################################

    def get_timederiv_mapR( self ):
        
        if (self.spin_map == True):

            self.d_spinX = 2 * np.sum(self.NAC * self.nucP) * self.spinZ / self.m - 2 * self.potential * self.spinY
            self.d_spinY = 2 * self.potential * self.spinX
            self.d_spinZ = - 2 * np.sum(self.NAC * self.nucP) * self.spinX / self.m

    #####################################################################

    def get_timederiv_mapP( self ):
        pass

    #####################################################################

    def get_2nd_timederiv_mapR( self, d_mapP ):
        #Subroutine to calculate the second time-derivative of just the mapping positions for each bead
        #This assumes that the nuclei are fixed - used in vv style integrators

        d2_mapR = np.einsum( 'inm,im->in', self.potential.Hel, d_mapP )

        return d2_mapR

    #####################################################################

    def get_PE( self ):
        #Subroutine to calculate potential energy associated with mapping variables and nuclear position

        #Internal ring-polymer modes, 0 if there is only one bead (i.e., LSC-IVR)
        if self.nbds > 1:
            engpe = self.potential.calc_rp_harm_eng( self.nucR, self.beta_p, self.mass )
        else:
            engpe = 0

        #State independent term
        engpe += self.potential.calc_state_indep_eng( self.nucR )

        #Update electronic Hamiltonian matrix
        self.potential.calc_Hel(self.nucR)

        #MMST Term
        engpe += 0.5 * np.sum( np.einsum( 'in,inm,im -> i', self.mapR, self.potential.Hel, self.mapR ) )
        engpe += 0.5 * np.sum( np.einsum( 'in,inm,im -> i', self.mapP, self.potential.Hel, self.mapP ) )

        if (self.potype != 'harm_lin_cpl_symmetrized' or self.potype != 'harm_lin_cpl_sym_2'):
            engpe += -0.5 * np.sum( np.einsum( 'inn -> i', self.potential.Hel ) )

        return engpe

    #####################################################################

    def init_spin_variables(self):
        pass

        
