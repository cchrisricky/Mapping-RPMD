#Define class for NRPMD in an optical cavity
#It's a child-class of the map_rpmd parent class

import numpy as np
import utils
import map_rpmd
import sys
from scipy.linalg import expm

class NRPMD_InC(map_rpmd.map_rpmd):

    ############################################################
    def __init__( self, nstates=1, nnuc=1, nbds=1, beta=1.0, mass=1.0, potype=None, potparams=None, mapR=None, mapP=None, nucR=None, nucP=None ):

        super().__init__( 'NRPMD_InC', nstates, nnuc, nbds,beta, mass, potype, potparams, mapR, mapP, nucR, nucP )
        #initilaize Theta
        self.theta = None
        self.W     = None

    ############################################################
        
    def get_timederivs( self ):
        #Subroutine to calculate the time-derivatives of all position/momenta

        #Update electronic Hamiltonian matrix
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
    
    def get_theta( self ):
        #Subroutine to calculate Theta (minus the Gaussian term and prefactor) following Huo's 2019 paper https://dx.doi.org/10.1063/1.5096276

        #Electronic trace over the product of C matrices
        halfeye = 0.5 * np.eye(self.nstates)
        prod = np.outer(( self.mapR[0] + 1j*self.mapP[0] ), ( self.mapR[0] - 1j*self.mapP[0] )) - halfeye
        if self.nbds > 1:
            for i in range(1, self.nbds):
                prod = prod @ ( np.outer(( self.mapR[i] + 1j*self.mapP[i] ), ( self.mapR[i] - 1j*self.mapP[i] )) - halfeye )

        self.theta = np.real( np.trace( prod ) )

    #####################################################################

    def get_W( self ):
        #Subroutine to calculate W following Richardson's 2013 paper https://dx.doi.org/10.1063/1.4816124

        #Update electronic Hamiltonian matrix
        self.potential.calc_Hel( self.nucR )

        #M matrices
        Ms = np.zeros( [self.nbds, self.nstates, self.nstates] )
        for i in range( self.nbds ): Ms[i] = expm(-self.beta_p / 2 * self.potential.Hel[i])

        #Loop over each bead to calculate and multiply W
        self.W = 1.0
        for i in range( self.nbds ):
            self.W *= ( self.mapP[i - 1] @ Ms[i] @ self.mapR[i]) * ( self.mapR[i] @ Ms[i] @ self.mapP[i] )

    #####################################################################

    def get_timederiv_nucR( self ):
        #Subroutine to calculate the time-derivative of the nuclear positions for each bead

        return self.nucP / self.mass

    #####################################################################

    def get_timederiv_nucP( self, intRP_bool=True ):
        #Subroutine to calculate the time-derivative of the nuclear momenta for each bead

        #Force associated with harmonic springs between beads and the state-independent portion of the potential
        #This is dealt with in the parent class
        #If intRP_bool is False it does not calculate the contribution from the harmonic ring polymer springs

        if (self.nbds > 1):
            d_nucP = super().get_timederiv_nucP( intRP_bool )
        else:
            d_nucP = super().get_timederiv_nucP( intRP_bool = False )

        #Calculate nuclear derivative of electronic Hamiltonian matrix
        self.potential.calc_Hel_deriv( self.nucR )

        #Calculate contribution from MMST term
        #XXX could maybe make this faster getting rid of double index in einsum
        d_nucP += -0.5 * np.einsum( 'in,ijnm,im -> ij', self.mapR, self.potential.d_Hel, self.mapR )
        d_nucP += -0.5 * np.einsum( 'in,ijnm,im -> ij', self.mapP, self.potential.d_Hel, self.mapP )

        #add the state-average potential
        if (self.potype != 'harm_lin_cpl_symmetrized' or self.potype != 'harm_lin_cpl_sym_2'):
            d_nucP +=  0.5 * np.einsum( 'ijnn -> ij', self.potential.d_Hel )

        return d_nucP

    #####################################################################

    def get_timederiv_mapR( self ):
        #Subroutine to calculate the time-derivative of just the mapping position for each bead

        d_mapR =  np.einsum( 'inm,im->in', self.potential.Hel, self.mapP )

        return d_mapR

    #####################################################################

    def get_timederiv_mapP( self ):
        #Subroutine to calculate the time-derivative of just the mapping momentum for each bead

        d_mapP = -np.einsum( 'inm,im->in', self.potential.Hel, self.mapR )

        return d_mapP

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
    

            
