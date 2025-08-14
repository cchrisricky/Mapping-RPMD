#subroutine to calculate potential terms for nuclear only potentials

import numpy as np
import utils
from abc import ABC, abstractmethod

######### PARENT POTENTIAL CLASS ##########

class nuc_only_potential(ABC):

    #####################################################################

    @abstractmethod
    def __init__( self, potname, potparams, nnuc, nbds ):

        self.potname   = potname #string corresponding to the name of the potential
        self.potparams = potparams #array defining the necessary constants for the potential
        self.nnuc      = nnuc #number of nuclei
        self.nbds      = nbds #number of beads

    #####################################################################

    def calc_rp_harm_eng( self, nucR, beta_p, mass ):

        #Calculate potential energy associated with harmonic springs between beads

        engpe = 0.0
        for i in range(self.nbds):
            if( i == 0 ):
                #periodic boundary conditions for the first bead
                engpe += 0.5 * (1.0/beta_p)**2 * np.sum( mass * ( nucR[i] - nucR[self.nbds-1] )**2 )
            else:
                engpe += 0.5 * (1.0/beta_p)**2 * np.sum( mass * ( nucR[i] - nucR[i-1] )**2 )

        return engpe

    ###############################################################

    def calc_rp_harm_force( self, nucR, beta_p, mass ):

        #Calculate force associated with harmonic springs between beads

        Fharm = np.zeros( [self.nbds, self.nnuc] )

        for i in range(self.nbds):
            if( i == 0 ):
                #periodic boundary conditions for the first bead
                Fharm[i] = -mass * (1.0/beta_p)**2 * ( 2.0 * nucR[i] - nucR[self.nbds-1] - nucR[i+1] )
            elif( i == self.nbds-1 ):
                #periodic boundary conditions for the last bead
                Fharm[i] = -mass * (1.0/beta_p)**2 * ( 2.0 * nucR[i] - nucR[i-1] - nucR[0] )
            else:
                Fharm[i] = -mass * (1.0/beta_p)**2 * ( 2.0 * nucR[i] - nucR[i-1] - nucR[i+1] )

        return Fharm

    #####################################################################

    def calc_tot_PE( self, nucR, beta_p, mass ):

        #Calculate total potential energy including ring-polymer and external terms

        engpe  = self.calc_rp_harm_eng( nucR, beta_p, mass )

        engpe += self.calc_external_eng( nucR )

        return engpe

    #####################################################################

    def calc_nuc_KE( self, nucP, mass ):

        #Calculate kinetic energy associated with nuclear beads

        engke = 0.5 * np.sum( nucP**2 / mass )

        return engke

    ###############################################################

    def error_wrong_param_numb( self, num ):

        print("ERROR: List potparams does not have enough entries (",num,") for", self.potname,"potential")
        exit()

    ###############################################################

    @abstractmethod
    def calc_external_eng( self, nucR ):
        pass

    ###############################################################

    @abstractmethod
    def calc_external_force( self, nucR ):
        pass

    ###############################################################

    @abstractmethod
    def error_check( self ):
        pass

    ###############################################################


####### DEFINED POTENTIALS AS INSTANCES OF PARENT POTENTIAL CLASS #######

class nuc_only_harm(nuc_only_potential):

    #Harmonic potential with different force-constants and equilibrium position for different nuclei
    #V = \sum_i 0.5 * k_i * ( x_i - x0_i )**2

    ###############################################################

    def __init__( self, potparams, nnuc, nbds ):

        super().__init__( 'Nuclear Only Harmonic', potparams, nnuc, nbds )

        #Set appropriate potential parameters
        if( len(potparams) != 2 ):
            super().error_wrong_param_numb(2)

        self.kvec  = potparams[0] #force constants, size nnuc
        self.R0vec = potparams[1] #equilibrium positions, size nnuc

        #Input error check
        self.error_check()

    ###############################################################

    def calc_external_eng( self, nucR ):
        #Subroutine to calculate the energy associated with the external potential

        #harmonic term with different k for each nuclei
        eng = 0.5 * np.sum( self.kvec * ( nucR - self.R0vec )**2 )

        return eng

    ###############################################################

    def calc_external_force( self, nucR ):
        #Subroutine to calculate the force associated with the external potential

        #force from harmonic term with different k for each nuclei
        force = -self.kvec * ( nucR - self.R0vec )

        return force

    ###############################################################

    def error_check( self ):

        if( self.kvec.size != self.nnuc ):
            print("ERROR: 1st entry of list potparams should correspond to nnuc k-vector for", self.potname, "potential")
            exit()

        if( self.R0vec.size != self.nnuc ):
            print("ERROR: 2nd entry of list potparams should correspond to nnuc equilibrium distance vector for", self.potname, "potential")
            exit()

#########################################################################

class nuc_only_Marcus_withbath(nuc_only_potential):

    # State-independent Marcus parabola with bath
    # Followed the Lawrence, Mannouch, Richardson 2024 JPCL paper
    ###############################################################

    def __init__( self, potparams, nnuc, nbds ):

        super().__init__( 'Nuclear Only Marcus parabola with bath', potparams, nnuc, nbds )

        #Set appropriate potential parameters
        if( len(potparams) != 5 ):
            super().error_wrong_param_numb(5)

        self.kvec = potparams[0] # force constant of the Marcus parabolas, size 1
        self.epsil = potparams[1] # driving force of the reaction, size 1
        self.lbd = potparams[2] # lambda, Marcus reorganization energy, size 1
        self.delta = potparams[3] # constant diabatic coupling, size 1
        self.bathvec = potparams[4] #bath terms including all other vibronic modes, following Ohmic spectral density, has the form of [N_bath, mass, gamma (friction coefficient, gamma=Xi*pi/2, Xi: Kondo parameter), w_b (characteristic freq)]

        #Input error check
        self.error_check()

        #the bath potential follows the Ohmic spectral density
        if self.bathvec[0] != 0:
            self.omega_k = - self.bathvec[3] * np.log( (np.arange(self.bathvec[0])+0.5) / self.bathvec[0] )
            self.c_k     = np.sqrt( 2*self.bathvec[1]*self.bathvec[2]*self.bathvec[3] / np.pi / self.bathvec[0] ) * self.omega_k
            self.k_bath_vec = self.omega_k**2 * self.bathvec[1]
            
    ###############################################################

    def calc_external_eng( self, nucR ):
        #Subroutine to calculate the energy associated with the external potential

        #harmonic term with different k for each nuclei
        rxnR = np.copy( nucR[:,0] )
        eng = 0.5 * self.kvec * np.sum( ( rxnR + np.sqrt(self.lbd/self.kvec/2) )**2 + self.epsil/2 )

        if (self.nnuc == 1):
            return eng #only reaction coordinate. No contribution from bath modes.
        else:
            for ibd in range(self.nbds):
                for inuc in range(self.nnuc-1):
                    eng += 0.5 * self.k_bath_vec[inuc] * ( nucR[ibd, inuc+1] + self.c_k[inuc] / self.k_bath_vec[inuc] * rxnR[ibd] )**2

            return eng

    ###############################################################

    def calc_external_force( self, nucR ):
        #Subroutine to calculate the force associated with the external potential

        #force from harmonic term with different k for each nuclei
        force = np.zeros((self.nbds, self.nnuc))
        rxnR = np.copy( nucR[:,0])
        force[:,0] = -self.kvec * ( rxnR + np.sqrt(self.lbd/self.kvec/2) )
        if (self.nnuc == 1):
            return force #only reaction coordinate.

        else:
            for ibd in range(self.nbds):
                for inuc in range(self.nnuc-1):

                    force[ibd, 0] -= self.c_k[inuc] * (self.c_k[inuc]/self.k_bath_vec[inuc] * rxnR[ibd] + nucR[ibd,inuc+1])
                    force[ibd, inuc+1] = - self.k_bath_vec[inuc] * nucR[ibd, inuc+1] - self.c_k[inuc] * rxnR[ibd]

            return force

    ###############################################################

    def error_check( self ):

        if( not isinstance(self.kvec, float) ):
            print("ERROR: 1st entry of list potparams should be the force constant of the PESs")
            exit()

        if( not isinstance(self.epsil, float) ):
            print("ERROR: 2nd entry of list potparams should be the ET driving force")
            exit()

        if( self.nnuc != self.bathvec[0] + 1):
            print("ERROR: bath mode number does not equal total nuclear modes -1")
            exit()

#########################################################################

