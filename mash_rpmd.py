import numpy as np
import utils
import sys
from scipy.linalg import expm
from scipy.special import erf
import math

#Define class for Mash-rpmd
#It's a child-class of the rpmd parent class

import map_rpmd
import integrator

class mash_rpmd( map_rpmd.map_rpmd ):

    def __init__( self, nstates=1, nnuc=1, nbds=1, beta=1.0, mass=1.0, potype=None, potparams=None, mapR=None, mapP=None, mapSx=None, mapSy=None, mapSz=None, nucR=None, nucP=None, spinmap_bool=False, functional_param=1.):

        super().__init__( 'RP-MASH', nstates, nnuc, nbds, beta, mass, potype, potparams, mapR, mapP, nucR, nucP )

        if (spinmap_bool == True):

            if (nstates != 2):
                print('ERROR: The spin mapping variable is only for 2-state systems')
                exit()

            self.mapSx = mapSx
            self.mapSy = mapSy
            self.mapSz = mapSz

            self.spin_map_error_check()
        
        self.spin_map = spinmap_bool #Boolean that decide if we use spin mapping variables
        self.functional_param = functional_param

    #####################################################################

    def run_dynamics( self, Nsteps=None, Nprint=100, delt=None, intype=None, init_time=0.0, small_dt_ratio=1 ):

        #Top-level routine to run dynamics specific to MASH

        #Number of decimal places when printing current time
        #modf splits the floating number into integer and decimal components
        #converting to string, taking the length, and subtracting 2 (for the 0.) gives us the length of just the decimal component
        tDigits = len( str( math.modf(Nprint * delt)[0] ) ) - 2

        #Error checks REMINDER TO REWITE THIS GUY FOR MASH
        #self.dynam_error_check( Nsteps, delt, intype )

        #Initialize the integrator
        self.integ = integrator.integrator( self, delt, intype, small_dt_ratio )

        #Automatically initialize nuclear momentum from MB distribution if none have been specified
        if( self.nucP is None ):
            print('Automatically initializing nuclear momentum to Maxwell-Boltzmann distribution at beta_p = ', self.beta_p*self.nbds ,' / ',self.nbds)
            self.get_nucP_MB()

        print()
        print( '#########################################################' )
        print( 'Running', self.methodname, 'Dynamics for', Nsteps, 'Steps' )
        print( '#########################################################' )
        print()

        #Open output files
        self.file_output = open( 'output.dat', 'w' )
        self.file_nucR   = open( 'nucR.dat','w' )
        self.file_nucP   = open( 'nucP.dat', 'w' )
        self.file_mapSx   = open( 'mapSx.dat','w' )
        self.file_mapSy   = open( 'mapSy.dat','w' )
        self.file_mapSz   = open( 'mapSz.dat','w' )
        #self.file_mapR   = open( 'mapR.dat','w' )
        #self.file_mapP   = open( 'mapP.dat', 'w' )
        #self.file_Q      = open( 'Q.dat', 'w')
        #self.file_phi    = open( 'phi.dat', 'w')
        #self.file_semi   = open( 'mvsq.dat', 'w')

        current_time = init_time
        step = 0
        for step in range( Nsteps ):

            #Print data starting with initial time
            if( np.mod( step, Nprint ) == 0 ):
                print('Writing data at step', step, 'and time', format(current_time, '.'+str(tDigits)+'f'), 'for', self.methodname, 'Dynamics calculation')
                self.print_data( current_time )
                sys.stdout.flush()

            #Integrate EOM by one time-step
            self.integ.onestep( self, step )

            #Increase current time
            current_time = init_time + delt * (step+1)

        #Print data at final step regardless of Nprint
        print('Writing data at step ', step+1, 'and time', format(current_time, '.'+str(tDigits)+'f'), 'for', self.methodname, 'Dynamics calculation')
        self.print_data( current_time )
        sys.stdout.flush()

        #Close output files
        self.file_output.close()
        self.file_nucR.close()
        self.file_nucP.close()
        self.file_mapSx.close()
        self.file_mapSy.close()
        self.file_mapSz.close()
        #self.file_mapR.close()
        #self.file_mapP.close()
        #self.file_Q.close()
        #self.file_phi.close()
        #self.file_semi.close()

        print()
        print( '#########################################################' )
        print( 'END', self.methodname, 'Dynamics' )
        print( '#########################################################' )
        print()
        
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
        if (self.spin_map == False):
            dmapR = self.get_timederiv_mapR()
            dmapP = self.get_timederiv_mapP()

            return dnucR, dnucP, dmapR, dmapP
        
        else:
            dmapSx = self.get_timederiv_mapSx()
            dmapSy, dmapSz = self.get_timederiv_mapSyz()

            return dnucR, dnucP, dmapSx, dmapSy, dmapSz

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

        if (self.spin_map==False):
            #Calculate contribution from MMST term
            #XXX could maybe make this faster getting rid of double index in einsum
            d_nucP += -0.5 * np.einsum( 'in,ijnm,im -> ij', self.mapR, self.potential.d_Hel, self.mapR )
            d_nucP += -0.5 * np.einsum( 'in,ijnm,im -> ij', self.mapP, self.potential.d_Hel, self.mapP )

            #add the state-average potential
            if (self.potype != 'harm_lin_cpl_symmetrized' or self.potype != 'harm_lin_cpl_sym_2'):
                d_nucP +=  0.5 * np.einsum( 'ijnn -> ij', self.potential.d_Hel )
        else:
            #The MASH nuclear force, note that Hel here are adiabatic surfaces
            d_Vz = self.potential.get_bopes_derivs()[:,:,1]
            #Calculate delta function force as functional limit or with adaptive timestep
            if (self.functional_param != None):
                Vz = self.potential.get_bopes()[:,1]
                NAC = self.potential.calc_NAC()
                d_nucP += -d_Vz * erf(self.mapSz / self.functional_param) + 4 * Vz * NAC * self.mapSx * np.exp(-(self.mapSz / self.functional_param)**2) / (self.functional_param * np.sqrt(np.pi))
            else:
                d_nucP += -d_Vz * np.sign(self.mapSz)

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

    def get_timederiv_mapSx( self ):

        Vz = self.potential.get_bopes()[:,1]
        NAC = self.potential.calc_NAC()

        d_mapSx = 2 * np.sum(NAC * self.nucP / self.mass, axis = 1) * self.mapSz - 2 * Vz * self.mapSy

        return d_mapSx

   #####################################################################

    def get_timederiv_mapSyz(self):

        Vz = self.potential.get_bopes()[:,1]
        NAC = self.potential.calc_NAC()

        d_mapSy = 2 * np.sum(NAC * self.nucP / self.mass, axis = 1) * self.mapSx
        d_mapSz = -2 * Vz * self.mapSx

        return d_mapSy, d_mapSz

   #####################################################################
    
    def get_2nd_timederiv_mapSx( self, d_mapSy, d_mapSz):

        Vz = self.potential.get_bopes()[:,1]
        NAC = self.potential.calc_NAC()

        print(np.sum(NAC * self.nucP / self.mass, axis = 1))

        d2_mapSx = 2 * np.sum(NAC * self.nucP / self.mass, axis = 1) * d_mapSz - 2 * Vz * d_mapSy

        return d2_mapSx

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

        if(self.spin_map==False):
            #MMST Term
            engpe += 0.5 * np.sum( np.einsum( 'in,inm,im -> i', self.mapR, self.potential.Hel, self.mapR ) )
            engpe += 0.5 * np.sum( np.einsum( 'in,inm,im -> i', self.mapP, self.potential.Hel, self.mapP ) )

            if (self.potype != 'harm_lin_cpl_symmetrized' or self.potype != 'harm_lin_cpl_sym_2'):
                engpe += -0.5 * np.sum( np.einsum( 'inn -> i', self.potential.Hel ) )

        else:
            engpe += np.sum(self.potential.Hel * np.sign(self.mapSz))
        return engpe

    #####################################################################

    def init_spin_variables(self):
        pass

    def spin_map_error_check(self):

        if( self.mapSx is not None and self.mapSx.shape != (self.nbds,) ):
            print('ERROR: Size of spin mapping variable Sx doesnt match bead number')
            exit()

        if( self.mapSy is not None and self.mapSy.shape != (self.nbds,) ):
            print('ERROR: Size of spin mapping variable Sy doesnt match bead number')
            exit()

        if( self.mapSz is not None and self.mapSz.shape != (self.nbds,) ):
            print('ERROR: Size of spin mapping variable Sz doesnt match bead number')
            exit()

    #####################################################################

    def get_sampling_eng(self):
        return None

    #####################################################################

    def print_data( self, step ):
        print(self.mapSz)
        return None


