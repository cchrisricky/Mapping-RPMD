#Define class for discrete variable representation calculation
#It's a child-class of the map_rpmd parent class

import numpy as np
import utils
import map_rpmd
import sys
import time
from scipy.linalg import expm

class dvr( map_rpmd.map_rpmd ):
    
    ################################################################################
    
    def __init__( self, nstates=1, nnuc=1, nbds=1, beta=1.0, mass=1.0, potype=None, potparams=None, mapR=None, mapP=None, nucR=None, nucP=None, Ngrid=None, Rmin=None, Rmax=None ):

        super().__init__( 'NRPMD', nstates, nnuc, nbds, beta, mass, potype, potparams, mapR, mapP, nucR, nucP )

        #initialize the grid
        self.Ngrid = Ngrid
        self.Rmin = Rmin
        self.Rmax = Rmax
        self.delR = ( Rmax - Rmin ) / ( Ngrid - 1 ) #grid point spacing

        self.Hdvr = np.zeros( [self.nstates * self.Ngrid, self.nstates * self.Ngrid] ) #define the dvr Hamiltonian matrix

    ################################################################################

    def kernel(self):

        self.get_Hdvr()

    ################################################################################

    def calc_Hdvr(self):
        """the routine to get the grided DVR Hamiltonian matrix\n
        output: xpos - the 1D np array of grid points\n
                Hdvr - the nstates*Ngrid-times-nstates*Ngrid grid point represented Hamiltonian matrix\n
                Q - the density between grid points
        """

        Rvec = np.zeros( self.Ngrid )
        #Calculate DVR Hamiltonian
        for state1 in range(self.nstates):
            for i in range(self.Ngrid):

                indx1 = i + state1*self.Ngrid

                Ri = self.Rmin + i * self.delR
                Rvec[i] = Ri
                Hel = self.getHel(Ri)

                for state2 in range(state1, self.nstates):
                    for j in range(i, self.Ngrid):

                        indx2 = j + state2 * self.Ngrid

                        fctr = (-1)**(i-j) / (2.0 * self.mass * self.delR**2)

                        if indx1 == indx2: #diagonal terms of dvr matrix
                            self.Hdvr[indx1,indx2] = fctr * np.pi**2/3.0 #kinetic term
                            self.Hdvr[indx1,indx2] += Hel[state1,state1] #state-dependent potential term
                        else:
                            if state1 == state2: #Kinetic contribution from nuclei, so off-diagonal in nulei, but still diagonal for electronic state
                                self.Hdvr[indx1,indx2] = fctr * 2.0 / (i-j) ** 2
                            elif i == j:
                                self.Hdvr[indx1,indx2] = Hel[state1, state2]

        self.Hdvr = self.Hdvr + np.transpose( np.triu( self.Hdvr, 1 ) )
        densmat = expm( -self.beta * self.Hdvr )
        self.Q = np.trace( densmat ) * self.delR

    ################################################################################

    def get_pop( self ):
        
        pass

    ################################################################################

    def get_cRR( self ):

        pass