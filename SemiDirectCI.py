import numpy as np
from opt_einsum import contract

"""
Class for running DirectCI calculations.

If only the number of alpha electrons is specified, it is assumed to be the total number of electrons
and they will be distributed evenly over the two spins. If both num_alpha_electrons and num_beta_electrons
are specified, the total number of electrons will be num_alpha_electrons + num_beta_electrons.
"""

class SemiDirectCI:
    def __init__(self, h, g, num_spatial_orbitals: int, num_alpha_electrons: int, num_beta_electrons: int = None):
        self.h = h
        self.g = g
        self.num_spatial_orbitals = num_spatial_orbitals

        if num_beta_electrons == None:
            if num_alpha_electrons%2 == 0:
                self.num_alpha_electrons = int(num_alpha_electrons/2)
                self.num_beta_electrons = self.num_alpha_electrons
            else:
                raise TypeError('An uneven number of electrons requires specifying' +
                                ' both num_alpha_electrons and num_beta_electrons.')
        else:
            self.num_alpha_electrons = num_alpha_electrons
            self.num_beta_electrons = num_beta_electrons
 

        # alpha determinants
        self.I_alpha = self.get_slater_dets(num_spatial_orbitals, num_alpha_electrons)
        self.num_alpha_dets = self.I_alpha.shape[0]
        self.W_alpha = self.get_address_weights(num_spatial_orbitals, num_alpha_electrons)
        
        if self.num_beta_electrons == self.num_alpha_electrons:
            self.I_beta = self.I_alpha
            self.num_beta_dets = self.num_alpha_dets
            self.W_beta = self.W_alpha
            self.equal_spins = True
            self.E_pq_alpha = self._calculate_Epq(self.num_alpha_dets, self.I_alpha, self.W_alpha)
            self.E_pq_beta = self.E_pq_alpha
            self.E_pqrs_alpha = self._calculate_Epqrs(self.num_alpha_dets, self.I_alpha, self.W_alpha)
            self.E_pqrs_beta = self.E_pqrs_alpha
        else:
            self.I_beta = self.get_slater_dets(num_spatial_orbitals, num_beta_electrons)
            self.num_beta_dets = self.I_beta.shape[0]
            self.W_beta = self.get_address_weights(num_spatial_orbitals, self.num_beta_electrons)
            self.equal_spins = False
            self.E_pq_alpha = self._calculate_Epq(self.num_alpha_dets, self.I_alpha, self.W_alpha)
            self.E_pq_beta = self._calculate_Epq(self.num_beta_dets, self.I_beta, self.W_beta)
            self.E_pqrs_alpha = self._calculate_Epqrs(self.num_alpha_dets, self.I_alpha, self.W_alpha)
            self.E_pqrs_beta = self._calculate_Epqrs(self.num_beta_dets, self.I_beta, self.W_beta)

        # In the pink book this should be used instead of h, but then it doesn't work
        # Hochstuhl uses h as is, so maybe there is some different convention on
        # how the two-body integrals are calculated?
        self.k = h #-0.5*np.einsum('prrq->pq', g) 

    def update_electron_integrals(self, h, g):
        self.h = h
        self.k = h
        self.g = g

    # Generate a list of all Slater determinants. Order agrees with the address function.
    @staticmethod
    def get_slater_dets(num_orbitals, num_electrons):
        dets = []
        if num_electrons == 0:
            dets.append(np.zeros(num_orbitals))
        elif num_electrons == num_orbitals:
            dets.append(np.ones(num_orbitals))
        else:
            for d in SemiDirectCI.get_slater_dets(num_orbitals-1, num_electrons):
                dets.append(np.concatenate((d, [0])))
            for d in SemiDirectCI.get_slater_dets(num_orbitals-1, num_electrons-1):
                dets.append(np.concatenate((d, [1])))

        return np.array(dets)
    
    @staticmethod
    def get_address_weights(num_orbitals, num_electrons):
        # Calculate node weights, see Fig. 19 and equation 3.70 in Hochstuhl
        W = np.zeros((num_orbitals+1, num_electrons+1))
        W[:,0] = 1

        for m in range(1,num_orbitals+1):
            for k in range(1,num_electrons+1):
                W[m,k] = W[m-1,k] + W[m-1, k-1]

        return W

    # Use node weights to calculate address, see equation 3.73 in Hochstuhl
    def address(self, n, W):
        res = 0
        for m in range(self.num_spatial_orbitals):
            res += n[m]*W[m, int(np.sum(n[:m+1]))]
        return int(res)
    

    # Define excitation operators
    def single_exc(self, p,q,n):
        n_pq = np.copy(n)
        n_pq[q] = 0

        if n_pq[p]==1:
            return 0, 0

        n_pq[p] = 1
        gamma = (-1)**(np.sum(n[:q])+np.sum(n_pq[:p]))

        return gamma, n_pq

    def double_exc(self, p, q, s, r, n):
        if s==r or p == q:
            return 0, 0

        n_pqsr = np.copy(n)
        n_pqsr[r] = 0
        gamma = np.sum(n_pqsr[:r])
        n_pqsr[s] = 0
        gamma += np.sum(n_pqsr[:s])

        if n_pqsr[q]==1 or n_pqsr[p]==1:
            return 0, 0

        n_pqsr[q] = 1
        gamma += np.sum(n_pqsr[:q])
        n_pqsr[p] = 1
        gamma += np.sum(n_pqsr[:p])

        return (-1)**gamma, n_pqsr
    
    def _calculate_Epq(self, num_dets, spin_strings, W):
        E_pq = np.zeros((num_dets,)*2+(self.num_spatial_orbitals,)*2)
        for J in spin_strings:
            j = self.address(J, W)
            occ = np.flatnonzero(J)
            for p in range(self.num_spatial_orbitals): # Is this possible to do better?
                for q in occ:
                    xi, I = self.single_exc(p,q,J)
                    if xi != 0:
                        i = self.address(I, W)
                        E_pq[i,j,p,q] += xi
        return E_pq
    
    def _calculate_Epqrs(self, num_dets, spin_strings, W):
        E_pqrs = np.zeros((num_dets,)*2+(self.num_spatial_orbitals,)*4)

        for J in spin_strings:
            j = self.address(J, W)
            occ = np.flatnonzero(J)
            for p in range(self.num_spatial_orbitals):
                for r in occ:
                    for q in range(self.num_spatial_orbitals):
                        for s in occ:
                            xi, I = self.double_exc(p,q,s,r,J)
                            if xi != 0:
                                i = self.address(I, W)
                                E_pqrs[i,j,p,q,r,s] += xi
        return E_pqrs
    
            
    def get_sigma_alpha(self, C):
        return contract('ijpq, pq, jk -> ik', self.E_pq_alpha, self.k, C)
    
    def get_sigma_beta(self, C):
        return contract('ijpq, pq, kj -> ki', self.E_pq_beta, self.k, C)
    
    def get_sigma_alpha2(self, C):
        return contract('ijpqrs, pqrs, jk -> ik', self.E_pqrs_alpha, self.g, C)
    
    def get_sigma_beta2(self, C):
        return contract('ijpqrs, pqrs, kj -> ki', self.E_pqrs_beta, self.g, C)
    

    def get_sigma_alphabeta(self, C):
        return contract('pqrs, ijpq, klrs, jl -> ik', self.g, self.E_pq_alpha, self.E_pq_beta, C)
    

    def get_sigma(self, C):
        if self.equal_spins:
            sigma = 2*(self.get_sigma_alpha(C) + self.get_sigma_alpha2(C)) + self.get_sigma_alphabeta(C)
        else:
            sigma = (self.get_sigma_alpha(C) + self.get_sigma_beta(C)
                     + self.get_sigma_alpha2(C) + self.get_sigma_beta2(C)
                     + self.get_sigma_alphabeta(C))
        return sigma
    
    def get_1p_RDM(self, C):
        if self.equal_spins:
            D = 2*contract('ijpq, ia, ja -> pq', self.E_pq_alpha, C.conj(), C)
        else:
            D = (contract('ijpq, ia, ja -> pq', self.E_pq_alpha, C.conj(), C) + 
                 contract('ijpq, ai, aj -> pq', self.E_pq_beta, C.conj(), C))
        return D


    def get_2p_RDM(self, C):
        if self.equal_spins:
            return (2*contract('ijpqrs, ia, ja -> pqrs', self.E_pqrs_alpha, C.conj(), C) + # Same spin term
                    2*contract('ik, ijpq, klrs, jl -> pqrs', C.conj(), self.E_pq_alpha, self.E_pq_beta, C)) # Mixed spin term, is it needed when the spins are equal?
        else:
            return (contract('ijpqrs, ia, ja', self.E_pqrs_alpha, C.conj(), C) + # Both spin alpha
                    contract('ijpqrs, ai, aj', self.E_pqrs_beta, C.conj(), C) + # Both spin beta
                    2*contract('ik, ijpq, klrs, jl -> pqrs', C.conj(), self.E_pq_alpha, self.E_pq_beta, C)) # Mixed spin term, is it needed when the spins are equal?
        
    def get_RDMs(self, C):
        return self.get_1p_RDM(C), self.get_2p_RDM(C)
    
    def calculate_energy(self, C):
        D, d = self.get_RDMs(C)
        return self.calculate_energy_from_RDMs(D, d)
    
    def calculate_energy_from_RDMs(self, D, d):
        return contract('pq, pq', D, self.h) + 0.5 * contract('pqrs, pqrs', self.g, d)
    