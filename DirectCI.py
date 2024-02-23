import numpy as np
from opt_einsum import contract

"""
Class for running DirectCI calculations.

If only the number of alpha electrons is specified, it is assumed to be the total number of electrons
and they will be distributed evenly over the two spins. If both num_alpha_electrons and num_beta_electrons
are specified, the total number of electrons will be num_alpha_electrons + num_beta_electrons.
"""

class DirectCI:
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
        else:
            self.I_beta = self.get_slater_dets(num_spatial_orbitals, num_beta_electrons)
            self.num_beta_dets = self.I_beta.shape[0]
            self.W_beta = self.get_address_weights(num_spatial_orbitals, self.num_beta_electrons)
            self.equal_spins = False

        # In the pink book this should be used instead of h, but then it doesn't work
        # Hochstuhl uses h as is, so maybe there is some different convention on
        # how the two-body integrals are calculated?
        self.k = h #-0.5*np.einsum('prrq->pq', g) 


    # Generate a list of all Slater determinants. Order agrees with the address function.
    @staticmethod
    def get_slater_dets(num_orbitals, num_electrons):
        dets = []
        if num_electrons == 0:
            dets.append(np.zeros(num_orbitals))
        elif num_electrons == num_orbitals:
            dets.append(np.ones(num_orbitals))
        else:
            for d in DirectCI.get_slater_dets(num_orbitals-1, num_electrons):
                dets.append(np.concatenate((d, [0])))
            for d in DirectCI.get_slater_dets(num_orbitals-1, num_electrons-1):
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
    
    def get_k_alpha(self):
        return self._get_k(self.num_alpha_dets, self.I_alpha, self.W_alpha)
    
    def get_k_beta(self):
        return self._get_k(self.num_beta_dets, self.I_beta, self.W_beta)
    
    def _get_k(self, num_dets, spin_strings, W):
        k = np.zeros((num_dets, )*2)
        for J in spin_strings:
            j = self.address(J, W)
            occ = np.flatnonzero(J)
            for p in range(self.num_spatial_orbitals): # Is this possible to do better?
                for q in occ:
                    xi, I = self.single_exc(p,q,J)
                    if xi != 0:
                        i = self.address(I, W)
                        k[i,j] += xi*self.k[p,q]
        return k
    
    def _get_G(self, num_dets, spin_strings, W):
        G = np.zeros((num_dets, )*2)
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
                                G[i,j] += 0.5*xi*self.g[p,q,r,s]

        return G
    
    def get_G_alpha(self):
        return self._get_G(self.num_alpha_dets, self.I_alpha, self.W_alpha)
    
    def get_G_beta(self):
        return self._get_G(self.num_beta_dets, self.I_beta, self.W_beta)
            
    def get_sigma_alpha(self, C):
        return contract('aj, jb -> ab', self.get_k_alpha(), C)
    
    def get_sigma_beta(self, C):
        return contract('bj, aj -> ab', self.get_k_beta(), C)
    
    def get_sigma_alpha2(self, C):
        return contract('aj,jb -> ab', self.get_G_alpha(), C)
    
    def get_sigma_beta2(self, C):
        return contract('bj,aj -> ab', self.get_G_beta(), C)
    
    def get_Dpq(self, C):
        Dpq = np.zeros((self.num_alpha_dets, self.num_beta_dets) + (self.num_spatial_orbitals,)*2, dtype=np.complex128)
        for J in self.I_alpha:
            j = self.address(J, self.W_alpha)
            occ = np.flatnonzero(J)
            for p in range(self.num_spatial_orbitals): # Is this possible to do better?
                for q in occ:
                    xi, I = self.single_exc(p,q,J)
                    if xi != 0:
                        i = self.address(I, self.W_alpha)
                        Dpq[i,:,p,q] += xi*C[j,:]

        return Dpq
    
    def get_Gpq(self):
        Gpq = np.zeros((self.num_beta_dets,)*2+(self.num_spatial_orbitals,)*2, dtype=np.complex128)
        for J in self.I_beta:
            j = self.address(J, self.W_beta)
            occ = np.flatnonzero(J)
            for r in range(self.num_spatial_orbitals): # Is this possible to do better?
                for s in occ:
                    xi, I = self.single_exc(r,s,J)
                    if xi != 0:
                        i = self.address(I, self.W_beta)
                        Gpq[i,j,:,:] += xi*self.g[:,r,:,s]

        return Gpq

    def get_sigma_alphabeta(self, C):
        return contract('ajpq, bjpq -> ab', self.get_Dpq(C), self.get_Gpq())
    

    def get_sigma(self, C):
        if self.equal_spins:
            sigma = 2*(self.get_sigma_alpha(C) + self.get_sigma_alpha2(C)) + self.get_sigma_alphabeta(C)
        else:
            sigma = (self.get_sigma_alpha(C) + self.get_sigma_beta(C)
                     + self.get_sigma_alpha2(C) + self.get_sigma_beta2(C)
                     + self.get_sigma_alphabeta(C))
        return sigma
    
    def get_1p_RDM_spin(self, spin_strings, W, C):
        D = np.zeros((self.num_spatial_orbitals,)*2)
        for J in spin_strings:
            j = self.address(J, W)
            occ = np.flatnonzero(J)
            for p in range(self.num_spatial_orbitals): # Is this possible to do better?
                for q in occ:
                    xi, I = self.single_exc(p,q,J)
                    if xi != 0:
                        i = self.address(I, W)
                        D[p,q] += np.sum(xi*C[:,i]*C[:,j])
        return D
    
    def get_1p_RDM(self, C):
        if self.equal_spins:
            D = 2*self.get_D_spin(self.I_alpha, self.W_alpha, C)
        else:
            D = (self.get_D_spin(self.I_alpha, self.W_alpha, C) + 
                self.get_D_spin(self.I_beta, self.W_beta, C))
        return D
    
    def get_2p_RDM_equal()
    