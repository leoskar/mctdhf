import numpy as np
from opt_einsum import contract
from jax import jit
from functools import partial



class SemiDirectCI:
    """Class for running SemiDirectCI calculations.

    The products sigma = HC are evaluated without directly, however the 
    excitation operators are stored as tensors.

    If only the number of alpha electrons is specified, it is assumed to be the
    total number of electrons and they will be distributed evenly over the two 
    spins. If both num_alpha_electrons and num_beta_electrons are specified, the 
    total number of electrons will be num_alpha_electrons + num_beta_electrons.
    """

    def __init__(self, num_spatial_orbitals: int, num_alpha_electrons: int, num_beta_electrons: int = None):
        if num_beta_electrons == None:
            if num_alpha_electrons%2 == 0:
                num_alpha_electrons = int(num_alpha_electrons/2)
                num_beta_electrons = num_alpha_electrons
            else:
                raise TypeError('An uneven number of electrons requires specifying' +
                                ' both num_alpha_electrons and num_beta_electrons.')
 

        # alpha determinants
        I_alpha = self.get_slater_dets(num_spatial_orbitals, num_alpha_electrons)
        num_alpha_dets = I_alpha.shape[0]
        W_alpha = self.get_address_weights(num_spatial_orbitals, num_alpha_electrons)
        
        if num_beta_electrons == num_alpha_electrons:
            self.E_pq_alpha = self._calculate_Epq(num_spatial_orbitals, num_alpha_dets, I_alpha, W_alpha)
            self.E_pq_beta = self.E_pq_alpha
            self.E_pqrs_alpha = self._calculate_Epqrs(num_spatial_orbitals, num_alpha_dets, I_alpha, W_alpha)
            self.E_pqrs_beta = self.E_pqrs_alpha
        else:
            I_beta = self.get_slater_dets(num_spatial_orbitals, num_beta_electrons)
            num_beta_dets = I_beta.shape[0]
            W_beta = self.get_address_weights(num_spatial_orbitals, num_beta_electrons)
            self.E_pq_alpha = self._calculate_Epq(num_spatial_orbitals, num_alpha_dets, I_alpha, W_alpha)
            self.E_pq_beta = self._calculate_Epq(num_spatial_orbitals, num_beta_dets, I_beta, W_beta)
            self.E_pqrs_alpha = self._calculate_Epqrs(num_spatial_orbitals, num_alpha_dets, I_alpha, W_alpha)
            self.E_pqrs_beta = self._calculate_Epqrs(num_spatial_orbitals, num_beta_dets, I_beta, W_beta)

        # In the pink book this should be used instead of h, but then it doesn't work
        # Hochstuhl uses h as is, so maybe there is some different convention on
        # how the two-body integrals are calculated?
        # k = h #-0.5*np.einsum('prrq->pq', g) 


    
    @staticmethod
    def get_slater_dets(num_orbitals, num_electrons):
        """ Generate a list of all Slater determinants. Order agrees with the address function."""

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

        return np.array(dets, dtype=bool) 
    
    @staticmethod
    def get_address_weights(num_orbitals, num_electrons):
        """Calculate node weights, see Fig. 19 and equation 3.70 in Hochstuhl et al."""

        W = np.zeros((num_orbitals+1, num_electrons+1))
        W[:,0] = 1

        for m in range(1,num_orbitals+1):
            for k in range(1,num_electrons+1):
                W[m,k] = W[m-1,k] + W[m-1, k-1]

        return W

    def address(self, n, num_spatial_orbitals, W):
        """Use node weights to calculate address, see equation 3.73 in Hochstuhl."""

        res = 0
        for m in range(num_spatial_orbitals):
            res += n[m]*W[m, int(np.sum(n[:m+1]))]
        return int(res)
    

    def single_exc(self, p, q, n):
        """Evaluate a single particle excitation a^+_pa_q."""

        n_pq = np.copy(n)
        n_pq[q] = 0

        if n_pq[p]==1:
            return 0, 0

        n_pq[p] = 1
        gamma = (-1)**(np.sum(n[:q])+np.sum(n_pq[:p]))

        return gamma, n_pq

    def double_exc(self, p, q, s, r, n):
        """Evaluate a two particle excitation a^+_pa^+_qa_sa_r."""

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
    
    def _calculate_Epq(self, num_spatial_orbitals, num_dets, spin_strings, W):
        """Calculate matrix of single particle excitations."""

        E_pq = np.zeros((num_dets,)*2+(num_spatial_orbitals,)*2)
        for J in spin_strings:
            j = self.address(J, num_spatial_orbitals, W)
            occ = np.flatnonzero(J)
            for p in range(num_spatial_orbitals): # Is this possible to do better?
                for q in occ:
                    xi, I = self.single_exc(p,q,J)
                    if xi != 0:
                        i = self.address(I, num_spatial_orbitals, W)
                        E_pq[i,j,p,q] += xi
        return E_pq
    
    def _calculate_Epqrs(self, num_spatial_orbitals, num_dets, spin_strings, W):
        """Calculate matrix of two particle excitations."""

        E_pqrs = np.zeros((num_dets,)*2+(num_spatial_orbitals,)*4)

        for J in spin_strings:
            j = self.address(J, num_spatial_orbitals, W)
            occ = np.flatnonzero(J)
            for p in range(num_spatial_orbitals):
                for r in occ:
                    for q in range(num_spatial_orbitals):
                        for s in occ:
                            xi, I = self.double_exc(p,q,s,r,J)
                            if xi != 0:
                                i = self.address(I, num_spatial_orbitals, W)
                                E_pqrs[i,j,p,q,r,s] += xi
        return E_pqrs
    
    @partial(jit, static_argnums=0)
    def get_sigma_alpha(self, C, h):
        """Get single particle sigma with alpha spin."""
        return contract('ijpq, pq, cjk -> cik', self.E_pq_alpha, h, C, backend='jax')
    
    @partial(jit, static_argnums=0)
    def get_sigma_beta(self, C, h):
        """Get single particle sigma with beta spin."""
        return contract('ijpq, pq, ckj -> cki', self.E_pq_beta, h, C, backend='jax')
    
    @partial(jit, static_argnums=0)
    def get_sigma_alpha2(self, C, g):
        """Get two particle sigma, both with alpha spin."""
        return 0.5*contract('ijpqrs, pqrs, cjk -> cik', self.E_pqrs_alpha, g, C, backend='jax') 
    
    @partial(jit, static_argnums=0)
    def get_sigma_beta2(self, C, g):
        """Get two particle sigma, both with beta spin."""
        return 0.5*contract('ijpqrs, pqrs, ckj -> cki', self.E_pqrs_beta, g, C, backend='jax') 
    
    @partial(jit, static_argnums=0)
    def get_sigma_alphabeta(self, C, g):
        """Get two particle sigma, with mixed alpha and beta spin."""
        return 0.5*contract('pqrs, ijpr, klqs, cjl -> cik', g, self.E_pq_alpha, self.E_pq_beta, C, backend='jax') 
    
    @partial(jit, static_argnums=0)
    def get_sigma(self, C, h, g):  
        """Total value of sigma = HC."""

        return (self.get_sigma_alpha(C, h) + self.get_sigma_beta(C, h)
                + self.get_sigma_alpha2(C, g) + self.get_sigma_beta2(C, g)
                + 2*self.get_sigma_alphabeta(C, g))
    

    @partial(jit, static_argnums=0)
    def get_1p_RDM(self, C):
        """Calculate single particle reduced density matrix."""

        return (contract('ijpq, cia, cja -> cpq', self.E_pq_alpha, C.conj(), C, backend='jax') + 
                contract('ijpq, cai, caj -> cpq', self.E_pq_beta, C.conj(), C, backend='jax'))


    @partial(jit, static_argnums=0)
    def get_2p_RDM(self, C):
        """Calculate two particle reduced density matrix."""

        return (contract('ijpqrs, cia, cja -> cpqrs', self.E_pqrs_alpha, C.conj(), C, backend='jax') + # Both spin alpha
                contract('ijpqrs, cai, caj -> cpqrs', self.E_pqrs_beta, C.conj(), C, backend='jax') + # Both spin beta
                contract('cik, ijpr, klqs, cjl -> cpqrs', C.conj(), self.E_pq_alpha, self.E_pq_beta, C, backend='jax') + # Mixed spin term
                contract('cki, ijpr, klqs, clj -> cpqrs', C.conj(), self.E_pq_beta, self.E_pq_alpha, C, backend='jax')) # Mixed spin term


    @partial(jit, static_argnums=0)    
    def get_RDMs(self, C):
        """Get one and two particle reduced density matrix."""
        return self.get_1p_RDM(C), self.get_2p_RDM(C)

    @partial(jit, static_argnums=0)  
    def calculate_energy(self, C, h, g):
        D, d = self.get_RDMs(C)
        return self.calculate_energy_from_RDMs(D, d, h, g)
    
    @partial(jit, static_argnums=0)
    def calculate_energy_from_RDMs(self, D, d, h, g):
        return contract('cpq, pq -> c', D, h) + 0.5 * contract('pqrs, cpqrs -> c', g, d, backend='jax')
    