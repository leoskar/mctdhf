import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from jax import jit
from functools import partial
from scipy.special import comb
from opt_einsum import contract

from .SemiDirectCI_jax import SemiDirectCI


class MCTDHF:
    """Class for running MCTDHF calculations.

    ...

    Attributes
    ----------
    h : ndarray (num_spatial_orbitals,)*2
        Time-independent part of the single particle electron integrals
    ht : function of t, returning ndarray (num_spatial_orbitals,)*2
        Time-dependent part of the single particle electron integrals
    g : ndarray (num_spatial_orbitals,)*4
        Two-particle electron integrals.
    num_mctdhf_orbitals : int
        The number of MCTHF orbitals.
    num_spatial_orbitals: int
        The number of spatial orbitals, i.e. the size of the full basis set.
    num_alpha_electrons : int
        The number of electrons with alpha spin.
    num_beta_electrons : int
        The number of electrons with beta spin.
    imag_time : bool
        Whether to perform the time evolution in imaginary time, default is false.
    S: ndarray (num_spatial_orbitals,)*2
        Overlap matrix for the basis function. The basis is assumed to be 
        ortonormal if no overlap matrix is provided.
    
    """

    def __init__(self, h, ht, g, num_mctdhf_orbitals, num_spatial_orbitals, 
                 num_alpha_electrons, num_beta_electrons, 
                 num_states = 1, imag_time=False, S = None):
        
        self.h = h
        self.ht = ht
        self.g = g
        self.num_mctdhf_orbitals = num_mctdhf_orbitals
        self.num_spatial_orbitals = num_spatial_orbitals
        self.num_alpha_electrons = num_alpha_electrons
        self.num_beta_electrons = num_beta_electrons
        self.num_states = num_states


        self.num_alpha_dets = int(comb(num_mctdhf_orbitals, num_alpha_electrons))
        self.num_beta_dets = int(comb(num_mctdhf_orbitals, num_beta_electrons))

        self.num_slater_dets = self.num_alpha_dets*self.num_beta_dets

        self.dCI = SemiDirectCI(self.num_mctdhf_orbitals, 
                                num_alpha_electrons, 
                                num_beta_electrons)
        
        self.imag_time = imag_time
        self.time = 1 if imag_time else 1j

        self.S_inv = np.eye(num_spatial_orbitals) if S is None else np.linalg.inv(S)
        
    
    @partial(jit, static_argnums=0)
    def transform_h(self, h, b, bc):
        """Transform the single particle electron integrals to he MCTDHF basis."""

        h_1 = contract('jm, ij -> im', b, h, backend='jax')
        h_2 = contract('in, im -> nm', bc, h_1, backend='jax')
        h_3 = contract('in, nm -> im', b, h_2, backend='jax')
        
        return h_1,h_2,h_3
    

    @partial(jit, static_argnums=0)
    def transform_g(self, g, b, bc):
        """Transform the two particle electron integrals to he MCTDHF basis."""

        g_2 = contract('jq, ls, ijkl -> iqks', bc, b, g, backend='jax')
        g_3 = contract('kr, iqks -> iqrs', b, g_2, backend='jax')
        g_4 = contract('ip, iqrs -> pqrs', bc, g_3, backend='jax')
        g_5 = contract('ip, pqrs -> iqrs', b, g_4, backend='jax')
        
        return g_3,g_4,g_5
    

    @partial(jit, static_argnums=0)
    def get_E(self, t, b, C):
        """Calculate the energy."""

        bc = b.conj()

        h = self.h + self.ht(t)
        h_1 = contract('jm, ij -> im', b, h, backend='jax')
        h_2 = contract('in, im -> nm', bc, h_1, backend='jax')

        g_2 = contract('jq, ls, ijkl -> iqks', bc, b, self.g, backend='jax')
        g_3 = contract('kr, iqks -> iqrs', b, g_2, backend='jax')
        g_4 = contract('ip, iqrs -> pqrs', bc, g_3, backend='jax')

        return self.dCI.calculate_energy(C, h_2, g_4)
    

    @partial(jit, static_argnums=0)
    def _Cb_to_y(self, C, b):
        return jnp.concatenate([C.flatten(), b.flatten()])


    @partial(jit, static_argnums=0)
    def _y_to_Cb(self, y):
        C, b = jnp.split(y, [self.num_slater_dets*self.num_states])
        C = jnp.reshape(C, (self.num_states, self.num_alpha_dets, self.num_beta_dets))
        b = jnp.reshape(b, (self.num_spatial_orbitals, self.num_mctdhf_orbitals))

        return C, b
    
    @partial(jit, static_argnums=0)
    def ortonorm_b(self, b):
        """Function to ortonormalise the transformation coefficents b
        
        The function uses the overlap algorithm from Lehtovaara et al., 
        DOI: 10.1016/j.jcp.2006.06.006
        """

        b = jnp.divide(b, jnp.sqrt(contract('ij, ij -> j', b.conj(), b, backend='jax')))
        ## Overlap renormalisation
        # u, _, vh = jnp.linalg.svd(b, full_matrices=False)
        # return u@vh

        ## Gram-Schmidt renormalisation
        for i in range(b.shape[1]):
            orto_adjustment = 0
            for j in range(0, i):
                orto_adjustment += jnp.dot(b[:, i], b[:, j])*b[:, j]

            b = b.at[:,i].add(-orto_adjustment)

        return b

    @partial(jit, static_argnums=0)
    def ortonorm_C(self, C):
        """Function to ortonormalise C
        
        The function uses the overlap algorithm from Lehtovaara et al., 
        DOI: 10.1016/j.jcp.2006.06.006
        """

        C = jnp.divide(C, jnp.sqrt(contract('cij, cij -> c', C.conj(), C, backend='jax')).reshape(self.num_states,1,1))

        ## Overlap renormalisation
        # u, _, vh = jnp.linalg.svd(C.reshape(self.num_states, -1), full_matrices=False)
        # return jnp.reshape(u@vh, (self.num_states, self.num_alpha_dets, self.num_beta_dets))
    
        # Gram-Schmidt renormalisation
        for i in range(C.shape[0]):
            orto_adjustment = 0
            for j in range(0, i):
                orto_adjustment += contract('ij,ij', C[i], C[j])*C[j]

            C = C.at[i].add(-orto_adjustment)

        return C


    @partial(jit, static_argnums=0)
    def __call__(self, t, y):
        """RHS of the MCTDHF problem."""

        C, b = self._y_to_Cb(y)
        bc = b.conj()

        h = self.h + self.ht(t)

        h_1, h_2, h_3 = self.transform_h(h, b, bc)
        g_3, g_4, g_5 = self.transform_g(self.g, b, bc)

        D, d = self.dCI.get_RDMs(C)
        D_inv = jnp.linalg.pinv(D[0])
      
        Sh_1 = contract('kl, lm -> km', self.S_inv, h_1, backend='jax')
        Sg_3 = contract('kl, lqrs -> kqrs', self.S_inv, g_3, backend='jax')

        b_dot = -self.time*(Sh_1 - h_3 + contract('np, pqrs, iqrs -> in', D_inv, d[0], Sg_3-g_5, backend='jax'))

        sigma = self.dCI.get_sigma(C, h_2, g_4)
        C_dot = -self.time*sigma

        return self._Cb_to_y(C_dot, b_dot)
    


    def integrate(self, integrator, num_steps: int, t_init = 0, b_init = None, C_init = None, normalize = None):
        """Integrate the MCTDHF problem.

        Args:
            integrator: The integrator used to solve the problem. 
            num_steps (int): Number of integration steps
            t_init (int, optional): Initial time. Defaults to 0.
            b_init (optional): Initial transformation coefficients. Defaults to random.
            C_init (optional): Initial CI coefficients. Defaults to random.
            normalize (optional): Whether to normalize during the integration. Defaults to imag_time.

        Returns:
            ts, Es, Cs, bs: Lists of the values at each time step.
        """

        if b_init is None:
            # Generate random orthonormal vectors
            rng = np.random.default_rng()
            r = rng.random((self.num_spatial_orbitals, self.num_mctdhf_orbitals))
            u, _, vh = np.linalg.svd(r, full_matrices=False)
            b_init = jnp.array((u@vh), jnp.complex64)

        if C_init is None:
            # Random initial C
            rng = np.random.default_rng()
            r = rng.random((self.num_states, self.num_alpha_dets, self.num_beta_dets))
            u, _, vh = np.linalg.svd(r, full_matrices=False)
            C_init = jnp.array((u@vh), jnp.complex64)


        if normalize is None:
            normalize = self.imag_time

        y = self._Cb_to_y(C_init, b_init)

        E_init = self.get_E(0, b_init, C_init)
        Es = [E_init]
        Cs = [C_init]
        bs = [b_init]
        ts = [t_init]

        t = t_init
        for _ in range(num_steps):
            t, E, C, b = self._step(t, y, integrator, normalize)

            Es.append(E)
            Cs.append(C)
            bs.append(b)
            ts.append(t)

            y = self._Cb_to_y(C, b)

        return np.array(ts), np.array(Es), np.array(Cs), np.array(bs)
    

    @partial(jit, static_argnums=(0,3,4))
    def _step(self, t, y, integrator, normalize):
        t, y = integrator(t, y)
        C, b = self._y_to_Cb(y)

        if normalize:
            C = self.ortonorm_C(C)
            b = self.ortonorm_b(b)

        E = self.get_E(t, b, C)

        return t, E, C, b
    

    @partial(jit, static_argnums = 0)
    def calculate_overlap(self, b1, C1, b2, C2):
        """Calculate the wave function overlap between two sets of b and C"""

        I_alpha = SemiDirectCI.get_slater_dets(self.num_mctdhf_orbitals, self.num_alpha_electrons)

        s_a = contract('ai,ki,kj,bj->abij', I_alpha, b1, b2, I_alpha, backend='jax')
        s_a = s_a[jnp.nonzero(s_a, size=(self.num_alpha_dets*self.num_alpha_electrons)**2)]
        S_a = jnp.linalg.det(s_a.reshape((self.num_alpha_dets,)*2+(self.num_alpha_electrons,)*2))

        if self.num_beta_electrons == self.num_alpha_electrons:
            return contract('cik,cjl,ij,kl->c', C1.conj(), C2, S_a, S_a, backend='jax')
    
        I_beta = SemiDirectCI.get_slater_dets(self.num_mctdhf_orbitals, self.num_beta_electrons)

        s_b = contract('ai,ki,kj,bj->abij', I_beta, b1, b2, I_beta, backend='jax')
        s_b = s_b[jnp.nonzero(s_b, size=(self.num_beta_dets*self.num_beta_electrons)**2)]
        S_b = jnp.linalg.det(s_b.reshape((self.num_beta_dets,)*2+(self.num_beta_electrons,)*2))

        return contract('cik,cjl,ij,kl->c', C1.conj(), C2, S_a, S_b, backend='jax')
    

    def calculate_particle_densities(self, b, C, spf):
        """Calculate the particle density"""

        Ds = self.dCI.get_1p_RDM(C)
        return contract('ix,in,snm,jm,jx->sx',spf,b,Ds,b,spf)
    
    def plot_Cbe(self, ts, Cs, bs, Es):
        fig, axs = plt.subplots(1, 3, figsize=(16,6))
        axs[0].plot(ts, np.real(Cs.reshape(Cs.shape[0],-1)))
        axs[0].set_title('Cs')
        axs[1].plot(ts, np.real(bs.reshape(-1, self.num_spatial_orbitals, self.num_mctdhf_orbitals)[:,:,0]))
        axs[1].set_title('bs')
        axs[2].plot(ts, np.real(Es))
        #axs[2].semilogy()
        #axs[2].set_ylim([0,5])
        axs[2].set_title('E')
        
        plt.show()
        
    


class MCTDHF_sinc(MCTDHF):
    @partial(jit, static_argnums=0)
    def transform_g(self, g, b, bc):
        g_2 = contract('jq, js, ij -> iqs', bc, b, g, backend='jax')
        g_3 = contract('ir, iqs -> iqrs', b, g_2, backend='jax')
        g_4 = contract('ip, iqrs -> pqrs', bc, g_3, backend='jax')
        g_5 = contract('ip, pqrs -> iqrs', b, g_4, backend='jax')
        
        return g_3,g_4,g_5
    
    @partial(jit, static_argnums=0)
    def get_E(self, t, b, C):
        bc = b.conj()

        h = self.h + self.ht(t)
        h_1 = contract('jm, ij -> im', b, h, backend='jax')
        h_2 = contract('in, im -> nm', bc, h_1, backend='jax')

        g_2 = contract('jq, js, ij -> iqs', bc, b, self.g, backend='jax')
        g_3 = contract('ir, iqs -> iqrs', b, g_2, backend='jax')
        g_4 = contract('ip, iqrs -> pqrs', bc, g_3, backend='jax')

        return self.dCI.calculate_energy(C, h_2, g_4) 
    
