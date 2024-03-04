import jax.numpy as jnp

from jax import jit
from functools import partial
from scipy.special import comb
from opt_einsum import contract

from integrators import *


@jit
def Cb_to_y(C, b):
    return jnp.concatenate([C.flatten(), b.flatten()])

@partial(jit, static_argnums=(1,2,3))
def y_to_Cb(y, num_mctdhf_orbitals, num_spin_orbitals, num_slater_dets):
    C, b = jnp.split(y, [num_slater_dets])
    b = jnp.reshape(b, (num_spin_orbitals, num_mctdhf_orbitals))

    return C, b

@jit
def gram_schmidt(b):
    b = jnp.divide(b, jnp.sqrt(contract('ij, ij -> j', b.conj(), b, backend='jax')))
    for i in range(b.shape[1]):
        orto_adjustment = 0
        for j in range(0, i):
            orto_adjustment += jnp.dot(b[:, i], b[:, j])*b[:, j]

        b.at[:,i].add(-orto_adjustment)

    return b

@jit
def overlap_ortnorm(b):
    b = jnp.divide(b, jnp.sqrt(contract('ij, ij -> j', b.conj(), b, backend='jax')))
    u, _, vh = jnp.linalg.svd(b, full_matrices=False)
    return u@vh

@jit
def normalize_C(C):
    return jnp.divide(C, jnp.sqrt(C.T.conj()@C))

@partial(jit, static_argnums=(0,5))
def get_E(dCI, h, g, b, C, C_shape):
    bc = b.conj()
    h_1 = contract('jm, ij -> im', b, h, backend='jax')
    h_2 = contract('in, im -> nm', bc, h_1, backend='jax')

    g_2 = contract('jq, ls, ijkl -> iqks', bc, b, g, backend='jax')
    g_3 = contract('kr, iqks -> iqrs', b, g_2, backend='jax')
    g_4 = contract('ip, iqrs -> pqrs', bc, g_3, backend='jax')

    return dCI.calculate_energy(jnp.reshape(C, C_shape), h_2, g_4)

class MCDTHF_imaginary_time:
    def __init__(self, h, g, num_mctdhf_orbitals, num_spatial_orbitals, num_alpha_electrons, num_beta_electrons):
        self.h = h
        self.g = g
        self.num_mctdhf_orbitals = num_mctdhf_orbitals
        self.num_spatial_orbitals = num_spatial_orbitals

        num_alpha_dets = int(comb(num_mctdhf_orbitals, num_alpha_electrons))
        num_beta_dets = int(comb(num_mctdhf_orbitals, num_beta_electrons))
        self.num_slater_dets = num_alpha_dets*num_beta_dets
        self.C_shape = (num_alpha_dets, num_beta_dets)

        self.dCI = SemiDirectCI(self.num_mctdhf_orbitals, 
                                num_alpha_electrons, 
                                num_beta_electrons)
        
    @partial(jit, static_argnums=0)
    def __call__(self, t, y):
        C, b = y_to_Cb(y, self.num_mctdhf_orbitals, self.num_spatial_orbitals, self.num_slater_dets)
        bc = b.conj()

        h_1 = contract('jm, ij -> im', b, self.h, backend='jax')
        h_2 = contract('in, im -> nm', bc, h_1, backend='jax')
        h_3 = contract('in, nm -> im', b, h_2, backend='jax')

        g_2 = contract('jq, ls, ijkl -> iqks', bc, b, self.g, backend='jax')
        g_3 = contract('kr, iqks -> iqrs', b, g_2, backend='jax')
        g_4 = contract('ip, iqrs -> pqrs', bc, g_3, backend='jax')
        g_5 = contract('ip, pqrs -> iqrs', b, g_4, backend='jax')

        D, d = self.dCI.get_RDMs(jnp.reshape(C, self.C_shape))
        #D_inv = np.linalg.pinv(D)
        # Regularize
        #eps = 1e-10
        #D_reg = D + eps*expm(-D/eps)
        D_inv = jnp.linalg.pinv(D)#_reg)
        #D_inv = pinvh(D_reg)
      
        b_dot = -(h_1 - h_3 + contract('np, pqrs, iqrs -> in', D_inv, d, g_3-g_5, backend='jax'))

        #sigma = self.dCI.get_sigma_equal_spins(C.reshape(self.C_shape)).flatten()
        sigma = self.dCI.get_sigma(jnp.reshape(C, self.C_shape), h_2, g_4).flatten()

        ## "Renormalization" from (127) in Beck et al. rewritten to direct CI
        #E = self.dCI.calculate_energy_from_RDMs(D,d)
        #C_dot = -(sigma-E*C) 

        C_dot = -sigma

        return Cb_to_y(C_dot, b_dot)
    
