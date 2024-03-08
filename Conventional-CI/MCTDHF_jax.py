import numpy as np
import jax.numpy as jnp
from jax import jit
from scipy.special import comb

from functools import partial

from integrators import *

from CI_physicist import *

class MCTDHF:

    def __init__(self, h: jnp.ndarray, g: jnp.ndarray, num_mctdhf_orbitals: np.integer, 
                 num_spin_orbitals: np.integer, num_electrons: np.integer):
        self.h = h
        self.g = g
        self.num_mctdhf_orbitals = num_mctdhf_orbitals
        self.num_spin_orbitals = num_spin_orbitals
        self.num_electrons = num_electrons
        self.num_slater_dets = comb(num_mctdhf_orbitals, num_electrons, exact = True)

        self.mctdhf_slater_dets = np.array(CIHamiltonian.get_slater_dets(num_mctdhf_orbitals, num_electrons))

    def get_random_guess(self):
        # Generate random orthonormal vectors
        rng = np.random.default_rng()
        r = rng.random((self.num_spin_orbitals, self.num_mctdhf_orbitals))
        u, _, vh = np.linalg.svd(r, full_matrices=False)
        b_init = (u@vh).astype(np.cdouble)

        # Random initial C
        C_init = rng.random(self.num_slater_dets).astype(np.cdouble)
        # Normalize
        C_init = np.divide(C_init, np.sqrt(C_init.T.conj()@C_init))

        return self.Cb_to_y(C_init, b_init)
    
    #@partial(jit, static_argnums=0)
    def get_RDMs(self, C):
        C_conj = C.conj()

        D = jnp.zeros((self.num_mctdhf_orbitals,)*2 , dtype=jnp.complex64)
        d = jnp.zeros((self.num_mctdhf_orbitals,)*4, dtype=jnp.complex64)

        for n, det_n in enumerate(self.mctdhf_slater_dets):
            for m, det_m in enumerate(self.mctdhf_slater_dets):
                num_differences = np.sum(np.abs(det_n-det_m))
                match(num_differences):
                    case 0:
                        for p, n_p in enumerate(det_n):
                            D.at[p,p].add(n_p*C_conj[m]*C[n])
                            for r, n_r in enumerate(det_n):
                                d.at[p,r,p,r].add(n_p*n_r*C_conj[m]*C[n])
                                d.at[p,r,r,p].add(-n_p*n_r*C_conj[m]*C[n]) # Note the sign                 
                    case 2:
                        p = np.flatnonzero(np.asarray((det_m-det_n)==1))[0]
                        q = np.flatnonzero(np.asarray((det_n-det_m)==1))[0]
                        gamma = (-1)**(np.sum(det_n[:q])+np.sum(det_m[:p]))
                        D.at[p,q].add(gamma*C_conj[m]*C[n])

                        for r, n_r in enumerate(det_n):
                            val = gamma*n_r*C_conj[m]*C[n]
                            d.at[p,r,q,r].add(val)
                            d.at[r,p,q,r].add(-val) # Note the sign
                            d.at[p,r,r,q].add(-val) # Note the sign
                            d.at[r,p,r,q].add(val)

                    case 4:
                        p,q = np.flatnonzero(np.asarray((det_m-det_n)==1))
                        r,s = np.flatnonzero(np.asarray((det_n-det_m)==1))
                        
                        gamma = np.sum(det_n[:r])
                        gamma += np.sum(det_n[:s])-1 # -1 since r<s 
                        gamma += np.sum(det_n[:q])-int(s<q)-int(r<q)
                        gamma += np.sum(det_n[:p])-int(s<p)-int(r<p) # No additional term since p<q

                        val = C_conj[m]*C[n]*(-1)**gamma
                        d.at[p,q,r,s].add(val)
                        d.at[q,p,r,s].add(-val) # Note the sign
                        d.at[p,q,s,r].add(-val) # Note the sign
                        d.at[q,p,s,r].add(val)

        return D, d


    # Helper functions to put the problem on a form handled by standard SciPy ODE solvers 
    @partial(jit, static_argnums=0)
    def Cb_to_y(self, C, b):
        return jnp.concatenate([C.flatten(), b.flatten()])

    @partial(jit, static_argnums=0)
    def y_to_Cb(self, y):

        C, b = jnp.split(y, [self.num_slater_dets])
        b = b.reshape((self.num_spin_orbitals, self.num_mctdhf_orbitals))

        return C, b

    @partial(jit, static_argnums=0)
    def imag_time_int(self, t, y):
        C, b = self.y_to_Cb(y)
        bc = b.conj()

        h_1 = jnp.einsum('jm, ij -> im', b, self.h)
        h_2 = jnp.einsum('in, im -> nm', bc, h_1)
        h_3 = jnp.einsum('in, nm -> im', b, h_2)

        g_2 = jnp.einsum('jq, ls, ijkl -> iqks', bc, b, self.g)
        g_3 = jnp.einsum('kr, iqks -> iqrs', b, g_2)
        g_4 = jnp.einsum('ip, iqrs -> pqrs', bc, g_3)
        g_5 = jnp.einsum('ip, pqrs -> iqrs', b, g_4)

        
        D, d = self.get_RDMs(C)
        D_inv = jnp.linalg.pinv(D)

        b_dot = -(h_1 - h_3 + jnp.einsum('np, pqrs, iqrs -> in', D_inv, d, g_3-g_5))


        H = SlaterCondonHamiltonian(self.num_mctdhf_orbitals, self.num_electrons, h_2, g_4).get_hamiltonian()

        ## From Beck paper, replacing HC with (H-IE)C should keep the wave function normalized
        E = C.conj().T@H@C/(C.conj().T@C)
        C_dot = - (H-self.I*E)@C 

        return self.Cb_to_y(C_dot, b_dot)
    
    
    def run_imag_time_prop(self, t_init, t_final, dt):

        integrate = get_runge_kutta_4_solver(self.imag_time_int, dt)
        y = self.get_random_guess()
        t = t_init

        print(self.imag_time_int(t,y))

        num_steps = int((t_final-t_init)/dt)

        for i in range(t_init, t_final, num_steps):
            t, y = integrate(t, y)