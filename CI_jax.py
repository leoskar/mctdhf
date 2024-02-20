import numpy as np
import jax.numpy as jnp

class CIHamiltonian:
    def __init__(self, num_orbitals, num_electrons, h, g):
        self.num_orbitals = num_orbitals
        self.num_electrons = num_electrons
        self.h = h
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
            for d in CIHamiltonian.get_slater_dets(num_orbitals-1, num_electrons):
                dets.append(np.concatenate((d, [0])))
            for d in CIHamiltonian.get_slater_dets(num_orbitals-1, num_electrons-1):
                dets.append(np.concatenate((d, [1])))

        return dets

class SlaterCondonHamiltonian(CIHamiltonian):
    # Helper functions for slater condon rules
    def one_pair_diff(self, p, q, det_n, det_m):
        gamma = (-1)**(jnp.sum(det_n[:q])+jnp.sum(det_m[:p]))
        H = self.h[p,q]
        for r, n_r in enumerate(det_n):
            H += n_r*(self.g[p,r,q,r]-self.g[p,r,r,q])
        return gamma*H

    def two_pair_diff(self, p, q, s, r, det_n, det_m):
        n_prqs = jnp.copy(det_n)
        n_prqs[r] = 0
        gamma = jnp.sum(n_prqs[:r])
        n_prqs[s] = 0
        gamma += jnp.sum(n_prqs[:s])
        n_prqs[q] = 1
        gamma += jnp.sum(n_prqs[:q])
        n_prqs[p] = 1
        gamma += jnp.sum(n_prqs[:p])
        return (self.g[p,q,r,s]-self.g[p,q,s,r])*(-1)**gamma

    # Slater condon rules
    def slater_condon(self, det_n, det_m):
        num_differences = jnp.sum(np.abs(det_n-det_m))
        match(num_differences):
            case 0:
                H=jnp.cdouble(0j)
                for p, n_p in enumerate(det_n):
                    H += n_p*self.h[p,p]
                    for r, n_r in enumerate(det_n):
                        H += 0.5*n_p*n_r*(self.g[p,r,p,r]-self.g[p,r,r,p])
                return H
                
            case 2:
                p = np.flatnonzero(np.asarray((det_m-det_n)==1))[0]
                q = np.flatnonzero(np.asarray((det_n-det_m)==1))[0]
                return self.one_pair_diff(p,q,det_n,det_m)

            case 4:
                # Ensures that p,r and q,s are in the correct order 
                # and corresponds to valid indicies to create/annihilate.
                p,q = np.flatnonzero(np.asarray((det_m-det_n)==1))
                r,s = np.flatnonzero(np.asarray((det_n-det_m)==1))
                return self.two_pair_diff(p,q,r,s,det_n,det_m)

            case _:
                return 0

    def get_hamiltonian(self):
        slater_dets = self.get_slater_dets(self.num_orbitals, self.num_electrons)
        n_dets = len(slater_dets)
        H = np.zeros((n_dets, n_dets), dtype=np.cdouble)

        for n, det_n in enumerate(slater_dets):
            for m, det_m in enumerate(slater_dets):
                H[n,m] = self.slater_condon(det_n, det_m)

        return H