import numpy as np

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

class AddressHamiltonian(CIHamiltonian):

    def __init__(self, num_orbitals, num_electrons, h, g):
        super().__init__(num_orbitals, num_electrons, h, g)

        # Calculate node weights, see Fig. 19 and equation 3.70 in Hochstuhl
        self.W = np.zeros((num_orbitals+1, num_electrons+1))
        self.W[:,0] = 1

        for m in range(1,num_orbitals+1):
            for k in range(1,num_electrons+1):
                self.W[m,k] = self.W[m-1,k] + self.W[m-1, k-1]

    # Use node weights to calculate address, see equation 3.73 in Hochstuhl
    def add(self, n):
        res = 0
        for m in range(self.num_orbitals):
            res += n[m]*self.W[m, int(np.sum(n[:m+1]))]
        return int(res)

    # Define excitation operators
    def single_exc(self, p,q,n):
        #if n[q]==0:
        #    return 0, 0
        n_pq = np.copy(n)
        n_pq[q] = 0

        if n_pq[p]==1:
            return 0, 0

        n_pq[p] = 1
        gamma = (-1)**(np.sum(n[:q])+np.sum(n_pq[:p]))

        return gamma, n_pq

    def double_exc(self, p, r, s, q, n):
        if s==q or p == r:
            return 0, 0
        #if n[s]==0 or n[q]==0:
        #    return 0, 0

        n_prqs = np.copy(n)
        n_prqs[q] = 0
        gamma = (-1)**np.sum(n_prqs[:q])
        n_prqs[s] = 0
        gamma *= (-1)**np.sum(n_prqs[:s])

        if n_prqs[r]==1 or n_prqs[p]==1:
            return 0, 0

        n_prqs[r] = 1
        gamma *= (-1)**np.sum(n_prqs[:r])
        n_prqs[p] = 1
        gamma *= (-1)**np.sum(n_prqs[:p])

        return gamma, n_prqs

    def get_hamiltonian(self):
        slater_dets = self.get_slater_dets(self.num_orbitals, self.num_electrons)
        n_dets = len(slater_dets)
        H = np.zeros((n_dets, n_dets))

        for det_n in slater_dets:
            n = self.add(det_n)
            occ = np.flatnonzero(det_n)
            for p in range(self.num_orbitals):
                for q in occ:
                    xi, n_pq = self.single_exc(p,q,det_n)
                    if xi != 0:
                        m = self.add(n_pq)
                        H[n,m] += xi*self.h[p,q]

                    for r in range(self.num_orbitals):
                        for s in occ:
                            xi, n_pqrs = self.double_exc(p,r,s,q,det_n)
                            if xi != 0:
                                m = self.add(n_pqrs)
                                H[n,m] += 0.5*xi*self.g[p,q,r,s]

        return H

class SlaterCondonHamiltonian(CIHamiltonian):
    # Helper functions for slater condon rules
    def one_pair_diff(self, p, q, det_n, det_m):
        gamma = (-1)**(np.sum(det_n[:q])+np.sum(det_m[:p]))
        H = self.h[p,q]
        for r, n_r in enumerate(det_n):
            H += n_r*(self.g[p,q,r,r]-self.g[p,r,r,q])
        return gamma*H

    def two_pair_diff(self, p, r, s, q, det_n, det_m):
        n_prqs = np.copy(det_n)
        n_prqs[q] = 0
        gamma = np.sum(n_prqs[:q])
        n_prqs[s] = 0
        gamma += np.sum(n_prqs[:s])
        n_prqs[r] = 1
        gamma += np.sum(n_prqs[:r])
        n_prqs[p] = 1
        gamma += np.sum(n_prqs[:p])
        return (self.g[p,q,r,s]-self.g[p,s,r,q])*(-1)**gamma

    # Slater condon rules
    def slater_condon(self, det_n, det_m):
        num_differences = np.sum(np.abs(det_n-det_m))
        match(num_differences):
            case 0:
                H=0
                for p, n_p in enumerate(det_n):
                    H += n_p*self.h[p,p]
                    for r, n_r in enumerate(det_n):
                        H += 0.5*n_p*n_r*(self.g[p,p,r,r]-self.g[p,r,r,p])
                return H
                
            case 2:
                p = np.flatnonzero(np.asarray((det_m-det_n)==1))[0]
                q = np.flatnonzero(np.asarray((det_n-det_m)==1))[0]
                return self.one_pair_diff(p,q,det_n,det_m)

            case 4:
                # Ensures that p,r and q,s are in the correct order 
                # and corresponds to valid indicies to create/annihilate.
                p,r = np.flatnonzero(np.asarray((det_m-det_n)==1))
                q,s = np.flatnonzero(np.asarray((det_n-det_m)==1))
                return self.two_pair_diff(p,r,q,s,det_n,det_m)

            case _:
                return 0

    def get_hamiltonian(self):
        slater_dets = self.get_slater_dets(self.num_orbitals, self.num_electrons)
        n_dets = len(slater_dets)
        H = np.zeros((n_dets, n_dets), dtype=np.complex_)

        for n, det_n in enumerate(slater_dets):
            for m, det_m in enumerate(slater_dets):
                H[n,m] = self.slater_condon(det_n, det_m)

        return H