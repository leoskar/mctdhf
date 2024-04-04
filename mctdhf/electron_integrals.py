import numpy as np
from scipy.linalg import eigh_tridiagonal

class GaussianWell:
    def __init__(self, a=1, w=1, center=0):
        self.a = a
        self.w = w
        self.center = center
        
    def __call__(self,x):
        return -self.w * np.exp(-self.a * (x - self.center)**2)

class HOPotential:
    def __init__(self, omega=1):
        self.omega = omega

    def __call__(self, x):
        return 0.5 * self.omega**2 * x**2

def get_spf_and_diag_h(l, grid, potential, w=0):
    num_grid_points = len(grid)

    # Assume even spacing
    dx = abs(grid[1] - grid[0])

    # Construct tridiagonal matrix (from finite-differences)
    h_diag = 1.0 / (dx**2) + potential(grid[1:-1])
    h_off_diag = -1.0 / (2 * dx**2) * np.ones(num_grid_points - 3)
    
    # Diagonalize the one-body Hamiltonian
    eigen_energies, eigen_states = eigh_tridiagonal(h_diag, h_off_diag, select='i', select_range=(0, l-1))

    # Set up single-particle functions (evaluated at the grid)
    spf = np.zeros((l, num_grid_points))

    # Note that we explicitly keep the boundaries at exactly zero
    # Also, the normalization should in principle be done by an integral

    spf[:, 1:-1] = eigen_states.T / np.sqrt(dx)
    h = np.diag(eigen_energies)

    return spf, h


def shielded_coulomb(x_1, x_2, kappa, a):
    return kappa / np.sqrt((x_1-x_2)**2+a**2)


def coulomb_interaction_matrix_elements(spf_1, spf_2, grid_1, grid_2, kappa, a):

    u = shielded_coulomb(grid_1[:, None], grid_2[None, :], kappa, a)

    # Assuming even spacing
    dx2 = (grid_1[1]-grid_1[0])*(grid_2[1]-grid_2[0])
    
    u_abcd = dx2*np.einsum('ap, bq, cp, dq, pq -> abcd', spf_1, spf_2, spf_1, spf_2, u, optimize=True)

    return u_abcd