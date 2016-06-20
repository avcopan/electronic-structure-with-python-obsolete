import psi4
import numpy as np
from scipy       import linalg as la
from index       import Index
from permutation import permute as P

class SpinOrbital:

    def __init__(self, scfwfn, mints):
      nocc = scfwfn.nalpha() + scfwfn.nbeta()
      dim  = mints.basisset().nbf() * 2
      Vnu  = psi4.get_active_molecule().nuclear_repulsion_energy()
      indx = Index(dim, 'pqrsPQRS')
      indx.add_index_range(0   , nocc, 'ijkl')
      indx.add_index_range(nocc,  dim, 'abcd')
      # grab integrals and project out symmetry
      C1 = mints.petite_list().sotoao() # Sym -> C1 projection matrix
      Fa = psi4.Matrix(dim/2, dim/2)
      Fb = psi4.Matrix(dim/2, dim/2)
      Fa.remove_symmetry(scfwfn.Fa(), C1)
      Fb.remove_symmetry(scfwfn.Fb(), C1)
      Sa = mints.ao_overlap()
      Ta = mints.ao_kinetic()
      Va = mints.ao_potential()
      Ga = mints.ao_eri()
      # build and diagonalize spin-orbital Fock matrix
      F     = block_matrix_ab(Fa, Fb)
      S     = block_matrix_aa(Sa)
      s, U  = la.eigh(S)
      U     = np.matrix(U)
      x     = 1./np.sqrt(s).real
      X     = U * np.diag(x) * U.T
      tF    = X * F * X
      e, tC = la.eigh(tF)
      C     = X * tC
      # compute integrals
      H = block_matrix_aa(Ta) + block_matrix_aa(Va)
      G = block_4darray(Ga).swapaxes(1,2) # < mu nu | rh si >, phys. notation
      self.nocc, self.dim, self.Vnu, self.indx = nocc, dim, Vnu, indx
      self.e, self.C, self.S, self.H, self.G, self.F = e, C, S, H, G, F

    def compute_Escf(self):
      h, f, K = self.build_mo_H(), self.build_mo_F(), self.build_mo_K()
      return np.trace(1./2*(h+f)*K) + self.Vnu

    def build_Ep1(self, fockmat=None):
      indx, nocc = self.indx, self.nocc
      if fockmat is None: fockmat = self.build_mo_F()
      D        = indx.zeros("ia")
      fpp      = np.diag(fockmat)
      fii, faa = fpp[:nocc], fpp[nocc:]
      D.block  = 1./(fii.reshape(-1,1) - faa.reshape(1,-1))
      return D

    def build_Ep2(self, fockmat=None):
      indx, nocc = self.indx, self.nocc
      if fockmat is None: fockmat = self.build_mo_F()
      D        = indx.zeros("ijab")
      fpp      = np.diag(fockmat)
      fii, faa = fpp[:nocc], fpp[nocc:]
      D.block  = 1./( fii.reshape(-1,1,1,1) + fii.reshape(1,-1,1,1)
                    - faa.reshape(1,1,-1,1) - faa.reshape(1,1,1,-1) )
      return D

    def build_Ep3(self, fockmat=None):
      indx, nocc = self.indx, self.nocc
      if fockmat is None: fockmat = self.build_mo_F()
      D        = indx.zeros("ijkabc")
      fpp      = np.diag(fockmat)
      fii, faa = fpp[:nocc], fpp[nocc:]
      D.block  = 1./( fii.reshape(-1,1,1,1,1,1) + fii.reshape(1,-1,1,1,1,1) + fii.reshape(1,1,-1,1,1,1)
                    - faa.reshape(1,1,1,-1,1,1) - faa.reshape(1,1,1,1,-1,1) - faa.reshape(1,1,1,1,1,-1) )
      return D

    def build_mo_K(self):
      K = np.identity(self.dim)
      K[self.nocc:,self.nocc:] = 0
      return K

    def build_mo_F(self):
      F, C = np.matrix(self.F), np.matrix(self.C)
      return C.T * F * C

    def build_mo_H(self):
      H, C = np.matrix(self.H), np.matrix(self.C)
      return C.T * H * C

    def build_mo_antisymmetrized_G(self):
      G, C, indx = self.G, self.C, self.indx
      return indx.meinsum('pqrs', 1, P("rs"), (G,"PQRS"), (C,"Pp"), (C,"Qq"), (C,"Rr"), (C,"Ss"))

    def rotate_orbitals(self, U):
      C, U = np.matrix(self.C), np.matrix(U)
      self.C = C * U

def block_matrix_aa(A):
  A  = np.matrix(A)
  I2 = np.identity(2)
  return np.matrix( np.kron(I2, A) )

def block_matrix_ab(A, B):
  A      , B       = np.matrix(A)   , np.matrix(B)
  IA     , IB      = np.zeros((2,2)), np.zeros((2,2))
  IA[0,0], IB[1,1] = 1              , 1
  return np.matrix( np.kron(IA, A) + np.kron(IB, B) )

def block_4darray(T):
  t  = np.array(T)
  n  = t.shape[0]
  I2 = np.identity(2)
  T  = np.zeros((2*n,2*n,2*n,2*n))
  for p in range(n):
    for q in range(n):
      T[p,q] = np.kron(I2, t[p,q])
  T[n:,n:] = T[:n,:n]
  return T

