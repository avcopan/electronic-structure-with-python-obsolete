import psi4
import numpy as np
from scipy import linalg as la
from index import Index
from permutation import Permute as P

class SpinOrbital:

    def __init__(self, scfwfn, mints):
      nocc = scfwfn.nalpha() + scfwfn.nbeta()
      dim  = mints.basisset().nbf() * 2
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
      X     = np.matrix(la.inv(la.sqrtm(S)))
      tF    = X * F * X
      e, tC = la.eigh(tF)
      C     = X * tC
      # compute integrals
      H = block_matrix_aa(Ta) + block_matrix_aa(Va)
      G = block_4darray(Ga).swapaxes(1,2) # < mu nu | rh si >, phys. notation
      self.nocc, self.dim, self.indx = nocc, dim, indx
      self.e, self.C, self.S, self.H, self.G, self.F = e, C, S, H, G, F

    def build_Ep1(self, fockmat=None):
      indx, nocc = self.indx, self.nocc
      if fockmat is None: fockmat = self.build_mo_F()
      D        = indx.zeros("ia")
      fpp      = np.diag(fockmat)
      fii, faa = fpp[:nocc], fpp[nocc:]
      D.array  = 1./(fii.reshape(-1,1) - faa.reshape(1,-1))
      return D

    def build_Ep2(self, fockmat=None):
      indx, nocc = self.indx, self.nocc
      if fockmat is None: fockmat = self.build_mo_F()
      D        = indx.zeros("ijab")
      fpp      = np.diag(fockmat)
      fii, faa = fpp[:nocc], fpp[nocc:]
      D.array  = 1./( fii.reshape(-1,1,1,1) + fii.reshape(1,-1,1,1)
                    - faa.reshape(1,1,-1,1) - faa.reshape(1,1,1,-1) )
      return D

    def build_Ep3(self, fockmat=None):
      indx, nocc = self.indx, self.nocc
      if fockmat is None: fockmat = self.build_mo_F()
      D        = indx.zeros("ijkabc")
      fpp      = np.diag(fockmat)
      fii, faa = fpp[:nocc], fpp[nocc:]
      D.array  = 1./( fii.reshape(-1,1,1,1,1,1) + fii.reshape(1,-1,1,1,1,1) + fii.reshape(1,1,-1,1,1,1)
                    - faa.reshape(1,1,1,-1,1,1) - faa.reshape(1,1,1,1,-1,1) - faa.reshape(1,1,1,1,1,-1) )
      return D

    def build_mo_K(self):
      nocc, S, C = self.nocc, np.matrix(self.S), np.matrix(self.C)
      oC = C[:,:nocc]
      return C.T * S * oC * oC.T * S * C

    def build_mo_F(self):
      F, C = np.matrix(self.F), np.matrix(self.C)
      return C.T * F * C

    def build_mo_H(self):
      H, C = np.matrix(self.H), np.matrix(self.C)
      return C.T * H * C

    def build_mo_antisymmetrized_G(self):
      G, C, indx = self.G, self.C, self.indx
      return indx.meinsum('pqrs', 1, P("rs"), (G,"PQRS"), (C,"Pp"), (C,"Qq"), (C,"Rr"), (C,"Ss"))

    def rotate_orbitals(self, X):
      C = np.matrix(self.C)
      U = la.expm(X - X.transpose())
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

