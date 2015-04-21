import psi4
import numpy as np
from scipy           import linalg as la
from lib.diis        import DIIS
from lib.spinorbital import SpinOrbital
from lib.index       import Index
from lib.permutation import Permute as P, Identity as I, Transpose as Tp


class SpinOrbODC12:

    def __init__(self, scfwfn, mints):
      diis      = DIIS()
      spinorb   = SpinOrbital(scfwfn, mints)
      Ep2       = spinorb.build_Ep2()                  # Ep2 = 1/(fii+fjj-faa-fbb)
      K         = spinorb.build_mo_K()                 # K   = <p|Phi><Phi|q> single-det density matrix
      h         = spinorb.build_mo_H()                 # h   = <p|T+V|q>      one-electron integrals
      g         = spinorb.build_mo_antisymmetrized_G() # g   = <pq||rs>       two-electron integrals
      nocc, dim = spinorb.nocc, spinorb.dim
      indx      = Index(dim, 'pqrst')
      indx.add_index_range(   0, nocc, 'ijklm')
      indx.add_index_range(nocc,  dim, 'abcde')
      # save what we need to object
      self.diis, self.spinorb, self.indx = diis, spinorb, indx
      self.Ep2, self.K, self.h, self.g   = Ep2, K, h, g
      self.E, self.Vnu = 0.0, psi4.get_active_molecule().nuclear_repulsion_energy()

    def odc12_energy(self):
      diis, spinorb, indx     = self.diis, self.spinorb, self.indx
      Ep2, K, h, g, Vnu, nocc = self.Ep2, self.K, self.h, self.g, self.Vnu, spinorb.nocc
      L = indx.einsum('ijab', (g,"ijab"), (Ep2,"ijab"))

      for i in range(maxiter):

        if do_diis and i >= diis_start: L, = diis.extrapolate()

        T = indx.meinblock('pq',
              ['ij',-1./2, I, (L,"ikab"), (L,"jkab")],
              ['ab', 1./2, I, (L,"ijac"), (L,"ijbc")] )

        for j in range(maxiter):
          T0 = T
          T  = indx.meinblock('pq',
                 ['ij',-1.  , I, (T,"ik"  ), (T,"kj"  )],
                 ['ij',-1./2, I, (L,"ikab"), (L,"jkab")],
                 ['ab', 1.  , I, (T,"ac"  ), (T,"cb"  )],
                 ['ab', 1./2, I, (L,"ijac"), (L,"ijbc")] )
          if(la.norm(T-T0) < r_conv): break

        G1 = K + T

        G2 = indx.meinblock('pqrs',
               ['pqrs', 1.  , P("rs")    , (G1,"pr" ), (G1,"qs" )],
               ['ijab', 1.  , Tp("ij|ab"), (L,"ijab")            ],
               ['ijkl', 1./2, I          , (L,"ijab"), (L,"klab")],
               ['abcd', 1./2, I          , (L,"ijab"), (L,"ijcd")],
               ['aibj',-1.  , P("ai|bj") , (L,"jkac"), (L,"ikbc")] )

        F  = indx.meinsums('pq',
               [ 1.  , I, (h,"pr"  ), (G1,"rq"  )],
               [ 1./2, I, (g,"prst"), (G2,"stqr")] )

        Ep1 = spinorb.build_Ep1()
        w   =  2 * (F - F.T)
        X   = -1./2 * indx.einsum('ai', (w,"ia"), (Ep1,"ia"))

        spinorb.rotate_orbitals(X - X.T)

        h = spinorb.build_mo_H()
        g = spinorb.build_mo_antisymmetrized_G()
        f = indx.meinsums('pq',
              [1., I, (h,"pq"  )           ],
              [1., I, (g,"prqs"), (G1,"sr")] )

        t, U      = la.eigh(T)
        t[nocc:] *= -1.
        tf        = (U.T * f * U) / (1 + t.reshape(-1,1) + t.reshape(1,-1))
        tF        = U * tf * U.T
        Ep2       = spinorb.build_Ep2(fockmat = tF)

        R = indx.meinsums('ijab',
              [ 1.  , P("ab")   , (tF,"ac"  ), (L,"ijcb")],
              [-1.  , P("ij")   , (tF,"ki"  ), (L,"kjab")],
              [ 1.  , I         , (g ,"ijab")            ],
              [ 1./2, I         , (g ,"ijkl"), (L,"klab")],
              [ 1./2, I         , (g ,"abcd"), (L,"ijcd")],
              [-1.  , P("ij|ab"), (g ,"kbjc"), (L,"ikac")] )

        L = L + Ep2 * R

        if do_diis: diis.add_vec((L, R.block))

        E = indx.meinsums('',
              [1.  , I, (h,"pq"  ), (G1,"pq"  )],
              [1./4, I, (g,"pqrs"), (G2,"pqrs")] ) + Vnu

        dE     = E - self.E
        self.E = E

        psi4.print_out('\n@ODC12{:-3d}{:20.15f}{:20.15f}'.format(i, E, dE))
        if(abs(dE) < e_conv): break

      return self.E

# keyword values
maxiter    = psi4.get_global_option('MAXITER')
do_diis    = psi4.get_global_option('DIIS')
diis_start = psi4.get_global_option('DIIS_START')
r_conv     = psi4.get_global_option('R_CONVERGENCE')
e_conv     = psi4.get_global_option('E_CONVERGENCE')
