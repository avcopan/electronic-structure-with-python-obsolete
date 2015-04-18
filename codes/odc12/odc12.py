import psi4
import numpy as np
import scipy.linalg as la
from spinorbital import SpinOrbital
from index       import Index
from permutation import Permute as P, Identity as I, Transpose as Tp


class SpinOrbODC12:

    def __init__(self, scfwfn, mints):
      spinorb = SpinOrbital(scfwfn, mints)
      nocc    = spinorb.nocc
      dim     = spinorb.dim
      Ep2     = spinorb.build_Ep2()                  # Ep2 = 1/(fii+fjj-faa-fbb)
      h       = spinorb.build_mo_H()                 # h  = <p|T+V|q>  one-electron integrals
      g       = spinorb.build_mo_antisymmetrized_G() # g  = <pq||rs>   two-electron integrals
      indx    = Index(dim, 'pqrst')
      indx.add_index_range(   0, nocc, 'ijklm')
      indx.add_index_range(nocc,  dim, 'abcde')
      # save what we need to object
      self.spinorb, self.indx, self.Ep2, self.h, self.g = spinorb, indx, Ep2, h, g
      self.E, self.Vnu = 0.0, psi4.get_active_molecule().nuclear_repulsion_energy()

    def odc12_energy(self):
      spinorb, indx, Ep2, h, g = self.spinorb, self.indx, self.Ep2, self.h, self.g
      L = indx.einsum('ijab', (g,"ijab"), (Ep2,"ijab")) # MP2 guess

      for i in range( psi4.get_global_option('MAXITER') ):

        K = spinorb.build_mo_K()

        T = indx.meinblock('pq',
              ['ij',-1./2, I, (L,"ikab"), (L,"jkab")],
              ['ab', 1./2, I, (L,"ijac"), (L,"ijbc")] )

        for j in range( psi4.get_global_option('MAXITER') ):
          T0 = T
          T  = indx.meinblock('pq',
                 ['ij',-1.  , I, (T,"ik"  ), (T,"kj"  )],
                 ['ij',-1./2, I, (L,"ikab"), (L,"jkab")],
                 ['ab', 1.  , I, (T,"ac"  ), (T,"cb"  )],
                 ['ab', 1./2, I, (L,"ijac"), (L,"ijbc")] )
          if(la.norm((T-T0).array) < psi4.get_global_option('R_CONVERGENCE')): break

        G1  = K + T

        G2  = indx.meinblock('pqrs',
                ['pqrs', 1.  , P("rs")    , (G1,"pr" ), (G1,"qs" )],
                ['ijab', 1.  , Tp("ij|ab"), (L,"ijab")            ],
                ['ijkl', 1./2, I          , (L,"ijab"), (L,"klab")],
                ['abcd', 1./2, I          , (L,"ijab"), (L,"ijcd")],
                ['aibj',-1.  , P("ai|bj") , (L,"jkac"), (L,"ikbc")] )

        F   = indx.meinsums('pq',
                [ 1.  , I, (h,"pr"  ), (G1,"rq"  )],
                [ 1./2, I, (g,"prst"), (G2,"stqr")] )

        Ep1 = spinorb.build_Ep1()
        w   = indx.meinsum('ia', 2., P("ia"), (F,"ai"), extend='pq')
        X   = -1./2 * indx.einsum('ia', (w,"ia"), (Ep1,"ia"))

        spinorb.rotate_orbitals(X)

        h = spinorb.build_mo_H()
        g = spinorb.build_mo_antisymmetrized_G()
        f = np.matrix( indx.meinsums('pq',
                         [1., I, (h,"pq"  )           ],
                         [1., I, (g,"pqrs"), (G1,"rs")] ) )

        nocc = spinorb.nocc
        t, U = la.eigh(T.array)
        t[nocc:] *= -1.
        tf  = (U.T * f * U) / (1 + t.reshape(-1,1) + t.reshape(1,-1))
        tF  = U * tf * U.T
        Ep2 = spinorb.build_Ep2(fockmt = tF)

        R   = indx.meinsums('ijab',
                [ 1.  , P("ab")   , (tF,"ac"  ), (L,"ijcb")],
                [-1.  , P("ij")   , (tF,"ki"  ), (L,"kjab")],
                [ 1.  , I         , (g ,"ijab")            ],
                [ 1./2, I         , (g ,"ijkl"), (L,"klab")],
                [ 1./2, I         , (g ,"abcd"), (L,"ijcd")],
                [-1.  , P("ij|ab"), (g ,"kbjc"), (L,"ikac")] )

        L   = L + Ep2 * R

        E   = indx.meinsums('',
                [1.  , I, (h,"pq"  ), (G1,"pq"  )],
                [1./4, I, (g,"pqrs"), (G2,"pqrs")] ) + self.Vnu

        dE     = E - self.E
        self.E = E

        psi4.print_out('\n@ODC12{:-3d}{:20.15f}{:20.15f}'.format(i, E, dE))
        if(abs(dE) < psi4.get_global_option('E_CONVERGENCE')): break

      return self.E

