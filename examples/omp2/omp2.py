import psi4
import numpy as np
from meinsum.spinorbital import SpinOrbital
from meinsum.index       import Index
from meinsum.permutation import Permute as P, Identity as I, Transpose as Tp


class SpinOrbOMP2:

    def __init__(self, scfwfn, mints):
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
      self.spinorb, self.indx, self.Ep2, self.K, self.h, self.g = spinorb, indx, Ep2, K, h, g
      self.E, self.Vnu = 0.0, psi4.get_active_molecule().nuclear_repulsion_energy()

    def omp2_energy(self):
      spinorb, indx     = self.spinorb, self.indx
      Ep2, K, h, g, Vnu = self.Ep2, self.K, self.h, self.g, self.Vnu
      t = indx.einsum('ijab', (g,"ijab"), (Ep2,"ijab"))

      for i in range(maxiter):

        T  = indx.meinblock('pq',
               ['ij',-1./2, I, (t,"ikab"), (t,"jkab")],
               ['ab', 1./2, I, (t,"ijac"), (t,"ijbc")])

        G1 = K + T

        G2 = indx.meinblock('pqrs',
               ['pqrs', 1.  , P("rs")    , (K ,"pr" ), (K,"qs" )],
               ['pqrs', 1.  , P("pq|rs") , (K ,"pr" ), (T,"qs" )],
               ['ijab', 1.  , Tp("ij|ab"), (t,"ijab")           ])

        F  = indx.meinsums('pq',
               [ 1.  , I, (h,"pr"  ), (G1,"rq"  )],
               [ 1./2, I, (g,"prst"), (G2,"stqr")])

        Ep1 = spinorb.build_Ep1()
        w   =  2 * (F - F.T)
        X   = -1./2 * indx.einsum('ai', (w,"ia"), (Ep1,"ia"))

        spinorb.rotate_orbitals(X - X.T)

        h   = spinorb.build_mo_H()
        g   = spinorb.build_mo_antisymmetrized_G()
        f   = spinorb.build_mo_F()
        Ep2 = spinorb.build_Ep2()
        np.fill_diagonal(f, 0.0)

        t = Ep2 * indx.meinsums('ijab',
              [ 1.  , I      , (g,"ijab")            ],
              [ 1.  , P("ab"), (f,"ac"  ), (t,"ijcb")],
              [-1.  , P("ij"), (f,"ki"  ), (t,"kjab")])

        E = indx.meinsums('',
              [1.  , I, (h,"pq"  ), (G1,"pq"  )],
              [1./4, I, (g,"pqrs"), (G2,"pqrs")]) + Vnu

        dE     = E - self.E
        self.E = E

        psi4.print_out('\n@OMP2{:-3d}{:20.15f}{:20.15f}'.format(i, E, dE))
        if(abs(dE) < e_conv): break

      return self.E

# keyword values
maxiter    = psi4.get_global_option('MAXITER')
e_conv     = psi4.get_global_option('E_CONVERGENCE')
