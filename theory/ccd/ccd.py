import psi4
from lib.spinorbital import SpinOrbital
from lib.index       import Index
from lib.permutation import Permute as P, Identity as I


class SpinOrbCCD:

    def __init__(self, scfwfn, mints):
      spinorb = SpinOrbital(scfwfn, mints)
      nocc    = spinorb.nocc
      dim     = spinorb.dim
      Ep      = spinorb.build_Ep2()                  # Ep = 1/(fii+fjj-faa-fbb)
      g       = spinorb.build_mo_antisymmetrized_G() # g  = <pq||rs>
      indx    = Index(dim, 'pqrs')
      indx.add_index_range(   0, nocc, 'ijkl')
      indx.add_index_range(nocc,  dim, 'abcd')
      # save what we need to object
      self.indx, self.Ep, self.g = indx, Ep, g
      self.E  = 0.0

    def ccd_energy(self):
      indx, Ep, g = self.indx, self.Ep, self.g
      t = indx.einsum('ijab', (g,"ijab"), (Ep,"ijab"))

      for i in range(maxiter):

        t = Ep * indx.meinsums('ijab',# Bartlett, p. 288
                   [ 1.  , I         , (g,"abij")                        ],#t01
                   [ 1./2, I         , (g,"abcd"), (t,"ijcd")            ],#t04
                   [ 1./2, I         , (g,"klij"), (t,"klab")            ],#t05
                   [ 1.  , P("ij|ab"), (g,"kbcj"), (t,"ikac")            ],#t06
                   [ 1./4, I         , (g,"klcd"), (t,"ijcd"), (t,"klab")],#t07
                   [ 1.  , P("ij")   , (g,"klcd"), (t,"ikac"), (t,"jlbd")],#t08
                   [-1./2, P("ij")   , (g,"klcd"), (t,"ikdc"), (t,"ljab")],#t09
                   [-1./2, P("ab")   , (g,"klcd"), (t,"lkac"), (t,"ijdb")])#t10

        E      = indx.meinsum('', 1./4, I, (g,"ijab"), (t,"ijab"))
        dE     = E - self.E
        self.E = E

        psi4.print_out('\n@CCD{:-3d}{:20.15f}{:20.15f}'.format(i, E, dE))
        if(abs(dE) < econv): break

      return self.E

# keyword values
maxiter = psi4.get_global_option('MAXITER')
econv   = psi4.get_global_option('E_CONVERGENCE')