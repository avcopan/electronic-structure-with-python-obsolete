import psi4
from meinsum.spinorbital import SpinOrbital
from meinsum.index       import Index
from meinsum.permutation import Permute as P, Identity as I


class SpinOrbMP2:

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

    def mp2_energy(self):
      indx, Ep, g = self.indx, self.Ep, self.g

      t = indx.einsum ('ijab',      (g,"ijab"), (Ep,"ijab"))
      E = indx.meinsum('', 1./4, I, (g,"ijab"), (t ,"ijab"))

      psi4.print_out('\n@MP2   {:20.15f}'.format(E))
      return self.E

