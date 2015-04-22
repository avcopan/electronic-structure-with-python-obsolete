import psi4
from theory.ccsd.ccsd_diis import SpinOrbCCSD
from lib.index             import Index
from lib.permutation       import Permute as P, Identity as I


class SpinOrbCCSDparT:

    def __init__(self, scfwfn, mints):
      ccsd    = SpinOrbCCSD(scfwfn, mints)
      spinorb = ccsd.spinorb
      Esd     = ccsd.ccsd_energy()
      t1      = ccsd.t1
      t2      = ccsd.t2
      g       = ccsd.g                # g   = <pq||rs>
      Ep3     = spinorb.build_Ep3()   # Ep3 = 1/(fii+fjj+fkk-faa-fbb-fcc)
      dim     = spinorb.dim
      nocc    = spinorb.nocc
      indx    = Index(dim, 'pqrstu')
      indx.add_index_range(   0, nocc, 'ijklmn')
      indx.add_index_range(nocc,  dim, 'abcdef')
      # save what we need to object
      self.indx, self.Esd, self.t1, self.t2, self.g, self.Ep3 = indx, Esd, t1, t2, g, Ep3

    def ccsdpart_energy(self):
      indx, Esd, t1, t2, f, Ep3 = self.indx, self.Esd, self.t1, self.t2, self.g, self.Ep3

      Et = Ep3 * indx.meinsums('',
                   [ 1./4, P("i/jk|a/bc"), (g,"eibc"), (g ,"fibc"), (t2,"aejk"), (t2,"ajfk")],
                   [-1./2, P("i/jk|a/bc"), (g,"eibc"), (t2,"aejk"), (g ,"majk"), (t2,"imbc")],
                   [ 1./4, P("i/jk|a/bc"), (g,"majk"), (g ,"najk"), (t2,"imbc"), (t2,"inbc")],


