import psi4
from theory.ccsd.ccsd_diis import SpinOrbCCSD
from lib.index             import Index
from lib.permutation       import Permute as P, Identity as I


class SpinOrbCCSDbT:

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

    def ccsdbt_energy(self):
      indx, Esd, t1, t2, g, Ep3 = self.indx, self.Esd, self.t1, self.t2, self.g, self.Ep3

      t3 = Ep3 * indx.meinsums('ijkabc',
                   [ 1., P("i/jk|a/bc"), (g,"eibc"), (t2,"jkae")],
                   [-1., P("i/jk|a/bc"), (g,"majk"), (t2,"imbc")])

      Et  = indx.meinsum('', 1./36, I, (t3/Ep3,"ijkabc"), (t3,"ijkabc"))

      psi4.print_out('\n@CCSD[T] Esd   = {:20.15f}'.format(Esd   ))
      psi4.print_out('\n@CCSD[T] Et    = {:20.15f}'.format(Et    ))
      psi4.print_out('\n@CCSD[T] Ecorr = {:20.15f}'.format(Esd+Et))

      return Esd + Et

