import psi4
from lib.spinorbital import SpinOrbital
from lib.index       import Index
from lib.permutation import Permute as P, Identity as I


class SpinOrbCCSD:

    def __init__(self, scfwfn, mints):
      spinorb = SpinOrbital(scfwfn, mints)
      nocc    = spinorb.nocc
      dim     = spinorb.dim
      Ep1     = spinorb.build_Ep1()                  # Ep1 = 1/(fii-faa)
      Ep2     = spinorb.build_Ep2()                  # Ep2 = 1/(fii+fjj-faa-fbb)
      g       = spinorb.build_mo_antisymmetrized_G() # g   = <pq||rs>
      indx    = Index(dim, 'pqrstu')
      indx.add_index_range(   0, nocc, 'ijklmn')
      indx.add_index_range(nocc,  dim, 'abcdef')
      # save what we need to object
      self.indx, self.Ep1, self.Ep2, self.g = indx, Ep1, Ep2, g
      self.E  = 0.0

    def ccsd_energy(self):
      indx, Ep1, Ep2, g = self.indx, self.Ep1, self.Ep2, self.g
      t1 = indx.zeros('ia')
      t2 = indx.einsum('ijab', (g,"ijab"), (Ep2,"ijab"))

      for i in range(maxiter):

        t1 = Ep1 * indx.meinsums('ia',# Bartlett p. 304
                     [ 1./2, I, (g,"akcd"), (t2,"ikcd")                          ],#t03
                     [-1./2, I, (g,"klic"), (t2,"klac")                          ],#t04
                     [ 1.  , I, (g,"akic"), (t1,"kc"  )                          ],#t07
                     [-1./2, I, (g,"klcd"), (t1,"ic"  ), (t2,"klad")             ],#t08
                     [-1./2, I, (g,"klcd"), (t1,"ka"  ), (t2,"ilcd")             ],#t09
                     [ 1.  , I, (g,"klcd"), (t1,"kc"  ), (t2,"lida")             ],#t10
                     [ 1.  , I, (g,"akcd"), (t1,"ic"  ), (t1,"kd"  )             ],#t12
                     [-1.  , I, (g,"klic"), (t1,"ka"  ), (t1,"lc"  )             ],#t13
                     [-1.  , I, (g,"klcd"), (t1,"ic"  ), (t1,"ka"  ), (t1,"ld"  )])#t14

        t2 = Ep2 * indx.meinsums('ijab',# Bartlett p. 288(') (CCD), p. 307('')-308(''') (CCSD)
                     [ 1.  , I         , (g,"abij")                                                  ],#t01'
                     [ 1./2, I         , (g,"abcd"), (t2,"ijcd")                                     ],#t04'
                     [ 1./2, I         , (g,"klij"), (t2,"klab")                                     ],#t05'
                     [ 1.  , P("ij|ab"), (g,"kbcj"), (t2,"ikac")                                     ],#t06'
                     [ 1./4, I         , (g,"klcd"), (t2,"ijcd"), (t2,"klab")                        ],#t07'
                     [ 1.  , P("ij")   , (g,"klcd"), (t2,"ikac"), (t2,"jlbd")                        ],#t08'
                     [-1./2, P("ij")   , (g,"klcd"), (t2,"ikdc"), (t2,"ljab")                        ],#t09'
                     [-1./2, P("ab")   , (g,"klcd"), (t2,"lkac"), (t2,"ijdb")                        ],#t10'
                     [ 1.  , P("ij")   , (g,"abcj"), (t1,"ic"  )                                     ],#t01''
                     [-1.  , P("ab")   , (g,"kbij"), (t1,"ka"  )                                     ],#t02''
                     [ 1.  , P("ij|ab"), (g,"akcd"), (t1,"ic"  ), (t2,"kjdb")                        ],#t05''
                     [-1.  , P("ij|ab"), (g,"klic"), (t1,"ka"  ), (t2,"ljcb")                        ],#t06''
                     [-1./2, P("ab")   , (g,"kbcd"), (t1,"ka"  ), (t2,"ijcd")                        ],#t07''
                     [ 1./2, P("ij")   , (g,"klcj"), (t1,"ic"  ), (t2,"klab")                        ],#t01'''
                     [ 1.  , P("ab")   , (g,"kacd"), (t1,"kc"  ), (t2,"ijdb")                        ],#t02'''
                     [-1.  , P("ij")   , (g,"klci"), (t1,"kc"  ), (t2,"ljab")                        ],#t03'''
                     [ 1.  , I         , (g,"abcd"), (t1,"ic"  ), (t1,"jd"  )                        ],#t04'''
                     [ 1.  , I         , (g,"klij"), (t1,"ka"  ), (t1,"lb"  )                        ],#t05'''
                     [-1.  , P("ij|ab"), (g,"kbcj"), (t1,"ic"  ), (t1,"ka"  )                        ],#t06'''
                     [ 1./2, I         , (g,"klcd"), (t1,"ic"  ), (t1,"jd"  ), (t2,"klab")           ],#t07'''
                     [ 1./2, I         , (g,"klcd"), (t1,"ka"  ), (t1,"lb"  ), (t2,"ijcd")           ],#t08'''
                     [-1.  , P("ij|ab"), (g,"klcd"), (t1,"ic"  ), (t1,"ka"  ), (t2,"ljdb")           ],#t09'''
                     [-1.  , P("ij")   , (g,"klcd"), (t1,"kc"  ), (t1,"id"  ), (t2,"ljab")           ],#t10'''
                     [-1.  , P("ab")   , (g,"klcd"), (t1,"kc"  ), (t1,"la"  ), (t2,"ijdb")           ],#t11'''
                     [ 1.  , P("ab")   , (g,"kbcd"), (t1,"ic"  ), (t1,"ka"  ), (t1,"jd"  )           ],#t12'''
                     [ 1.  , P("ij")   , (g,"klcj"), (t1,"ic"  ), (t1,"ka"  ), (t1,"lb"  )           ],#t13'''
                     [ 1.  , I         , (g,"klcd"), (t1,"ic"  ), (t1,"jd"  ), (t1,"ka"  ), (t1,"lb")])#t14'''

        E      = indx.meinsums('',
                   [1./4, I, (g,"ijab"), (t2,"ijab")            ],
                   [1./2, I, (g,"ijab"), (t1,"ia"  ), (t1,"jb" )])
        dE     = E - self.E
        self.E = E

        psi4.print_out('\n@CCSD{:-3d}{:20.15f}{:20.15f}'.format(i, E, dE))
        if(abs(dE) < econv): break

      return self.E

# keyword values
maxiter = psi4.get_global_option('MAXITER')
econv   = psi4.get_global_option('E_CONVERGENCE')
