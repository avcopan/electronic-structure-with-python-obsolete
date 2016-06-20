import psi4
from meinsum.meinsum.spinorbital import SpinOrbital
from meinsum.meinsum.index       import Index
from meinsum.meinsum.permutation import permute as P, identity as I


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

        t1 = Ep1 * indx.meinsums('ia',# Crawford p. 59
                     [ 1.  , I, (g,"kaci"), (t1,"kc"  )                          ],
                     [ 1./2, I, (g,"kacd"), (t2,"kicd")                          ],
                     [-1./2, I, (g,"klci"), (t2,"klca")                          ],
                     [-1.  , I, (g,"klci"), (t1,"kc"  ), (t1,"la"  )             ],
                     [ 1.  , I, (g,"kacd"), (t1,"kc"  ), (t1,"id"  )             ],
                     [-1.  , I, (g,"klcd"), (t1,"kc"  ), (t1,"id"  ), (t1,"la"  )],
                     [ 1.  , I, (g,"klcd"), (t1,"kc"  ), (t2,"lida")             ],
                     [-1./2, I, (g,"klcd"), (t2,"kicd"), (t1,"la"  )             ],
                     [-1./2, I, (g,"klcd"), (t2,"klca"), (t1,"id"  )             ])

        t2 = Ep2 * indx.meinsums('ijab',# Crawford, p. 59-60
                     [ 1.  , I         , (g,"abij")                                                    ],
                     [ 1./2, I         , (g,"klij"), (t2,"klab")                                       ],
                     [ 1./2, I         , (g,"abcd"), (t2,"ijcd")                                       ],
                     [ 1.  , P("ij|ab"), (g,"kbcj"), (t2,"ikac")                                       ],
                     [ 1.  , P("ij")   , (g,"abcj"), (t1,"ic"  )                                       ],
                     [-1.  , P("ab")   , (g,"kbij"), (t1,"ka"  )                                       ],
                     [ 1./2, P("ij|ab"), (g,"klcd"), (t2,"ikac"), (t2,"ljdb")                          ],
                     [ 1./4, I         , (g,"klcd"), (t2,"ijcd"), (t2,"klab")                          ],
                     [-1./2, P("ab")   , (g,"klcd"), (t2,"ijac"), (t2,"klbd")                          ],
                     [-1./2, P("ij")   , (g,"klcd"), (t2,"ikab"), (t2,"jlcd")                          ],
                     [ 1./2, P("ab")   , (g,"klij"), (t1,"ka"  ), (t1,"lb"  )                          ],
                     [ 1./2, P("ij")   , (g,"abcd"), (t1,"ic"  ), (t1,"jd"  )                          ],
                     [-1.  , P("ij|ab"), (g,"kbic"), (t1,"ka"  ), (t1,"jc"  )                          ],
                     [-1.  , P("ij")   , (g,"klci"), (t1,"kc"  ), (t2,"ljab")                          ],
                     [ 1.  , P("ab")   , (g,"kacd"), (t1,"kc"  ), (t2,"ijdb")                          ],
                     [ 1.  , P("ij|ab"), (g,"akdc"), (t1,"id"  ), (t2,"jkbc")                          ],
                     [ 1.  , P("ij|ab"), (g,"klic"), (t1,"la"  ), (t2,"jkbc")                          ],
                     [ 1./2, P("ij")   , (g,"klcj"), (t1,"ic"  ), (t2,"klab")                          ],
                     [-1./2, P("ab")   , (g,"kbcd"), (t1,"ka"  ), (t2,"ijcd")                          ],
                     [-1./2, P("ij|ab"), (g,"kbcd"), (t1,"ic"  ), (t1,"ka"  ), (t1,"jd"  )             ],
                     [ 1./2, P("ij|ab"), (g,"klcj"), (t1,"ic"  ), (t1,"ka"  ), (t1,"lb"  )             ],
                     [-1.  , P("ij")   , (g,"klcd"), (t1,"kc"  ), (t1,"id"  ), (t2,"ljab")             ],
                     [-1.  , P("ab")   , (g,"klcd"), (t1,"kc"  ), (t1,"la"  ), (t2,"ijdb")             ],
                     [ 1./4, P("ij")   , (g,"klcd"), (t1,"ic"  ), (t1,"jd"  ), (t2,"klab")             ],
                     [ 1./4, P("ab")   , (g,"klcd"), (t1,"ka"  ), (t1,"lb"  ), (t2,"ijcd")             ],
                     [ 1.  , P("ij|ab"), (g,"klcd"), (t1,"ic"  ), (t1,"lb"  ), (t2,"kjad")             ],
                     [ 1./4, P("ij|ab"), (g,"klcd"), (t1,"ic"  ), (t1,"ka"  ), (t1,"jd"  ), (t1,"lb"  )])

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
