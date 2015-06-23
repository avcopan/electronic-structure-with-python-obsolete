from meinsum.permutation import Permute as P

def P_test():
  assert set((sgn, per('ijk')) for sgn, per in P('ijk')) ==\
         set([(1, 'ijk'), (-1, 'ikj'), (-1, 'jik'), (1, 'jki'), (1, 'kij'), (-1, 'kji')])


