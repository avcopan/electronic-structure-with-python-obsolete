from meinsum.permutation import permute as P

def test_permutation_permute_1():
  assert set((sgn, per('ijk')) for sgn, per in P('ijk')) ==\
         set([(1, 'ijk'), (-1, 'ikj'), (-1, 'jik'), (1, 'jki'), (1, 'kij'), (-1, 'kji')])


def test_permutation_permute_2():
  assert set((sgn, per('ijab')) for sgn, per in P('ij|ab')) ==\
         set([(1, 'ijab'), (-1, 'ijba'), (-1, 'jiab'), (1, 'jiba')])
