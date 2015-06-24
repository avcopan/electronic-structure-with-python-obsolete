from meinsum.permutation import identity, transpose, permute

def test_permutation_identity():
  assert [(sgn, per('ijab'))   for sgn, per in identity            ] == [( 1, 'ijab'  )]
def test_permutation_transpose():
  assert [(sgn, per('ijab'))   for sgn, per in transpose('ij|ab')  ] == [( 1, 'ijab'  ),
                                                                         ( 1, 'abij'  )]
def test_permutation_permute_1():
  assert [(sgn, per('ijk'))    for sgn, per in permute('ijk')      ] == [( 1, 'ijk'   ),
                                                                         (-1, 'ikj'   ),
                                                                         (-1, 'jik'   ),
                                                                         ( 1, 'jki'   ),
                                                                         ( 1, 'kij'   ),
                                                                         (-1, 'kji'   )]
def test_permutation_permute_2():
  assert [(sgn, per('ijk'))    for sgn, per in permute('ij/k')     ] == [( 1, 'ijk'   ),
                                                                         (-1, 'ikj'   ),
                                                                         (-1, 'kji'   )]
def test_permutation_permute_3():
  assert [(sgn, per('ijab'))   for sgn, per in permute('ij|ab')    ] == [( 1, 'ijab'  ),
                                                                         (-1, 'ijba'  ),
                                                                         (-1, 'jiab'  ),
                                                                         ( 1, 'jiba'  )]
def test_permutation_permute_4():
  assert [(sgn, per('ijkabc')) for sgn, per in permute('ij/k|ab/c')] == [( 1, 'ijkabc'),
                                                                         (-1, 'ijkacb'),
                                                                         (-1, 'ijkcba'),
                                                                         (-1, 'ikjabc'),
                                                                         ( 1, 'ikjacb'),
                                                                         ( 1, 'ikjcba'),
                                                                         (-1, 'kjiabc'),
                                                                         ( 1, 'kjiacb'),
                                                                         ( 1, 'kjicba')]
