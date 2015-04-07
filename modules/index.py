import numpy as np


'''Index class -- allows specification of index ranges for numpy.einsum, as
   well as general linear combinations of numpy.einsum contractions with
   permutation and weighting coefficients.'''
class Index:

  def __init__(self, ngen, nocc, genstring, occstring, virstring):
    self.ngen = ngen
    self.nocc = nocc
    self.nvir = ngen - nocc
    self.gen  = [ char for char in genstring.lower() + genstring.upper() ]
    self.occ  = [ char for char in occstring.lower() + occstring.upper() ]
    self.vir  = [ char for char in virstring.lower() + virstring.upper() ]

  class ArrayBlock(np.ndarray):
