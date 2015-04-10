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


class BlockArray(np.ndarray):

  def __new__(cls, array, index):
    obj = np.asarray(array).view(cls)
    obj.index = index
    return obj
 
  def __array_finalize__(self, obj):
    if obj is None: return
    self.index = getattr(obj, 'index', None)

  def __array_wrap__(self, array, context=None):
    return np.ndarray.__array_wrap__(self, array, context)


