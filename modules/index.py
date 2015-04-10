import numpy as np

'''Index class -- allows specification of index ranges for numpy.einsum, as
   well as general linear combinations of numpy.einsum contractions with
   permutation and weighting coefficients.'''
class Index:

  def __init__(self, ngen, nocc, genstring, occstring, virstring):
    slicedict = {}
    slicedict.update( { gen:slice(0   , ngen) for gen in genstring } )
    slicedict.update( { occ:slice(0   , nocc) for occ in occstring } )
    slicedict.update( { vir:slice(nocc, ngen) for vir in virstring } )
    self.ngen, self.nocc, self.slicedict = ngen, nocc, slicedict

  def indexslice(self, index):
    return tuple( self.slicedict(char) for char in index )

  



class ArrayBlock(np.ndarray):

  def __new__(cls, array, index):
    obj = np.asarray(array).view(cls)
    obj.index = index
    return obj
 
  def __array_finalize__(self, obj):
    if obj is None: return
    self.index = getattr(obj, 'index', None)

  def __array_wrap__(self, array, context=None):
    return np.asarray(array)


