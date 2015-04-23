import numpy     as np
import itertools as it
import lib.block as bl

class Index:

    def __init__(self, dim, genchars):
      self.dim       = dim
      self.genchars  = genchars
      self.rangedict = {}
      self.add_index_range(0, dim, genchars)

    def add_index_range(self, start, stop, chars):
      check_args(self, start, stop, chars)
      self.rangedict.update({char:(start, stop) for char in chars})

    def ranges       (self, index): return [self.rangedict[char] for char in index]
    def arrayshape   (self, index): return (self.dim,) * len(index)
    def zeros        (self, index): return bl.zeros (self.ranges(index), self.arrayshape(index))
    def trim  (self, array, index): return bl.trim  (array, self.ranges(index))
    def extend(self, array, index): return bl.extend(array, self.ranges(index))

    def get_cost(self, *arrayindexpairs):
      arrays, indices = zip(*arrayindexpairs)
      cost = sum( np.product([stop-start for start, stop in self.ranges(index)])
                  for index in accumulate_indices_pairwise(indices) )
      return cost

    def optimize_order(self, *arrayindexpairs):
      reorderedpairs = min(it.permutations(arrayindexpairs), key=lambda x:self.get_cost(*x))
      return [(self.trim(array, index).block, index) for array, index in reorderedpairs]

    def einsum(self, targetindex, *arrayindexpairs):
      arrayarg     = tuple(self.trim(array, index).block for array, index in arrayindexpairs)
      indexarg     = ','.join(index for array, index in arrayindexpairs) + '->' + targetindex
      block        = np.einsum(indexarg, *arrayarg)
      ranges       = self.ranges    (targetindex)
      arrayshape   = self.arrayshape(targetindex)
      return bl.ArrayBlock(block, ranges, arrayshape).view()

    def eindot(self, targetindex, *arrayindexpairs):
      pairs        = self.optimize_order(*arrayindexpairs)
      array, index = pairs.pop(0)
      for pair in pairs: array, index = tensordot((array, index), pair)
      block        = transpose(array, targetindex, index)
      ranges       = self.ranges    (targetindex)
      arrayshape   = self.arrayshape(targetindex)
      return bl.ArrayBlock(block, ranges, arrayshape).view()

    def meinsum(self, targetindex, coefficient, permutations, *arrayindexpairs):
      block = coefficient * self.eindot(targetindex, *arrayindexpairs)
      return sum( sgn * transpose(block, p(targetindex), targetindex) for sgn, p in permutations ).view()

    def meinsums(self, targetindex, *args):
      return sum( self.meinsum(targetindex, *arg) for arg in args ).view()

    def meinblock(self, targetindex, *args):
      return sum( self.extend(self.meinsum(*arg), targetindex) for arg in args ).view()


def tensordot((array1, index1), (array2, index2)):
  axes1 = [index1.index(char) for char in index1 if char in index2]
  axes2 = [index2.index(char) for char in index1 if char in index2]
  array = np.tensordot(array1, array2, (axes1, axes2))
  index = get_uncontracted_indices(index1, index2)
  return array, index

def transpose(array, targetindex, index):
  axes = tuple(index.index(char) for char in targetindex)
  return array.transpose(*axes)

def accumulate_indices_pairwise(indextuple):
  indexlist = list(indextuple)
  index     = indexlist.pop(0)
  for nextindex in indexlist:
    contracted = get_contracted_indices  (index, nextindex)
    index      = get_uncontracted_indices(index, nextindex)
    yield index+contracted

def get_uncontracted_indices(index1, index2):
  return ''.join(char for char in     index1+index2  if not (char in index1 and char in index2))

def get_contracted_indices(index1, index2):
  return ''.join(char for char in set(index1+index2) if     (char in index1 and char in index2))

def check_args(indx, start, stop, chars):
  if not 0 <= start <= stop <= indx.dim:
    raise Exception('Inconsistent index range {} for dimension {:d}'.format((start, stop), indx.dim))
  if any(char in indx.rangedict for char in chars):
    raise Exception('Some of the characters in {} have already been assigned ranges'.format(chars))
