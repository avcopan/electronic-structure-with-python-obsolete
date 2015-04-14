import numpy as np
from block import ArrayBlock, get_slices

'''Index class -- allows specification of index ranges for einsum/tensordot,
   as well as general linear combinations of tensordot contractions with
   permutation and weighting coefficients.'''
class Index:

    def __init__(self, dim, genchars):
      self.dim = dim
      self.genchars = genchars
      self.rangedict = {}
      self.add_index_range(0, dim, genchars)
 
    def add_index_range(self, start, stop, chars):
      if 0 <= start <= stop <= self.dim:
        self.rangedict.update( { char:(start,stop) for char in chars } )
      else:
        raise Exception('Inconsistent index range {} for dimension {:d}'.format((start, stop), self.dim))

    def get_ranges(self, index):
      return [ self.rangedict[char] for char in index ]

    def get_zeros_block(self, index):
      ranges = self.get_ranges(index)
      shape  = tuple(stop-start for start, stop in ranges)
      return ArrayBlock( np.zeros(shape), ranges )
 
    def trim_block(self, block, index):
      if type(block) in [np.ndarray, np.matrix]: block = ArrayBlock( block, [ (0, self.dim) ] * block.ndim )
      subranges = self.get_ranges(index)
      slices    = get_slices(subranges, block.ranges)
      return ArrayBlock( block.array[slices], subranges )

    def extend_block(self, subblock, index):
      block               = self.get_zeros_block(index)
      slices              = get_slices(subblock.ranges, block.ranges)
      block.array[slices] = subblock.array
      return block

    def einsum(self, targetindex, *blockindexpairs):
      arrayarg    = tuple( self.trim_block(block, index).array for block, index in blockindexpairs )
      indexarg    = ",".join( index for block, index in blockindexpairs ) + "->" + targetindex
      array       = np.einsum(indexarg,*arrayarg)
      ranges      = self.get_ranges(targetindex)
      return ArrayBlock(array, ranges)

    def eindot(self, targetindex, *blockindexpairs):
      pairs        = [ (self.trim_block(block,index).array, index) for block, index in blockindexpairs ]
      array, index = pairs.pop(0)
      for pair in pairs: array, index = tensordot((array, index), pair)
      array        = reorder_axes(array,targetindex,index)
      ranges       = self.get_ranges(targetindex)
      return ArrayBlock(array, ranges)

    def meinsum(self, targetindex, coefficient, permutations, *blockindexpairs):
      array  = coefficient * self.eindot(targetindex, *blockindexpairs).array
      array  = sum( parity * reorder_axes(permute(targetindex), targetindex, array)
                    for parity, permute in permutations )
      ranges = self.get_ranges(targetindex)
      return ArrayBlock(array, ranges)

    def meinsums(self, targetindex, *meinsumargs):
      array  = sum( self.meinsum(targetindex, *tuple(meinsumarg)) for meinsumarg in meinsumargs )
      ranges = self.get_ranges(targetindex)
      return ArrayBlock(array, ranges)

    def meinblock(self, targetindex, *meinsumargs):
      array  = sum( self.extend_block(self.meinsum(*tuple(meinsumarg)), targetindex) for meinsumarg in meinsumargs )
      ranges = self.get_ranges(targetindex)
      return ArrayBlock(array, ranges)

    



def tensordot((array0, index0), (array1, index1)):
  axes0 = [ index0.index(char) for char in index0 if char in index1 ]
  axes1 = [ index1.index(char) for char in index0 if char in index1 ]
  array = np.tensordot(array0, array1, (axes0, axes1))
  index = ''.join( char for char in index0+index1 if not (char in index0 and char in index1) )
  return array, index

def reorder_axes(array, newindex, oldindex):
  axistuple = tuple( oldindex.index(char) for char in newindex )
  return array.transpose(axistuple)
  

