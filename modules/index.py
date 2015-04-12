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

    def get_zeros_block(self, index):
      ranges = [ self.rangedict[char] for char in index ]
      shape = tuple(stop-start for start, stop in ranges)
      return ArrayBlock( np.zeros(shape), ranges )
 
    def trim_block(self, block, index):
      if not type(block) is ArrayBlock: block = ArrayBlock( block, [ (0, self.dim) ] * block.ndim )
      subblock      = self.get_zeros_block(index)
      slices        = get_slices(subblock.axisranges, block.axisranges)
      subblock      = block[slices]
      return subblock

    def extend_block(self, subblock, index):
      block         = self.get_zeros_block(index)
      slices        = get_slices(subblock.axisranges, block.axisranges)
      block[slices] = subblock
      return block
      

