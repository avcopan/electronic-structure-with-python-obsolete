import numpy as np

class ArrayBlock(np.ndarray):

    def __new__(cls, array, axisranges):
      if not check_ranges(array.shape, axisranges):
        raise Exception('Inconsistent shape {} for block ranges {}'.format(array.shape, axisranges))
      block = np.asarray(array).view(cls)
      block.axisranges = axisranges
      return block
  
    def __array_finalize__(self, block):
      if block is None: return
      self.axisranges = getattr(block, 'axisranges', None)

    def __array_wrap__(self, outblock, context=None):
      if check_ranges( outblock.shape, outblock.axisranges ):
        return np.ndarray.__array_wrap__(self, outblock, context)
      else:
        return np.array(outblock)

def check_ranges(shape, ranges):
  for i, (start, stop) in enumerate(ranges):
    if not stop-start == shape[i]:
      return False
  return True

def get_slices(subranges, blkranges):
  return tuple( get_slice(*subrange+blkrange) for subrange, blkrange in zip(subranges, blkranges) )

def get_slice(substart, substop, blkstart, blkstop):
  if blkstart <= substart <= substop <= blkstop:
    return slice(substart-blkstart, substop-blkstart)
  else:
    raise Exception('Block range ({:d},{:d}) out of bounds ({:d},{:d})'.format(substart, substop, blkstart, blkstop))

