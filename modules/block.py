import numpy as np

class ArrayBlock:
    def __init__(self, array, ranges):
      if not check_ranges(array.shape, ranges):
        raise Exception('Inconsistent shape {} for block ranges {}'.format(array.shape, ranges))
      self.array  = array
      self.ranges = ranges


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

