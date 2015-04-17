import numpy as np

class ArrayBlock:

    def __init__(self, array, ranges):
      if not type(array) in (np.ndarray, np.matrix):
        raise Exception('Incorrect type {} for ArrayBlock.array'.format(type(array)))
      if not check_ranges(array.shape, ranges):
        raise Exception('Inconsistent shape {} for block ranges {}'.format(array.shape, ranges))
      self.array  = array
      self.ranges = ranges

    def __mul__(self, other):
      if isinstance(other, ArrayBlock):
        return ArrayBlock(other.array * self.array, self.ranges)
      else:
        return ArrayBlock(other       * self.array, self.ranges)

    def __rmul__(self, other):
      if isinstance(other, ArrayBlock):
        return ArrayBlock(self.array * other.array, self.ranges)
      else:                              
        return ArrayBlock(self.array * other      , self.ranges)

    def __add__(self, other):
      if isinstance(other, ArrayBlock):
        return ArrayBlock(other.array + self.array, self.ranges)
      else:
        return ArrayBlock(other       + self.array, self.ranges)

    def __radd__(self, other):
      if isinstance(other, ArrayBlock):
        return ArrayBlock(other.array + self.array, self.ranges)
      else:
        return ArrayBlock(other       + self.array, self.ranges)

    def __sub__(self, other):
      if isinstance(other, ArrayBlock):
        return ArrayBlock(other.array - self.array, self.ranges)
      else:
        return ArrayBlock(other       - self.array, self.ranges)

    def __rsub__(self, other):
      if isinstance(other, ArrayBlock):
        return ArrayBlock(other.array - self.array, self.ranges)
      else:
        return ArrayBlock(other       - self.array, self.ranges)

    def transpose(self, axistuple):
      return ArrayBlock(self.array.transpose(axistuple), [self.ranges[axis] for axis in axistuple])

    def trim_block(self, ranges):
      slices = get_slices(ranges, self.ranges)
      return ArrayBlock(self.array[slices], ranges)

    def extend_block(self, ranges):
      slices = get_slices(self.ranges, ranges)
      array  = np.zeros(tuple(stop-start for start, stop in ranges))
      array[slices] = self.array
      return ArrayBlock(array, ranges)


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

