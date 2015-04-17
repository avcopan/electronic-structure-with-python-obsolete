import numpy as np

class ArrayBlock:

    def __init__(self, array, ranges):
      if not type(array) in (np.ndarray, np.matrix):
        raise Exception('Incorrect type {} for ArrayBlock.array'.format(type(array)))
      if not check_ranges(array.shape, ranges):
        raise Exception('Inconsistent shape {} for block ranges {}'.format(array.shape, ranges))
      self.array  = array
      self.ranges = ranges

    def binaryop(self, other, operation):
      if not isinstance(other, ArrayBlock):
        return ArrayBlock(operation(self.array, other), self.ranges)
      elif self.ranges == other.ranges:
        return ArrayBlock(operation(self.array, other.array), self.ranges)
      else:
        ranges = [(min(start1,start2), max(stop1,stop2)) for (start1,start2),(stop1,stop2) in zip(self.ranges, other.ranges)]

    def __pos__(self): return ArrayBlock(+self.array, self.ranges)
    def __neg__(self): return ArrayBlock(-self.array, self.ranges)

    def __add__ (self, other): return self.binaryop(other, np.ndarray.__add__ )
    def __sub__ (self, other): return self.binaryop(other, np.ndarray.__sub__ )
    def __mul__ (self, other): return self.binaryop(other, np.ndarray.__mul__ )
    def __div__ (self, other): return self.binaryop(other, np.ndarray.__div__ )
    def __radd__(self, other): return self.binaryop(other, np.ndarray.__radd__)
    def __rsub__(self, other): return self.binaryop(other, np.ndarray.__rsub__)
    def __rmul__(self, other): return self.binaryop(other, np.ndarray.__rmul__)
    def __rdiv__(self, other): return self.binaryop(other, np.ndarray.__rdiv__)

    def transpose(self, axistuple):
      return ArrayBlock(self.array.transpose(axistuple), [self.ranges[axis] for axis in axistuple])

    def trim_block(self, ranges):
      slices = get_slices(ranges, self.ranges)
      return ArrayBlock(self.array[slices], ranges)

    def extend_block(self, ranges):
      slices = get_slices(self.ranges, ranges)
      array  = zeros(ranges)
      array[slices] = self.array
      return ArrayBlock(array, ranges)


def zeros(ranges):
  array = np.zeros(tuple(stop-start for start, stop in ranges))
  return ArrayBlock(array, ranges)

def get_slices(subranges, blkranges):
  return tuple( get_slice(*subrange+blkrange) for subrange, blkrange in zip(subranges, blkranges) )

def get_slice(substart, substop, blkstart, blkstop):
  if blkstart <= substart <= substop <= blkstop:
    return slice(substart-blkstart, substop-blkstart)
  else:
    raise Exception('Block range ({:d},{:d}) out of bounds ({:d},{:d})'.format(substart, substop, blkstart, blkstop))

def check_ranges(shape, ranges):
  for i, (start, stop) in enumerate(ranges):
    if not stop-start == shape[i]:
      return False
  return True

