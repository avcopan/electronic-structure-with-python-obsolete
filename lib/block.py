import numpy as np

def zeros(ranges, arrayshape):
  block = np.zeros(get_shape(ranges))
  return ArrayBlock(block, ranges, arrayshape)

def trim(arrblock, ranges):
  arrblock      = asblock(arrblock)
  slices        = get_slices(ranges, arrblock.ranges)
  block         = arrblock.block[slices]
  return ArrayBlock(block, ranges, arrblock.arrayshape)

def extend(arrblock, ranges):
  arrblock      = asblock(arrblock)
  slices        = get_slices(arrblock.ranges, ranges)
  block         = np.zeros(get_shape(ranges))
  block[slices] = arrblock.block
  return ArrayBlock(block, ranges, arrblock.arrayshape)


class ArrayBlock:

    __array_priority__ = 1000 # force my overloaded operations to be called

    def __init__(self, block, ranges, arrayshape):
      check_args(block, ranges, arrayshape)
      self.block       = block
      self.shape       = block.shape
      self.ranges      = ranges
      self.arrayshape  = arrayshape
      self.arrayranges = [(0, dim) for dim in arrayshape]
      self.ndim        = len(arrayshape)

    def view(self):
      if not self.shape in [self.arrayshape, ()]: return self
      elif self.ndim is 2:                        return np.matrix(self.block)
      else:                                       return self.block

    def transpose(self, *axes):
      if axes is (): axes = tuple(range(self.block.ndim)[::-1])
      if type(axes[0]) is tuple: axes = axes[0]
      block      = self.block.transpose(*axes)
      ranges     = [self.ranges[axis] for axis in axes]
      arrayshape = tuple(self.arrayshape[axis] for axis in axes)
      return ArrayBlock(block, ranges, arrayshape)

    def get_T(self): return self.transpose()
    T = property(get_T, None)

    def __pos__(self): return ArrayBlock(+self.block, self.ranges, self.arrayshape)
    def __neg__(self): return ArrayBlock(-self.block, self.ranges, self.arrayshape)
    def __add__ (self, other): return self.binary_operation(other, np.ndarray.__add__ )
    def __sub__ (self, other): return self.binary_operation(other, np.ndarray.__sub__ )
    def __mul__ (self, other): return self.binary_operation(other, np.ndarray.__mul__ )
    def __div__ (self, other): return self.binary_operation(other, np.ndarray.__div__ )
    def __radd__(self, other): return self.binary_operation(other, np.ndarray.__radd__)
    def __rsub__(self, other): return self.binary_operation(other, np.ndarray.__rsub__)
    def __rmul__(self, other): return self.binary_operation(other, np.ndarray.__rmul__)
    def __rdiv__(self, other): return self.binary_operation(other, np.ndarray.__rdiv__)

    def binary_operation(self, other, operation):
      if hasattr(other, 'shape') and not other.shape is ():
        other  = asblock(other)
        ranges = get_containing_ranges(self.ranges, other.ranges)
        block1 = extend( self, ranges).block
        block2 = extend(other, ranges).block
      else:
        ranges = self.ranges
        block1 = self.block
        block2 = other
      block = operation(block1, block2)
      return ArrayBlock(block, ranges, self.arrayshape).view()


def asblock(array):
  if isinstance(array, ArrayBlock): return array
  ranges = [(0, dim) for dim in array.shape]
  return ArrayBlock(array, ranges, array.shape)

def asarray(arrblock):
  return extend(arrblock, arrblock.arrayranges).view()

def get_shape(ranges):
  return tuple(stop-start for start, stop in ranges)

def get_containing_ranges(ranges1, ranges2):
  return [(min(start1,start2), max(stop1,stop2)) for (start1,stop1),(start2,stop2) in zip(ranges1, ranges2)]

def get_slices(subranges, supranges=None):
  if supranges is None: supranges = [(0, substop) for substart, substop in subranges]
  return tuple( get_slice(subrange, suprange) for subrange, suprange in zip(subranges, supranges) )

def get_slice((substart, substop), (supstart, supstop)):
  if not supstart <= substart <= substop <= supstop:
    raise Exception('Block range {} out of bounds {}'.format((substart, substop), (supstart, supstop)))
  return slice(substart-supstart, substop-supstart)

def check_args(block, ranges, arrayshape):
  if not type(block) in (np.ndarray, np.matrix):
    raise Exception('Incorrect type {} for ArrayBlock.block'.format(type(block)))
  if not all(0 <= blkdim <= arrdim for blkdim, arrdim in zip(block.shape, arrayshape)):
    raise Exception('Inconsistent block shape {} for array shape {}'.format(block.shape, arrayshape))
  if not all(stop-start == blkdim for (start,stop), blkdim in zip(ranges, block.shape)):
    raise Exception('Inconsistent shape {} for block ranges {}'.format(block.shape, ranges))
