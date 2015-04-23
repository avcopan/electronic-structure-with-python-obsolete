import numpy     as np
import itertools as it
from scipy.misc      import comb
from lib.index       import Index
from lib.permutation import Permute as P

'''Compress/uncompress T-amplitude-like tensors -- nice for DIIS storage of T3 = Tijkabc.
   Compressed tensor is 1d array of unique elements, i.e. i<j<k, a<b<c'''

class Compressor:

    def __init__(self, dim, nocc):
      indx = Index(dim, 'pqr')
      indx.add_index_range(   0, nocc, 'ijk')
      indx.add_index_range(nocc,  dim, 'abc')
      self.indx, self.no, self.nv = indx, nocc, dim - nocc

    def compress(self, T, index):
      indx, no, nv = self.indx, self.no, self.nv
      T, k = indx.trim(T, index).block, len(index)/2
      o  = it.combinations(range(no), k)
      v  = it.combinations(range(nv), k)
      cT = np.zeros(comb(no, k) * comb(nv, k))
      for ind, (oind, vind) in enumerate(it.product(o, v)): cT[ind] = T[oind+vind]
      return cT

    def uncompress(self, cT, index):
      indx, no, nv = self.indx, self.no, self.nv
      bT, k = indx.zeros(index), len(index)/2
      T  = bT.block
      o  = it.combinations(range(no), k)
      v  = it.combinations(range(nv), k)
      for ind, (oind, vind) in enumerate(it.product(o, v)): T[oind+vind] = cT[ind]
      pindex = '|'.join((index[:k], index[k:]))
      return indx.meinsum(index, 1., P(pindex), (bT, index))

