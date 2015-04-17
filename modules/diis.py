import numpy as np
from scipy import linalg as la

class DIIS:

    def __init__(self, maxvecs=6, minvecs=3):
      self.maxvecs = maxvecs
      self.minvecs = minvecs
      self.paramss = []
      self.errorss = []

    def add_vec(self, *paramerrortuples):
      paramss, errorss, maxvecs = self.paramss, self.errorss, self.maxvecs
      params, errors = zip(*paramerrortuples)
      paramss.append(params)
      errorss.append(errors)
      while(len(errors) > maxvecs): paramss.pop(0); errorss.pop(0)

    def get_extrapolation_coefficients(self):
      errors, dim = self.errors, len(self.errors)
      # define blocks of A in Ax=b
      B = np.zeros((m, m))
      for i in range(m):
        for j in range(i,m):
          B[i,j] = B[j,i] = vdot_arraytuples(errorss[i], errorss[j])
      o = np.matrix( [-1.]*m )
      z = np.matrix( [ 0.]   )
      # build A, where A is the block matrix [[B,-1], [-1,0]]
      A = np.bmat( [[B, o.T], [o, z]] )
      b = np.array( [ 0.]*m + [-1.] )
      # solve for x
      x = la.solve(A, b)
      c, errnorm = x[:m], x[m]
      return c, errnorm

    def extrapolate(self):
      paramss, minvecs = self.paramss, self.minvecs
      if len(paramss) < minvecs: return paramss[-1]
      c, errnorm = self.get_extrapolation_coefficients()
      return tuple( sum(ci*vi for ci, vi in zip(c, v)) for v in zip(*paramss) )

def vdot_arraytuples(arraytuple1, arraytuple2):
  return sum( np.vdot(array1, array2) for array1, array2 in zip(arraytuple1, arraytuple2) )
