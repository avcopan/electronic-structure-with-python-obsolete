import numpy        as np
import scipy.linalg as la


class DIIS:

    def __init__(self, maxvecs=6, minvecs=3):
      self.maxvecs = maxvecs
      self.minvecs = minvecs
      self.arrayss = []
      self.errorss = []

    def add_vec(self, *arrayerrorpairs):
      arrayss, errorss, maxvecs = self.arrayss, self.errorss, self.maxvecs
      arrays , errors = zip(*arrayerrorpairs)
      arrayss.append(arrays)
      errorss.append(errors)
      while(len(errorss) > maxvecs): arrayss.pop(0); errorss.pop(0)

    def get_extrapolation_coefficients(self):
      errorss, m = self.errorss, len(self.errorss)
      B = np.zeros((m, m)) # define blocks of A in Ax=b
      for i in range(m):
        for j in range(i, m):
          B[i,j] = B[j,i] = sum(np.vdot(e1.view(np.ndarray), e2.view(np.ndarray))
                                for e1, e2 in zip(errorss[i], errorss[j]))
      o = np.matrix([-1.]*m)
      z = np.matrix([ 0.]  )
      A = np.bmat ([[B, o.T], [o, z]]) # A = [[B, -1], [-1, 0]]
      b = np.array( [ 0.]*m + [-1.]  ) # b = [ 0, ...,    0,  -1]
      x = la.solve(A, b)
      c, errnorm = x[:m], x[m]         # x = [c1, ..., cm-1, err]
      return c, errnorm

    def extrapolate(self):
      arrayss, minvecs = self.arrayss, self.minvecs
      if len(arrayss) < minvecs: return arrayss[-1]
      c, errnorm = self.get_extrapolation_coefficients()
      return tuple(sum(ci*vi for ci, vi in zip(c, v)) for v in zip(*arrayss))

