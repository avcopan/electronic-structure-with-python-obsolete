import itertools as it
import string    as st

'''helper functions'''

def permute_nonequivalent(string):
  if not '/' in string: string = '/'.join(string)
  subs  = st.split(string, '/')
  ref   = st.translate(string, None, '/')
  pools = [st.replace(ref, sub, char) for sub in subs for char in sub]
  for prod in it.product(*pools):
    if len(prod) == len(set(prod)):
      yield prod

def parity(ref, per):
  sgn, per = +1, list(per)
  for c in ref:
    i, j = ref.index(c), per.index(c)
    sgn *= -1 if not i is j else +1
    per[i], per[j] = per[j], per[i]
  return sgn

def pair(ref, per, sign=False):
  sgn = parity(ref, per) if sign else +1
  pmt = lambda x: st.translate(x, st.maketrans(ref, per))
  return sgn, pmt

'''use these'''

identity = [pair('','')]

def permute(string):
  subs    = st.split(string, '|')
  ref     = st.translate(string, None, '/|')
  subpers = [permute_nonequivalent(sub) for sub in subs]
  for prod in it.product(*subpers):
    per = ''.join(it.chain(*prod))
    yield   pair(ref, per, sign=True)

def transpose(string):
  subs    = st.split(string, '|')
  ref     = st.translate(string, None, '|')
  for per in it.permutations(subs):
    per = ''.join(per)
    yield   pair(ref, per)

