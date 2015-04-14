import itertools as it
from string import maketrans

Identity = [( 1, lambda x: x )]

def Permute(barstring):
  strings = tuple(barstring.split("|"))
  refstring  = ''.join(strings).replace('/','')
  permutationlists = tuple( string_to_permutationlist(string) for string in strings )
  for p in it.product( *permutationlists ):
    perstring = ''.join(subperstrings for subperstrings in p)
    yield ( parity(refstring, perstring), lambda x: x.translate(maketrans(refstring, perstring)) )

def Transpose(barstring):
  lower, upper = tuple(barstring.split("|")) # argument should have the form "pqr|stu"
  return [( 1, lambda x: x.translate(maketrans(lower+upper,lower+upper)) ),
          ( 1, lambda x: x.translate(maketrans(lower+upper,upper+lower)) )]



'''HELPER FUNCTIONS'''

def string_to_permutationlist( string ):
  slashstring = string if '/' in string else '/'.join(list(string))
  return [ permutation for permutation in restricted_permutations( slashstring ) ]

def restricted_permutations( slashstring ):
  substrings = slashstring.split('/')
  refstring  = ''.join(substrings)
  poolbyposition = tuple( refstring.replace(substring,char) for substring in substrings for char in substring )
  # for "ij/k": the allowed values at position 1 are i and k, so poolbyposition[1] = 'ik'
  n = len(refstring)
  for p in it.product( *poolbyposition ): # cartesian product of position values -- only keep the permutations
    if len(set(p)) == n: yield ''.join(p)

def parity( refstring, perstring ):
  n = len(refstring)
  sgn = 1.0
  for i in range(n): # loop over string positions, counting the number of transposed elements
    perchar, refchar = perstring[i], refstring[i]
    if perchar != refchar:
      sgn *= -1.0 # flip sign, then remove transposition by swapping characters to reference position
      perstring = perstring.translate(maketrans( perchar+refchar, refchar+perchar ))
  return sgn



'''
Permute() function

   Explanation for future reference: for an argument *strings = "ijk", "ab/cd", ...
   permutationlists will be a tuple of lists returned by string_to_permutationlist
   calls
           string_to_permutationlist("ijk")
           string_to_permutationlist("ab/cd") ...
   It will look like this:
           ( ["ijk", "ikj", "jik", "jki", "kij", "kji"],
             ["abcd", "acbd", "adcb", "cbad", "cdab", "cdba", "dbca", "dcab", "dcba"],
             .... )
   We then loop over the Cartesian product of these lists to determine
           P("ijk|ab/cd|...") = P("ijk") * P("ab/cd") * ...
   which looks like this:
           ("ijk", "abcd", ...),
           ("ijk", "acbd", ...),
           ("ijk", "adcb", ...),
           ("ijk", "cbad", ...),
           ("ijk", "cdab", ...),
           ("ijk", "cdba", ...),
           ("ijk", "dbca", ...),
           ("ijk", "dcab", ...),
           ("ijk", "dcba", ...),
           ("ikj", "abcd", ...), ...
   Then we join the elements in each tuple and determine the parity and translation
   table from each overall permutation, by comparing with the reference string 
   "ijkabcd...".  The parity gets passed back as a float and the permutation itself
   gets passed as a string translation operator.

   To test it, use

def check_permutations(targetstring, barstring):
  for parity, permute in Permute(barstring):
    print '{:2d} {:s}'.format( int(parity), permute(targetstring) )

   For example:

>>> check_permutations("ijab","ij|ab")
 1 ijab
-1 ijba
-1 jiab
 1 jiba

'''
