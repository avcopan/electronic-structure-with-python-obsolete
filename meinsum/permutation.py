import itertools as it
import numpy     as np
import string
from tensorshuffle.permutations import BlockPermutations

def get_block_permutations_from_index_string(index_string):
  if not '/' in index_string: index_string = '/'.join(index_string)
  index_equivalent_subsets = [list(equivalent_index_string) for equivalent_index_string in index_string.split('/')]
  elements = tuple(sum(index_equivalent_subsets, []))
  composition = tuple(len(index_equivalent_subset) for index_equivalent_subset in index_equivalent_subsets)
  return BlockPermutations(elements, composition).iter_permutations_with_signature()

def make_string_permuter(reference_string, permuted_string):
  return lambda s: string.translate(s, string.maketrans(reference_string, permuted_string))

def permute(index_strings_with_bars):
  index_strings = index_strings_with_bars.split('|')
  reference_string = string.translate(index_strings_with_bars, None, '/|')
  pools = [get_block_permutations_from_index_string(index_string) for index_string in index_strings]
  for prod in it.product(*pools):
    signs, permuted_subsets = zip(*prod)
    sign = np.product(signs)
    permuted_string = ''.join(sum(permuted_subsets, ()))
    permuter = make_string_permuter(reference_string, permuted_string)
    yield sign, permuter

identity = [(+1, lambda s:s)]

def transpose(index_string_with_bar):
  try:
    upper_indices, lower_indices = index_string_with_bar.split('|')
    reference_string = upper_indices + lower_indices
    transpose_string = lower_indices + upper_indices
    return identity + [(+1, make_string_permuter(reference_string, transpose_string))]
  except:
    raise Exception("Invalid argument {:s} passed to transpose().".format(index_string_with_bar))

