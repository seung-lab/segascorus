#!/usr/bin/env python
__doc__ = '''
General Utilities - utils.py

Nicholas Turner, Jingpeng Wu June-October 2015
'''

import timeit
import repres

def choose_two(n):
	'''
	Returns n choose 2. A faster version for np arrays
	is implemented within the cy module
	'''
	return (n / 2.0) * (n-1)


def calc_overlap_matrix(seg1, seg2, split_zeros):
	'''Calculates the overlap matrix of two segmentations'''

	print("Finding overlap matrix...")
	start = timeit.default_timer()
	om = repres.OverlapMatrix( seg1.ravel(), seg2.ravel(), "coo", split_zeros )
	end = timeit.default_timer()
	print("Completed in %f seconds" % (end-start))


	return om


def print_metrics(metrics):
	keys = list(metrics.keys())
	keys.sort()

	for key in keys:
		print("{}: {}".format(key, metrics[key]))


def overflow_warning_check(n_val):
	'''
	The Python numbers.Integral class can represent more values than NumPy arrays,
	so it's convenient to use the square/choose_two operation above to check for overflow
	errors (which can be silent bugs under Cython).
	'''

	warning_string = ("WARNING: total number of pairs exceeds bit length.\n",
		" You may see overflow errors.\n")

	if DTYPE == 'uint64':
		if (n_val ** 2) > 2 ** 32:
			print(warning_string)
	if DTYPE == 'uint32':
		if (n_val ** 2) > 2 ** 16:
			print(warning_string)


