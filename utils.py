#!/usr/bin/env python
__doc__ = '''
General Utilities - utils.py

Some of these are common functions / structures
within both error.py and curve.py

Nicholas Turner, Jingpeng Wu June-October 2015
'''

import timeit
import data_prep, repres
from metrics import *
import numpy as np


metric_fns = [
    (
      "Rand F-Score",
        lambda om, full_name :
            om_metric(om_rand_f_score, full_name, om, True, True, 0.5)
    ),
    (
      "Rand Error"  ,
        lambda om, full_name :
            om_metric(om_rand_error, full_name, om, True, True)
    ),
    (
      "VI F-Score"  ,
        lambda om, full_name :
            om_metric(om_variation_f_score, full_name, om, True, True, 0.5)
    ),
    (
      "Variation of Information" ,
        lambda om, full_name :
            om_metric(om_variation_information, full_name, om, True, True)
    )
]


prep_fns = [
    ( "2D Relabeling", data_prep.relabel2d ),
    ( "Foreground Restriction", data_prep.foreground_restriction )
]


def parse_fns( fns, bools ):
    return [fns[i] for i in range(len(bools)) if bools[i]]


def run_preprocessing( seg1, seg2, fns ):
    for (name,fn) in fns:
      seg1, seg2 = fn(seg1, seg2)
    return seg1, seg2


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

def crop(vol, target_shape, pick_right=None):
    '''Currently only returns value of crop3d'''
    return crop3d(vol, target_shape, pick_right=pick_right)


def crop3d(vol, target_shape, round_up=None, pick_right=None):
	'''
	Crops the input 3d volume to fit to the target 3d shape

	round_up: Whether to crop an extra voxel in the case of an odd dimension
	difference
	pick_right: Whether to prefer keeping the earlier index voxel in the case of
	an odd dimension difference
	'''
	dim_diffs = np.array(vol.shape) - np.array(target_shape)

	#Error checking
	odd_dim_diff_exists = any([dim_diffs[i] % 2 == 1 for i in range(len(dim_diffs))])
	if odd_dim_diff_exists and round_up == None and pick_right == None:
		raise ValueError('Odd dimension difference between volume shape and target' +
	    				 ' with no handling specified')

	if any([vol.shape[i] < target_shape[i] for i in range(len(target_shape))]):
		raise ValueError('volume already smaller that target volume!')


	#Init
	margin = np.zeros(dim_diffs.shape)
	if round_up:
	    margin = np.ceil(dim_diffs / 2.0).astype(np.int)

	#round_up == False || round_up == None
	elif pick_right != None:
	    #voxel selection option will handle the extra
	    margin = np.ceil(dim_diffs / 2.0).astype(np.int)

	else: #round_up == None and pick_right == None => even dim diff
	    margin = dim_diffs / 2

	zmin = margin[0]; zmax = vol.shape[0] - margin[0]
	ymin = margin[1]; ymax = vol.shape[1] - margin[1]
	xmin = margin[2]; xmax = vol.shape[2] - margin[2]

	#THIS SECTION NOT ENTITRELY CORRECT YET
	# DOESN'T TAILOR 'SELECTION' TO AXES WITH THE ODD DIM DIFFS
	if odd_dim_diff_exists and pick_right:

	    zmax += 1; ymax += 1; xmax += 1

	elif odd_dim_diff_exists and pick_right != None:
	    #pick_right == False => pick_left

	    zmin -= 1; ymin -= 1; xmin -= 1

	return vol[zmin:zmax, ymin:ymax, xmin:xmax]
