#!/usr/bin/env python
__doc__ = '''
SeungLab Error Metrics for Volummetric Segmentation

 This module computes the error between two segmentations of a volume.
The metrics are either based on the Rand Index or Variation of Information, and
features a few customizable options. These include:

- Foreground Restriction- computing error over voxels where the
  second segmentation (presumed to be ground truth) != 0.
  This is applied as a default option.

- Splitting '0' Segment into Singletons- The segment id of 0 often
  corresponds to a background segment, however it can also represent
  singleton voxels after processing by watershed. This option re-splits
  all segments labelled 0 into new segment ids, recovering the singletons.
  This is applied as a default option, and disabled by -no_split0

- Boundary Thinning- (not complete yet)

Inputs

	- First Segmentation File (seg1, as .tif file)
	- Second Segmentation File (seg2, as .tif file)
	  This should be the "ground truth" segmentation if applicable
	- Foreground Restriction (optional flag -nofr, default=on)
	- Boundary Thinning (optional flag -bt, not complete yet)
	- Split 0 Segment (optional flag -no_split0, default=on)

	- Metric Types
	  (these are all calculated by default)
		- Rand F Score - ISBI 2012 Error Metric
		- Rand Error - 1 - RandIndex
		- Variation F Score - ISBI 2012 Information Theoretic Error Metric
		- Variation of Information
	- 2D Metric Types
	  (not calculated by default)
		- Rand F Score         -rfs2d
		- Rand Error           -re2d
		- Variation F Score    -vifs2d
		- Variation of Info    -vi2d

Main Outputs:

	-Reports requested error metrics by a print

Nicholas Turner, Jingpeng Wu June-October 2015
'''

import timeit
import argparse
from os import path

#Dependencies
import tifffile
import numpy as np
import scipy.sparse as sp

#Cython functions
import cy

#This should be an int type, and consistent with cy.pyx
DTYPE= cy.DTYPE

def choose_two(n):
	'''
	Returns n choose 2. A faster version for np arrays
	is implemented within the cy module
	'''
	return (n / 2.0) * (n-1)


def om_rand_f_score(om, alpha=0.5, merge_score=False, split_score=False):
	'''
	Calculates the Rand F-Score (from ISBI2012) of an unnormalized (raw counts) overlap matrix

	Can also return split and/or merge error separately assuming that the "ground-truth"
	segmentation is represented in the 2nd axis (columns)

	In order to use multiple alpha values, pass in an np.array of the desired values

	alpha corresponds to the weight of the split score, where 1-alpha
	refers to the merge score
	'''

	col_counts = om.sum(0) #t_j * N
	row_counts = om.sum(1) #s_i * N

	#float conversion for division below
	N = float( col_counts.sum() )

	#implicit conversion to float64
	t_term = np.sum( np.square(col_counts / N) )
	s_term = np.sum( np.square(row_counts / N) )

	#p term requires a bit more work with sparse matrix
	p_term = np.sum( np.square(np.copy(om.data) / N) )

	split_sc = (p_term / t_term)
	merge_sc = (p_term / s_term)

	full_sc = p_term / (alpha * s_term + (1-alpha) * t_term)	

	if split_score and merge_score:
		return full_sc, merge_sc, split_sc
	elif split_score:
		return full_sc, split_sc
	elif merge_score:
		return full_sc, merge_sc
	else:
		return full_sc


def om_rand_error(om, merge_error=False, split_error=False):
	'''
	Calculates the rand error (= 1 - RandIndex) of an unnormalized (raw counts) overlap matrix

	Can also return split and/or merge error separately assuming that the "ground-truth"
	segmentation is represented in the 2nd axis (columns)
	'''

	#Converting to np.array makes ravel work properly
	# (sp version seems buggy)
	col_counts = np.array( om.sum(0) )
	row_counts = np.array( om.sum(1) )

	#for float division below
	N = float(col_counts.sum())
	#Numpy has built-in warnings for float overflows,
	# but not int, so we may have to make do with our own
	# overflow_warning_check(long(N))

	#TP - True Positive pairs
	#FP - False Positive pairs
	#FN - False Negative pairs

	#Pre-emptive dividing by N and N-1 helps dodge overflows,
	# both by keeping values small, and recruiting 
	# NumPy's float overflow warnings
	TPplusFP_norm = np.sum(
		  (row_counts / N) *
		  ((row_counts - 1) / (N-1))
		)
	TPplusFN_norm = np.sum(
		  (col_counts / N) *
		  ((col_counts - 1) / (N-1))
		)

	c_ij_vals = np.copy(om.data)
	TP_norm = np.sum(
		  (c_ij_vals / N) *
		  ((c_ij_vals - 1) / (N-1))
		)

	split_err = TPplusFN_norm - TP_norm
	merge_err = TPplusFP_norm - TP_norm

	full_err  = TPplusFP_norm + TPplusFN_norm - 2*TP_norm

	if split_error and merge_error:
		return full_err, merge_err, split_err
	elif split_error:
		return full_err, split_err
	elif merge_error:
		return full_err, merge_err
	else:
		return full_err


def om_variation_f_score(om, alpha=0.5, merge_score=False, split_score=False):
	'''
	Calculates the variation of information F-Score (from ISBI2012) of an unnormalized (raw counts) overlap matrix

	Can also return split and/or merge error separately assuming that the "ground-truth"
	segmentation is represented in the 2nd axis (columns)
	
	In order to use multiple alpha values, pass in an np array of the desired values

	alpha corresponds to the weight of the split score, where 1-alpha
	refers to the merge score
	'''

	#Transforming to np.array makes ravel work properly
	# (sp version seems buggy)
	col_counts = np.array( om.sum(0) ) #t_j * N
	row_counts = np.array( om.sum(1) ) #s_i * N

	#float conversion for division below (float64)
	N = float( col_counts.sum() )

	#implicitly converts to float64
	t_j = col_counts / N
	s_i = row_counts / N

	HT = np.sum(cy.shannon_entropy( t_j.ravel() ))
	HS = np.sum(cy.shannon_entropy( s_i.ravel() ))

	if HT == 0:
		print "WARNING: HT equals zero! You will likely see a RuntimeWarning"
	if HS == 0:
		print "WARNING: HS equals zero! You will likely see a RuntimeWarning"

	p_ij_vals = (-1.0) * cy.shannon_entropy( (np.copy(om.data) / N) )
	Hp = np.sum(p_ij_vals)

	I = Hp + HT + HS

	split_sc = I / HS
	merge_sc = I / HT

	full_sc = I / ( (1 - alpha) * HS + alpha * HT )

	if split_score and merge_score:
		return full_sc, merge_sc, split_sc
	elif split_score:
		return full_sc, split_sc
	elif merge_score:
		return full_sc, merge_sc
	else:
		return full_sc


def om_variation_information(om, merge_error=False, split_error=False):
	'''
	Calculates the variation of information of an unnormalized (raw counts) overlap matrix

	Can also return split and/or merge error separately, yet this assumes that the "ground-truth"
	segmentation is represented in the 2nd axis (columns)
	
	In order to use multiple alpha values, pass in a np array of the desired values
	'''

	#Transforming to np.array makes ravel work properly
	# (sp version seems buggy)
	col_counts = np.array( om.sum(0) ) #t_j * N
	row_counts = np.array( om.sum(1) ) #s_i * N

	#for float division below (float64)
	N = float( col_counts.sum() )

	#implicitly converts to float64
	# again, conversion to np.array fixes sp weirdness
	t_j = np.array( col_counts / N )
	s_i = np.array( row_counts / N )

	rows, cols, vals = sp.find(om)
	vals = vals / N #p_ij

	split_err = np.sum( cy.conditional_entropy( vals, cols, t_j.ravel() ) )
	merge_err = np.sum( cy.conditional_entropy( vals, rows, s_i.ravel() ) )

	full_err = split_err + merge_err

	if split_error and merge_error:
		return full_err, merge_err, split_err
	elif split_error:
		return full_err, split_err
	elif merge_error:
		return full_err, merge_err
	else:
		return full_err



def calc_overlap_matrix(seg1, seg2, split_zeros):
	'''Calculates the overlap matrix of two segmentations'''

	print "Finding overlap matrix..."
	start = timeit.default_timer()
	om = cy.overlap_matrix_dok( seg1.ravel(), seg2.ravel(), split_zeros )
	end = timeit.default_timer()
	print "Completed in %f seconds" % (end-start)

	return om


def relabel2d(seg1, seg2):
	'''Relabels segmentations to be 2d for 2d-based error metrics'''
	print "Relabelling segments for 2d metrics..."
	return cy.relabel2d(seg1), cy.relabel2d(seg2)


def foreground_restriction(seg1, seg2):
	'''Performs foreground restriction on seg2's foreground'''

	print "Performing foreground-restriction"
	seg1_fr = seg1[seg2 != 0]
	seg2_fr = seg2[seg2 != 0]

	return seg1_fr, seg2_fr


def om_score(om_score_function, score_name,
	om, merge_score=False, split_score=False, alpha=0.5):
	'''Runs a score function, times the calculation, and returns the result'''
	
	print "Calculating {}...".format(score_name)
	start = timeit.default_timer()
	score = om_score_function(om, alpha, merge_score, split_score)
	end = timeit.default_timer()
	print "Completed in %f seconds" % (end-start)

	return score


def om_error(om_error_function, error_name,
	om, merge_err=False, split_err=False):
	'''Runs an error function, times the calculation, and returns the result'''

	print "Calculating {}...".format(error_name)
	start = timeit.default_timer()
	score = om_error_function(om, merge_err, split_err)
	end = timeit.default_timer()
	print "Completed in %f seconds" % (end-start)
	
	return score


def seg_score(om_score_function, score_name, 
	seg1, seg2, merge_score=False, split_score=False, 
	alpha=0.5, split0=True):
	'''High-level function which handles segmentations'''

	assert seg1.dtype == DTYPE
	assert seg2.dtype == DTYPE

	om = calc_overlap_matrix(seg1, seg2, split0)

	score = om_score(om_score_function, score_name,
		om, merge_score, split_score, alpha)

	return score


def seg_error(om_error_function, error_name,
	seg1, seg2, merge_err=False, split_err=False, split0=True):
	'''High-level function which handles segmentations'''

	assert seg1.dtype == DTYPE
	assert seg2.dtype == DTYPE

	om = calc_overlap_matrix(seg1, seg2, split0)

	score = om_error(om_error_function, error_name,
		om, merge_err, split_err)

	return score

	
#=====================================================================
#Functions for interactive module use

def seg_rand_f_score(seg1, seg2, merge_score=False, split_score=False, 
	alpha=0.5, split0=True):
	'''Computes the Rand F Score for a segmentation'''
	return seg_score(om_rand_f_score, "Rand Score",
		seg1, seg2, merge_score, split_score, alpha, split0)


def seg_rand_error(seg1, seg2, merge_err=False, split_err=False, split0=True):
	'''Computes the Rand Error for a segmentation'''
	return seg_error(om_rand_error, "Rand Error",
		seg1, seg2, merge_err, split_err, split0)


def seg_variation_f_score(seg1, seg2, merge_score=False, split_score=False, 
	alpha=0.5, split0=True):
	'''Computes the Variation of Information F Score for a segmentation'''
	return seg_score(om_variation_f_score, "VI Score",
		seg1, seg2, merge_score, split_score, alpha, split0)


def seg_variation_information(seg1, seg2, merge_err=False, split_err=False, split0=True):
	'''Computes the Variation of Information for a segmentation'''
	return seg_error(om_variation_information, "Variation of Information",
		seg1, seg2, merge_err, split_err, split0)


def seg_fr_rand_f_score(seg1,seg2, merge_score=False, split_score=False, 
	alpha=0.5, split0=True):
	'''Computes the Rand F Score for a segmentation w/ foreground restriction'''
	seg1, seg2 = foreground_restriction(seg1, seg2)
	return seg_rand_f_score(seg1, seg2, merge_score, split_score, alpha, split0)


def seg_fr_rand_error(seg1, seg2, merge_error=False, split_error=False, split0=True):
	'''Computes the Rand Error for a segmentation w/ foreground restriction'''
	seg1, seg2 = foreground_restriction(seg1, seg2)
	return seg_rand_error(seg1, seg2, merge_error, split_error, split0)


def seg_fr_variation_f_score(seg1, seg2, merge_score=False, split_score=False, 
	alpha=0.5, split0=True):
	'''Computes the Variation of Information F Score for a segmentation w/ foreground restriction'''
	seg1, seg2 = foreground_restriction(seg1, seg2)
	return seg_variation_f_score(seg1, seg2, merge_score, split_score, alpha, split0)


def seg_fr_variation_information(seg1, seg2, merge_error=False, split_error=False, split0=True):
	'''Computes the Variation of Information for a segmentation w/ foreground restriction'''
	seg1, seg2 = foreground_restriction(seg1, seg2)
	return seg_variation_information(seg1, seg2, merge_error, split_error, split0)


#== 2d versions ==#
def seg_2d_rand_error(seg1, seg2, merge_err=False, split_err=False, split0=True):

	print "Relabelling segmentations for 2d comparison"
	seg1, seg2 = relabel2d(seg1, seg2)

	return seg_rand_error(seg1, seg2, merge_err, split_err, split0)
#INCOMPLETE

#=====================================================================
#Utility Functions

def print_metrics(metrics):
	keys = metrics.keys()
	keys.sort()

	for key in keys:
		print "{}: {}".format(key, metrics[key])


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
			print warning_string
	if DTYPE == 'uint32':
		if (n_val ** 2) > 2 ** 16:
			print warning_string


def import_tif(filename):
    return tifffile.imread(filename).astype(DTYPE)


def import_h5(filename):
    import h5py

    f = h5py.File(filename)
    return f['/main'].value.astype(DTYPE)


#=====================================================================
#Main function (script functionality)

def main(seg1_fname, seg2_fname, 
	calc_rand_score=True,
	calc_2d_rand_score=False,
	calc_rand_error=True, 
	calc_2d_rand_error=False,
	calc_variation_score=True,
	calc_2d_variation_score=False,
	calc_variation_information=True,
	calc_2d_variation_information=False,
	foreground_restricted=True,
	split_0_segment=True):
	'''
	Script functionality, computes the overlap matrix, computes any specified metrics,
	and prints the results nicely
	'''

	print "Loading Data..."
	seg1 = import_tif(seg1_fname)
	seg2 = import_tif(seg2_fname)

	results = {}

	#Whether or not we plan to calc a 2d metric
	calc_2d_metric = any((calc_2d_rand_score, calc_2d_rand_error,
		calc_2d_variation_score, calc_2d_variation_information))

	#Whether or not we plan to calc a 3d metric
	calc_3d_metric = any((calc_rand_score, calc_rand_error,
		calc_variation_score, calc_variation_information))

	#relabelling segmentation for 2d metrics (if applicable)
	if calc_2d_metric:
		seg1_2d, seg2_2d = relabel2d(seg1, seg2)

	#foreground restriction
	if foreground_restricted:
		if calc_3d_metric:
			seg1, seg2 = foreground_restriction(seg1, seg2)

		if calc_2d_metric:
			seg1_2d, seg2_2d = foreground_restriction(seg1_2d, seg2_2d)

	#Calculating the necessary overlap matrices for the
	# desired output (2d vs. 3d)
	if calc_3d_metric:
		om = calc_overlap_matrix(seg1, seg2, split_0_segment)
	if calc_2d_metric:
		om_2d = calc_overlap_matrix(seg1_2d, seg2_2d, split_0_segment)

	#Calculating each desired metric
	if calc_rand_score:
		(f, m, s) = om_score(om_rand_f_score, "Rand F-Score", om, True, True, 0.5)

		results['Rand F Score Full'] = f
		results['Rand F Score Split'] = s
		results['Rand F Score Merge'] = m

	if calc_2d_rand_score:
		(f, m, s) = om_score(om_rand_f_score, "2D Rand F-Score", om_2d, True, True, 0.5)

		results['2D Rand F Score Full'] = f
		results['2D Rand F Score Split'] = s
		results['2D Rand F Score Merge'] = m

	if calc_rand_error:
		(f, m, s) =  om_error(om_rand_error, "Rand Error", om, True, True)

		results['Rand Error Full'] = f
		results['Rand Error Split'] = s
		results['Rand Error Merge'] = m

	if calc_2d_rand_error:

		(f, m, s) = om_error(om_rand_error, "2D Rand Error", om_2d, True, True)

		results['2D Rand Error Full'] = f
		results['2D Rand Error Split'] = s
		results['2D Rand Error Merge'] = m	

	if calc_variation_score:
		(f, m, s) = om_score(om_variation_f_score, "Variation F-Score", om, True, True, 0.5)

		results['Variation F Score Full'] = f
		results['Variation F Score Split'] = s
		results['Variation F Score Merge'] = m

	if calc_2d_variation_score:
		(f, m, s) = om_score(om_variation_f_score, "2D Variation F-Score", om_2d, True, True, 0.5)

		results['2D Variation F Score Full'] = f
		results['2D Variation F Score Split'] = s
		results['2D Variation F Score Merge'] = m

	if calc_variation_information:
		(f, m, s) =  om_error(om_variation_information, "Variation of Information", om, True, True)

		results['Variation of Information Full'] = f
		results['Variation of Information Split'] = s
		results['Variation of Information Merge'] = m

	if calc_2d_variation_information:
		(f, m, s) =  om_error(om_variation_information, "2D Variation of Information", om_2d, True, True)

		results['2D Variation of Information Full'] = f
		results['2D Variation of Information Split'] = s
		results['2D Variation of Information Merge'] = m

	print
	print_metrics(results)

if __name__ == '__main__':

	parser = argparse.ArgumentParser(
		description=__doc__,
		formatter_class=argparse.RawDescriptionHelpFormatter)

	parser.add_argument('seg1_filename', 
		help="Filename of the output image")
	parser.add_argument('seg2_filename',
		help='Filename of the labels for comparison- "ground truth" if available')

	#NOTE: "No" args store whether or not to calc the metric
	# the 'no' part of the flag is for command-line semantics
	parser.add_argument('-no_rfs','-no_rand_f_score',
		default=False, action='store_true')
	parser.add_argument('-rfs2d','-calc_2d_rand_f_score',
		default=False, action='store_true')

	parser.add_argument('-no_re','-no_rand_error',
		default=False, action='store_true')
	parser.add_argument('-re2d','-calc_2d_rand_error',
		default=False, action='store_true')

	parser.add_argument('-no_vifs','-no_variation_f_score',
		default=False, action='store_true')
	parser.add_argument('-vifs2d','-calc_2d_variation_f_score',
		default=False, action='store_true')

	parser.add_argument('-no_vi','-no_variation_information',
		default=False, action='store_true')
	parser.add_argument('-vi2d','-calc_2d_variation_information',
		default=False, action='store_true')

	parser.add_argument('-no_fr','-no_foreground_restriction',
		default=False, action='store_true')
	parser.add_argument('-no_split0','-dont_split_0_segment',
		default=False, action='store_true')

	args = parser.parse_args()

	rfs     = not args.no_rfs
	re      = not args.no_re
	vi      = not args.no_vi
	vifs    = not args.no_vifs
	fr      = not args.no_fr
	split0  = not args.no_split0

	main(args.seg1_filename,
	     args.seg2_filename,
	     rfs,  args.rfs2d,
	     re,   args.re2d,
	     vifs, args.vifs2d,
	     vi,   args.vi2d,
	     fr,
	     sp0)
