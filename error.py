#!/usr/bin/env python
__doc__ = '''
SeungLab Error Metrics for Volummetric Segmentation

 This module computes the error between two segmentations of a volume.
The metrics are either based on the Rand Index or Variation of Information, and
features a few customizable options. These include:

- Foreground Restriction- computing error over voxels where the
  second segmentation (presumed to be ground truth) != 0.
- Boundary Thinning- (not complete yet)

Inputs

	- First Segmentation File (seg1, as .tif file)
	- Second Segmentation File (seg2, as .tif file)
	  This should be the "ground truth" segmentation if applicable
	- Foreground Restriction (optional flag -fr, default=true)
	- Boundary Thinning (optional flag -bt, not complete yet)
	- Metric Types
		- Rand Score - ISBI 2012 Error Metric
		- Rand Error - 1 - RandIndex
		- Variation Score - ISBI 2012 Information Theoretic Error Metric
		- Variation of Information

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

DTYPE='uint64'

def choose_two(n):
	'''
	Returns n choose 2. A faster version for np arrays
	is implemented within the cy module
	'''
	# = (n * (n-1)) / 2.0, with fewer overflows
	return (n / 2.0) * (n-1)

def om_rand_score(om, alpha=0.5, merge_score=False, split_score=False):
	'''
	Calculates the rand SCORE (from ISBI2012) of an unnormalized (raw counts) overlap matrix

	Can also return split and/or merge error separately assuming that the "ground-truth"
	segmentation is represented in the 2nd axis (columns)

	In order to use multiple alpha values, pass in an np.array of the desired values
	'''

	col_counts = om.sum(1) #t_j * N
	row_counts = om.sum(0) #s_i * N

	#for float division below
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
	col_counts = np.array( om.sum(1) )
	row_counts = np.array( om.sum(0) )

	#for float division below
	N = float(col_counts.sum())

	#TP - True Positive pairs
	#FP - False Positive pairs
	#FN - False Negative pairs
	TPplusFP = np.sum(cy.choose_two( col_counts.ravel() ))
	TPplusFN = np.sum(cy.choose_two( row_counts.ravel() ))

	p_ij_vals = cy.choose_two( np.copy(om.data) )
	TP = np.sum(p_ij_vals)

	total_pairs = choose_two(N)
	overflow_warning_check(total_pairs)

	split_err = (TPplusFN - TP) / total_pairs
	merge_err = (TPplusFP - TP) / total_pairs

	full_err = (TPplusFP + TPplusFN - 2*TP) / total_pairs

	if split_error and merge_error:
		return full_err, merge_err, split_err
	elif split_error:
		return full_err, split_err
	elif merge_error:
		return full_err, merge_err
	else:
		return full_err

def om_variation_score(om, alpha=0.5, merge_score=False, split_score=False):
	'''
	Calculates the variation of information SCORE (from ISBI2012) of an unnormalized (raw counts) overlap matrix

	Can also return split and/or merge error separately assuming that the "ground-truth"
	segmentation is represented in the 2nd axis (columns)
	
	In order to use multiple alpha values, pass in an np array of the desired values
	'''

	#Transforming to np.array makes ravel work properly
	# (sp version seems buggy)
	col_counts = np.array(om.sum(1)) #t_j * N
	row_counts = np.array(om.sum(0)) #s_i * N

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
	col_counts = np.array(om.sum(1)) #t_j * N
	row_counts = np.array(om.sum(0)) #s_i * N

	#for float division below (float64)
	N = float( col_counts.sum() )

	#implicitly converts to float64
	# again, conversion to np.array fixes sp weirdness
	t_j = np.array( col_counts / N )
	s_i = np.array( row_counts / N )

	rows, cols, vals = sp.find(om)
	vals = vals / N #p_ij

	split_err = np.sum( cy.om_VI_byaxis( cols, vals, s_i.ravel() ) )
	merge_err = np.sum( cy.om_VI_byaxis( rows, vals, t_j.ravel() ) )

	full_err = split_err + merge_err

	if split_error and merge_error:
		return full_err, merge_err, split_err
	elif split_error:
		return full_err, split_err
	elif merge_error:
		return full_err, merge_err
	else:
		return full_err

def calc_overlap_matrix(seg1, seg2):
	'''Calculates the overlap matrix of two segmentations'''

	print "Finding overlap matrix..."
	start = timeit.default_timer()
	om = cy.overlap_matrix( seg1.ravel(), seg2.ravel() )
	end = timeit.default_timer()
	print "Completed in %f seconds" % (end-start)

	return om

def relabel2d(seg1, seg2):
	'''Relabels segmentations to be 2d for 2d-based error metrics'''
	return cy.relabel1N(seg1), cy.relabel1N(seg2)

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
	seg1, seg2, merge_score=False, split_score=False, alpha=0.5):
	'''High-level function which handles segmentations'''

	om = calc_overlap_matrix(seg1, seg2)

	score = om_score(om_score_function, score_name,
		om, merge_score, split_score, alpha)

	return score

def seg_error(om_error_function, error_name,
	seg1, seg2, merge_err=False, split_err=False):
	'''High-level function which handles segmentations'''

	om = calc_overlap_matrix(seg1, seg2)

	score = om_error(om_error_function, error_name,
		om, merge_err, split_err)

	return score
	
#=====================================================================
#Functions for interactive module use

def seg_rand_score(seg1, seg2, merge_score=False, split_score=False, alpha=0.5):
	return seg_score(om_rand_score, "Rand Score",
		seg1, seg2, merge_score, split_score, alpha)

def seg_rand_error(seg1, seg2, merge_err=False, split_err=False):
	return seg_error(om_rand_error, "Rand Error",
		seg1, seg2, merge_err, split_err)

def seg_2d_rand_error(seg1, seg2, merge_err=False, split_err=False):

	print "Relabelling segmentations for 2d comparison"
	seg1, seg2 = relabel2d(seg1, seg2)

	return seg_rand_error(seg1, seg2, merge_err, split_err)

def seg_variation_score(seg1, seg2, merge_score=False, split_score=False, alpha=0.5):
	return seg_score(om_variation_score, "VI Score",
		seg1, seg2, merge_score, split_score, alpha)

def seg_variation_information(seg1, seg2, merge_err=False, split_err=False):
	return seg_error(om_variation_information, "Variation of Information",
		seg1, seg2, merge_err, split_err)

def seg_fr_rand_score(seg1,seg2, merge_score=False, split_score=False, alpha=0.5):
	seg1, seg2 = foreground_restriction(seg1, seg2)
	return seg_rand_score(seg1, seg2, merge_score, split_score, alpha)

def seg_fr_rand_error(seg1, seg2, merge_error=False, split_error=False):
	seg1, seg2 = foreground_restriction(seg1, seg2)
	return seg_rand_error(seg1, seg2, merge_error, split_error)

def seg_fr_variation_score(seg1, seg2, merge_score=False, split_score=False, alpha=0.5):
	seg1, seg2 = foreground_restriction(seg1, seg2)
	return seg_variation_score(seg1, seg2, merge_score, split_score, alpha=0.5)

def seg_fr_variation_information(seg1, seg2, merge_error=False, split_error=False):
	seg1, seg2 = foreground_restriction(seg1, seg2)
	return seg_variation_information(seg1, seg2, merge_error, split_error)

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
	errors (often silent under Cython). 
	'''

	warning_string = ("WARNING: total number of pairs exceeds bit length.",
		"\n You may see overflow errors.\n")

	if DTYPE == 'uint64':
		if n_val > 2 ** 64:
			print warning_string
	if DTYPE == 'uint32':
		if n_val > 2 ** 32:
			print warning_string

def import_tif(filename):
    return tifffile.imread(filename).astype(DTYPE)

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
	foreground_restricted=True,):
	'''
	Script functionality, computes the overlap matrix, computes any specified metrics,
	and prints the results nicely
	'''

	print "Loading Data..."
	seg1 = import_tif(seg1_fname)
	seg2 = import_tif(seg2_fname)

	results = {}

	calc_2d_metric = any((calc_2d_rand_score, calc_2d_rand_error,
		calc_2d_variation_score, calc_2d_variation_information))

	calc_3d_metric = any((calc_rand_score, calc_rand_error,
		calc_variation_score, calc_variation_information))

	if calc_2d_metric:
		seg1_2d, seg2_2d = relabel2d(seg1, seg2)

	if foreground_restricted:
		if calc_3d_metric:
			seg1, seg2 = foreground_restriction(seg1, seg2)

		if calc_2d_metric:
			seg1_2d, seg2_2d = foreground_restriction(seg1_2d, seg2_2d)

	if calc_3d_metric:
		om = calc_overlap_matrix(seg1, seg2)

	if calc_2d_metric:
		om_2d = calc_overlap_matrix(seg1_2d, seg2_2d)

	if calc_rand_score:
		(f, m, s) = om_score(om_rand_score, "Rand Score", om, True, True, 0.5)

		results['Rand Score Full'] = f
		results['Rand Score Split'] = s
		results['Rand Score Merge'] = m

	if calc_2d_rand_score:
		(f, m, s) = om_score(om_rand_score, "Rand Score", om_2d, True, True, 0.5)

		results['2D Rand Score Full'] = f
		results['2D Rand Score Split'] = s
		results['2D Rand Score Merge'] = m

	if calc_rand_error:
		(f, m, s) =  om_error(om_rand_error, "Rand Error", om, True, True)

		results['Rand Error Full'] = f
		results['Rand Error Split'] = s
		results['Rand Error Merge'] = m

	if calc_2d_rand_error:

		(f, m, s) = om_error(om_rand_error, "Rand Error", om_2d, True, True)

		results['2D Rand Error Full'] = f
		results['2D Rand Error Split'] = s
		results['2D Rand Error Merge'] = m	

	if calc_variation_score:
		(f, m, s) = om_score(om_variation_score, "Variation Score", om, True, True, 0.5)

		results['Variation Score Full'] = f
		results['Variation Score Split'] = s
		results['Variation Score Merge'] = m

	if calc_2d_variation_score:
		(f, m, s) = om_score(om_variation_score, "Variation Score", om_2d, True, True, 0.5)

		results['2D Variation Score Full'] = f
		results['2D Variation Score Split'] = s
		results['2D Variation Score Merge'] = m

	if calc_variation_information:
		(f, m, s) =  om_error(om_variation_information, "Variation of Information", om, True, True)

		results['Variation of Information Full'] = f
		results['Variation of Information Split'] = s
		results['Variation of Information Merge'] = m

	if calc_2d_variation_information:
		(f, m, s) =  om_error(om_variation_information, "Variation of Information", om_2d, True, True)

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
	parser.add_argument('-no_rs','-no_rand_score',
		default=True, action='store_true')
	parser.add_argument('-rs2d','-calc_2d_rand_score',
		default=False, action='store_true')

	parser.add_argument('-no_re','-no_rand_error',
		default=True, action='store_false')
	parser.add_argument('-re2d','-calc_2d_rand_error',
		default=False, action='store_true')

	parser.add_argument('-no_vs','-no_variation_score',
		default=True, action='store_false')
	parser.add_argument('-vs2d','-calc_2d_variation_score',
		default=False, action='store_true')

	parser.add_argument('-no_vi','-no_variation_information',
		default=True, action='store_false')
	parser.add_argument('-vi2d','-calc_2d_variation_information',
		default=False, action='store_true')

	parser.add_argument('-no_fr','-foreground_restriction',
		default=True, action='store_false')

	args = parser.parse_args()

	main(args.seg1_filename,
	     args.seg2_filename,
	     args.no_rs,
	     args.rs2d,
	     args.no_re,
	     args.re2d,
	     args.no_vs,
	     args.vs2d,
	     args.no_vi,
	     args.vi2d,
	     args.no_fr)

