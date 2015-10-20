#!/usr/bin/env python
__doc__ = '''
SeungLab Error Metrics for Volummetric Segmentation

 This module computes the error between two segmentations of a volume.
The metric can either be rand error (or soon variation of information), and
features a few customizable options. These include

- Foreground Restriction- computing error over voxels where the
  second segmentation (presumed to be ground truth) != 0.
- Boundary Thinning- (not complete yet)

Inputs

	- First Segmentation File (seg1, as .tif file)
	- Second Segmentation File (seg2, as .tif file)
	  This should be the "ground truth" segmentation if applicable
	- Foreground Restriction (flag, optional)
	- Boundary Thinning (int, optional, not complete yet)
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
import tifffile #dependency
import numpy as np
import scipy.sparse as sp
from os import path

#Cython functions
import cy

def choose_two(n):
	'''
	Returns n choose 2. A faster version for np arrays
	is implemented within the cy module
	'''
	# = (n * (n-1)) / 2.0, with fewer overflows
	return (n / 2.0) * (n-1)

def om_rand_score(om, alpha=0.5, merge_err=False, split_err=False):
	'''
	Calculates the rand SCORE (from ISBI2012) of an unnormalized (raw counts) overlap matrix

	Can also return split and/or merge error separately ASSUMING that the "ground-truth"
	segmentation is represented in the 2nd axis (columns)

	In order to use multiple alpha values, pass in an np.array of the desired values
	'''

	col_counts = om.sum(1) #t_j * N from preprint
	row_counts = om.sum(0) #s_i * N from preprint

	#for float division below
	N = float( col_counts.sum() )

	t_term = np.sum( np.square(col_counts / N) )
	s_term = np.sum( np.square(row_counts / N) )

	#p term requires a bit more work with sparse matrix
	p_term = np.sum( np.square(np.copy(om.data) / N) )

	split_error = (p_term / t_term)
	merge_error = (p_term / s_term)

	full_error = p_term / (alpha * s_term + (1-alpha) * t_term)	

	if split_err and merge_err:
		return full_error, merge_error, split_error
	elif split_err:
		return full_error, split_error
	elif merge_err:
		return full_error, merge_error
	else:
		return full_error

def om_rand_error(om, merge_err=False, split_err=False):
	'''
	Calculates the rand error (= 1 - RandIndex) of an unnormalized (raw counts) overlap matrix

	Can also return split and/or merge error separately ASSUMING that the "ground-truth"
	segmentation is represented in the 2nd axis (columns)
	'''

	#Transforming to np.array makes ravel work properly
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

	split_error = (TPplusFN - TP) / total_pairs
	merge_error = (TPplusFP - TP) / total_pairs

	full_error = (TPplusFP + TPplusFN - 2*TP) / total_pairs

	if split_err and merge_err:
		return full_error, merge_error, split_error
	elif split_err:
		return full_error, split_error
	elif merge_err:
		return full_error, merge_error
	else:
		return full_error

def om_variation_score(om, alpha=0.5, merge_score=False, split_score=False):
	'''
	Calculates the variation of information SCORE (from preprint) of an unnormalized (raw counts) overlap matrix

	Can also return split and/or merge error separately ASSUMING that the "ground-truth"
	segmentation is represented in the 2nd axis (columns)
	
	In order to use multiple alpha values, pass in an np array of the desired values
	'''

	#Transforming to np.array makes ravel work properly
	# (sp version seems buggy)
	col_counts = np.array(om.sum(1)) #t_j * N
	row_counts = np.array(om.sum(0)) #s_i * N

	#for float division below (float64)
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

	split_score = I / HS
	merge_score = I / HT

	full_score = I / ( (1 - alpha) * HS + alpha * HT )

	if split_score and merge_score:
		return full_score, merge_score, split_score
	elif split_score:
		return full_score, split_score
	elif merge_score:
		return full_score, merge_score
	else:
		return full_score

def om_variation_information(om, merge_err=False, split_err=False):
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

	split_error = np.sum( cy.om_VI_byaxis( rows, vals, t_j.ravel() ) )
	merge_error = np.sum( cy.om_VI_byaxis( cols, vals, s_i.ravel() ) )

	full_error = split_error + merge_error

	if split_err and merge_err:
		return full_error, merge_error, split_error
	elif split_err:
		return full_error, split_error
	elif merge_err:
		return full_error, merge_error
	else:
		return full_error

def calc_overlap_matrix(seg1, seg2):
	'''Calculates the overlap matrix of two segmentations'''

	print "Finding overlap matrix..."
	start = timeit.default_timer()
	om = cy.overlap_matrix( seg1.ravel(), seg2.ravel() )
	end = timeit.default_timer()
	print "Completed in %f seconds" % (end-start)

	return om

def foreground_restriction(seg1, seg2):
	'''Performs foreground restriction on seg2's foreground'''

	print "Performing foreground-restriction"
	seg1_fr = seg1[seg2 != 0]
	seg2_fr = seg2[seg2 != 0]

	return seg1_fr, seg2_fr

def om_score(om_score_function, score_name,
	om, merge_score=False, split_score=False, alpha=0.5):
	
	print "Calculating {}...".format(score_name)
	start = timeit.default_timer()
	score = om_score_function(om, alpha, merge_score, split_score)
	end = timeit.default_timer()
	print "Completed in %f seconds" % (end-start)

	return score

def om_error(om_error_function, error_name,
	om, merge_err=False, split_err=False):

	print "Calculating {}...".format(error_name)
	start = timeit.default_timer()
	score = om_error_function(om, merge_err, split_err)
	end = timeit.default_timer()
	print "Completed in %f seconds" % (end-start)
	
	return score

def seg_score(om_score_function, score_name, 
	seg1, seg2, merge_score=False, split_score=False, alpha=0.5):
	'''Higher-level function which handles segmentations'''

	om = calc_overlap_matrix(seg1, seg2)

	score = om_score(om_score_function, score_name,
		om, merge_score, split_score, alpha)

	return score

def seg_error(om_error_function, error_name,
	seg1, seg2, merge_err=False, split_err=False):
	'''Higher-level function which handles segmentations'''

	om = calc_overlap_matrix(seg1, seg2)

	score = om_error(om_error_function, error_name,
		om, merge_err, split_err)

	return score
	
def seg_rand_score(seg1, seg2, merge_score=False, split_score=False, alpha=0.5):
	return seg_score(om_rand_score, "Rand Score",
		seg1, seg2, merge_score, split_score, alpha)

def seg_rand_error(seg1, seg2, merge_err=False, split_err=False):
	return seg_error(om_rand_error, "Rand Error",
		seg1, seg2, merge_err, split_err)

def seg_variation_score(seg1, seg2, merge_score=False, split_score=False, alpha=0.5):
	return seg_score(om_variation_score, "VI Score",
		seg1, seg2, merge_score, split_score, alpha)

def seg_variation_information(seg1, seg2, merge_err=False, split_err=False):
	return seg_error(om_variation_information, "Variation of Information",
		seg1, seg2, merge_err, split_err)

def print_metrics(metrics):

	keys = metrics.keys()
	keys.sort()

	for key in keys:
		print "{}: {}".format(key, metrics[key])

def main(seg1_fname, seg2_fname, 
	calc_rand_score=False,
	calc_rand_error=True, 
	calc_variation_score=False,
	calc_variation_information=False,
	foreground_restricted=True):
	'''
	Script functionality, computes the overlap matrix, computes any specified metrics,
	and prints the results nicely
	'''

	print "Loading Data..."
	seg1 = tifffile.imread(seg1_fname).astype('uint32')
	seg2 = tifffile.imread(seg2_fname).astype('uint32')

	results = {}

	if foreground_restricted:
		seg1, seg2 = foreground_restriction(seg1, seg2)

	om = calc_overlap_matrix(seg1, seg2)

	if calc_rand_score:
		(f, m, s) = om_score(om_rand_score, "Rand Score", om, True, True, 0.5)

		results['Rand Score Full'] = f
		results['Rand Score Split'] = s
		results['Rand Score Merge'] = m

	if calc_rand_error:
		(f, m, s) =  om_error(om_rand_error, "Rand Error", om, True, True)

		results['Rand Error Full'] = f
		results['Rand Error Split'] = s
		results['Rand Error Merge'] = m

	if calc_variation_score:
		(f, m, s) = om_score(om_variation_score, "Variation Score", om, True, True, 0.5)

		results['Variation Score Full'] = f
		results['Variation Score Split'] = s
		results['Variation Score Merge'] = m

	if calc_variation_information:
		(f, m, s) =  om_error(om_variation_information, "Variation of Information", om, True, True)

		results['Variation of Information Full'] = f
		results['Variation of Information Split'] = s
		results['Variation of Information Merge'] = m

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
	parser.add_argument('-rs','-calc_rand_score',
		default=False, action='store_true')
	parser.add_argument('-re','-calc_rand_error',
		default=True, action='store_false')
	parser.add_argument('-vs','-calc_variation_score',
		default=False, action='store_true')
	parser.add_argument('-vi','-calc_variation_information',
		default=False, action='store_true')
	parser.add_argument('-fr','-foreground_restriction',
		default=True, action='store_false')

	args = parser.parse_args()

	main(args.seg1_filename,
	     args.seg2_filename,
	     args.rs,
	     args.re,
	     args.vs,
	     args.vi,
	     args.fr)

