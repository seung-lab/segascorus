#!/usr/bin/env python
__doc__ = '''
Error Metrics - metrics.py

Nicholas Turner October 2016
'''

import timeit

#Dependencies
import numpy as np
import scipy.sparse as sp
import data_prep, repres, utils, global_vars
import metrics_u


DTYPE= global_vars.DTYPE


#=====================================================================
#Functions for interactive module use

def seg_rand_f_score(seg1, seg2, merge_score=False, split_score=False,
	alpha=0.5, split0=True):
	'''Computes the Rand F Score for a segmentation'''
	return seg_metric( seg1, seg2, split0, om_rand_f_score, "Rand Score",
		           merge_score, split_score, alpha )


def seg_rand_error(seg1, seg2, merge_score=False, split_score=False, split0=True):
	'''Computes the Rand Error for a segmentation'''
	return seg_metric( seg1, seg2, split0, om_rand_error, "Rand Error",
	                   merge_score, split_score )


def seg_variation_f_score(seg1, seg2, merge_score=False, split_score=False,
	alpha=0.5, split0=True):
	'''Computes the Variation of Information F Score for a segmentation'''
	return seg_metric( seg1, seg2, split0, om_variation_f_score, "VI Score",
	                   merge_score, split_score )


def seg_variation_information(seg1, seg2, merge_score=False, split_score=False, split0=True):
	'''Computes the Variation of Information for a segmentation'''
	return seg_metric( seg1, seg2, split0, om_variation_information, "Variation of Information",
	                   merge_score, split_score )


def seg_fr_rand_f_score(seg1,seg2, merge_score=False, split_score=False,
	alpha=0.5, split0=True):
	'''Computes the Rand F Score for a segmentation w/ foreground restriction'''
	seg1, seg2 = data_prep.foreground_restriction(seg1, seg2)
	return seg_rand_f_score(seg1, seg2, merge_score, split_score, alpha, split0)


def seg_fr_rand_error(seg1, seg2, merge_error=False, split_error=False, split0=True):
	'''Computes the Rand Error for a segmentation w/ foreground restriction'''
	seg1, seg2 = data_prep.foreground_restriction(seg1, seg2)
	return seg_rand_error(seg1, seg2, merge_error, split_error, split0)


def seg_fr_variation_f_score(seg1, seg2, merge_score=False, split_score=False,
	alpha=0.5, split0=True):
	'''Computes the Variation of Information F Score for a segmentation w/ foreground restriction'''
	seg1, seg2 = data_prep.foreground_restriction(seg1, seg2)
	return seg_variation_f_score(seg1, seg2, merge_score, split_score, alpha, split0)


def seg_fr_variation_information(seg1, seg2, merge_error=False, split_error=False, split0=True):
	'''Computes the Variation of Information for a segmentation w/ foreground restriction'''
	seg1, seg2 = data_prep.foreground_restriction(seg1, seg2)
	return seg_variation_information(seg1, seg2, merge_error, split_error, split0)


#== 2d versions ==#
def seg_2d_rand_error(seg1, seg2, merge_err=False, split_err=False, split0=True):

	print("Relabelling segmentations for 2d comparison")
	seg1, seg2 = relabel2d(seg1, seg2)

	return seg_rand_error(seg1, seg2, merge_err, split_err, split0)
#INCOMPLETE

#============
#Mid-level fns


def om_metric( om_metric_fn, metric_name, *args ):
	'''Runs an error function, times the calculation, and returns the result'''

	print("Calculating {}...".format(metric_name))
	start = timeit.default_timer()
	score = om_metric_fn(*args)
	end = timeit.default_timer()
	print("Completed in %f seconds" % (end-start))

	return score


def seg_metric( seg1, seg2, split0, *args ):
	'''High-level function which handles segmentations'''

	assert seg1.dtype == DTYPE
	assert seg2.dtype == DTYPE

	om = utils.calc_overlap_matrix(seg1, seg2, split0)

	om_metric_fn, metric_name = args[:2]
	score = om_metric(om_metric_fn, metric_name, om, *args[2:])

	return score


#====================
#Overlap Matrix (low-level) fns


shannon_entropy = metrics_u.shannon_entropy
conditional_entropy = metrics_u.conditional_entropy


def om_rand_f_score( om, merge_score=False, split_score=False, alpha=0.5,
                     other_label=None ):
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

	if other_label is not None:
            col_counts[:,other_label] = 0

	#float conversion for division below
	N  = float( row_counts.sum() )
	Nc = float( col_counts.sum() )


	#implicit conversion to float64
	t_term = np.sum( np.square(col_counts / Nc) )
	s_term = np.sum( np.square(row_counts / N) )

	rows, cols, vals = om.find()
	full_p_term = np.sum( np.square(vals / N) )

	if other_label is not None:
	  vals[cols == other_label] = 0
	res_p_term  = np.sum( np.square(vals / Nc) )


	split_sc = (res_p_term / t_term)
	merge_sc = (full_p_term / s_term)

	full_sc = 1. / ( alpha / merge_sc + (1-alpha) / split_sc)

	if split_score and merge_score:
		return full_sc, merge_sc, split_sc
	elif split_score:
		return full_sc, split_sc
	elif merge_score:
		return full_sc, merge_sc
	else:
		return full_sc


def om_rand_error(om, merge_error=False, split_error=False, other_label=None):
	'''
	Calculates the rand error (= 1 - RandIndex) of an unnormalized (raw counts) overlap matrix

	Can also return split and/or merge error separately assuming that the "ground-truth"
	segmentation is represented in the 2nd axis (columns)
	'''

	#Converting to np.array makes ravel work properly
	# (sp version seems buggy)
	col_counts = np.array( om.sum(0) )
	row_counts = np.array( om.sum(1) )

	if other_label is not None:
            col_counts[:,other_label] = 0

	#for float division below
	N  = float(row_counts.sum())
	Nc = float(col_counts.sum())

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
		  (col_counts / Nc) *
		  ((col_counts - 1) / (Nc-1))
		)


	rows, cols, c_ij_vals = om.find()


	TP_norm_full = np.sum(
		  (c_ij_vals / N) *
		  ((c_ij_vals - 1) / (N-1))
		)

	if other_label is not None:
	  c_ij_vals[cols == other_label] = 0

	TP_norm_res = np.sum(
		  (c_ij_vals / Nc) *
		  ((c_ij_vals - 1) / (Nc-1))
		)


	split_err = TPplusFN_norm - TP_norm_res
	merge_err = TPplusFP_norm - TP_norm_full

	#full_err  = TPplusFP_norm + TPplusFN_norm - 2*TP_norm
	full_err  = split_err + merge_err


	if split_error and merge_error:
		return full_err, merge_err, split_err
	elif split_error:
		return full_err, split_err
	elif merge_error:
		return full_err, merge_err
	else:
		return full_err


def om_variation_f_score(om, merge_score=False, split_score=False, alpha=0.5,
        other_label=None):
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

	if other_label is not None:
	  col_counts[:,other_label] = 0

	#float conversion for division below (float64)
	N = float( row_counts.sum() )
	Nc = float( col_counts.sum() )

	#implicitly converts to float64
	t_j = col_counts / Nc
	s_i = row_counts / N

	HT = shannon_entropy( t_j.ravel() )
	HS = shannon_entropy( s_i.ravel() )

	if HT == 0:
		print("WARNING: H(T) equals zero! You will likely see a RuntimeWarning")
	if HS == 0:
		print("WARNING: H(S) equals zero! You will likely see a RuntimeWarning")

	rows,cols,vals = om.find()

	#p_ij_vals_f = (-1.0) * shannon_entropy( (vals / N) )

	HTgS = conditional_entropy( vals/N, rows, s_i.ravel() )

	if other_label is not None:
	  vals[cols == other_label] = 0

	HSgT = conditional_entropy( vals/Nc, cols, t_j.ravel() )

	split_sc = 1 - HSgT / HS
	merge_sc = 1 - HTgS / HT
	#p_ij_vals_r = (-1.0) * shannon_entropy( (vals / N) )

	#Hpf = np.sum(p_ij_vals_f)
	#Hpr = np.sum(p_ij_vals_r)

	#If = Hpf + HT + HS
	#Ir = Hpr + HT + HS

	#print(Hpf); print(Hpr)
	#print(HS); print(HT); print(Ir); print(If)
	#split_sc = Ir / HS
	#merge_sc = If / HT

	#full_sc = I / ( (1 - alpha) * HS + alpha * HT )
	full_sc = 1. / ( alpha / merge_sc + (1-alpha) / split_sc )

	if split_score and merge_score:
		return full_sc, merge_sc, split_sc
	elif split_score:
		return full_sc, split_sc
	elif merge_score:
		return full_sc, merge_sc
	else:
		return full_sc


def om_variation_information(om, merge_error=False, split_error=False,
        other_label=None):
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

	if other_label is not None:
	  col_counts[:,other_label] = 0

	#for float division below (float64)
	N  = float( row_counts.sum() )
	Nc = float( col_counts.sum() )

	#implicitly converts to float64
	# again, conversion to np.array fixes sp weirdness
	t_j = np.array( col_counts / Nc )
	s_i = np.array( row_counts / N )

	rows, cols, vals = om.find()

	merge_err = conditional_entropy( vals/N, rows, s_i.ravel() )

	if other_label is not None:
	  vals[cols == other_label] = 0

	split_err = conditional_entropy( vals/Nc, cols, t_j.ravel() )

	full_err = split_err + merge_err

	if split_error and merge_error:
		return full_err, merge_err, split_err
	elif split_error:
		return full_err, split_err
	elif merge_error:
		return full_err, merge_err
	else:
		return full_err
