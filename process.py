#!/usr/bin/env python
__doc__ = '''
Command-line processing - process.py

Nicholas Turner, Jingpeng Wu June-October 2015
'''


import argparse
import io_utils
import utils
from metrics import *

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

	print("Loading Data...")
	seg1 = io_utils.import_file(seg1_fname)
	seg2 = io_utils.import_file(seg2_fname)

	results = {}

	#Whether or not we plan to compute a 2d metric
	calc_2d_metric = any((calc_2d_rand_score, 
                              calc_2d_rand_error,
		              calc_2d_variation_score, 
                              calc_2d_variation_information))

	#Whether or not we plan to compute a 3d metric
	calc_3d_metric = any((calc_rand_score, 
                              calc_rand_error,
		              calc_variation_score, 
                              calc_variation_information))


	#relabelling segmentation for 2d metrics (if applicable)
	if calc_2d_metric:
		seg1_2d, seg2_2d = data_prep.relabel2d(seg1, seg2)


	#foreground restriction
	if foreground_restricted:
		if calc_3d_metric:
			seg1, seg2 = data_prep.foreground_restriction(seg1, seg2)

		if calc_2d_metric:
			seg1_2d, seg2_2d = data_prep.foreground_restriction(seg1_2d, seg2_2d)


	#Calculating the necessary overlap matrices for the
	# desired output (2d vs. 3d)
	if calc_3d_metric:
		om = utils.calc_overlap_matrix(seg1, seg2, split_0_segment)
	if calc_2d_metric:
		om_2d = utils.calc_overlap_matrix(seg1_2d, seg2_2d, split_0_segment)


	#Calculating each desired metric
	if calc_rand_score:
		(f, m, s) = om_metric(om_rand_f_score, "Rand F-Score", om, True, True, 0.5)

		results['Rand F Score Full'] = f
		results['Rand F Score Split'] = s
		results['Rand F Score Merge'] = m

	if calc_2d_rand_score:
		(f, m, s) = om_metric(om_rand_f_score, "2D Rand F-Score", om_2d, True, True, 0.5)

		results['2D Rand F Score Full'] = f
		results['2D Rand F Score Split'] = s
		results['2D Rand F Score Merge'] = m

	if calc_rand_error:
		(f, m, s) =  om_metric(om_rand_error, "Rand Error", om, True, True)

		results['Rand Error Full'] = f
		results['Rand Error Split'] = s
		results['Rand Error Merge'] = m

	if calc_2d_rand_error:

		(f, m, s) = om_metric(om_rand_error, "2D Rand Error", om_2d, True, True)

		results['2D Rand Error Full'] = f
		results['2D Rand Error Split'] = s
		results['2D Rand Error Merge'] = m

	if calc_variation_score:
		(f, m, s) = om_metric(om_variation_f_score, "Variation F-Score", om, True, True, 0.5)

		results['Variation F Score Full'] = f
		results['Variation F Score Split'] = s
		results['Variation F Score Merge'] = m

	if calc_2d_variation_score:
		(f, m, s) = om_metric(om_variation_f_score, "2D Variation F-Score", om_2d, True, True, 0.5)

		results['2D Variation F Score Full'] = f
		results['2D Variation F Score Split'] = s
		results['2D Variation F Score Merge'] = m

	if calc_variation_information:
		(f, m, s) =  om_metric(om_variation_information, "Variation of Information", om, True, True)

		results['Variation of Information Full'] = f
		results['Variation of Information Split'] = s
		results['Variation of Information Merge'] = m

	if calc_2d_variation_information:
		(f, m, s) =  om_metric(om_variation_information, "2D Variation of Information", om_2d, True, True)

		results['2D Variation of Information Full'] = f
		results['2D Variation of Information Split'] = s
		results['2D Variation of Information Merge'] = m

	print()
	utils.print_metrics(results)

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
	     split0)
