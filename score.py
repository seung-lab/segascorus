#!/usr/bin/env python
__doc__ = '''
Command-line processing - score.py

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


Inputs

	- First Segmentation File (seg1, as .tif or hdf5 file)
	- Second Segmentation File (seg2, as .tif or hdf5 file)
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

Nicholas Turner, Jingpeng Wu 2015-2016
'''


import argparse
import io_utils
import utils
from metrics import *


def main(seg1_fname, seg2_fname,

	calc_rand_score=True,
	calc_rand_error=True,
	calc_variation_score=True,
	calc_variation_information=True,

	relabel2d=False,
	foreground_restricted=True,
	split_0_segment=True,

	other=None):
	'''
	Script functionality, computes the overlap matrix,
    computes any specified metrics,
	and prints the results nicely
	'''

	print("Loading Data...")
	seg1 = io_utils.import_file(seg1_fname)
	seg2 = io_utils.import_file(seg2_fname)


	prep = utils.parse_fns( utils.prep_fns,
                             [relabel2d, foreground_restricted ] )
	seg1, seg2 = utils.run_preprocessing( seg1, seg2, prep )


	om = utils.calc_overlap_matrix(seg1, seg2, split_0_segment)


	#Calculating each desired metric
	metrics = utils.parse_fns( utils.metric_fns,
                               [calc_rand_score,
                                calc_rand_error,
                                calc_variation_score,
                                calc_variation_information] )

	results = {}
	for (name,metric_fn) in metrics:
	  if relabel2d:
	    full_name = "2D {}".format(name)
	  else:
	    full_name = name

	  (f,m,s) = metric_fn( om, full_name, other )
	  results["{} Full".format(name)] = f
	  results["{} Merge".format(name)] = m
	  results["{} Split".format(name)] = s


	print("")
	utils.print_metrics(results)


def compute_all(om):
    """Useful for benchmarking"""

    om_rand_f_score( om, True, True )
    om_rand_error( om, True, True )
    om_variation_f_score( om, True, True )
    om_variation_information( om, True, True )


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
	parser.add_argument('-no_re','-no_rand_error',
		default=False, action='store_true')
	parser.add_argument('-no_vifs','-no_variation_f_score',
		default=False, action='store_true')
	parser.add_argument('-no_vi','-no_variation_information',
		default=False, action='store_true')

	parser.add_argument('-rel2d','-2d_relabeling',
		default=False, action="store_true")
	parser.add_argument('-no_fr','-no_foreground_restriction',
		default=False, action='store_true')
	parser.add_argument('-no_split0','-dont_split_0_segment',
		default=False, action='store_true')

	parser.add_argument('-other', type=int,
		default=None)

	args = parser.parse_args()

	rfs     = not args.no_rfs
	re      = not args.no_re
	vi      = not args.no_vi
	vifs    = not args.no_vifs
	rel2d   =     args.rel2d
	fr      = not args.no_fr
	split0  = not args.no_split0

	main(args.seg1_filename,
	     args.seg2_filename,
	     rfs,
	     re,
	     vifs,
	     vi,
	     rel2d,
	     fr,
	     split0,
             args.other)
