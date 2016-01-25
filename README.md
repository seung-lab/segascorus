SegError
========

SeungLab Error Metrics for Volummetric Segmentation
---------------------------------------------------

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

Inputs:
-------
- First Segmentation File (seg1, as .tif file)
- Second Segmentation File (seg2, as .tif file)
 This should be the "ground truth" segmentation if applicable
- Foreground Restriction (optional flag -nofr, default=on)
- Boundary Thinning (not complete yet)
- Split 0 Segment (optional flag -no_split0, default=on)

- Metric Types
  (all calculated by default)
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


Dependencies:
-------------
|Library|
|:-----:|
|[Cython](http://cython.org/) >= 0.23.4 |
|[python.tifffile](https://pypi.python.org/pypi/tifffile)|
|[NumPy](http://www.numpy.org/)|
|[Scipy](http://www.scipy.org/)|

Installation (compiling Cython module):
-------------
    make
    
NOTE: You will see a harmless warning when compiling the Cython functions. See (http://docs.cython.org/src/reference/compilation.html)

If you'd like to use segerror as a python module, rename/move the init.py file within the current directory after compilation

