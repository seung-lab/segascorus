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
- Boundary Thinning- (not complete yet)

Inputs:
-------
- First Segmentation File (seg1, as .tif file)
- Second Segmentation File (seg2, as .tif file)
 This should be the "ground truth" segmentation if applicable
- Foreground Restriction (optional flag -nofr, default=on)
- Boundary Thinning (not complete yet)
- Metric Types
 - Rand Score - ISBI 2012 Error Metric
 - Rand Error - 1 - RandIndex
 - Variation Score - ISBI 2012 Information Theoretic Error Metric
 - Variation of Information


Dependencies:
-------------
|Library|
|:-----:|
|[Cython](http://cython.org/)|
|[python.tifffile](https://pypi.python.org/pypi/tifffile)|
|[NumPy](http://www.numpy.org/)|
|[Scipy](http://www.scipy.org/)|

Installation (compiling Cython module):
-------------
    make
    
NOTE: You will see a harmless warning when compiling the Cython functions. See (http://docs.cython.org/src/reference/compilation.html)
