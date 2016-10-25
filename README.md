Segascorus
========

[![Build Status](https://travis-ci.org/seung-lab/segascorus.svg?branch=master)](https://travis-ci.org/seung-lab/segascorus)

SeungLab Error Metrics for Volummetric Segmentation
---------------------------------------------------

 This package computes the error between two segmentations of a volume.
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

See the [manual](https://github.com/seung-lab/segascorus/blob/master/segerror-manual.pdf) for detailed descriptions of each metric and its computation.

The main executable modules are:
- score.py - One-shot scoring/comparison of two segmentations
- curve.py - Computing error curves over the threshold a watershed MST (see [Watershed](https://github.com/seung-lab/Watershed.jl))
- plot.py  - Basic plotting functionality of error curves from curve.py

You can learn more about each of these modules by using a help flag.

    python{3} score.py --help
    python{3} curve.py --help
    python{3} plot.py --help
    
metrics.py can also be used as an importable module for more flexible metric computation.


Dependencies:
-------------
|Library|
|:-----:|
|[Cython](http://cython.org/) >= 0.23.4 |
|[python.tifffile](https://pypi.python.org/pypi/tifffile)|
|[NumPy](http://www.numpy.org/)|
|[Scipy](http://www.scipy.org/)|
|[matplotlib](http://matplotlib.org/)|

Installation (compiling Cython modules):
-------------
    make
    
NOTE: You will see a harmless warning when compiling the Cython functions. See (http://docs.cython.org/src/reference/compilation.html)

The codebase is now compatible with python3, in which case you can compile the Cython functions with

    make python3

If you'd like to use segerror as a python module, rename/move the init.py file within the current directory after compilation



