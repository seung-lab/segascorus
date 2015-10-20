# segerror

SeungLab Error Metrics for Volummetric Segmentation

 This module computes the error between two segmentations of a volume.
The metric can either be rand error (or soon variation of information), and
features a few customizable options. These include

- Foreground Restriction- computing error over voxels where the
  second segmentation (presumed to be ground truth) != 0.
- Boundary Thinning- (precise specification TBD)

Inputs

        - First Segmentation File (seg1, as .tif file)
        - Second Segmentation File (seg2, as .tif file)
          This should be the "ground truth" segmentation if applicable
        - Foreground Restriction (flag, optional)
        - Boundary Thinning (int, optional, not complete yet (need direction))
        - Metric Types
                - Rand Score - ISBI 2012 Error Metric
                - Rand Error - 1 - RandIndex
                - Variation Score - ISBI 2012 Information Theoretic Error Metric
                - Variation of Information

Dependencies:

Cython
python.tifffile

Installation:

(from segerror directory)
python setup.py build_ext --inplace
