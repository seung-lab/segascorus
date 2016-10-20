#!/usr/bin/env python
__doc__ = '''
Data Preprocessing - data_prep.py

Nicholas Turner 2016
'''


import data_prep_u as u


def relabel2d(seg1, seg2):
    '''
    Relabels segmentations by their contiguous 2d segments using
    6-connectivity connected components
    '''
    print("Relabeling segments for 2d metrics...")
    return u.relabel2d(seg1), u.relabel2d(seg2)


def relabel2d_byid(seg1, seg2):
    '''
    Relabels segmentations by their ids within 2d slices. This preserves
    the equality of non-contiguous 2d segments within a slice.
    '''
    print("Relabeling segments for 2d metrics by id...")
    return u.relabel1N(seg1), u.relabel1N(seg2)


def foreground_restriction(seg1, seg2):
    '''Performs foreground restriction on seg2's foreground'''

    print("Performing foreground-restriction")
    seg1_fr = seg1[seg2 != 0]
    seg2_fr = seg2[seg2 != 0]

    return seg1_fr, seg2_fr


