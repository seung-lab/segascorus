# -*- coding: utf-8 -*-
"""
Data Preprocessing Utilities - data_prep_u.pyx

Nicholas Turner, 2015
"""
#defining NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
import numpy as np
cimport numpy as np
import scipy.sparse as sp
cimport cython


from libc.math cimport log
#@cython.boundscheck(False) # turn off bounds-checking for entire function
#@cython.wraparound(False)
#@cython.nonecheck(False)


include "global_vars.pyx"


cpdef np.ndarray[DTYPE_t, ndim=3] relabel_segmentation(np.ndarray[DTYPE_t, ndim=3] seg, np.ndarray[DTYPE_t, ndim=1] relabelling):
    '''
    Takes a segmentation volume, along with an array encoding 
    a mapping from segment ids (encoded by index) to new segment ids
    (encoded by the value at that index), and maps the old values 
    to the new throughout the volume
    '''

    sz = seg.shape[0]
    sy = seg.shape[1]
    sx = seg.shape[2]

    cdef np.ndarray[DTYPE_t, ndim=3] result = np.empty((sz, sy, sx), dtype=DTYPE)

    cdef int z, y, x

    for z in xrange(sz):
        for y in xrange(sy):
            for x in xrange(sx):

                result[z,y,x] = relabelling[seg[z,y,x]]

    return result


#TO DO make option for 2d/3d
#This will also be useful once I incorporate functions for 
# mapping NN output to segmentations
cdef dfs(np.ndarray[DTYPE_t, ndim=3] orig_seg,\
        np.ndarray[DTYPE_t, ndim=3] seg2,\
        np.ndarray[np.uint8_t,  ndim=3, cast=True] mask, \
        np.uint32_t relid, \
        np.uint32_t label, \
        int z, int y, int x):
    '''
    Performs an iteration of depth-first search over voxels with
    the same segment id. 

    It will also stay within the limits of a mask volume,
    where True indicates either that the search should not go to this location,
    or it already has traversed it.

    Currently hard-coded to follow a 2d traversal, though this will be
    modified when enough people complain.
    '''
    cdef list seeds = []
    seeds.append((z,y,x))
    while seeds:
        z,y,x = seeds.pop()
        seg2[z,y,x] = relid
        mask[z,y,x] = True
        
        #2d relabelling for now
        # if z+1<seg.shape[0] and seg[z+1,y,x] == label and not mask[z+1,y,x] :
            # seeds.append((z+1,y,x))
        # if z-1>=0    and seg[z-1,y,x] == label and not mask[z-1,y,x] :
            # seeds.append((z-1,y,x))
        if y+1<orig_seg.shape[1] and orig_seg[z,y+1,x] == label and not mask[z,y+1,x] :
            seeds.append((z,y+1,x))
        if y-1>=0    and orig_seg[z,y-1,x] == label and not mask[z,y-1,x] :
            seeds.append((z,y-1,x))          
        if x+1<orig_seg.shape[2] and orig_seg[z,y,x+1] == label and not mask[z,y,x+1] :
            seeds.append((z,y,x+1))
        if x-1>=0    and orig_seg[z,y,x-1] == label and not mask[z,y,x-1] :
            seeds.append((z,y,x-1))       
    return seg2, mask


#Another function designated to be modified for NN output
cpdef np.ndarray[DTYPE_t, ndim=3] relabel1N(np.ndarray[DTYPE_t, ndim=3] seg):
    '''
    Modifies the labels of a segmentation to range between 1 and N
    where N is the number of nonzero segments. 

    Currently, the dfs function used here only traverses the segmentation
    in 2d, so this results layers of 2d segments, and with N higher
    than the original number of segments.

    This function also ignores the '0' segment, leaving it as passed in.
    ''' 

    cdef np.ndarray[np.uint8_t,    ndim=3, cast=True] mask 
    mask = (seg==0)
    
    sz = seg.shape[0]
    sy = seg.shape[1]
    sx = seg.shape[2]

    cdef np.ndarray[DTYPE_t, ndim=3] seg2 = np.zeros((sz, sy, sx), dtype=DTYPE)

    # relabel ID
    cdef np.uint32_t relid = 0
    cdef np.uint32_t z,y,x

    for z in xrange(sz):
        for y in xrange(sy):
            for x in xrange(sx):

                if mask[z,y,x]:
                    continue

                relid += 1

                # flood fill
                seg2, mask = dfs(seg, seg2, mask, relid, seg[z,y,x], z,y,x)

    print "number of segments: {}-->{}".format( np.unique(seg).shape[0], np.unique(seg2).shape[0] )
    return seg2


cpdef np.ndarray[DTYPE_t, ndim=3] relabel2d(np.ndarray[DTYPE_t, ndim=3] seg):
    '''
    This results layers of 2d segments, and with N higher
    than the original number of segments.

    This function also ignores the '0' segment, leaving it as passed in.
    ''' 

    sz = seg.shape[0]
    sy = seg.shape[1]
    sx = seg.shape[2]

    cdef np.ndarray[DTYPE_t, ndim=3] seg2 = np.zeros((sz, sy, sx), dtype=DTYPE)

    cdef np.ndarray[DTYPE_t, ndim=1] unique_vals
    cdef int num_unique_vals
    
    cdef np.ndarray[DTYPE_t, ndim=2] source_slice
    cdef np.ndarray[DTYPE_t, ndim=2] dest_slice

    cdef np.uint32_t new_id = 1
    cdef np.uint32_t z,y,x

    for z in xrange(sz):

        source_slice = seg[z,:,:]
        dest_slice   = seg2[z,:,:]

        unique_vals = np.unique( source_slice )

        num_unique_vals = unique_vals.shape[0]

        for i in xrange(num_unique_vals):

            if unique_vals[i] == 0:
                continue

            dest_slice[source_slice == unique_vals[i]] = new_id
            new_id += 1

    print "number of segments: {}-->{}".format( np.unique(seg).shape[0], np.unique(seg2).shape[0] )
    return seg2


