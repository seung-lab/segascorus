# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 16:40:02 2015

Cython Tools for error.py

Nicholas Turner, 2015
"""
#defining NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
import numpy as np
cimport numpy as np
import scipy.sparse as sp
cimport cython

from libc.math cimport log
#@cython.boundscheck(False) # turn of bounds-checking for entire function
#@cython.wraparound(False)
#@cython.nonecheck(False)

DTYPE = np.uint32
ctypedef np.uint32_t DTYPE_t

cpdef np.ndarray[np.float64_t, ndim=1] shannon_entropy(np.ndarray[np.float64_t, ndim=1] arr):
    '''
    Calculates the Shannon Entropy for a given set of [0,1] (probability) values 

    Returns a 1d array of the same size as the input. The total H(S) can be computed by
    '''

    sx = arr.shape[0]
    
    cdef np.ndarray[np.float64_t, ndim=1] result = np.empty((sx,), dtype=np.float64)

    cdef np.float64_t val
    for i in xrange(sx):

        val = arr[i]

        if val == 0:
            result[i] = 0

        else:
            result[i] = (-1.0) * val * log(val)

    return result

cpdef np.ndarray[np.float64_t, ndim=1] om_VI_byaxis(np.ndarray[np.int32_t, ndim=1] axis_indices,
    np.ndarray[np.float64_t, ndim=1] vals,
    np.ndarray[np.float64_t, ndim=1] axis_sum):
    '''
    The Variation of Information can be split between split error and merge error, yet
     the calculation of each is the same for a specific axis (row/col respectively). This function calculates
     the VI terms for a type of error (split/merge) given

    - an array of the indices into the proper axis for each value
    - the array of values in the overlap matrix
    - the sum over the values in a particular row/col 
    '''

    sx = vals.shape[0]

    cdef np.ndarray[np.float64_t, ndim=1] result = np.empty((sx,), dtype=np.float64)

    cdef np.float64_t val
    cdef np.int32_t axis_index
    cdef int i
    for i in xrange(sx):

        val = vals[i]

        if val == 0:
            result[i] = 0.0

        else:
            axis_index = axis_indices[i]
            result[i] = (-1.0) * val * log( val / axis_sum[axis_index] )

    return result

cpdef np.ndarray[DTYPE_t, ndim=1] choose_two(np.ndarray[DTYPE_t] arr, ndim=1):

    sx = arr.shape[0]
    
    cdef np.ndarray[DTYPE_t, ndim=1] result = np.empty((sx,), dtype=DTYPE)

    cdef DTYPE_t val
    for i in xrange(sx):

        val = arr[i]
        result[i] = (val / 2) * (val-1)

    return result

cpdef np.ndarray[DTYPE_t, ndim=3] relabel_segmentation(np.ndarray[np.uint32_t, ndim=3] seg, np.ndarray[np.uint32_t, ndim=1] relabelling):

    sz = seg.shape[0]
    sy = seg.shape[1]
    sx = seg.shape[2]

    cdef np.ndarray[DTYPE_t, ndim=3] result = np.empty((sz, sy, sx), dtype=DTYPE)


    for z in xrange(sz):
        for y in xrange(sy):
            for x in xrange(sx):

                result[z,y,x] = relabelling[seg[z,y,x]]

    return result

#%% depth first search
#TO DO make option for 2d/3d
cdef dfs(np.ndarray[np.uint32_t, ndim=3] seg,\
        np.ndarray[np.uint32_t, ndim=3] seg2,\
        np.ndarray[np.uint8_t,  ndim=3, cast=True] mask, \
        np.uint32_t relid, \
        np.uint32_t label, \
        int z, int y, int x):
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
        if y+1<seg.shape[1] and seg[z,y+1,x] == label and not mask[z,y+1,x] :
            seeds.append((z,y+1,x))
        if y-1>=0    and seg[z,y-1,x] == label and not mask[z,y-1,x] :
            seeds.append((z,y-1,x))          
        if x+1<seg.shape[2] and seg[z,y,x+1] == label and not mask[z,y,x+1] :
            seeds.append((z,y,x+1))
        if x-1>=0    and seg[z,y,x-1] == label and not mask[z,y,x-1] :
            seeds.append((z,y,x-1))       
    return seg2, mask

#%% relabel by connectivity analysis
cpdef np.ndarray[DTYPE_t, ndim=3] relabel1N(np.ndarray[DTYPE_t, ndim=3] seg):
    print 'relabel by connectivity analysis ...'
    # masker for visiting
    cdef np.ndarray[np.uint8_t,    ndim=3, cast=True] mask 
    mask = (seg==0)
    
    sz = seg.shape[0]
    sy = seg.shape[1]
    sx = seg.shape[2]

    cdef np.ndarray[DTYPE_t, ndim=3] seg2 = np.zeros((sz, sy, sx), dtype=DTYPE)   # change to np.zeros ?
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

cpdef overlap_matrix(
    np.ndarray[DTYPE_t, ndim=1] seg1, 
    np.ndarray[DTYPE_t, ndim=1] seg2):
    '''Calculates the overlap matrix between two segmentations of a volume'''

    cdef DTYPE_t seg1max = np.max(seg1)
    cdef DTYPE_t seg2max = np.max(seg2)

    cdef int num_segs1 = seg1max + 1 #+1 accounts for '0' segment
    cdef int num_segs2 = seg2max + 1

    #Representing the sparse overlap matrix as row/col/val arrays
    cdef np.ndarray[DTYPE_t] om_vals
    om_vals = np.ones(seg1.size, dtype=DTYPE) #value for now will always be one
    
    return sp.coo_matrix((om_vals, (seg1, seg2)), shape=(num_segs1, num_segs2)).tocsr()

