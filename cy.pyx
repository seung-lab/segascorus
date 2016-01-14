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
#@cython.boundscheck(False) # turn off bounds-checking for entire function
#@cython.wraparound(False)
#@cython.nonecheck(False)

#THIS SHOULD BE AN INT TYPE
DTYPE = np.uint32
ctypedef np.uint32_t DTYPE_t

cpdef np.ndarray[np.float64_t, ndim=1] shannon_entropy(np.ndarray[np.float64_t, ndim=1] arr):
    '''
    Calculates the Shannon Entropy for a given set of [0,1] (probability) values 

    Returns a 1d array of the same size as the input. The total H(S) can be computed by
    a sum over the returned array

    will return nan values if arr contains zeros
    '''

    sx = arr.shape[0]
    
    cdef np.ndarray[np.float64_t, ndim=1] result = np.empty((sx,), dtype=np.float64)

    cdef np.float64_t val
    cdef int i

    for i in xrange(sx):

        val = arr[i]

        if val == 0:
            result[i] = 0

        else:
            result[i] = (-1.0) * val * log(val)

    return result


cpdef np.ndarray[np.float64_t, ndim=1] conditional_entropy(np.ndarray[np.float64_t, ndim=1] vals,
    np.ndarray[np.int32_t, ndim=1] axis_indices,
    np.ndarray[np.float64_t, ndim=1] axis_sum):
    '''
    Calculates the Conditional Shannon Entropy for values of a joint distribution (vals),
    along with the values of a marginal distribution (axis_sum), and the indices of the
    marginal distribution containing the joint values (axis_indices)

    Another way to think about these arguments is the following:
    - a 1D array of p_ij values in the overlap matrix (vals)
    - an array of the indices into the proper axis for each p_ij value (axis_indices)
    - the sum over the values in a particular row/col (s_i or t_j: axis_sum) 

    axis_indices is specified as int32 to fit with the result of sp.find
    vals and axis sum should be (float) probability values

    will return nans if vals contains 0
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


cpdef np.ndarray[DTYPE_t, ndim=1] choose_two(np.ndarray[DTYPE_t, ndim=1] arr):
    '''
    Vectorized version of (n choose 2) operation over a 1d numpy array

    Assumes values of the array are non-negative integers

    Does NOT report overflow errors
    '''

    sx = arr.shape[0]
    
    cdef np.ndarray[DTYPE_t, ndim=1] result = np.empty((sx,), dtype=DTYPE)

    cdef DTYPE_t val
    cdef int i

    for i in xrange(sx):

        val = arr[i]
        result[i] = int((val / 2.0) * (val-1))

    return result


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


cpdef split_zeros(np.ndarray[DTYPE_t, ndim=1] seg,
    DTYPE_t segmax):
    '''
    Relabels the zero segment of the passed array as
    singleton voxels (with new ids). Also returns the new
    maximum segment id.
    '''

    s = seg.shape[0]

    cdef np.ndarray[DTYPE_t, ndim=1] res = np.empty((s,), dtype=DTYPE)

    cdef int i

    for i in xrange(s):

        if seg[i] == 0:
            segmax += 1
            res[i] = segmax

        else:
            res[i] = seg[i]

    return res, segmax 


cpdef overlap_matrix(
    np.ndarray[DTYPE_t, ndim=1] seg1, 
    np.ndarray[DTYPE_t, ndim=1] seg2,
    bint split0):
    '''
    Calculates the overlap matrix between two segmentations of a volume

    Can also split the '0' segmentation of both arrays into new singleton
    segments to reflect the semantics of the current watershed code (using split0)
    '''

    cdef DTYPE_t seg1max = np.max(seg1)
    cdef DTYPE_t seg2max = np.max(seg2)

    #Debug output
    # print "max before split (including implicit 0)"
    # print seg1max
    # print seg2max

    if split0:
        seg1, seg1max = split_zeros(seg1, seg1max)
        seg2, seg2max = split_zeros(seg2, seg2max)

    cdef int num_segs1 = seg1max + 1 #+1 accounts for base 0 indexing
    cdef int num_segs2 = seg2max + 1

    #MORE lembas bread
    # print "max after split (including implicit 0)"
    # print seg1max
    # print seg2max

    #Representing the sparse overlap matrix as row/col/val arrays
    res = sp.dok_matrix( (num_segs1, num_segs2), DTYPE )

    cdef int i

    for i in xrange(seg1.size):
        res[ seg1[i], seg2[i] ] += 1
   
    return res.tocsr()


#===========================================
# Boundary Thinning Code - not quite finished
# (will also be finished when enough people complain)

cdef list neighbors(int i, int j, int k, 
    int si, int sj, int sk, bint two_dim):
    '''Returns a list of 4-connectivity neighbors within a 2d image'''

    # Init
    cdef list res = []
    cdef int[3] orig = (i, j, k)
    cdef int[3] shape = (si, sj, sk)

    #Index order: z,y,x
    cdef int dim

    cdef int[2] twod_dims = (1,2)
    cdef int[3] threed_dims = (0,1,2)

    cdef int[3] candidate
    
    for dim in threed_dims:

        candidate = (orig[0], orig[1], orig[2])
        candidate[dim] = candidate[dim] + 1

        if candidate[dim] < shape[dim]:
            res.append(candidate)

        candidate = (orig[0], orig[1], orig[2])
        candidate[dim] = candidate[dim] - 1

        if 0 <= candidate[dim]:
            res.append(candidate)

    return res

cdef bint all_equal(DTYPE_t[:] arr, int length, DTYPE_t value):
    '''Faster test for "all({arr}=={value})" within cython'''

    cdef int i = 0
    while i < length:
        if arr[i] != value:
            return False
        i += 1
    return True

cpdef np.ndarray[DTYPE_t, ndim=3] thin_boundary(np.ndarray[DTYPE_t, ndim=3] vol, bint twodim):
    '''
    Returns a copy of the passed volume with one pixel of boundary pixels
    from each nonzero compartment removed for either 2d or 3d connectivity.

    2d connectivity yields 4-connectivity
    3d connectivity yields 6-connectivity
    '''

    s0 = vol.shape[0]
    s1 = vol.shape[1]
    s2 = vol.shape[2]

    #Init
    cdef np.ndarray[DTYPE_t, ndim=3] res = np.empty((s0, s1, s2), dtype=DTYPE)
    cdef int z,y,x, num_neighbors
    cdef list neighbor_indices
    cdef DTYPE_t val
    #a structure holding the maximum number of neighbors for 3d connectivity
    cdef DTYPE_t[6] neighbor_values = (0,0,0,0,0,0)

    for z in xrange(s0):
        for y in xrange(s1):
            for x in xrange(s2):

                val = vol[z,y,x]
                #Find indices of neighbors
                neighbor_indices = neighbors(z,y,x, s0,s1,s2, twodim)

                #Find their values
                num_neighbors = len(neighbor_indices)
                for i in xrange(num_neighbors):
                    neighbor_values[i] = vol[
                        neighbor_indices[i][0],
                        neighbor_indices[i][1],
                        neighbor_indices[i][2]]

                #Assign new label according to collection of neighbor
                # values
                if all_equal(neighbor_values, num_neighbors, val):
                    res[z,y,x] = val
                else:
                    res[z,y,x] = 0
    return res

