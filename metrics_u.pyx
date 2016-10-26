# -*- coding: utf-8 -*-
"""
Metrics Utilities - metrics_u.pyx

Nicholas Turner, 2015-6
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


##ctypedef np.uint32_t DTYPE_t
include "global_vars.pyx"


cpdef np.float64_t shannon_entropy(np.ndarray[np.float64_t, ndim=1] arr):
    '''
    Calculates the Shannon Entropy for a given set of [0,1] (probability) values
    '''

    sx = arr.shape[0]

    cdef np.float64_t result = 0
    cdef np.float64_t val
    cdef int i

    for i in xrange(sx):

        val = arr[i]

        if val == 0:
            continue
        else:
            result += (-1.0) * val * log(val)

    return result


cpdef np.ndarray[np.float64_t, ndim=1] shannon_entropy_vec(np.ndarray[np.float64_t, ndim=1] arr):
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


cpdef np.float64_t conditional_entropy(np.ndarray[np.float64_t, ndim=1] vals,
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
    '''

    sx = vals.shape[0]

    cdef np.float64_t result = 0.0
    cdef np.float64_t val
    cdef np.int32_t axis_index
    cdef int i

    for i in xrange(sx):

        val = vals[i]

        if val == 0:
            continue
        else:
            axis_index = axis_indices[i]
            result += (-1.0) * val * log( val / axis_sum[axis_index] )

    return result

cpdef np.ndarray[np.float64_t, ndim=1] conditional_entropy_vec(np.ndarray[np.float64_t, ndim=1] vals,
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
