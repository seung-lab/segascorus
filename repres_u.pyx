# -*- coding: utf-8 -*-
__doc__ = """
Data Representation Utilities - repres_u.py
"""

#defining NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
import numpy as np
cimport numpy as np
import scipy.sparse as sp
cimport cython

#Defines DTYPE and DTYPE_t
include "global_vars.pyx"

#@cython.boundscheck(False) # turn off bounds-checking for entire function
#@cython.wraparound(False)
#@cython.nonecheck(False)


cpdef map_over_vals(
    np.ndarray[int, ndim=1] arr,
    dict d):
    """
    Maps the dict as a fn over the values of the arr
    """

    l = arr.shape[0]
    cdef int i

    for i in xrange(l):

      arr[i] = d.get(arr[i],arr[i])


cpdef overlap_matrix_coo(
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

    #Representing the sparse overlap matrix as row/col/val arrays
    cdef np.ndarray[DTYPE_t] om_vals = np.ones(seg1.size, dtype=DTYPE) #value for now will always be one

    return sp.coo_matrix((om_vals, (seg1, seg2)), shape=(num_segs1, num_segs2)).tocsr()


cpdef overlap_matrix_dok(
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
    cdef DTYPE_t v

    for i in xrange(seg1.size):
        v = res.get( (seg1[i],seg2[i]), 0  )
        res.update( {(seg1[i],seg2[i]):v+1})

    return res.tocsr()


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
