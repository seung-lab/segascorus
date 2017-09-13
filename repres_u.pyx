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


from libcpp.unordered_map cimport unordered_map
cpdef map_to_MST_thresh(
    np.ndarray[int, ndim=1] segids,
    np.ndarray[np.float32_t, ndim=1] segPairAffinities,
    np.ndarray[DTYPE_t, ndim=2] segPairs,
    float t):

    mapping = thresh_MST( segPairAffinities, segPairs, t )


    cdef int l = segids.shape[0]
    cdef int i, v

    for i in range(l):
      v = mapping[ segids[i] ]

      if v == 0:
        continue

      segids[i] = v


cpdef dict_thresh_MST(
    np.ndarray[np.float32_t, ndim=1] dend_values,
    np.ndarray[DTYPE_t, ndim=2] dend_pairs,
    np.float64_t t):

    mapping = thresh_MST(dend_values, dend_pairs, t)

    d = {}
    l = dend_values.shape[0]

    for i in range(l):
      child, parent = dend_pairs[:,i]

      d[child] = mapping[child]

    return d


cdef unordered_map[int,int] thresh_MST(
    np.ndarray[np.float32_t, ndim=1] dend_values,
    np.ndarray[DTYPE_t, ndim=2] dend_pairs,
    np.float64_t t):

    cdef unordered_map[int,int] assignment

    l = dend_values.shape[0]
    cdef int i
    cdef DTYPE_t child, parent

    for i in range(l):
      if dend_values[i] > t:

        child, parent = dend_pairs[:,i]

        if assignment[parent] == 0:
          assignment[child] = parent
        else:
          assignment[child] = assignment[parent]

    cdef DTYPE_t current, next_val

    for i in range(l):
      child, parent = dend_pairs[:,i]

      while True:
        current = assignment[child]

        if current == 0: #should only happen on the first iter
          break

        next_val = assignment[current]
        if next_val == 0:
          break

        assignment[child] = next_val 

    return assignment


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

cpdef thresh_seg(
      np.ndarray[DTYPE_t,ndim=3] seg,
      np.ndarray[np.float32_t, ndim=1] dend_values,
      np.ndarray[DTYPE_t, ndim=2] dend_pairs,
      np.float64_t t):

    mapping = thresh_MST( dend_values, dend_pairs, t )

    sz = seg.shape[0]
    sy = seg.shape[1]
    sx = seg.shape[2]

    cdef int z, y, x
    cdef DTYPE_t v

    for z in xrange(sz):
      for y in xrange(sy):
        for x in xrange(sx):

          v = seg[z,y,x];
          if mapping[v] == 0:
            continue

          seg[z,y,x] = mapping[v]


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


    #Representing the sparse overlap matrix as row/col/val arrays
    cdef np.ndarray[DTYPE_t] om_vals = np.ones(seg1.size, dtype=DTYPE) #value for now will always be one

    return sp.coo_matrix((om_vals, (seg1, seg2)),
                         shape=(num_segs1, num_segs2))


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

    #Debug output (lembas bread)
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

    return res.tocoo()


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

#Unused code below - may be useful later
cpdef _sum_duplicates(
    np.ndarray[int, ndim=1] rows,
    np.ndarray[int, ndim=1] cols,
    np.ndarray[DTYPE_t, ndim=1] data):

    order = np.lexsort((rows, cols))

    rows, cols, data = rows[order], cols[order], data[order]
    unique_mask = find_unique_coords(rows, cols)

    sum_to_unique_locs(data, unique_mask)

    return rows[unique_mask], cols[unique_mask], data[unique_mask]
    #return unique_mask, rows, cols


cdef find_unique_coords(
    np.ndarray[int, ndim=1] rows,
    np.ndarray[int, ndim=1] cols):

    cdef int s = rows.shape[0]

    cdef np.ndarray[np.uint8_t, ndim=1, cast=True] uniques = np.ones((s,), dtype=np.bool)

    cdef int i

    for i in range(1,s):
      if rows[i] != rows[i-1]:
        continue
      if cols[i] == cols[i-1]:
        uniques[i] = False

    return uniques


cdef sum_to_unique_locs(
    np.ndarray[DTYPE_t, ndim=1] data,
    np.ndarray[np.uint8_t, ndim=1, cast=True] unique):

    cdef int s = data.shape[0]
    cdef int last_unique = 0

    for i in range(s):
      if unique[i]:
        last_unique = i
        continue
      data[last_unique] += data[i]
