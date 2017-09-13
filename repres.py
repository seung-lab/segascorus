#!/usr/bin/env python
__doc__ = """
Data Representations - repres.py
"""
import numpy as np
import scipy.sparse as sp
import global_vars
import repres_u

DTYPE = global_vars.DTYPE

class OverlapMatrix:

    constructor_fns = {
      "coo": repres_u.overlap_matrix_coo,
      "dok": repres_u.overlap_matrix_dok
    }

    def __init__(self, seg1, seg2, mattype, split0=True):

        assert mattype in self.constructor_fns.keys()

        self.mat = self.constructor_fns[mattype](seg1.ravel(), seg2.ravel(), split0)
        #remove duplicates
        self.mat = self.mat.tocsr().tocoo(copy=False)



    def sum(self, axis):
        return self.mat.sum(axis)


    def find(self):
        #sp.find turns out to be a bottleneck in scipyV0.18 as it repeatedly
        # attempts to sum duplicates. There should be no duplicates after
        # __init__,
        # so we can remove that step if we're careful about maintaining it
        A = self.mat.tocoo(copy=True)
        return A.row,A.col,A.data


    def nonzeros(self):
        return np.copy(self.mat.data)


    def merge_to_thr(self, dend_pairs, dend_values, thresh):

        i,j,v = self.find()
        repres_u.map_to_MST_thresh( i, dend_values, dend_pairs, thresh )

        #can introduce duplicates here, tocsr should sum them
        self.mat = sp.coo_matrix( (v, (i,j)) ).tocsr().tocoo(copy=False)
