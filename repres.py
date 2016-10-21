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

        #Avoiding a little overhead by storing the
        # margin sums between calls
        #self.__sums = (None,None)


    def sum(self, axis):

        #if self.__sums[axis] == None:
        #    self.__sums[axis] == self.mat.sum(axis)

        #return self.__sums[axis]
        return self.mat.sum(axis)


    def find(self):
        return sp.find(self.mat)


    def nonzeros(self):
        return np.copy(self.mat.data)


    def map(self, mapping):

        i,j,v = self.find()
        repres_u.map_over_vals( i, mapping )

        self.mat = sp.coo_matrix( (v, (i,j)) ).tocsr()
