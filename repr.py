#!/usr/bin/env python
import numpy as np
import scipy.sparse as sp
import repr_utils as repr_u

class OverlapMatrix:

    constructor_fns = {
      "coo": repr_u.overlap_matrix_coo,
      "dok": repr_u.overlap_matrix_dok
    }

    def __init__(self, seg1, seg2, mattype, split0=True):

        assert mattype in constructor_fns.keys()

        self.mat = constructor_fns[mattype](seg1.ravel(), seg2.ravel(), split0)

        #Avoiding a little overhead by storing the
        # margin sums between calls
        self.__sums = (None,None)


    def sum(self, axis):

        if self.__sums[axis] == None:
            self.__sums[axis] == self.mat.sum(axis)

        return self.__sums[axis]


    def find(self):
        pass


    def values(self):
        pass


    def merge(self, groups):
        #Note, need to delete old axis sums here
        pass
