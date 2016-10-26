#!/usr/bin/env python
__doc__ = """
Plotting Functionality

usage: plot.py "${metric}" ${dset1_name} ${dset1_file} ${dset2_name} ...
"""
import numpy as np
import matplotlib.pyplot as plt
import h5py


def plot_all_curves( plots, limits=False ):
    """
    plots should be a dict from curve name to
    an array where the first column is filled with merge values,
    and the second split values
    """

    for (name, values) in plots.items():
      plt.plot( values[:,1], values[:,0], label=name )

    plt.legend()

    plt.xlabel("Split")
    plt.ylabel("Merge")

    if limits:
      plt.xlim((0,1))
      plt.ylim((0,1))

    plt.show()


def extract_values_from_file( filename, metric_name ):
    """
    """
    import h5py

    f = h5py.File(filename)

    full_name_split = "{}/Split".format(metric_name)
    full_name_merge = "{}/Merge".format(metric_name)

    split = f[full_name_split].value

    res = np.empty((split.size,2),split.dtype)

    res[:,1] = split
    res[:,0] = f[full_name_merge].value

    return res


def read_files( names, filenames, metric_name ):

    res = {}
    for i in range(len(names)):
      res[names[i]] = extract_values_from_file(filenames[i], metric_name)

    return res


def main( names, filenames, metric_name ):
    print("names: {}".format(names))
    print("fnames: {}".format(filenames))

    print("Reading files...")
    data = read_files(names, filenames, metric_name)

    print("Plotting files...")
    plot_all_curves( data )


if __name__ == '__main__':

    from sys import argv

    #odd num input arguments + default argv elem
    # = even argv len
    assert len(argv) % 2 == 0
    assert len(argv) >= 3

    num_args = len(argv)

    metric_name = argv[1]
    names  = argv[2:num_args:2]
    fnames = argv[3:num_args:2]

    main( names, fnames, metric_name )
