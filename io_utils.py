#!/usr/bin/env python
__doc__ = '''
I/O Utilities - io_utils.py

Nicholas Turner, Ignacio Tartavull 2016
'''


from os import path


def import_tif(filename):
    import tifffile

    return tifffile.imread(filename).astype(DTYPE)


def import_h5(filename):
    import h5py

    f = h5py.File(filename)
    return f['/main'].value.astype(DTYPE)


def import_file(filename):
	'''
	Takes a path to a file in the form of an string and decides which import
	method to use base on the filename extension.
	If it doesn't find a suitable import method raises ValueError
	'''
	valid_h5_extensions = ['.h5','.hdf5']
	valid_tif_extensions = ['.tif','.tiff']
	_, file_extension = path.splitext(filename)
	if file_extension in valid_h5_extensions:
		return import_h5(filename)
	elif file_extension in valid_tif_extensions:
		return import_tif(filename)
	else:
		raise ValueError('Failed to load filename {} ,Valid file extensions are {}'
			    			.format(filename, valid_h5_extensions+valid_tif_extensions))
