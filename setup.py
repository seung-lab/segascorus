from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np

ext_modules = [
    Extension("global_vars", ["global_vars.pyx"]),
    Extension("data_prep_u", ["data_prep_u.pyx"]),
    Extension("metrics_u", ["metrics_u.pyx"]),
    Extension("repres_u", ["repres_u.pyx"],
              language="c++",
              extra_compile_args=["-std=c++11"]),
]

setup(
	ext_modules = cythonize(ext_modules),
        include_dirs = [np.get_include()]
)

