
from distutils.core import setup
from Cython.Build import cythonize
import numpy as np

setup(
	name = "dynamic time warping",
	ext_modules = cythonize("dtw.pyx"),
	include_dirs = [np.get_include()]
	)
