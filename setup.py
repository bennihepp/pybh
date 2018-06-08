#!/usr/bin/env python

import numpy
from distutils.core import setup, Extension

transformations_module = Extension(
    'pybh.contrib._transformations',
    #define_macros = [('MAJOR_VERSION', '1'),
    #                 ('MINOR_VERSION', '0')],
    include_dirs = [numpy.get_include()],
    libraries = [],
    library_dirs = [],
    sources=['pybh/contrib/transformations.c'])

setup(name='pybh',
      version='0.2',
      description='Personal python utilities',
      author='Benjamin Hepp',
      author_email='benjamin.hepp@posteo.de',
      license='BSD 3 License',
      packages=['pybh'],
      ext_modules=[transformations_module],
      install_requires=[
          'numpy',
          'Pillow',
          # Requirements for rendering
          'msgpack',
          'moderngl',
          'pyglet',
          'pyassimp',
          'pywavefront',
      ]
)

