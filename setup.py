#!/usr/bin/env python

from distutils.core import setup

setup(name='pybh',
      version='0.2',
      description='Personal python utilities',
      author='Benjamin Hepp',
      author_email='benjamin.hepp@posteo.de',
      license='BSD 3 License',
      packages=['pybh'],
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

