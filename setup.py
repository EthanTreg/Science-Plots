"""
Package setup
"""
from setuptools import setup

import plots

setup(
    name='sciplots',
    version=plots.__version__,
    description='Utility to create scientific plots',
    url='https://github.com/EthanTreg/Science-Plots',
    author='Ethan Tregidga',
    author_email='ethan.tregidga@epfl.ch',
    license='MIT',
    packages=['plots'],
    install_requires=['numpy', 'scipy', 'matplotlib'],
)
