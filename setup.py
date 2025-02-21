"""
Package setup
"""
from setuptools import setup

import sciplots

setup(
    name='sciplots',
    version=sciplots.__version__,
    description='Utility to create scientific sciplots',
    url='https://github.com/EthanTreg/Science-Plots',
    author='Ethan Tregidga',
    author_email='ethan.tregidga@epfl.ch',
    license='MIT',
    packages=['sciplots'],
    install_requires=['numpy', 'scipy', 'matplotlib', 'SciencePlots'],
)
