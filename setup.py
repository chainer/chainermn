from setuptools import Extension
from setuptools import find_packages
from setuptools import setup

import os
import sys


install_requires = [
    'cffi',
    'chainer >=3.0.0',
    'mpi4py',
]


if os.environ.get('READTHEDOCS', None) == 'True':
    install_requires.remove('mpi4py')  # mpi4py cannot be installed without MPI

setup(
    name='chainermn',
    version='1.1.0',
    description='ChainerMN: Multi-node distributed training with Chainer',
    author='Takuya Akiba',
    author_email='akiba@preferred.jp',
    packages=find_packages(),
    install_requires=install_requires,
    test_requires=['pytest']
)
