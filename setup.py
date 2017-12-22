from Cython.Distutils import build_ext
from setuptools import Extension
from setuptools import find_packages
from setuptools import setup

import os
import sys


install_requires = [
    'cffi',
    'chainer >=3.0.0',
    'cython',
    'mpi4py',
]

ext_modules = [
    Extension(
        name='chainermn.nccl.nccl',
        sources=['chainermn/nccl/nccl.pyx'],
        libraries=['nccl'])
]

if '--no-nccl' in sys.argv:
    sys.argv.remove('--no-nccl')
    ext_modules = []
elif os.environ.get('READTHEDOCS', None) == 'True':
    ext_modules = []
    install_requires.remove('mpi4py')  # mpi4py cannot be installed without MPI
else:
    install_requires.append('cupy')

setup(
    name='chainermn',
    version='1.1.0',
    description='ChainerMN: Multi-node distributed training with Chainer',
    author='Takuya Akiba',
    author_email='akiba@preferred.jp',
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass={'build_ext': build_ext},
    install_requires=install_requires,
    test_requires=['pytest']
)
