from Cython.Distutils import build_ext
from setuptools import Extension
from setuptools import find_packages
from setuptools import setup
import sys

import setup_build

if False:
    if '--no-nccl' in sys.argv:
        sys.argv.remove('--no-nccl')
        ext_modules = []
    else:
        ext_modules = [
            Extension(
                name='chainermn.nccl.nccl',
                sources=['chainermn/nccl/nccl.pyx'],
                libraries=['nccl'])
        ]
else:
    ext_modules = setup_build.get_ext_modules()
    
    
setup(
    name='chainermn',
    version='0.0.1',
    description='ChainerMN: Multi-node distributed training with Chainer',
    author='Takuya Akiba',
    author_email='akiba@preferred.jp',
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass={'build_ext': build_ext}
)
