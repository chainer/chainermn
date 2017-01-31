from setuptools import find_packages
from setuptools import setup

setup(
    name='chainermn',
    version='0.0.1',
    description='ChainerMN: Multi-node distributed training with Chainer',
    author='Takuya Akiba',
    author_email='akiba@preferred.jp',
    packages=find_packages(),
)
