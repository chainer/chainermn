<div align="center"><img src="docs/image/chainermn1000px.png" width="800"/></div>

[![Build Status](https://travis-ci.org/chainer/chainermn.svg?branch=master)](https://travis-ci.org/chainer/chainermn)
[![Documentation Status](https://readthedocs.org/projects/chainermn/badge/?version=latest)](http://chainermn.readthedocs.io/en/latest/?badge=latest)
[![PyPI](https://img.shields.io/pypi/v/chainermn.svg)](https://pypi.python.org/pypi/chainermn)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

# ChainerMN: Distributed Deep Learning with Chainer

[Documentation](https://chainermn.readthedocs.io/en/latest) |
[Installation](https://chainermn.readthedocs.io/en/latest/installation/index.html) |
[Examples](examples) |
[Release Notes](https://github.com/pfnet/chainermn/releases)

*ChainerMN* is an additional package for [Chainer](https://github.com/pfnet/chainer), a flexible deep learning framework. ChainerMN enables multi-node distributed deep learning with the following features:

* **Scalable** --- it makes full use of the latest technologies such as NVIDIA NCCL and CUDA-Aware MPI,
* **Flexible** --- even dynamic neural networks can be trained in parallel thanks to Chainer's flexibility, and
* **Easy** --- minimal changes to existing user code are required.

[This blog post](http://chainer.org/general/2017/02/08/Performance-of-Distributed-Deep-Learning-Using-ChainerMN.html) provides our benchmark results using up to 128 GPUs.

## Installation

ChainerMN can be used for both inner-node (i.e., multiple GPUs inside a node) and inter-node settings.
For inter-node settings, we highly recommend to use high-speed interconnects such as InfiniBand.

In addition to Chainer, ChainerMN depends on the following software libraries: CUDA-Aware MPI, NVIDIA NCCL, and a few Python packages.
After setting them up, ChainerMN can be installed via PyPI:

```
pip install chainermn
```

Please refer to the [installation guide](https://chainermn.readthedocs.io/en/latest/installation/index.html) for more information.


## Getting Started

You can invoke MNIST example with four workers by the following command:

```
mpiexec -n 4 python examples/mnist/train_mnist.py
```

* **[Chainer Tutorial](http://docs.chainer.org/en/latest/tutorial/index.html)** --- If you are new to Chainer, we recommend to start from this.
* **[ChainerMN Tutorial](https://chainermn.readthedocs.org/en/latest/tutorial)** --- In this tutorial, we explain how to modify your existing code using Chainer to enable distributed training with ChainerMN in a step-by-step manner.
* **[Examples](examples)** --- The examples are based on the official examples of Chainer and the differences are highlighted.


## Contributing
Any contribution to ChainerMN would be highly appreciated.
Please refer to [Chainer Contribution Guide](http://docs.chainer.org/en/latest/contribution.html).


## License

[MIT License](LICENSE)

## Reference

Akiba, T., Fukuda, K. and Suzuki, S.,
ChainerMN: Scalable Distributed Deep Learning Framework,
*Proceedings of Workshop on ML Systems in
The Thirty-first Annual Conference on Neural Information Processing Systems (NIPS)*, (2017)
[URL](http://learningsys.org/nips17/assets/papers/paper_25.pdf), [BibTex](chainermn_bibtex.txt)