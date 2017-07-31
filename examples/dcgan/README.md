# DCGAN

This is an example implementation of DCGAN (https://arxiv.org/abs/1511.06434) 
trained with multi-GPU using `chainermn`.

This code uses Cifar-10 dataset by default.
You can use your own dataset by specifying `--dataset` argument to the directory consisting of image files for training.
The model assumes the resolution of an input image is 32x32.
If you want to use another image resolution, you need to change the network architecture in net.py.

Below is an example learning result using cifar-10 dataset after 200 epoch.

![example result](https://raw.githubusercontent.com/pfnet/chainer/master/examples/dcgan/example_image.png)

## How to run the code

For example, below command is to train the model using 4 GPUs (= processes).

```
mpiexec -n 4 python train_dcgan.py -g
```
