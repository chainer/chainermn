import chainer
import copy
import multiprocessing.pool


class _DoubleBufferingOptimizer(object):

    def __init__(self, actual_optimizer, communicator):
        super(_DoubleBufferingOptimizer, self).__setattr__(
            'communicator', communicator)
        super(_DoubleBufferingOptimizer, self).__setattr__(
            'actual_optimizer', actual_optimizer)
        super(_DoubleBufferingOptimizer, self).__setattr__(
            'needs_broadcast', True)
        super(_DoubleBufferingOptimizer, self).__setattr__(
            'needs_update', False)
        super(_DoubleBufferingOptimizer, self).__setattr__(
            'device', None)
        super(_DoubleBufferingOptimizer, self).__setattr__(
            'thread_pool', multiprocessing.pool.ThreadPool(1))
        super(_DoubleBufferingOptimizer, self).__setattr__(
            'communicated_target', None)
        super(_DoubleBufferingOptimizer, self).__setattr__(
            'target_params_list', [[], []])
        super(_DoubleBufferingOptimizer, self).__setattr__(
            'allreduce_grad_res', None)
        super(_DoubleBufferingOptimizer, self).__setattr__(
            'allreduce_grad_stream', chainer.cuda.Stream(non_blocking=True))

    def update(self, lossfun=None, *args, **kwds):
        target = self.target
        if lossfun is not None:
            use_cleargrads = getattr(self, '_use_cleargrads', False)
            loss = lossfun(*args, **kwds)
            if use_cleargrads:
                target.cleargrads()
            else:
                target.zerograds()
            loss.backward()
            del loss

        if self.needs_broadcast:
            self.communicator.broadcast_data(target)
            super(_DoubleBufferingOptimizer, self).__setattr__(
                'needs_broadcast', False)
            super(_DoubleBufferingOptimizer, self).__setattr__(
                'device',
                chainer.cuda.get_device_from_id(target._device_id))
            super(_DoubleBufferingOptimizer, self).__setattr__(
                'communicated_target', copy.deepcopy(target))
            super(_DoubleBufferingOptimizer, self).__setattr__(
                'target_params_list', [
                    list(sorted(self.target.namedparams())),
                    list(sorted(self.communicated_target.namedparams()))])
        else:
            self.wait()
            self.swap_grad(self.target_params_list[0],
                           self.target_params_list[1])
            super(_DoubleBufferingOptimizer, self).__setattr__(
                'allreduce_grad_res',
                self.thread_pool.apply_async(self.allreduce_grad))
            if self.needs_update:
                self.actual_optimizer.update(None, *args, **kwds)
            else:
                super(_DoubleBufferingOptimizer, self).__setattr__(
                    'needs_update', True)

    def allreduce_grad(self):
        chainer.cuda.get_device(self.device).use()
        self.communicator.allreduce_grad(
            self.communicated_target, self.allreduce_grad_stream)

    def swap_grad(self, target1_params, target2_params):
        for param1, param2 in zip(target1_params, target2_params):
            name1, var1 = param1
            name2, var2 = param2
            assert name1 == name2
            var1.grad, var2.grad = var2.grad, var1.grad

    def wait(self):
        if self.allreduce_grad_res is not None:
            self.allreduce_grad_res.get()

    def __getattr__(self, attr_name):
        return getattr(self.actual_optimizer, attr_name)

    def __setattr__(self, attr_name, value):
        setattr(self.actual_optimizer, attr_name, value)
