import chainer
import copy
import multiprocessing.pool
try:
    import cupy as cp
    _cupy_avail = True
except ImportError:
    _cupy_avail = False

class DoubleBufferingOptimizer(object):

    def __init__(self, actual_optimizer, communicator):
        super(DoubleBufferingOptimizer, self).__setattr__(
            'communicator', communicator)
        super(DoubleBufferingOptimizer, self).__setattr__(
            'actual_optimizer', actual_optimizer)
        super(DoubleBufferingOptimizer, self).__setattr__(
            'needs_broadcast', True)
         
        self.needs_update = False
        self.device = None
        self.thread_pool = multiprocessing.pool.ThreadPool(1)
        self.communicated_target = None
        self.target_params_list = [[], []]
        self.allreduce_grad_res = None
        self.allreduce_grad_stream = chainer.cuda.Stream(non_blocking=True)
        
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
            super(DoubleBufferingOptimizer, self).__setattr__(
                'needs_broadcast', False)
            self.device = cp.cuda.runtime.getDevice()
            self.communicated_target = copy.deepcopy(target)
            self.target_params_list = [list(sorted(self.target.namedparams())),
                                       list(sorted(self.communicated_target.namedparams()))]
        else:
            self.wait()
            self.swap_grad(self.target_params_list[0], self.target_params_list[1])
            self.allreduce_grad_res = self.thread_pool.apply_async(self.allreduce_grad)
        
            if self.needs_update:
                self.actual_optimizer.update(None, *args, **kwds)
            else:
                self.needs_update = True

    def allreduce_grad(self) :
        chainer.cuda.get_device(self.device).use()
        self.communicator.allreduce_grad(self.communicated_target, self.allreduce_grad_stream)

    def swap_grad(self, target1_params, target2_params) :
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

