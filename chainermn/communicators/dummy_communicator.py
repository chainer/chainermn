from chainermn.communicators import _base
from chainermn.communicators import _communication_utility
from chainermn.communicators import _memory_utility


class DummyCommunicator(_base.CommunicatorBase):

    """Dummy communicator that does not communicate at all.

    This class is intended to measure the overhead of packing and unpacking.
    This class does not pass the tests.
    """

    def __init__(self, mpi_comm):
        super(DummyCommunicator, self).__init__(mpi_comm, use_nccl=True)

        self.gpu_buffer_a = _memory_utility.DeviceMemory()

    def broadcast_data(self, model):
        _communication_utility.broadcast_naive(self.mpi_comm, model)

    def allreduce_grad(self, model):
        self._init_comms()

        params = _memory_utility.extract_params(model)
        itemsize = 4
        n_elems_total = sum(param.grad.size for param in params)
        n_bytes_total = n_elems_total * itemsize
        self.gpu_buffer_a.assign(n_bytes_total)

        _memory_utility.pack_params(
            params, itemsize, 'grad', self.gpu_buffer_a)

        _memory_utility.unpack_params(
            params, itemsize, 'grad', self.gpu_buffer_a)
