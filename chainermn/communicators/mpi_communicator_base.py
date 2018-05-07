import collections

import mpi4py
import numpy

import chainer.cuda
import chainer.utils
from chainermn.communicators import _communication_utility
from chainermn.communicators._communication_utility import chunked_bcast_obj
from chainermn.communicators import _memory_utility
from chainermn.communicators import communicator_base
from chainermn import nccl


def _cnt_to_dsp(cnt):
    """Utility to convert length array to cumulative array."""
    return [0] + numpy.cumsum(cnt)[:-1].tolist()


class _MessageType(object):

    def __init__(self, obj):
        if isinstance(obj, numpy.ndarray) \
                or chainer.cuda.get_array_module(obj) is not numpy:
            self.is_tuple = False
            self.narr = 1
            self.ndims = [obj.ndim]
            self.shapes = [obj.shape]
        elif isinstance(obj, collections.Iterable):
            self.is_tuple = True
            self.narr = len(obj)
            self.ndims = [x.ndim for x in obj]
            self.shapes = [x.shape for x in obj]
        else:
            raise ValueError(
                'Message object must be numpy/cupy array or tuple.')


class MpiCommunicatorBase(communicator_base.CommunicatorBase):
    '''MpiCommunicatorBase

    Implementation of communicator interface defined by
    :class:`CommunicatorBase`. This communicator assumes MPI4py and
    all ChainerMN processes are invoked by ``mpirun`` (``mpiexec``)
    command. Although this lacks several important methods such as
    ``allreduce_grad`` to be impelmented with speficic algorithm. See
    hierarcical communicator or pure_nccl communicator for example.

    '''

    def __init__(self, mpi_comm, use_nccl=False):
        self.mpi_comm = mpi_comm
        self._init_ranks()

        if use_nccl and not nccl._available:
            raise RuntimeError(
                'NCCL is not available. '
                'Please confirm that NCCL is enabled in CuPy.'
            )

        self.use_nccl = use_nccl

        # We have to delay the initialization of communicators. This is because
        # NCCL's communicators use the current CUDA devices at the time of
        # initialization. Therefore, we have to initialize NCCL communicators
        # after users set the devices to use.
        self.inter_mpi_comm = None
        self.intra_mpi_comm = None
        if self.use_nccl:
            self.intra_nccl_comm = None

    @property
    def rank(self):
        return self.mpi_comm.rank

    @property
    def intra_rank(self):
        return self._intra_rank

    @property
    def size(self):
        return self.mpi_comm.size

    def split(self, color, key):
        return self.__class__(mpi_comm=self.mpi_comm.Split(color, key))

    def alltoall(self, xs):
        """A primitive of inter-process all-to-all function.

        This method tries to invoke all-to-all communication within the
        communicator. All processes in the communicator are expected to
        invoke ``alltoall()``. This method relies on mpi4py fast communication
        optimized for numpy arrays, as well as ``send()`` and ``recv()``.

        Args:
            xs (tuple of numpy.ndarray)

        Returns:
            ys (tuple of numpy.ndarray):
                Received arrays. The length of tuple equals to
                the communicator size.
        """
        chainer.utils.experimental(
            'chainermn.communicators.MpiCommunicatorBase.alltoall')

        if len(xs) != self.size:
            raise ValueError(
                'The length of data must be same as communicator size.')

        # Type check.
        for x in xs:
            if x.dtype != numpy.float32:
                raise ValueError(
                    'alltoall only support dtype == numpy.float32')

        # Mediate #axes of arrays.
        sndims = numpy.array([x.ndim for x in xs], dtype=numpy.int32)
        rndims = numpy.empty(self.size, dtype=numpy.int32)
        self.mpi_comm.Alltoall(
            [sndims, mpi4py.MPI.INT],
            [rndims, mpi4py.MPI.INT])

        # Arbitrate shapes of arrays.
        sshapes = numpy.hstack([x.shape for x in xs]).astype(numpy.int32)
        rshapes = numpy.empty(sum(rndims), dtype=numpy.int32)
        self.mpi_comm.Alltoallv(
            [sshapes, (sndims, _cnt_to_dsp(sndims)), mpi4py.MPI.INT],
            [rshapes, (rndims, _cnt_to_dsp(rndims)), mpi4py.MPI.INT])
        shapes = [rshapes[i:i + l]
                  for i, l in zip(_cnt_to_dsp(rndims), rndims)]

        # Collective communication.
        slens = [numpy.prod(x.shape) for x in xs]
        xp = chainer.cuda.get_array_module(xs[0])
        sbuf = xp.hstack([x.reshape(-1) for x in xs])
        rlens = [numpy.prod(s) for s in shapes]
        rbuf = numpy.empty(sum(rlens), dtype=numpy.float32)
        if xp is not numpy:
            sbuf = _memory_utility.array_to_buffer_object(sbuf)[0]
            chainer.cuda.Stream.null.synchronize()
        self.mpi_comm.Alltoallv(
            [sbuf, (slens, _cnt_to_dsp(slens)), mpi4py.MPI.FLOAT],
            [rbuf, (rlens, _cnt_to_dsp(rlens)), mpi4py.MPI.FLOAT])
        ys = [rbuf[i:i + l].reshape(s)
              for i, l, s in zip(_cnt_to_dsp(rlens), rlens, shapes)]

        return tuple(ys)

    def send(self, data, dest, tag):
        """A primitive for inter-process transmitter.

        This method sends numpy-array to target process.
        The target process is expected to invoke ``recv()``.
        This method relies on mpi4py fast communication optimized for
        numpy arrays, which discards any information attached to
        chainer.Variable objects. Please be sure.

        Args:
            data: data to be sent (tuple, list or raw numpy/cupy array)
            dest (int): Target process specifier.
            tag (int): Message ID (MPI feature).

        """
        chainer.utils.experimental(
            'chainermn.communicators.MpiCommunicatorBase.send')

        msgtype = _MessageType(data)
        """We use ssend() instead of send() to pass unittests.
        If we don't use it, an error occurs in
        test_point_to_point_communication.py
        when using MVAPICH2-2.2 and GPUs.
        """
        self.mpi_comm.ssend(msgtype, dest=dest, tag=tag)

        # Type check.
        if not msgtype.is_tuple:
            data = [data]

        for x in data:
            if x.dtype != numpy.float32:
                raise ValueError('send only support dtype == numpy.float32')

        for array in data:
            if chainer.cuda.get_array_module(array) is not numpy:
                chainer.cuda.Stream.null.synchronize()

            buf = _memory_utility.array_to_buffer_object(array)
            """We use Ssend() for the same reason as using ssend()."""
            self.mpi_comm.Ssend(buf, dest=dest, tag=tag)

    def recv(self, source, tag):
        """A primitive of inter-process receiver.

        This method tries to receive numpy-array from target process.
        The target process is expected to invoke ``send()``.
        This method relies on mpi4py fast communication optimized for
        numpy arrays, which discards any information attached to
        chainer.Variable objects. Please be sure.

        Args:
            source (int): Target process specifier.
            tag (int): Message ID (MPI feature).

        """

        chainer.utils.experimental(
            'chainermn.communicators.MpiCommunicatorBase.recv')

        msgtype = self.mpi_comm.recv(source=source, tag=tag)

        if msgtype.is_tuple:
            msg = []
            for shape in msgtype.shapes:
                buf = numpy.empty(numpy.prod(shape), dtype=numpy.float32)
                self.mpi_comm.Recv(buf, source=source, tag=tag)
                msg.append(buf.reshape(shape))
            return tuple(msg)

        else:
            assert len(msgtype.shapes) == 1
            shape = msgtype.shapes[0]
            buf = numpy.empty(numpy.prod(shape), dtype=numpy.float32)
            self.mpi_comm.Recv(buf, source=source, tag=tag)
            return buf.reshape(shape)

    def bcast(self, x, root=0):
        """A primitive of inter-process broadcast communication.

        This method tries to invoke broadcast communication within the
        communicator. All processes in the communicator are expected to
        invoke ``broadcast()``. This method relies on mpi4py fast communication
        optimized for numpy arrays, as well as ``send()`` and ``recv()``.

        Args:
            x (numpy.array): Array to be broadcasted.
            root (int): Rank of root process.

        Returns:
            ys (tuple of numpy.ndarray): Received arrays.
        """
        chainer.utils.experimental(
            'chainermn.communicators.MpiCommunicatorBase.bcast')

        is_master = self.mpi_comm.rank == root

        if is_master:
            msgtype = _MessageType(x)
            if msgtype.is_tuple:
                raise TypeError('Tuple data cannot be broadcasted')

            elif x.dtype != numpy.float32:
                raise TypeError('bcast only supports dtype == numpy.float32')

            msgtype = self.mpi_comm.bcast(msgtype, root)
            shape = msgtype.shapes[0]
            buf = _memory_utility.array_to_buffer_object(x)
            self.mpi_comm.Bcast(buf, root)
            return x
        else:
            msgtype = None
            msgtype = self.mpi_comm.bcast(msgtype, root)
            shape = msgtype.shapes[0]
            buf = numpy.empty(numpy.prod(shape), dtype=numpy.float32)
            self.mpi_comm.Bcast(buf, root)
            return buf.reshape(shape)

    def gather(self, x, root=0):
        """A primitive of inter-process gather communication.

        This method tries to invoke gather communication within the
        communicator. All processes in the communicator are expected to
        invoke ``gather()``. This method relies on mpi4py fast communication
        optimized for numpy arrays, as well as ``send()`` and ``recv()``.

        Args:
            x (numpy.array): Array to be gathered.
            root (int): Rank of root process.

        Returns:
            ys (tuple of numpy.ndarray):
                Received arrays. ``None`` for non-root processes.
        """
        chainer.utils.experimental(
            'chainermn.communicators.MpiCommunicatorBase.gather')

        is_master = self.mpi_comm.rank == root

        msgtype = _MessageType(x)
        msgtypes = self.mpi_comm.gather(msgtype, root)

        # Type check.
        if x.dtype != numpy.float32:
            raise TypeError('gather only support dtype == numpy.float32')

        if is_master:
            for msgtype in msgtypes:
                if msgtype.is_tuple:
                    raise TypeError('gather cannot handle tuple data')

                assert len(msgtype.shapes) == 1

            sbuf = _memory_utility.array_to_buffer_object(x)
            shapes = [mty.shapes[0] for mty in msgtypes]
            rlens = [numpy.prod(s) for s in shapes]
            rbuf = numpy.empty(sum(rlens), dtype=numpy.float32)

            if chainer.cuda.get_array_module(x) is not numpy:
                chainer.cuda.Stream.null.synchronize()

            self.mpi_comm.Gatherv(
                sbuf,
                [rbuf, (rlens, _cnt_to_dsp(rlens)), mpi4py.MPI.FLOAT],
                root)

            ys = [rbuf[i:i + l].reshape(s)
                  for i, l, s in zip(_cnt_to_dsp(rlens), rlens, shapes)]
            return tuple(ys)

        else:
            sbuf = _memory_utility.array_to_buffer_object(x)
            self.mpi_comm.Gatherv(sbuf, None, root)
            return None

    def allgather(self, x):
        chainer.utils.experimental(
            'chainermn.communicators.MPICommunicatorBase.allgather')

        msgtype = _MessageType(x)
        msgtypes = self.mpi_comm.allgather(msgtype)

        # Type check.
        for msgtype in msgtypes:
            if msgtype.is_tuple:
                raise TypeError('allgather cannot handle tuple data')

            assert len(msgtype.shapes) == 1

        if x.dtype != numpy.float32:
            raise TypeError('allgather only support dtype == numpy.float32')

        # Collective communication.
        xp = chainer.cuda.get_array_module(x)
        shapes = [msgtype.shapes[0] for msgtype in msgtypes]
        sbuf = _memory_utility.array_to_buffer_object(x)
        rlens = [numpy.prod(s) for s in shapes]
        rbuf = numpy.empty(sum(rlens), dtype=numpy.float32)
        if xp is not numpy:
            chainer.cuda.Stream.null.synchronize()
        self.mpi_comm.Allgatherv(
            sbuf,
            [rbuf, (rlens, _cnt_to_dsp(rlens)), mpi4py.MPI.FLOAT])
        ys = [rbuf[i:i + l].reshape(s)
              for i, l, s in zip(_cnt_to_dsp(rlens), rlens, shapes)]

        return tuple(ys)

    # Objects
    def send_obj(self, obj, dest):
        self.mpi_comm.send(obj, dest=dest)

    def recv_obj(self, source):
        return self.mpi_comm.recv(source=source)

    def bcast_obj(self, obj, max_buf_len=256 * 1024 * 1024, root=0):
        return chunked_bcast_obj(obj, self.mpi_comm,
                                 max_buf_len=max_buf_len,
                                 root=root)

    def gather_obj(self, obj, root=0):
        return self.mpi_comm.gather(obj, root=root)

    def scatter(self, xs, root=0):
        """A primitive of inter-process scatter communication.

        This method tries to invoke scatter communication within the
        communicator. All processes in the communicator are expected to
        invoke ``scatter()``. This method relies on mpi4py fast communication
        optimized for numpy arrays, as well as ``send()`` and ``recv()``.

        If ``xs`` is tuple, each element is send to different processes.
        The length of the tuple must be the same as the communicator size.
        If ``xs`` is ``numpy.ndarrray``, it is splitted with the first
        axis and sent to different processes. For slave processes, ``xs``
        is allowed to be any value (will be ignored).

        Args:
            xs (tuple of numpy.array or numpy.array): Arrays to be scattered.
            root (int): Rank of root process.

        Returns:
            ys (numpy.ndarray): Received arrays.
        """
        chainer.utils.experimental(
            'chainermn.communicators.CommunicatorBase.scatter')

        is_master = self.mpi_comm.rank == root

        if is_master:
            # Type check.
            msgtype = _MessageType(xs)

            if msgtype.is_tuple:
                if xs[0].dtype != numpy.float32:
                    raise TypeError(
                        'scatter only support dtype == numpy.float32')

                if len(msgtype.shapes) != self.size:
                    raise ValueError(
                        'the length of xs must be consistent '
                        'with communicator size')

                xp = chainer.cuda.get_array_module(*xs)
                msgtype = tuple([_MessageType(x) for x in xs])
                shapes = [mty.shapes[0] for mty in msgtype]
                # concatenate([x.reshape(-1) ... ], axis=0) will fail
                xs = xp.concatenate([x.reshape(1, -1) for x in xs], axis=1)

            else:
                assert len(msgtype.shapes) == 1

                if xs.dtype != numpy.float32:
                    raise TypeError(
                        'scatter only support dtype == numpy.float32')

                if msgtype.shapes[0][0] != self.mpi_comm.size:
                    raise ValueError(
                        'scatter received inconsistent number of inputs '
                        'with communicator size')

                xp = chainer.cuda.get_array_module(xs)
                msgtype = tuple([_MessageType(xs[0])
                                 for _ in range(self.size)])
                shapes = [xs.shape[1:] for _ in range(self.size)]

            msgtype = self.mpi_comm.scatter(msgtype, root)
            shape = msgtype.shapes[0]

            # Collective communication.
            slens = [numpy.prod(s) for s in shapes]
            sbuf = _memory_utility.array_to_buffer_object(xs)[0]
            rbuf = numpy.empty(numpy.prod(shape), dtype=numpy.float32)
            if xp is not numpy:
                chainer.cuda.Stream.null.synchronize()

            self.mpi_comm.Scatterv(
                [sbuf, (slens, _cnt_to_dsp(slens)), mpi4py.MPI.FLOAT],
                rbuf, root)

            return rbuf.reshape(shape)

        else:  # slave processes
            msgtypes = self.mpi_comm.scatter(None, root)
            shape = msgtypes.shapes[0]
            rbuf = numpy.empty(numpy.prod(shape), dtype=numpy.float32)
            self.mpi_comm.Scatterv(None, rbuf, root)
            return rbuf.reshape(shape)

    def allreduce_obj(self, obj):
        # Summation by default
        return self.mpi_comm.allreduce(obj)

    def bcast_data(self, model):
        for _, param in sorted(model.namedparams()):
            buf = _memory_utility.array_to_buffer_object(param.data)
            self.mpi_comm.Bcast(buf)

    # Private methods
    def _init_ranks(self):
        my_ranks = _communication_utility.init_ranks(self.mpi_comm)
        assert my_ranks[0] == self.mpi_comm.rank
        self._intra_rank = my_ranks[1]
        self.intra_size = my_ranks[2]
        self.inter_rank = my_ranks[3]
        self.inter_size = my_ranks[4]

    def _init_comms(self):
        if self.inter_mpi_comm is not None:
            assert self.intra_mpi_comm is not None
            assert not self.use_nccl or self.intra_nccl_comm is not None
            return

        comms = _communication_utility.init_comms(
            self.mpi_comm, self.intra_rank, self.intra_size, self.inter_rank,
            use_nccl=self.use_nccl)
        self.intra_mpi_comm = comms[0]
        self.inter_mpi_comm = comms[1]
        if self.use_nccl:
            self.intra_nccl_comm = comms[2]
