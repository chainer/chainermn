import collections
import mpi4py.MPI

from chainermn.communicators import _memory_utility


def init_ranks(mpi_comm):
    """Returns rank information of the local process in `mpi_comm`.

    Args:
        mpi_comm (type:TODO)
                 MPI Communicator from mpi4py

    Returns:
        rank_info (list):
            Elements are:
                * rank (`mpi_comm.rank`)
                * intra_rank (rank within the local computing node)
                * intra_size (number of processes on the node)
                * inter_rank (rank of the node)
                * inter_size (number of computing nodes)
    """

    global_names = mpi_comm.gather(mpi4py.MPI.Get_processor_name())

    if mpi_comm.rank == 0:
        name_to_global_ranks = collections.defaultdict(list)
        for global_rank, name in enumerate(global_names):
            name_to_global_ranks[name].append(global_rank)

        for global_ranks in name_to_global_ranks.values():
            global_ranks.sort()

        inter_names = sorted(
            set(global_names), key=lambda name: name_to_global_ranks[name])
        name_to_inter_rank = {
            name: inter_rank
            for inter_rank, name in enumerate(inter_names)
        }
        inter_size = len(inter_names)

        all_ranks = []
        for global_rank, name in enumerate(global_names):
            ranks = name_to_global_ranks[name]
            intra_rank = ranks.index(global_rank)
            intra_size = len(ranks)
            inter_rank = name_to_inter_rank[name]
            all_ranks.append((
                global_rank, intra_rank, intra_size,
                inter_rank, inter_size))
        my_ranks = mpi_comm.scatter(all_ranks)
    else:
        my_ranks = mpi_comm.scatter(None)

    assert my_ranks[0] == mpi_comm.rank
    return my_ranks


def init_comms(mpi_comm, intra_rank, intra_size, inter_rank, use_nccl=True):
    intra_mpi_comm = mpi_comm.Split(inter_rank, intra_rank)
    inter_mpi_comm = mpi_comm.Split(intra_rank, inter_rank)
    if use_nccl:
        from chainermn import nccl
        intra_nccl_comm_id = intra_mpi_comm.bcast(nccl.get_unique_id())
        intra_nccl_comm = nccl.NcclCommunicator(
            intra_size, intra_nccl_comm_id, intra_rank)
        if nccl.get_version() >= 2000:
            nccl_comm_id = mpi_comm.bcast(nccl.get_unique_id())
            nccl_comm = nccl.NcclCommunicator(
                mpi_comm.size, nccl_comm_id, mpi_comm.rank)
        else:
            nccl_comm = None
        return intra_mpi_comm, inter_mpi_comm, intra_nccl_comm, nccl_comm
    else:
        return intra_mpi_comm, inter_mpi_comm


def broadcast_naive(mpi_comm, model):
    for _, param in sorted(model.namedparams()):
        buf = _memory_utility.array_to_buffer_object(param.data)
        mpi_comm.Bcast(buf)


def inter_allreduce_gpu(
        inter_mpi_comm, size, gpu_buffer_a, gpu_buffer_b,
        n_bytes_buffer, n_elems_per_node, n_bytes_per_node, cuda_stream):
    inter_size = inter_mpi_comm.size

    # Exchange all data to get own region data (bufferB -> bufferA)
    cuda_stream.synchronize()
    inter_mpi_comm.Alltoall(
        [gpu_buffer_b.buffer(n_bytes_buffer), mpi4py.MPI.FLOAT],
        [gpu_buffer_a.buffer(n_bytes_buffer), mpi4py.MPI.FLOAT])

    # Reduce own region data (inplace bufferA) and averaging
    ret = gpu_buffer_a.array(inter_size * n_elems_per_node) \
        .reshape(inter_size, n_elems_per_node) \
        .sum(axis=0) * (1.0 / size)

    # Gather others' region data (bufferA -> bufferB)
    for i in range(0, inter_size):
        gpu_buffer_a.from_device(
            ret, n_bytes_per_node, i * n_bytes_per_node)

    cuda_stream.synchronize()
    inter_mpi_comm.Alltoall(
        [gpu_buffer_a.buffer(n_bytes_buffer), mpi4py.MPI.FLOAT],
        [gpu_buffer_b.buffer(n_bytes_buffer), mpi4py.MPI.FLOAT])
