import collections
import mpi4py.MPI


def init_ranks(mpi_comm):
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

    if intra_rank == 0:
        inter_ranks = mpi_comm.allreduce([mpi_comm.rank])
    else:
        inter_ranks = mpi_comm.allreduce([])

    world_group = mpi_comm.Get_group()
    inter_group = world_group.Incl(inter_ranks)
    inter_mpi_comm = mpi_comm.Create(inter_group)

    if use_nccl:
        from chainermn import nccl
        nccl_comm_id = intra_mpi_comm.bcast(nccl.NcclCommunicatorId())
        intra_nccl_comm = nccl.NcclCommunicator(
            intra_size, nccl_comm_id, intra_rank)
        return intra_mpi_comm, inter_mpi_comm, intra_nccl_comm
    else:
        return intra_mpi_comm, inter_mpi_comm
