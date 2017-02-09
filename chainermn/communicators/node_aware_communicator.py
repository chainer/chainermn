import collections
import mpi4py.MPI

from chainermn.communicators import mpi_based_communicator
from chainermn import nccl


class NodeAwareCommunicator(mpi_based_communicator.MPIBasedCommunicator):

    def __init__(self, mpi_comm=mpi4py.MPI.COMM_WORLD):
        super(NodeAwareCommunicator, self).__init__(mpi_comm)
        self._init_ranks()
        self._init_comms()

    def _init_ranks(self):
        global_names = self.mpi_comm.gather(mpi4py.MPI.Get_processor_name())

        if self.mpi_comm.rank == 0:
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
            my_ranks = self.mpi_comm.scatter(all_ranks)
        else:
            my_ranks = self.mpi_comm.scatter(None)

        assert my_ranks[0] == self.mpi_comm.rank
        self.intra_rank = my_ranks[1]
        self.intra_size = my_ranks[2]
        self.inter_rank = my_ranks[3]
        self.inter_size = my_ranks[4]

    def _init_comms(self):
        self.intra_mpi_comm = self.mpi_comm.Split(self.inter_rank, self.intra_rank)

        if self.intra_rank == 0:
            inter_ranks = self.mpi_comm.allreduce([self.rank])
        else:
            inter_ranks = self.mpi_comm.allreduce([])

        world_group = self.mpi_comm.Get_group()
        inter_group = world_group.Incl(inter_ranks)
        self.inter_mpi_comm = self.mpi_comm.Create(inter_group)

        nccl_comm_id = self.intra_mpi_comm.bcast(nccl.NcclCommunicatorId())
        self.intra_nccl_comm = nccl.NcclCommunicator(self.intra_size, nccl_comm_id, self.intra_rank)
