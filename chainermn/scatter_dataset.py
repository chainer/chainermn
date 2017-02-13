import chainer.datasets


def scatter_dataset(dataset, comm):
    # TODO(akiba): write why we do not use mpi_comm.scatter

    if hasattr(comm, 'mpi_comm'):
        comm = comm.mpi_comm
    assert hasattr(comm, 'send')
    assert hasattr(comm, 'recv')

    if comm.rank == 0:
        mine = None
        n_samples = len(dataset)
        for i in range(comm.size):
            b = n_samples * i // comm.size
            e = n_samples * (i + 1) // comm.size
            subds = chainer.datasets.SubDataset(dataset, b, e)
            if i == 0:
                mine = subds
            else:
                comm.send(subds, dest=i)
        return mine
    else:
        return comm.recv(source=0)
