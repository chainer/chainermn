import os

_mpi_rank = (os.environ.get('MV2_COMM_WORLD_RANK') or
             os.environ.get('OMPI_COMM_WORLD_RANK'))

if _mpi_rank is None:
    _print_report = True
else:
    if int(_mpi_rank) == 0:
        _print_report = True
    else:
        _print_report = False


def pytest_configure(config):
    if not _print_report:
        config.option.verbose = -1
