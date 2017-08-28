import pkg_resources

from chainermn.communicators import create_communicator  # NOQA
from chainermn.datasets import scatter_dataset  # NOQA
from chainermn.extensions import create_multi_node_evaluator  # NOQA
from chainermn.links import MultiNodeChainList  # NOQA
from chainermn.multi_node_optimizer import create_multi_node_optimizer  # NOQA


__version__ = pkg_resources.get_distribution('chainermn').version
