.. module:: chainermn

API Reference
=============


Communicators
~~~~~~~~~~~~~

.. autofunction:: create_communicator
.. autoclass:: CommunicatorBase
    :members: rank, intra_rank, size, alltoall, split, send, recv,
              bcast, gather, allreduce, send_obj, recv_obj, bcast_obj,
              gather_obj, allreduce_obj, bcast_data, allreduce_grad


Optimizers and Evaluators
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: create_multi_node_optimizer
.. autofunction:: create_multi_node_evaluator


Dataset Utilities
~~~~~~~~~~~~~~~~~

.. autofunction:: scatter_dataset
.. autofunction:: chainermn.datasets.create_empty_dataset


Links
~~~~~

.. autoclass:: MultiNodeChainList
    :members: add_link
.. autoclass:: chainermn.links.MultiNodeBatchNormalization


Functions
~~~~~~~~~

.. autofunction:: chainermn.functions.send
.. autofunction:: chainermn.functions.recv
.. autofunction:: chainermn.functions.pseudo_connect
.. autofunction:: chainermn.functions.all_to_all
.. autofunction:: chainermn.functions.bcast


Iterators
~~~~~~~~~

.. autofunction:: chainermn.iterators.create_multi_node_iterator


Trainer extensions
~~~~~~~~~~~~~~~~~~

.. autoclass:: chainermn.extensions.AllreducePersistent
.. autofunction:: chainermn.create_multi_node_checkpointer
