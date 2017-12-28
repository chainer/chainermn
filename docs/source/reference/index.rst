.. module:: chainermn

API Reference
=============


Communicators
~~~~~~~~~~~~~

.. autofunction:: create_communicator


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

Trainer extensions
~~~~~~~~~~~~~~~~~~

.. autoclass:: chainermn.extensions.AllreducePersistent
.. autofunction:: chainermn.create_multi_node_checkpointer
