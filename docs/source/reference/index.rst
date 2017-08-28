.. module:: chainermn

API Reference
=============


Communicators
~~~~~~~~~~~~~

.. autofunction:: create_communicator


Links
~~~~~

.. autoclass:: MultiNodeChainList
    :members: add_link
.. autoclass:: chainermn.links.MultiNodeBatchNormalization


Optimizers and Evaluators
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: create_multi_node_optimizer
.. autofunction:: create_multi_node_evaluator


Multi Node Utility Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: chainermn.functions.send
.. autofunction:: chainermn.functions.recv
.. autofunction:: chainermn.functions.pseudo_connect


Dataset Utilities
~~~~~~~~~~~~~~~~~

.. autofunction:: scatter_dataset
.. autofunction:: chainermn.datasets.create_empty_dataset
