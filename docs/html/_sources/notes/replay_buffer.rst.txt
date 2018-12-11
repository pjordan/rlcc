:github_url: https://github.com/pjordan/rlcc

Replay Buffer
=============

The `rlcc.storage` and `rlcc.observe` packages contains classes that can be used to create replay buffers from transition observations.

For example, use the following to create a size-limited buffer.

.. code-block:: python

    from rlcc import storage

    buffer_size = int(1e5) 
    buffer = storage.buffer(buffer_size=buffer_size)

This buffer can now be used in an observer as the backing storage.

.. code-block:: python

    from rlcc.observe import BufferedObserver
    observer = BufferedObserver(buffer)

For learning, the buffer is input into a `rlcc.learn.ReplayLearner`.

