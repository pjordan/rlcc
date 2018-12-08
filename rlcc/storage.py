"""Strorage-related functions"""

from collections import deque

def buffer(buffer_size=int(1e5)):
    r"""Creates a buffer.

    Arguments:
        buffer_size (int): the buffer size.
    """
    return deque(maxlen=buffer_size)