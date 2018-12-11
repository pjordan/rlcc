"""Strorage-related functions"""

from collections import deque

def buffer(buffer_size=int(1e5)):
    r"""Creates a buffer.

    :param buffer_size: the additive delta
    :type buffer_size: int, optional
    """
    return deque(maxlen=buffer_size)