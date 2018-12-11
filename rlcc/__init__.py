r"""
RL Common Components

The rlcc package contains data structures and algorithms for common components 
of reinforcement learning.  

The package uses pytorch for underlying tensor operations.
"""


from collections import namedtuple
import torch

__author__       = 'Patrick R. Jordan'
__email__        = 'patrick.r.jordan@gmail.com'
__version__      = '0.1.3'
__url__          = 'https://github.com/pjordan/rlcc/',
__description__  = 'RL Common Components'

__all__ = [
    'transition', 'tensor', 
    'Transition'
]

# Import the basic transition operations
Transition = namedtuple(
    "Transition",
    field_names=["state", "action", "reward", "next_state", "is_terminal"])
Transition.state.__doc__ += ': The state the transition is from.'
Transition.action.__doc__ += ': The action taken in the transition.'
Transition.reward.__doc__ += ': The reward for transitioning.'
Transition.next_state.__doc__ += ': The state the transition is to.'
Transition.is_terminal.__doc__ += ': If terminal then 1.0, 0.0 otherwise.'


def transition(*args):
    r"""Creates a transition object from arguments.

    :param args: tuple of (state, action, reward, next_state, is_terminal) tensors 
    :type args: tensors   
    """
    return Transition(*args)


# Define standard tensor operations
def tensor(data):
    r"""Constructs a torch `tensor` with data that returns
    a tensor of float data type.

    :param data: data for the tensor. 
    :type data: Can be a list, tuple, NumPy ndarray, scalar, and other types.
    """
    return torch.tensor(data, dtype=torch.float)


def to_device(iterable, device):
    r"""Send iterable contents to device.

    :param iterable: the iterable
    :type iterable: iterable
    :param device: the torch device
    :type device: str
    """
    return map(lambda x: x.to(device), iterable)