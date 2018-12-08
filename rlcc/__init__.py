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


def transition(*args):
    r"""Creates a transition object from arguments.

    Arguments:
        state (tensor): The state the transition is from.
        action (tensor): The action taken in the transition.
        reward (tensor): The reward for transitioning.
        next_state (tensor): The state the transition is to.
        is_terminal (tensor): If terminal then 1.0, 0.0 otherwise.
    """
    return Transition(*args)


# Define standard tensor operations
def tensor(data):
    r"""Constructs a torch `tensor` with data that returns
    a tensor of float data type.

    Arguments:
        data (array_like) – Initial data for the tensor. 
        Can be a list, tuple, NumPy ndarray, scalar, and other types.
    """
    return torch.tensor(data, dtype=torch.float)


def to_device(iterable, device):
    r"""Creates a transition object from non-tensor arguments.

    Arguments:
        state (numpy array): The state the transition is from.
        action (numpy array): The action taken in the transition.
        reward (numpy array): The reward for transitioning.
        next_state (numpy array): The state the transition is to.
        is_terminal (numpy array): If terminal then 1.0, 0.0 otherwise.
    """
    return map(lambda x: x.to(device), iterable)