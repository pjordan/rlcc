"""Transition-related functions"""

from collections import deque,namedtuple
import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader

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


def from_numpy(*args):
    r"""Creates a transition object from numpy arguments.

    Arguments:
        state (numpy array): The state the transition is from.
        action (numpy array): The action taken in the transition.
        reward (numpy array): The reward for transitioning.
        next_state (numpy array): The state the transition is to.
        is_terminal (numpy array): If terminal then 1.0, 0.0 otherwise.
    """
    return transition(*map(lambda x: torch.from_numpy(x).float(), args))


def from_primitives(state, action, reward, next_state, is_terminal):
    r"""Creates a transition object from primitive arguments.

    Arguments:
        state (array): The state the transition is from.
        action (array): The action taken in the transition.
        reward (array): The reward for transitioning.
        next_state (array): The state the transition is to.
        is_terminal (array): If terminal then 1.0, 0.0 otherwise.
    """
    return from_numpy(
        np.array(state),
        np.array(action),
        np.array(reward),
        np.array(next_state),
        np.array(is_terminal).astype(np.uint8))


def collate(transitions):
    r"""Creates a collated transition object from a list of
    transitions.

    The resulting transition will contain a tensor for each element
    that is the stack of the corresponding list of respective elements.

    Arguments:
        transitions (array): list of transitions.
    """
    return transition(*map(torch.stack, zip(*transitions)))


def replay_buffer(buffer_size=int(1e5)):
    r"""Creates a replay buffer.

    Arguments:
        buffer_size (int): the buffer size.
    """
    return deque(maxlen=buffer_size)


def data_loader(transitions, batch_size=128):
    r"""Creates a ``DataLoader`` for transitions from a list of transitions.

    Arguments:
        batch_size (int): the batch size.
    """
    return DataLoader(transitions, batch_size=batch_size, collate_fn=collate)
