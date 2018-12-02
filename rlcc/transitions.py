"""Transition-related functions"""

from collections import deque, namedtuple
import random
import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader

Transition = namedtuple(
    "Transition",
    field_names=["state", "action", "reward", "next_state", "is_terminal"])

def _make_tensor(x):
    return torch.tensor(x, dtype=torch.float)

def to_device(transition, device):
    map(lambda x: x.to(device), transition)

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


def make(*args):
    r"""Creates a transition object from non-tensor arguments.

    Arguments:
        state (numpy array): The state the transition is from.
        action (numpy array): The action taken in the transition.
        reward (numpy array): The reward for transitioning.
        next_state (numpy array): The state the transition is to.
        is_terminal (numpy array): If terminal then 1.0, 0.0 otherwise.
    """
    return transition(*map(_make_tensor, args))


def buffer(buffer_size=int(1e5)):
    r"""Creates a buffer.

    Arguments:
        buffer_size (int): the buffer size.
    """
    return deque(maxlen=buffer_size)

def default_collate(transitions):
    r"""Creates a collated transition object from a list of
    transitions.

    The resulting transition will contain a tensor for each element
    that is the stack of the corresponding list of respective elements.

    Arguments:
        transitions (array): list of transitions.
    """
    return transition(*map(torch.stack, zip(*transitions)))


class _TransitionReplayerIter(object):
    r"""Iterates over the TransitionReplayer's transitions."""
    def __init__(self, replayer):
        self.transitions = replayer.transitions
        self.batch_size = replayer.batch_size
        self.collate_fn = replayer.collate_fn
        self.device = replayer.device

    def __len__(self):
        return len(self.transitions)
    
    def __next__(self):
        batch = self.collate_fn(random.sample(self.transitions, k=self.batch_size))
        if self.device:
            to_device(batch, self.device)
        return batch
        
    def __iter__(self):
        return self


class TransitionReplayer(object):
    r"""
    Transition Replayer. 

    Arguments:
        transitions: dataset from which to replay the transitions.
        device (string, optional): the device to send the transition data to
        batch_size (int, optional): how many samples per batch to load
            (default: 1).
        collate_fn (callable, optional): merges a list of samples to form a mini-batch.
    """

    __initialized = False

    def __init__(self, transitions, device=None, batch_size=1, collate_fn=default_collate):
        self.transitions = transitions
        self.batch_size = batch_size
        self.collate_fn = collate_fn
        self.device = device
        self.__initialized = True

    def __setattr__(self, attr, val):
        if self.__initialized and attr in ('batch_size', 'collate_fn'):
            raise ValueError('{} attribute should not be set after {} is '
                             'initialized'.format(attr, self.__class__.__name__))

        super(TransitionReplayer, self).__setattr__(attr, val)

    def __iter__(self):
        return _TransitionReplayerIter(self)
    
    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.transitions)    