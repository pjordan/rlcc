r"""Transition-related functions"""


import random
import numpy as np
import torch
from . import transition


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
            return to_device(batch, self.device)
        else:
            return batch
        
    def __iter__(self):
        return self


class TransitionReplayer(object):
    r"""
    Transition Replayer. 

    Arguments:
        transitions: dataset from which to replay the transitions.
        device (string, optional): the device to send the transition data to
        batch_size (int, optional): how many samples per batch to load (default: 1).
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