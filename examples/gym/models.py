"""Model utility functions"""


import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class ContinuousActor(nn.Module):
    """ContinuousActor network
    
    :param state_size: the size of the state space 
    :type state_size: int
    :param hidden1_size: the size of the first hidden network
    :type hidden1_size: int
    :param hidden2_size: the size of the second hidden network
    :type hidden2_size: int
    :param action_size: the size of the action space
    :type action_size: int
    """
    def __init__(self, state_size, hidden1_size, hidden2_size, action_size):
        super(ContinuousActor, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden1_size)
        self.fc2 = nn.Linear(hidden1_size, hidden2_size)
        self.fc3 = nn.Linear(hidden2_size, action_size)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        r"""Build an actor network that maps states to actions.
    
        :param state: the state 
        :type state: :class:`torch.Tensor`
        :return: action tensor
        :rtype: :class:`torch.Tensor`
        """
        x = F.leaky_relu(self.fc1(state))
        x = F.leaky_relu(self.fc2(x))
        return F.tanh(self.fc3(x))

class ContinuousCritic(nn.Module):
    """ContinuousCritic network
    
    :param state_size: the size of the state space 
    :type state_size: int
    :param hidden1_size: the size of the first hidden network
    :type hidden1_size: int
    :param hidden2_size: the size of the second hidden network
    :type hidden2_size: int
    :param action_size: the size of the action space
    :type action_size: int
    """
    def __init__(self, state_size, hidden1_size, hidden2_size, action_size):
        super(ContinuousCritic, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden1_size)
        self.fc2 = nn.Linear(hidden1_size + action_size, hidden2_size)
        self.fc3 = nn.Linear(hidden2_size, 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-4, 3e-4)

    def forward(self, state, action):
        r"""Build a critic network that maps (state,action) to utility.
    
        :param state: the state 
        :type state: :class:`torch.Tensor`
        :param action: action tensor
        :type action: :class:`torch.Tensor`
        :return: utility tensor
        :rtype: :class:`torch.Tensor`
        """
        xs = F.leaky_relu(self.fc1(state))
        x = torch.cat((xs, action), dim=1)
        x = F.leaky_relu(self.fc2(x))
        return self.fc3(x)