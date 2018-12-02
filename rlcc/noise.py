import numpy as np
import torch
import random

class NoiseProcess():
    r"""Base class for all noise processes."""

    def __init__(self):
        pass

    def reset(self):
        r"""Reset the noise process."""
        pass

    def sample(self):
        r"""Sample the noise process."""
        raise NotImplementedError

# Based on https://github.com/songrotek/DDPG/blob/master/ou_noise.py
class OUProcess(NoiseProcess):
    """Ornstein-Uhlenbeck process."""
    
    def __init__(self, action_dimension, scale=1.0, mu=0, theta=0.15, sigma=0.2):
        super(OUProcess, self).__init__()
        self.action_dimension = action_dimension
        self.scale = scale
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state * self.scale
