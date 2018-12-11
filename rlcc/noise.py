r"""Noise process functions."""

import torch

class NoiseProcess():
    r"""Base class for all noise processes.

    Your noise processes should also subclass this class.
    """
    def __init__(self):
        pass

    def reset(self):
        r"""Reset the noise process."""
        pass

    def sample(self):
        r"""Sample the noise process."""
        raise NotImplementedError

class DeviceNoiseProcess(NoiseProcess):
    r"""Device noise process

    :param process: the delegate process
    :type process: :class:`rlcc.noise.NoiseProcess`
    :param device: tensor device
    :type device: str
    """
    def __init__(self, process, device):
        super(DeviceNoiseProcess, self).__init__()
        self.process = process
        self.device = device

    def reset(self):
        self.process.reset()

    def sample(self):
        return self.process.sample().to(self.device)
    
class ScaledProcess(NoiseProcess):
    r"""Scaled process

    :param process: the delegate process
    :type process: :class:`rlcc.noise.NoiseProcess`
    :param scale: scale multiplier
    :type scale: float
    """
    def __init__(self, process, scale=1.0):
        super(ScaledProcess, self).__init__()
        self.process = process
        self.scale = scale

    def reset(self):
        self.process.reset()

    def sample(self):
        return self.process.sample() * self.scale

class ScheduledScaledProcess(NoiseProcess):
    r"""Scheduled scaled process

    :param process: the delegate process
    :type process: :class:`rlcc.noise.NoiseProcess`
    :param scale_schedule: scale_schedule
    :type scale_schedule: float
    """
    def __init__(self, process, scale_schedule):
        super(ScheduledScaledProcess, self).__init__()
        self.process = process
        self.scale_schedule = scale_schedule
        self.reset()

    def reset(self):
        self.process.reset()
        self.scale_schedule.reset()

    def sample(self):
        return self.process.sample() * self.scale_schedule.value()

# Based on https://github.com/songrotek/DDPG/blob/master/ou_noise.py
class OUProcess(NoiseProcess):
    r"""Ornstein-Uhlenbeck process: dx = theta * (mu-x) dt + sigma * dW

    :param action_dimension: action dimensions
    :type action_dimension: int
    :param mu: mu 
    :type mu: float
    :param theta: theta
    :type theta: float
    :param sigma: sigma
    :type sigma: float
    """
    def __init__(self, action_dimension, mu=0, theta=0.15, sigma=0.2, device=None):
        super(OUProcess, self).__init__()
        self.action_dimension = action_dimension
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        self.state = torch.ones(self.action_dimension) * self.mu

    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * torch.randn(self.action_dimension)
        self.state = x + dx
        return self.state
    
