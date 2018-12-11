r"""Actor and related functionality."""


import torch


class Actor():
    r"""Base class for all actors.
    
    Your actors should also subclass this class.
    """
    def __init__(self):
        r"""Constructor method"""
        pass

    def act(self, state):
        r"""Returns actions for given state as tensor."""
        """Returns an action tensor

        :param state: state tensor
        :type state: `torch.tensor`
        :return: action tensor
        :rtype: `torch.tensor`
        """
        raise NotImplementedError


class NetworkActor(Actor):
    r"""This is an implementation of an actor whose policy is encoded as a network.
    
    :param network: A network that returns an action for a state
    :type network: class:`torch.nn.Module`
    """

    def __init__(self, network):
        r"""Constructor method"""
        super(NetworkActor, self).__init__()
        self.network = network

    def act(self, state):
        r"""Returns actions for given state as tensor."""
        """Returns an action tensor

        :param state: state tensor
        :type state: `torch.tensor`
        :return: action tensor
        :rtype: `torch.tensor`
        """
        self.network.eval()
        with torch.no_grad():
            action = self.network(state.unsqueeze(0)).squeeze(0)
        self.network.train()
        return action


class NoisyActor(Actor):
    r"""Implements a noisy actor policy.

    :param base_actor: the actor whose actions noise is added.
    :type base_actor: class:`rlcc.act.Actor`
    :param noise_process: Device where the states should be sent
    :type noise_process: rlcc.noise.NoiseProcess
    """
    def __init__(self, base_actor, noise_process):
        r"""Constructor method"""
        super(NoisyActor, self).__init__()
        self.base_actor = base_actor
        self.noise_process = noise_process
 
    def act(self, state):
        r"""Returns actions for given state as tensor."""
        """Returns an action tensor

        :param state: state tensor
        :type state: `torch.tensor`
        :return: action tensor
        :rtype: `torch.tensor`
        """
        return self.base_actor.act(state) + self.noise_process.sample()


class ClippingActor(Actor):
    r"""Implements an actor policy that clips the actions.

    :param base_actor: the actor whose actions noise is added.
    :type network: class:`rlcc.act.Actor`
    :param action_min: Minimal action value
    :type action_min: scalar or numpy.Array, optional
    :param action_max: Maximal action value
    :type action_max: scalar or numpy.Array, optional
    """
    def __init__(self, base_actor, action_min=-1, action_max=1):
        """Constructor method"""
        super(ClippingActor, self).__init__()
        self.base_actor = base_actor
        self.action_min = action_min
        self.action_max = action_max
 
    def act(self, state):
        r"""Returns actions for given state as tensor."""
        """Returns an action tensor

        :param state: state tensor
        :type state: `torch.tensor`
        :return: action tensor
        :rtype: `torch.tensor`
        """
        return torch.clamp(self.base_actor.act(state), self.action_min, self.action_max)


class StackedActor(Actor):
    r"""Creates a network actor.

        :param actors: list of actors
        :type actors: class:`rlcc.act.Actor`
        """
    def __init__(self, actors):
        """Constructor method"""
        super(StackedActor, self).__init__()
        self.actors = actors
        
    def act(self, state):
        r"""Returns actions for given state as tensor."""
        """Returns an action tensor

        :param state: state tensor
        :type state: `torch.tensor`
        :return: action tensor
        :rtype: `torch.tensor`
        """
        return torch.stack([actor.act(state[i]) for i, actor in enumerate(self.actors)])
