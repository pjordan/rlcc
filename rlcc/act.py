import numpy as np
import torch

class Actor():
    r"""Base class for all actors.
    """

    def __init__(self):
        pass

    def act(self, state):
        """Returns actions for given state as numpy array."""
        raise NotImplementedError

class NetworkActor(Actor):
    """Implements Deterministic Policy Gradient.
    """

    def __init__(self, network, device):
        super(NetworkActor, self).__init__()
        self.network = network
        self.device = device

    def act(self, state):
        """Returns actions for given state as numpy array."""
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.network.eval()
        with torch.no_grad():
            action = self.network(state).cpu().data.numpy()
        self.network.train()
        return action

class NoisyActor(Actor):
    """Implements a noisy actor policy.
    """

    def __init__(self, base_actor, noise_process, action_min=-1, action_max=1):
        super(NoisyActor, self).__init__()
        self.base_actor = base_actor
        self.noise_process = noise_process
        self.action_min = action_min
        self.action_max = action_max
 
    def act(self, state):
        """Returns actions for given state as numpy array."""
        action = self.base_actor.act(state) + self.noise_process.sample()
        return np.clip(action, self.action_min, self.action_max)
    
class StackedActor(Actor):
    def __init__(self, actors):
        super(StackedActor, self).__init__()
        self.actors = actors
        
    def act(self, state):
        """Returns actions for given state as numpy array."""
        return np.vstack([actor.act(state[i]) for i, actor in enumerate(self.actors)])
