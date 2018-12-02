from itertools import islice 
import torch.nn.functional as F
from .transition_replayer import TransitionReplayer

class Learner():
    r"""Base class for all learners.
    """

    def __init__(self):
        pass

    def learn(self):
        r"""Performs a single learning step."""
        raise NotImplementedError

class ReplayLearner(Learner):
    """Implements a learner using a replay buffer.
    """

    def __init__(self, learning_strategy, transitions, batch_size=128, batches_per_step=1):
        super(ReplayLearner, self).__init__()
        self.learning_strategy = learning_strategy
        self.batches_per_step = batches_per_step
        self.replayer = TransitionReplayer(transitions, batch_size=batch_size)

    def learn(self):
        r"""Performs a single learning step."""
        for batch in islice(self.replayer, self.batches_per_step):
            self.learning_strategy.step(batch)

class LearningStrategy():
    r"""Base class for all learning strategies.
    """

    def __init__(self):
        pass

    def step(self, transitions):
        r"""Performs a single learning step.

        Arguments:
            transitions (list of Transition): transitions
        """
        raise NotImplementedError


class DPG(LearningStrategy):
    """Implements Deterministic Policy Gradient.
    """

    def __init__(self, actor_triad, critic_triad, device, gamma=0.99, tau=1e-3):
        super(DPG, self).__init__()
        self.gamma = gamma
        self.tau = tau
        self.actor_triad = actor_triad
        self.critic_triad = critic_triad

    def step(self, transitions):
        r"""Performs a single learning step.

        Arguments:
            transitions (list of Transition): transitions
        """
        # Unpack tuples
        states, actions, rewards, next_states, is_terminals = transitions
        actor_local, actor_target, actor_optimizer = self.actor_triad
        critic_local, critic_target, critic_optimizer = self.critic_triad

        # Get predicted next-state actions and Q values from target models
        actions_next = actor_target(next_states)
        q_targets_next = critic_target(next_states, actions_next)

        # Compute Q targets for current states
        q_targets = rewards + (self.gamma * q_targets_next * (1 - is_terminals))
        # Compute critic loss
        q_expected = critic_local(states, actions)
        critic_loss = F.mse_loss(q_expected, q_targets)
        # Minimize the loss
        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()

        # Compute actor loss
        actions_pred = actor_local(states)
        actor_loss = -critic_local(states, actions_pred).mean()
        # Minimize the loss
        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()

        # soft updates
        self.critic_triad.soft_update(self.tau)
        self.actor_triad.soft_update(self.tau)
