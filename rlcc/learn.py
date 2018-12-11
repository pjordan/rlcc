r"""Learning and related functionality."""


from itertools import islice 
import torch.nn.functional as F
import torch.nn.utils as nn_utils
from .replay import TransitionReplayer
from .model import soft_update

class Learner():
    r"""Base class for all learners.
    
    Your learners should also subclass this class.
    """
    def __init__(self):
        r"""Constructor method"""
        pass

    def learn(self):
        r"""Performs a single learning step."""
        raise NotImplementedError

class StackedLearner(Learner):
    r"""Implements a learner that delegates to other learners.

    :param learners: list of learners
    :type learners: list of :class:`rlcc.learn.Learner`
    """
    def __init__(self, learners):
        r"""Constructor method"""
        super(StackedLearner, self).__init__()
        self.learners = learners
        
    def learn(self):
        r"""Performs a single learning step."""
        for learner in self.learners:
            learner.learn()

class ReplayLearner(Learner):
    r"""Implements a learner using a replay buffer.
    
    :param learning_strategy: delegate learning strategy
    :type learning_strategy: :class:`rlcc.learn.LearningStrategy`
    :param transitions: list of transitions
    :type transitions: list of :class:`rlcc.Transition`
    :param device: the torch device
    :type device: str, optional
    :param batch_size: the batch size
    :type batch_size: int, optional
    :param batches_per_step: the number of batches per step
    :type batches_per_step: int, optional
    """
    def __init__(self, learning_strategy, transitions, batch_size=128, batches_per_step=1):
        r"""Constructor method"""
        super(ReplayLearner, self).__init__()
        self.learning_strategy = learning_strategy
        self.batch_size = batch_size
        self.batches_per_step = batches_per_step
        self.replayer = TransitionReplayer(transitions, batch_size=batch_size)

    def learn(self):
        r"""Performs a single learning step."""
        if len(self.replayer)>self.batch_size:
            for batch in islice(self.replayer, self.batches_per_step):
                self.learning_strategy.step(batch)

class LearningStrategy():
    r"""Base class for all learning strategies.

    Your learning strategies should also subclass this class.
    """

    def __init__(self):
        r"""Constructor method"""
        pass

    def step(self, transitions):
        r"""Performs a single learning step.

        :param transitions: a list of :class:`rlcc.Transition`
        :type transitions: :class:`rlcc.Transition`
        """
        raise NotImplementedError


class StackedLearningStrategy(LearningStrategy):
    r"""Implements stacked learning strategy

    :param strategies: list of learning strategies
    :type strategies: list of :class:`rlcc.learn.LearningStrategy`
    """
    def __init__(self, strategies):
        r"""Constructor method"""
        super(StackedLearningStrategy, self).__init__()
        self.strategies = strategies

    def step(self, transitions):
        r"""Performs a single learning step.

        :param transitions: a list of :class:`rlcc.Transition`
        :type transitions: :class:`rlcc.Transition`
        """
        for strategy in self.strategies:
            strategy.step(transitions)


class DoubleLearningStrategy(LearningStrategy):
    r"""Implements a double learning strategy.
    
    :param local: the local network
    :type local: :class:`torch.nn.Module`
    :param target: the target network
    :type target: :class:`torch.nn.Module`
    :param optimizer: the optimizer
    :type optimizer: :class:`torch.nn.Module`
    :param tau: the update weight
    :type tau: float, optional
    :param clip_norm: the gradient norm maximum
    :type clip_norm: float, optional
    :param clip_value: the gradient value maximum
    :type clip_value: float, optional
    """
    def __init__(self, 
                 local, 
                 target,
                 optimizer,
                 tau=1e-3, 
                 clip_norm=None,
                 clip_value=None):
        r"""Constructor method"""
        super(DoubleLearningStrategy, self).__init__()
        self.tau = tau
        self.local = local
        self.target = target
        self.optimizer = optimizer
        self.observers = []
        self.clip_norm = clip_norm
        self.clip_value = clip_value
        self.learning_steps = 0

    def step(self, transitions):
        r"""Performs a single learning step.

        :param transitions: a list of :class:`rlcc.Transition`
        :type transitions: :class:`rlcc.Transition`
        """
        # Compute actor loss
        loss = self.loss(transitions)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        # Clip the gradient
        if self.clip_norm:
            nn_utils.clip_grad_norm_(self.local.parameters(), self.clip_norm)
        if self.clip_value:
            nn_utils.clip_grad_value_(self.local.parameters(), self.clip_value)
        # Optimize
        self.optimizer.step()
        # Notify the observers
        for obs in self.observers:
            obs(transitions=transitions, local=self.local, loss=loss, learning_steps=self.learning_steps)
        # soft updates
        soft_update(self.target, self.local, self.tau)

        # update counts
        self.learning_steps += 1

    def loss(self, transitions):
        r"""Compute the loss of the transitions

        :param transitions: a list of :class:`rlcc.Transition`
        :type transitions: :class:`rlcc.Transition`
        """
        raise NotImplementedError


class DPGActor(DoubleLearningStrategy):
    r"""Implements Deterministic Policy Gradient Actor learner.
    
    :param actor_local: the local network
    :type actor_local: :class:`torch.nn.Module`
    :param actor_target: the target network
    :type actor_target: :class:`torch.nn.Module`
    :param actor_optimizer: the optimizer
    :type actor_optimizer: :class:`torch.nn.Module`
    :param critic: the critic
    :type critic: :class:`torch.nn.Module`
    :param tau: the update weight
    :type tau: float, optional
    :param observers: list of observers
    :type observers: list of observers, optional
    :param clip_norm: the gradient norm maximum
    :type clip_norm: float, optional
    :param clip_value: the gradient value maximum
    :type clip_value: float, optional
    """
    def __init__(self, 
                 actor_local,
                 actor_target,
                 actor_optimizer, 
                 critic,
                 tau=1e-3,
                 clip_norm=None,
                 clip_value=None):
        r"""Constructor method"""
        super(DPGActor, self).__init__(
            actor_local, 
            actor_target,
            actor_optimizer,
            tau=tau,
            clip_norm=clip_norm,
            clip_value=clip_value)
        self.critic = critic

    def loss(self, transitions):
        r"""Compute the loss of the transitions

        :param transitions: a list of :class:`rlcc.Transition`
        :type transitions: :class:`rlcc.Transition`
        """
        # Unpack tuples
        states, _, _, _, _ = transitions
        return -self.critic(states, self.local(states)).mean()


class DPGCritic(DoubleLearningStrategy):
    r"""Implements Deterministic Policy Gradient Critic learner.
    
    :param critic_local: the local network
    :type critic_local: :class:`torch.nn.Module`
    :param critic_target: the target network
    :type critic_target: :class:`torch.nn.Module`
    :param critic_optimizer: the optimizer
    :type critic_optimizer: :class:`torch.nn.Module`
    :param actor: the actor
    :type actor: :class:`torch.nn.Module`
    :param tau: the update weight
    :type tau: float, optional
    :param observers: list of observers
    :type observers: list of observers, optional
    :param clip_norm: the gradient norm maximum
    :type clip_norm: float, optional
    :param clip_value: the gradient value maximum
    :type clip_value: float, optional
    """
    def __init__(self, 
                 critic_local,
                 critic_target,
                 critic_optimizer, 
                 actor,
                 gamma=0.99, 
                 tau=1e-3,
                 clip_norm=None,
                 clip_value=None):
        r"""Constructor method"""
        super(DPGCritic, self).__init__(
            critic_local, 
            critic_target,
            critic_optimizer,
            tau=tau,
            clip_norm=clip_norm,
            clip_value=clip_value)
        self.gamma = gamma
        self.actor = actor

    def loss(self, transitions):
        r"""Compute the loss of the transitions

        :param transitions: a list of :class:`rlcc.Transition`
        :type transitions: :class:`rlcc.Transition`
        """
        # Unpack tuples
        states, actions, rewards, next_states, is_terminals = transitions
              
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor(next_states)
        q_targets_next = self.target(next_states, actions_next)
        
        # Compute Q targets for current states
        q_targets = rewards + (self.gamma * q_targets_next * (1 - is_terminals))
        
        # Compute critic loss
        q_expected = self.local(states, actions)
        return F.mse_loss(q_expected, q_targets)

