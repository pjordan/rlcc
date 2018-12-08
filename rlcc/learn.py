from itertools import islice 
import torch.nn.functional as F
import torch.nn.utils as nn_utils
from .replay import TransitionReplayer

class Learner():
    r"""Base class for all learners.
    """

    def __init__(self):
        pass

    def learn(self):
        r"""Performs a single learning step."""
        raise NotImplementedError

class StackedLearner(Learner):
    """Implements a learner that delegates to other learners.
    """

    def __init__(self, learners):
        super(StackedLearner, self).__init__()
        self.learners = learners
        
    def learn(self):
        r"""Performs a single learning step."""
        for learner in self.learners:
            learner.learn()

class ReplayLearner(Learner):
    """Implements a learner using a replay buffer.
    """

    def __init__(self, learning_strategy, transitions, device=None, batch_size=128, batches_per_step=1):
        super(ReplayLearner, self).__init__()
        self.learning_strategy = learning_strategy
        self.batch_size = batch_size
        self.batches_per_step = batches_per_step
        self.replayer = TransitionReplayer(transitions, device=device, batch_size=batch_size)

    def learn(self):
        r"""Performs a single learning step."""
        if len(self.replayer)>self.batch_size:
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

    def __init__(self, actor_triad, critic_triad, device,
                 gamma=0.99, tau=1e-3, writer=None, write_frequency=1, write_prefix='',
                 actor_norm_clip=None, critic_norm_clip=None):
        super(DPG, self).__init__()
        self.gamma = gamma
        self.tau = tau
        self.actor_triad = actor_triad
        self.critic_triad = critic_triad
        self.writer = writer
        self.learning_steps = 0
        self.write_frequency = write_frequency
        self.write_prefix = write_prefix
        self.actor_norm_clip = actor_norm_clip
        self.critic_norm_clip = critic_norm_clip

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
        if self.critic_norm_clip:
            nn_utils.clip_grad_norm_(critic_local.parameters(), self.critic_norm_clip)
        critic_optimizer.step()

        if self.writer is not None:
            self.writer.add_scalar('data/'+self.write_prefix+'critic_loss',critic_loss, self.learning_steps)
            if self.learning_steps % self.write_frequency == 0:
                for n, p in filter(lambda np: np[1].grad is not None, critic_local.named_parameters()):
                    self.writer.add_histogram(self.write_prefix+"critic_local."+n+".grad", p.grad.data.cpu().numpy(), global_step=self.learning_steps)
                    self.writer.add_histogram(self.write_prefix+"critic_local."+n, p.data.cpu().numpy(), global_step=self.learning_steps)
                    
        # Compute actor loss
        actions_pred = actor_local(states)
        actor_loss = -critic_local(states, actions_pred).mean()
        # Minimize the loss
        actor_optimizer.zero_grad()
        actor_loss.backward()
        if self.actor_norm_clip:
            nn_utils.clip_grad_norm_(actor_local.parameters(), self.actor_norm_clip)
        actor_optimizer.step()

        if self.writer is not None:
            self.writer.add_scalar('data/'+self.write_prefix+'actor_loss',actor_loss, self.learning_steps)
            if self.learning_steps % self.write_frequency == 0:
                for n, p in filter(lambda np: np[1].grad is not None, actor_local.named_parameters()):
                    self.writer.add_histogram(self.write_prefix+"actor_local."+n+".grad", p.grad.data.cpu().numpy(), global_step=self.learning_steps)
                    self.writer.add_histogram(self.write_prefix+"actor_local."+n, p.data.cpu().numpy(), global_step=self.learning_steps)
                    
        # soft updates
        self.critic_triad.soft_update(self.tau)
        self.actor_triad.soft_update(self.tau)
        
        # update counts
        self.learning_steps += 1
