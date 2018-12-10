import gym
env = gym.make('MountainCarContinuous-v0')



import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F


from rlcc import transition

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

action_size = env.action_space.shape[0]
state_size = env.observation_space.shape[0]

# Network architecture
hidden1_size = 400
hidden2_size = 300

# Network learning parameters
lr_actor = 1e-4
lr_critic = 1e-3
tau = 1e-3
batch_size = 128
gamma = 0.99

# Replay buffer parameters
buffer_size = int(1e5)

# Network definitions
from models import ContinuousActor, ContinuousCritic
from rlcc.model import hard_update
actor_local = ContinuousActor(state_size, hidden1_size, hidden2_size, action_size)
actor_target = ContinuousActor(state_size, hidden1_size, hidden2_size, action_size)
actor_optimizer = optim.Adam(actor_local.parameters(), lr=lr_actor)
hard_update(actor_target, actor_local)

critic_local = ContinuousCritic(state_size, hidden1_size, hidden2_size, action_size)
critic_target = ContinuousCritic(state_size, hidden1_size, hidden2_size, action_size)
critic_optimizer = optim.Adam(critic_local.parameters(), lr=lr_critic)
hard_update(critic_target, critic_local)

# Actor definitions
from rlcc.act import NoisyActor, NetworkActor
from rlcc.noise import OUProcess
base_actor = NetworkActor(actor_local)
noise_process = OUProcess(action_size)
actor = NoisyActor(base_actor, noise_process)

# Replay buffer configuration
from rlcc import storage
buffer = storage.buffer(buffer_size=buffer_size)

# Define the observer
from rlcc.observe import PreprocessingObserver, BufferedObserver
observer = BufferedObserver(buffer)

# Learner
from rlcc.learn import DPGActor, DPGCritic, StackedLearningStrategy, ReplayLearner
learning_strategy = StackedLearningStrategy([
    DPGActor(actor_local, actor_target, actor_optimizer, critic_local, tau=tau),
    DPGCritic(critic_local, critic_target, critic_optimizer, actor_local, gamma=gamma, tau=tau)])
learner = ReplayLearner(learning_strategy, buffer, batch_size=batch_size)

# Converters
def state_to_tensor(state):
    return torch.tensor(state, dtype=torch.float)

def action_to_numpy(action):
    return action.cpu().data.numpy()

from tqdm import trange

episode_scores = []
with trange(100, desc='episode') as episode_bar:
    for episode in episode_bar:
        state = state_to_tensor(env.reset())        # get the current state
        score = 0.0
        while True:
            action = actor.act(state)               # select an action
            # Prior to converting to tensors
            next_state, reward, is_terminal, _ = env.step(action_to_numpy(action))
            # Convert to tensors
            next_state = torch.tensor(next_state, dtype=torch.float)
            reward_t = torch.tensor([reward], dtype=torch.float)
            is_terminal_t = torch.tensor([is_terminal], dtype=torch.float)
            # Create a transition
            t = transition(state, action, reward_t, next_state, is_terminal_t)
            observer.observe(t)                     # observe
            state = next_state                      # roll over states to next time step
            score += reward                         # update the score (for each agent)
            if is_terminal:                         # exit loop if episode finished
                break
            learner.learn()
        
        episode_scores.append(score)                # save most recent score
        episode_bar.set_postfix(score=episode_scores[-1])
