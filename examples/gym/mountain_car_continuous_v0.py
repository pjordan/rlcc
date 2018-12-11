# https://github.com/openai/gym/wiki/MountainCarContinuous-v0

import gym
env = gym.make('MountainCarContinuous-v0')

import numpy as np

import torch
import torch.optim as optim

from rlcc import transition

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

action_size = env.action_space.shape[0]
state_size = env.observation_space.shape[0]

# Network architecture
hidden1_size = 20
hidden2_size = 10

# Network learning parameters
lr_actor = 1e-3
lr_critic = 1e-3
tau = 1e-2
batch_size = 128
gamma = 0.99

# Replay buffer parameters
buffer_size = int(1e5)

# Network definitions
from models import ContinuousActor, ContinuousCritic
from rlcc.model import hard_update
actor_local = ContinuousActor(state_size, hidden1_size, hidden2_size, action_size).to(device)
actor_target = ContinuousActor(state_size, hidden1_size, hidden2_size, action_size).to(device)
actor_optimizer = optim.Adam(actor_local.parameters(), lr=lr_actor)
hard_update(actor_target, actor_local)

critic_local = ContinuousCritic(state_size, hidden1_size, hidden2_size, action_size).to(device)
critic_target = ContinuousCritic(state_size, hidden1_size, hidden2_size, action_size).to(device)
critic_optimizer = optim.Adam(critic_local.parameters(), lr=lr_critic)
hard_update(critic_target, critic_local)

# Noise process and actor definitions
from rlcc.noise import OUProcess, ScaledProcess, DeviceNoiseProcess, ScheduledScaledProcess
from rlcc.schedule import ExponentialSchedule

noise_scale_schedule = ExponentialSchedule(2.0, 0.9)
noise_process = DeviceNoiseProcess(
    ScheduledScaledProcess(
        OUProcess(action_size), 
        noise_scale_schedule), 
    device)

from rlcc.act import NoisyActor, NetworkActor, ClippingActor
actor = ClippingActor(
    NoisyActor(
        NetworkActor(actor_local),
        noise_process),
    action_min=-1.0, action_max=1.0)

# Replay buffer configuration
from rlcc import storage
buffer = storage.buffer(buffer_size=buffer_size)

# Define the observer
from rlcc.observe import BufferedObserver
observer = BufferedObserver(buffer)

# Learning observers
from tensorboardX import SummaryWriter
writer = SummaryWriter(comment='MountainCarContinuous-v0')
write_frequency = 1000

def critic_observer(**kwargs):
    transitions = kwargs['transitions']
    learning_steps = kwargs['learning_steps']
    loss = kwargs['loss']
    local = kwargs['local']
    writer.add_scalar('data/critic_loss', loss, learning_steps)
    if learning_steps % write_frequency == 0:
        for n, p in filter(lambda np: np[1].grad is not None, local.named_parameters()):
            writer.add_histogram("critic_local."+n+".grad", p.grad.data.cpu().numpy(), global_step=learning_steps)
            writer.add_histogram("critic_local."+n, p.data.cpu().numpy(), global_step=learning_steps)

def actor_observer(**kwargs):
    transitions = kwargs['transitions']
    learning_steps = kwargs['learning_steps']
    loss = kwargs['loss']
    local = kwargs['local']
    writer.add_scalar('data/actor_loss', loss, learning_steps)
    if learning_steps % write_frequency == 0:
        for n, p in filter(lambda np: np[1].grad is not None, local.named_parameters()):
            writer.add_histogram("actor_local."+n+".grad", p.grad.data.cpu().numpy(), global_step=learning_steps)
            writer.add_histogram("actor_local."+n, p.data.cpu().numpy(), global_step=learning_steps)
        s, a, r, ns, term = transitions
        writer.add_histogram("transitions/states.loc", s[:,0].data.cpu().numpy(), global_step=learning_steps)
        writer.add_histogram("transitions/states.vel", s[:,1].data.cpu().numpy(), global_step=learning_steps)
        writer.add_histogram("transitions/actions", a.data.cpu().numpy(), global_step=learning_steps)
        writer.add_histogram("transitions/rewards", r.data.cpu().numpy(), global_step=learning_steps)
        writer.add_histogram("transitions/next_states.loc", ns[:,0].data.cpu().numpy(), global_step=learning_steps)
        writer.add_histogram("transitions/next_states.vel", ns[:,1].data.cpu().numpy(), global_step=learning_steps)
        writer.add_histogram("transitions/is_terminals", term.data.cpu().numpy(), global_step=learning_steps)

def write_buffer(steps):
    s, a, r, ns, term = map(torch.stack, zip(*buffer))
    writer.add_histogram("buffer/states.loc", s[:,0].data.cpu().numpy(), global_step=steps)
    writer.add_histogram("buffer/states.vel", s[:,1].data.cpu().numpy(), global_step=steps)
    writer.add_histogram("buffer/actions", a.data.cpu().numpy(), global_step=steps)
    writer.add_histogram("buffer/rewards", r.data.cpu().numpy(), global_step=steps)
    writer.add_histogram("buffer/next_states.loc", ns[:,0].data.cpu().numpy(), global_step=steps)
    writer.add_histogram("buffer/next_states.vel", ns[:,1].data.cpu().numpy(), global_step=steps)
    writer.add_histogram("buffer/is_terminals", term.data.cpu().numpy(), global_step=steps)
        
# Learner
from rlcc.learn import DPGActor, DPGCritic, StackedLearningStrategy, ReplayLearner
actor_strategy = DPGActor(actor_local, actor_target, actor_optimizer, critic_local, tau=tau)
actor_strategy.observers.append(actor_observer)

critic_strategy = DPGCritic(critic_local, critic_target, critic_optimizer, actor_target, gamma=gamma, tau=tau)
critic_strategy.observers.append(critic_observer)

learning_strategy = StackedLearningStrategy([critic_strategy,actor_strategy])
learner = ReplayLearner(learning_strategy, buffer, batch_size=batch_size, batches_per_step=1000)

# Converters
def state_to_tensor(state):
    return torch.tensor(state, dtype=torch.float).to(device)

def action_to_numpy(action):
    return action.cpu().data.numpy()

from tqdm import trange

episode_scores = []
with trange(1000, desc='episode') as episode_bar:
    for episode in episode_bar:
        state = state_to_tensor(env.reset())        # get the current state
        score = 0.0
        step = 0
        while True:
            action = actor.act(state)               # select an action
            # Prior to converting to tensors
            next_state, reward, is_terminal, _ = env.step(action_to_numpy(action))
            # Convert to tensors
            next_state = torch.tensor(next_state, dtype=torch.float).to(device)
            reward_t = torch.tensor([reward], dtype=torch.float).to(device)
            is_terminal_t = torch.tensor([is_terminal], dtype=torch.float).to(device)
            # Create a transition
            t = transition(state, action, reward_t, next_state, is_terminal_t)
            observer.observe(t)                     # observe
            state = next_state                      # roll over states to next time step
            score += reward                         # update the score (for each agent)
            step += 1
            if is_terminal:                         # exit loop if episode finished
                break
        learner.learn()
        write_buffer(episode)
        noise_scale_schedule.update()
        
        writer.add_scalar('data/score', score, episode)
        episode_scores.append(score)                # save most recent score
        episode_bar.set_postfix(score=episode_scores[-1], steps=step)
        

if max(episode_scores) >= 90.0:
    print("Congratulations, you have solved the MountainCarContinuous-v0.")
else:
    print("You have not solved the MountainCarContinuous-v0.")