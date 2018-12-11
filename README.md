--------------------------------------------------------------------------------

rlcc is a Python package provides common components for reinforcement learning.  

The package uses pytorch for underlying tensor operations.

Where to go from here:

- [Installation](#installation)
- [Getting Started](#getting-started)

## Installation

```bash
pip install rlcc
```

## Getting Started

- [The API Reference](https://pjordan.github.io/rlcc/)

## Learning

`rlcc` supports learning through two abstract classes: `rlcc.learn.Learner` and `rlcc.learn.LearningStrategy`.
The classes are designed to be used in a strategy pattern, where the learner delegates stepwise learning
to the learning strategy.

The `rlcc.learn.ReplayLearner` class provides an example of a concrete learner.  The replay learner accepts
a learning strategy and source of transitions to learn from (replay buffer for example).

```python

    from rlcc.learn import ReplayLearner

    transitions = [...] # collection of transition objects.
    learning_strategy = [...] # Learning strategy described later.
    learner = ReplayLearner(learning_strategy, transitions)
```

Within a learning loop, we simply call

```python

    learner.learn()
```

to iterate the learning process.

The `rlcc` contains several learning strategies.  Each learning strategy accepts 
a batch of transition examples (a torch tensor) to incrementally learn from.


A abstract `rlcc.learn.DoubleLearningStrategy` accepts a triple of (local network, target network, optimizer).
Implementing classes define a loss function.  For example, `rlcc.learn.DPGActor` defines the DPG actor loss fuction,
instantiated as follows:


```python

    from rlcc.learn import DPGActor

    actor_local = [...] # network that accepts states and outputs actions
    actor_target = [...] # network with identical architecture to the local network
    actor_optimizer = [...] # Local network optimizer
    critic = [...] # Network that accepts (state, action) pairs and outputs the q-estimates
    learner = DPGActor(actor_local, actor_target, actor_optimizer, critic)
```

Sometimes, as in DPG, we need to train multiple networks.  In this case, we can use an
`rlcc.learn.StackedLearningStrategy` to coordinate learning amonst multiple strategies.
For example, we can construct an actor-critic stacked learning strategy.


```python

    from rlcc.learn import StackedLearningStrategy

    actor_stategy = [...]
    critic_stategy = [...] 
    actor_critic_strategy = StackedLearningStrategy([critic_stategy, actor_stategy])
```

## Multi-Agent Learning

The `rlcc` packages supports multi-agent learning through stacked classes: 

- `rlcc.act.StackedActor`
- `rlcc.observe.StackedObserver`
- `rlcc.learn.StackedLearner`

Each of these classes makes an underlying assumption that the input will be a list,
one per agent.  Learning, observing, and acting can then be composed of independent
instances per agent.

## Exploration

The `rlcc.noise` package contains various classes that can add noise to actions.

For example, to create an Ornstein-Uhlenbeck process, use

```python

    from rlcc.noise import OUProcess

    action_dimension = [...] # size of the action space
    noise_process = OUProcess(action_dimension)
```

The noise process should then be added to an existing actor to create a noisy actor.

```python

    from rlcc.act import NoisyActor

    noise_process = [...] # noise process
    actor = [...] # base actor
    noisy_actor = NoisyActor(actor, noise_process)
```

Because the noise may cause actions to exceed valid ranges, we may need to clip the values.

```python

    from rlcc.act import ClippingActor

    noisy_actor = [...]
    action_min, action_max = [...]
    clipped_actor = ClippingActor(noisy_actor, action_min=action_min, action_max=action_max)
```

## Replay Buffer

The `rlcc.storage` and `rlcc.observe` packages contains classes that can be used to create replay buffers from transition observations.

For example, use the following to create a size-limited buffer.

```python

    from rlcc import storage

    buffer_size = int(1e5) 
    buffer = storage.buffer(buffer_size=buffer_size)
```

This buffer can now be used in an observer as the backing storage.

```python

    from rlcc.observe import BufferedObserver
    observer = BufferedObserver(buffer)
```
For learning, the buffer is input into an `rlcc.learn.ReplayLearner`.

