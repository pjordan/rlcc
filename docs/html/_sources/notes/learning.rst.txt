:github_url: https://github.com/pjordan/rlcc

Learning
========

`rlcc` supports learning through two abstract classes: `rlcc.learn.Learner` and `rlcc.learn.LearningStrategy`.
The classes are designed to be used in a strategy pattern, where the learner delegates stepwise learning
to the learning strategy.

The `rlcc.learn.ReplayLearner` class provides an example of a concrete learner.  The replay learner accepts
a learning strategy and source of transitions to learn from (replay buffer for example).

.. code-block:: python

    from rlcc.learn import ReplayLearner

    transitions = [...] # collection of transition objects.
    learning_strategy = [...] # Learning strategy described later.
    learner = ReplayLearner(learning_strategy, transitions)

Within a learning loop, we simply call

.. code-block:: python

    learner.learn()

to iterate the learning process.

The `rlcc` contains several learning strategies.  Each learning strategy accepts 
a batch of transition examples (a torch tensor) to incrementally learn from.


A abstract `rlcc.learn.DoubleLearningStrategy` accepts a triple of (local network, target network, optimizer).
Implementing classes define a loss function.  For example, `rlcc.learn.DPGActor` defines the DPG actor loss fuction,
instantiated as follows:


.. code-block:: python

    from rlcc.learn import DPGActor

    actor_local = [...] # network that accepts states and outputs actions
    actor_target = [...] # network with identical architecture to the local network
    actor_optimizer = [...] # Local network optimizer
    critic = [...] # Network that accepts (state, action) pairs and outputs the q-estimates
    learner = DPGActor(actor_local, actor_target, actor_optimizer, critic)


Sometimes, as in DPG, we need to train multiple networks.  In this case, we can use an
`rlcc.learn.StackedLearningStrategy` to coordinate learning amonst multiple strategies.
For example, we can construct an actor-critic stacked learning strategy.


.. code-block:: python

    from rlcc.learn import StackedLearningStrategy

    actor_stategy = [...]
    critic_stategy = [...] 
    actor_critic_strategy = StackedLearningStrategy([critic_stategy, actor_stategy])


