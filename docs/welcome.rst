Welcome to rlcc's documentation!
================================

The Reinforcement learning common components (rlcc) package
contains basic functionality to aid in creating RL agents.

Agent components are classified into three basic stages:

- act: take some action based on a state
- observe: observe the result of taking the action
- learn: learn from history

These stages form a loop while interacting with the environment.
The basic structure is shown below in pseudo-code.

.. code-block:: python

    state = env.reset()  # initial state for episode
    while True: 
        action = actor.act(state) 
        next_state, reward, is_terminal = env.step(action)
        transition = (state, action, reward, next_state, is_terminal)
        observer.observe(transition)                     
        state = next_state                      
        if is_terminal:                      
            break
    learner.learn()

Examples are available at `<https://github.com/pjordan/rlcc/examples/>`_

Installation
============

Install the package (or add it to your ``requirements.txt`` file):

.. code:: bash

    pip install rlcc


