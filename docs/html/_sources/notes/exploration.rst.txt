Exploration
===========

The `rlcc.noise` package contains various classes that can add noise to actions.

For example, to create an Ornstein-Uhlenbeck process, use

.. code-block:: python

    from rlcc.noise import OUProcess

    action_dimension = [...] # size of the action space
    noise_process = OUProcess(action_dimension)

The noise process should then be added to an existing actor to create a noisy actor.

.. code-block:: python

    from rlcc.act import NoisyActor

    noise_process = [...] # noise process
    actor = [...] # base actor
    noisy_actor = NoisyActor(actor, noise_process)

Because the noise may cause actions to exceed valid ranges, we may need to clip the values.

.. code-block:: python

    from rlcc.act import ClippingActor

    noisy_actor = [...]
    action_min, action_max = [...]
    clipped_actor = ClippingActor(noisy_actor, action_min=action_min, action_max=action_max)