r"""Model utility functions"""


def soft_update(target, local, tau):
    r"""Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        
    :param target: target network
    :type target: list of :class:`torch.nn.Module`
    :param local: target network
    :type local: list of :class:`torch.nn.Module`
    :param tau: the weight
    :type tau: float
    """
    for tparam, lparam in zip(target.parameters(), local.parameters()):
        tparam.data.copy_(tau*lparam.data + (1.0-tau)*tparam.data)


def hard_update(target, local):
    r"""Copy network parameters from source to target.
    
    :param target: target network
    :type target: list of :class:`torch.nn.Module`
    :param local: target network
    :type local: list of :class:`torch.nn.Module`
    """
    for tparam, lparam in zip(target.parameters(), local.parameters()):
        tparam.data.copy_(lparam.data)
