"""Local-Target-Optimizer functions"""


from collections import namedtuple


_LocalTargetOptimizer = namedtuple(
    "_LocalTargetOptimizer",
    field_names=["local", "target", "optimizer"])

class _ModelTriad(_LocalTargetOptimizer):
    def soft_update(self, tau):
        r"""Soft update model parameters.
            θ_target = τ*θ_local + (1 - τ)*θ_target
            Params
            ======
                tau (float): interpolation parameter
            """
        for tparam, lparam in zip(self.target.parameters(), self.local.parameters()):
            tparam.data.copy_(tau*lparam.data + (1.0-tau)*tparam.data)

    def hard_update(self):
        r"""Copy network parameters from source to target"""
        for tparam, lparam in zip(self.target.parameters(), self.local.parameters()):
            tparam.data.copy_(lparam.data)

def bind(local, target, optimizer):
    return _ModelTriad(local, target, optimizer)