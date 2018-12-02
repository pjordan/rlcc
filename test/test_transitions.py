import unittest
import rlcc.transitions as transitions
import torch

class TestExperiencesMethods(unittest.TestCase):

    def test_make(self):
        t = transitions.make(
            [1.0], # state
            [2.0], # action
            [3.0], # reward
            [4.0], # next_state
            [0] # is_terminal
            )
        self.assertEqual(t.state, torch.tensor([1.0], dtype=torch.float))
        self.assertEqual(t.action, torch.tensor([2.0], dtype=torch.float))
        self.assertEqual(t.reward, torch.tensor([3.0], dtype=torch.float))
        self.assertEqual(t.next_state, torch.tensor([4.0], dtype=torch.float))
        self.assertEqual(t.is_terminal, torch.tensor([0.0], dtype=torch.float))

if __name__ == '__main__':
    unittest.main()