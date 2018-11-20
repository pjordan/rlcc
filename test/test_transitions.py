import unittest
import numpy as np
import rlcc
import rlcc.transitions as transitions
import torch

class TestExperiencesMethods(unittest.TestCase):

    def test_from_primitives(self):
        t = transitions.from_primitives(
            [1.0], # state
            [2.0], # action
            [3.0], # reward
            [4.0], # next_state
            [0] # is_terminal
            )
        self.assertEqual(t.state, torch.from_numpy(np.array([1.0])).float())
        self.assertEqual(t.action, torch.from_numpy(np.array([2.0])).float())
        self.assertEqual(t.reward, torch.from_numpy(np.array([3.0])).float())
        self.assertEqual(t.next_state, torch.from_numpy(np.array([4.0])).float())
        self.assertEqual(t.is_terminal, torch.from_numpy(np.array([0.0])).float())

    def test_collate(self):
        t1 = transitions.from_primitives(
            [1.0], # state
            [2.0], # action
            [3.0], # reward
            [4.0], # next_state
            [0] # is_terminal
            )

        t2 = transitions.from_primitives(
            [5.0], # state
            [6.0], # action
            [7.0], # reward
            [8.0], # next_state
            [1] # is_terminal
            )
        
        t3 = transitions.collate([t1,t2])

        self.assertTrue(torch.equal(t3.state, torch.from_numpy(np.array([[1.0],[5.0]])).float()))
        self.assertTrue(torch.equal(t3.action, torch.from_numpy(np.array([[2.0],[6.0]])).float()))
        self.assertTrue(torch.equal(t3.reward, torch.from_numpy(np.array([[3.0],[7.0]])).float()))
        self.assertTrue(torch.equal(t3.next_state, torch.from_numpy(np.array([[4.0],[8.0]])).float()))
        self.assertTrue(torch.equal(t3.is_terminal, torch.from_numpy(np.array([[0.0],[1.0]])).float()))

if __name__ == '__main__':
    unittest.main()