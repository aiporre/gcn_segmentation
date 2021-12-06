from unittest import TestCase
import torch
from lib.process.evaluation import dice_coeff

class Test(TestCase):
    def test_dice_coeff(self):
        a = torch.tensor([[[0.0, 0.0], [0.0, 0.0]], [[0.1, 0.1], [0.9, 0.9]]])

        a_mask = (a>0.5).float()
        b = torch.tensor([[[0.0, 0.0], [0.0, 0.0]], [[1.0, 0.0], [1.0, 1.0]]])
        value = dice_coeff(a_mask,b)
        print(value)
        value = dice_coeff(b, b)
        print(value)
