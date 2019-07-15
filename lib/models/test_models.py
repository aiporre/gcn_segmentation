from unittest import TestCase
from .models import FCN

import torch
import torch.nn.functional as F

class TestFCN(TestCase):
    def setUp(self):
        self.model = FCN(1,1)

    def test_instance(self):
        input = torch.rand(1,1,512,512, dtype=torch.float)
        target = torch.randint(0,1,(1,1,512,512),dtype=torch.float)
        pred = self.model(input)
        self.assertEqual(pred.shape, (1,1,512,512))
        print('errors: ', (pred-target).sum())
        loss = F.binary_cross_entropy(pred,target)
        print('loss: ' , loss.item())

