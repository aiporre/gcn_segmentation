from unittest import TestCase
from .train import Trainer
from ..datasets.m2nist import M2NIST

dataset = M2NIST()
trainer = Trainer(model=None,dataset=dataset,batch_size=10)
class TrainerTest(TestCase):
    def test_init(self):
        print(len(trainer))