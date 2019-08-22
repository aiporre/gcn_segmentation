from unittest import TestCase

from lib.process.evaluation import KEvaluator
from .train import Trainer, KTrainer
from ..datasets.mnist import MNIST
from dvn import FCN, VNET, UNET  # import libraries
import numpy as np

import dvn.losses as ls
import dvn.misc as ms
dataset = MNIST()
trainer = Trainer(model=None,dataset=dataset,batch_size=10)
class TrainerTest(TestCase):
    def test_init(self):
        print(len(trainer.dataset))
    def test_vesselnet(self):
        dim = 2
        # print('Testing FCN Network')
        # print(('Data Information => ', 'volume size:', X.shape, ' labels:', np.unique(Y)))
        # net.fit(x=X, y=Y, epochs=30, batch_size=2, shuffle=True)
        net = FCN(dim=2, nchannels=1, nlabels=2)  # create the network object (You can replace FCN with VNET or UNET),

        trainer = KTrainer(model=net, dataset=dataset, batch_size=10)
        evaluator = KEvaluator(dataset)
        trainer.train_epoch()

        # net.compile()                               # compile the network (supports keras compile parameters)
        # net.fit(x=X, y=Y, epochs=10, batch_size=10) # train the network (supports keras fit parameters)
        preds = net.predict(x=dataset.test.get_images())
        print(preds.shape)
        v = evaluator.bin_scores(net)
        print(v)


        # predict (supports keras predict parameters)
        # net.save(filename='model.dat')              # save network params
        # net = FCN.load(filename='model.dat')        # Load network params  (You can replace FCN with VNET or UNET as used above)