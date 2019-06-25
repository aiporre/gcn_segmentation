import torch
from torch import optim
import torch.nn as nn
import os

from .progress_bar import printProgressBar


class Trainer(object):

    '''Class to train a model '''
    def __init__(self,model,dataset, **kwargs):
        self.model = model
        self.dataset = dataset.train
        self._batch_size = kwargs['batch_size'] if 'batch_size' in kwargs.keys() else 1
        self.dataset.enforce_batch(self._batch_size)
        self.to_tensor = kwargs['to_tensor'] if 'to_tensor' in kwargs.keys() else True
        self.device = kwargs['device'] if 'device' in kwargs.keys() else torch.device('cpu')

    def update_lr(self, lr):
        self.optimizer = optim.SGD(self.model.parameters(),
                              lr=lr,
                              momentum=0.9,
                              weight_decay=0.0005)

        self.criterion = nn.BCELoss()

    def train_batch(self):
        '''
        produce one batch iteration
        :return:
        '''
        images, labels = self.dataset.next_batch(batch_size=self._batch_size, shuffle=True)
        features = torch.tensor(images).float() if self.to_tensor else images
        target = torch.tensor(labels).float() if self.to_tensor else labels

        features = features.to(self.device)
        target = target.to(self.device)

        prediction = self.model(features)

        prediction_flat = prediction.view(-1)
        target_flat = target.view(-1)

        loss = self.criterion(prediction_flat,target_flat)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
    def train_epoch(self, lr=0.01, progress_bar=True):
        loss = []
        self.update_lr(lr=lr)
        L = self.dataset.num_batches
        print('--> L', L)
        if progress_bar:
            printProgressBar(0, L, prefix='Train Epoch:', suffix='Complete', length=50)
        for i in range(self.dataset.num_batches):
            loss_batch = self.train_batch()
            suffix = 'loss batch '+ str(loss_batch)
            if progress_bar:
                printProgressBar(i+1, L, prefix='Train Epoch:', suffix='Complete, '+suffix, length=50)
            else:
                print('Training Epoch: in batch ', i+1, ' out of ', L, '({})'.format(100.0*(i+1)/L) , 'status: '+suffix)
            loss.append(loss_batch)
        return loss

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, model, path):
        self.model = model
        if os.path.exists(path):
            self.model.load_state_dict(torch.load(path))
        else:
            print('Warning: there is no file :', path)