import torch
from torch import optim
import torch.nn as nn
import os

class Trainer(object):

    '''Class to train a model '''
    def __init__(self,model,dataset, **kwargs):
        self.model = model
        self.dataset = dataset.train
        self._batch_size = kwargs['batch_size'] if 'batch_size' in kwargs.keys() else 1

    def __len__(self):
        return self.dataset.num_examples

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
        features = torch.tensor(images).float()
        target = torch.tensor(labels).float()

        prediction = self.model(features)

        prediction_flat = prediction.view(-1)
        target_flat = target.view(-1)

        loss = self.criterion(prediction_flat,target_flat)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
    def train_epoch(self, lr=0.01):
        loss = []
        self.update_lr(lr=lr)
        currect_epochs_completed = self.dataset.epochs_completed
        while self.dataset.epochs_completed == currect_epochs_completed:
            loss_batch = self.train_batch()
            print('loss batch ', loss_batch)
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