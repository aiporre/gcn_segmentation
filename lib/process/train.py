import keras
import numpy as np
import torch
from torch import optim
import torch.nn as nn
import os
import re
import matplotlib.pyplot as plt
from torch_geometric.data import Data

from lib.utils import savefigs, get_npy_files, upload_training
from ..graph.batch import to_torch_batch

try:
    from dvn import FCN
except Exception as e:
    print('Warning: No module dvn. Failed to import deep vessel models (dvn.FCN), Exception: ', str(e))

try:
    import dvn.misc as ms

except Exception as e:
    print('Warning: No module dvn. Failed to import deep vessel models(dvn.misc), Exception: ', str(e))


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
        self.criterion = kwargs['criterion'] if 'criterion' in kwargs.keys() else nn.BCEWithLogitsLoss()
        self.lr = 0.01
        self.optimizer = None
        self._epoch = 0
        self._loss_per_iter = []
        # initialization of measurements collections
        self._measurements = {"train_loss": [],
                              "val_loss": [],
                              "DCS": []}
        for m in kwargs.get('measurements', []):
            self._measurements[m] = []

    def update_lr(self, lr):
        if not self.lr==lr or self.optimizer is None:
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
            self.lr = lr


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

        if isinstance(prediction, Data):
            prediction = to_torch_batch(prediction)
        loss = self.criterion(prediction, target)
        self.optimizer.zero_grad()
        loss.backward()
        # for param in self.model.parameters():
        #     if param.grad is not None:
        #         print('param.grad.data: ', param.grad.data.mean().item(),'±',param.grad.data.std().item())
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
                prtg = " ({0:3.2f}%) ".format(100.0*(i+1)/L)
                if (i+1)%10 == 0:
                    print('Training Epoch: in batch ', i+1, ' out of ', L, prtg, 'status: '+suffix)
            loss.append(loss_batch)
        return loss

    def update_loss_log(self, loss_values: list):
        assert isinstance(loss_values, list), "loss_values input must be a list or a tuple"
        self._loss_per_iter.extend(loss_values)

    def update_measurement(self, mea_dict):
        assert all([k in self._measurements.keys() for k in mea_dict.keys()]), \
            "All measurements must be updated at  same time. Given: {}, Expected: {},".format(mea_dict.keys(),
                                                                                              self._measurements.keys())
        for t, m in mea_dict.items():
            g = self._measurements[t]
            g.append(m)
            self._measurements = g

    def save_model(self, path):
        self.model.eval()
        torch.save(self.model.state_dict(), path)

    def load_model(self, model, path):
        self.model = model
        if os.path.exists(path):
            self.model.load_state_dict(torch.load(path))
        else:
            print('Warning: there is no file :', path)

    def get_range(self,EPOCHS):
        return range(self._epoch,EPOCHS)

    def load_checkpoint(self, prefix):
        files = get_npy_files()
        checkpoint_file = list(filter(lambda x: x.endswith('checkpoint.npy') and x.startswith(prefix), files))
        # finds the closest largest checkpoint file
        if len(checkpoint_file) > 1:
            e_list_strs = [re.search(r'_e(.*?)_lr', s).group(1) for s in checkpoint_file]
            # attempt to parse to int
            e_list_ints = []
            for ee  in e_list_strs:
                if ee.isdigit():
                    e_list_ints.append(int(ee))
                else:
                    raise ValueError(
                        'Something went wrong while extracting epoch list in your checkpoints {} with prefix {}'.format(
                            checkpoint_file, prefix))
            def argmax(iterable):
                return max(enumerate(iterable), key=lambda x: x[1])[0]
            checkpoint_file = [checkpoint_file[argmax(e_list_ints)]]
            prefix = checkpoint_file[0].split("_checkpoint.")[0]
        # presets the epochs
        if len(checkpoint_file)>0:
            d1 = np.load(checkpoint_file[0], allow_pickle=True)
            self._epoch = d1.item().get('e')
            best_metric = d1.item().get('best_metric')
            print('loaded checkpoint: ', self._epoch, 'best metric: ', best_metric)
        else:
            best_metric = None

        lossall_file = list(filter(lambda x: x.endswith('lossall.npy') and x.startswith(prefix), files))
        measurements_file = list(filter(lambda x: x.endswith('measurements.npy') and x.startswith(prefix), files))
        assert len(lossall_file) in [0,1], 'something failed while searching for the lossall file. ' \
                                           'Found: %s' % str(lossall_file)
        assert len(measurements_file) in [0,1], 'something failed while searching for the measurements file. ' \
                                                'Found: %s' % str(measurements_file)
        if len(lossall_file)==0 or len(measurements_file)==0:
            print("loss files and measurements were not found")
        else:
            print('Loading checkpoint')
            loss_all = np.load(lossall_file[0])
            self._loss_per_iter = loss_all.tolist()
            measurements = np.load(measurements_file[0], allow_pickle=True)
            measurements = measurements.item() # convert to dictionary
            for k,v in measurements.items():
                self._measurements[k] = v
        return best_metric
    def save_checkpoint(self, prefix, prefix_model, lr, e, EPOCHS, fig_dir, upload=False, best_metric=None):
        check_point = {'lr':lr,'e':e,'E':EPOCHS,'best_metric':best_metric}
        print('Saved checkpoint ', e,  '/', EPOCHS)
        np.save("{}_checkpoint.npy".format(prefix), check_point)
        np.save('{}_lossall'.format(prefix), self._loss_per_iter)
        np.save('{}_measurements'.format(prefix), self._measurements)
        if not len(self._loss_per_iter)==0:
            fig = plt.figure(figsize=(15, 10))

            plt.subplot(3, 1, 1)
            plt.plot(self._loss_per_iter)
            plt.xlabel('iterations')
            plt.ylabel('loss')
            plt.title('Loss history per iteration')

            plt.subplot(3, 1, 2)
            plt.plot(self._measurements["train_loss"], label="train loss")
            plt.plot(self._measurements["val_loss"], label="val loss")
            plt.xlabel('epochs')
            plt.ylabel('loss')
            plt.title('Loss history avg per epoch')
            plt.legend()

            plt.subplot(3, 1, 3)
            for target, mea in self._measurements.items():
                if target not in ["train_loss", "val_loss"]:
                    plt.plot(self._measurements[target], label=target)
            plt.xlabel('epochs')
            plt.ylabel('metrics')
            plt.title('Evaluation metrics')
            plt.legend()
            savefigs(fig_name='{}_loss_history'.format(prefix), fig_dir=fig_dir, fig=fig)
        if upload:
            print('Uploading training')
            upload_training(prefix=prefix,EPOCHS=EPOCHS,lr=lr)

class KTrainer(Trainer):
    def __init__(self,model,dataset,**kwargs):
        '''
        Keras trainer
        :param model:
        :param dataset:
        :param kwargs:
        '''
        super(KTrainer, self).__init__(model=model, dataset=dataset, **kwargs)
        self._model_compiled = False

    def update_lr(self, lr):
        if not self.lr==lr or self.optimizer is None:
            self.optimizer = keras.optimizers.SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)
            self.lr = lr

    def save_model(self, path):
        path = path.replace(".pth",".dat")
        self.model.save(filename=path)

    def load_model(self, model, path):
        self.model = model
        path = path.replace(".pth",".dat")

        if os.path.exists(path):
            self.model = FCN.load(filename=path)
        else:
            print('Warning: there is no file :', path)
    def train_batch(self):
        '''
        produce one batch iteration
        :return:
        '''
        X, Y = self.dataset.next_batch(batch_size=self._batch_size, shuffle=True)

        Y = Y.astype(int)
        Y = ms.to_one_hot(Y)
        dim = 2 # TODO:Hard-Coded
        Y = np.transpose(Y, axes=[0, dim+1]+list(range(1, dim+1)))
        B = self.dataset._batch_size
        history = self.model.fit(x=X, y=Y, epochs=1, batch_size=B,verbose=False)
        return history.history['loss'][0]

    def train_epoch(self, lr=0.01, progress_bar=True):
        if not self._model_compiled:
            self.update_lr(lr=lr)
            self.model.compile(optimizer=self.optimizer, metrics=[])  # compile the network (supports keras compile parameters)
            self._model_compiled=True

        return super(KTrainer,self).train_epoch(lr=lr,progress_bar=progress_bar)

        # self.update_lr(lr=lr)
        # X = self.dataset.get_images()
        # Y = self.dataset.get_labels()
        # Y = Y.astype(int)
        # Y = ms.to_one_hot(Y)
        # dim = 2 # TODO:Hard-Coded
        # Y = np.transpose(Y, axes=[0, dim+1]+list(range(1, dim+1)))
        # B = self.dataset._batch_size
        # history = self.model.fit(x=X, y=Y, epochs=1, batch_size=B)
        # return history.history['loss']
    def save_checkpoint(self, loss_all, measurements, prefix, prefix_model, lr, e, EPOCHS, fig_dir, upload=False, best_metric=None):
        super(KTrainer,self).save_checkpoint(loss_all=loss_all, measurements=measurements,prefix=prefix,
                                             lr=lr,  e=e,
                                             EPOCHS=EPOCHS, fig_dir=fig_dir, upload=False, best_metric=best_metric)
        if upload:
            print('Uploading training')
            upload_training(prefix_model=prefix_model, prefix=prefix, EPOCHS=EPOCHS,lr=lr,dataset_name=dataset_name,h5format=True)
    # def compile(self):
    #     self.update_lr(self.lr)
    #     self.model.compile(optimizer=self.optimizer,
    #                        metrics=[])  # compile the network (supports keras compile parameters)
