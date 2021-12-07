import shutil

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
from lib.process.evaluation import dice_coeff

try:
    from dvn import FCN
except Exception as e:
    print('Warning: No module dvn. Failed to import deep vessel models (dvn.FCN), Exception: ', str(e))

try:
    import dvn.misc as ms

except Exception as e:
    print('Warning: No module dvn. Failed to import deep vessel models(dvn.misc), Exception: ', str(e))

from .progress_bar import printProgressBar


class TrainingDir(object):
    net_name: str

    def __init__(self, root, net, dataset, id, EPOCHS, mode="best"):
        assert root is not None and os.path.exists(root), f"root directory {root} does not exist"
        assert mode in ["best", "last"], f"Invalid mode {mode} valid values are 'best', 'last'"
        self.root = root
        self.net_name = net
        self.dataset_name = dataset
        self.id = id
        self.mode = mode
        self.EPOCHS = EPOCHS
        self.legacy()

    def legacy(self):
        if self.mode == "best" and not os.path.exists(self.model_path):
            model_path_legacy = os.path.join(self.root, self.prefix_model + ".pth")
            if os.path.exists(model_path_legacy):
                print('WARNING: model path legacy was found and will be copied to its new destination', self.model_path)
                shutil.copyfile(model_path_legacy, self.model_path)

    @property
    def model_path(self):
        # Notice that ending is not _best because .pth was already used as ending for best mode
        ending = "_last.pth" if self.mode == "last" else "_best.pth"
        model_path = os.path.join(self.root, self.prefix_model + ending)
        return model_path

    @property
    def model_path_last(self):
        return os.path.join(self.root, self.prefix_model + "_last.pth")

    @property
    def model_path_best(self):
        # Notice that ending is not _best because .pth was already used as ending for best mode
        return os.path.join(self.root, self.prefix_model + "_best.pth")

    @property
    def checkpoint_path(self):
        return os.path.join(self.root, "{}_checkpoint.npy".format(self.prefix))


    @property
    def lossall_path(self):
        return os.path.join(self.root, "{}_lossall.npy".format(self.prefix))

    @property
    def measurements_path(self):
        return os.path.join(self.root, "{}_measurements.npy".format(self.prefix))

    @property
    def prefix_model(self):
        return "{}-ds{}-id{}".format(self.net_name, self.dataset_name, self.id)

    @property
    def prefix(self):
        return f"{self.net_name}_e{self.EPOCHS}_ds{self.dataset_name}_id{self.id}"

    @property
    def fig_dir(self):
        return os.path.join(self.root, f"fig_{self.prefix}")

    @property
    def history_plot_path(self):
        return os.path.join(self.fig_dir, "{}_checkpoint.npy".format(self.prefix))


class Trainer(object):
    '''Class to train a model '''

    def __init__(self, model, dataset, **kwargs):
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

    def update_lr(self, lr):
        if not self.lr == lr or self.optimizer is None:
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
            suffix = 'loss batch ' + str(loss_batch)
            if progress_bar:
                printProgressBar(i + 1, L, prefix='Train Epoch:', suffix='Complete, ' + suffix, length=50)
            else:
                prtg = " ({0:3.2f}%) ".format(100.0 * (i + 1) / L)
                if (i + 1) % 10 == 0:
                    print('Training Epoch: in batch ', i + 1, ' out of ', L, prtg, 'status: ' + suffix)
            loss.append(loss_batch)
        return loss

    def save_model(self, path):
        self.model.eval()
        torch.save(self.model.state_dict(), path)

    def load_model(self, model, path):
        self.model = model
        if os.path.exists(path):
            self.model.load_state_dict(torch.load(path))
        else:
            print('Warning: there is no file :', path)

    def get_range(self, EPOCHS):
        return range(self._epoch, EPOCHS)

    def load_checkpoint(self, root, prefix, eval_logging):
        files = get_npy_files(path=root)
        checkpoint_file = list(filter(lambda x: x.endswith('checkpoint.npy') and x.startswith(prefix), files))

        # finds the closest largest checkpoint file
        def argmax(iterable):
            return max(enumerate(iterable), key=lambda x: x[1])[0]

        if len(checkpoint_file) > 1:
            e_list_strs = [re.search(r'_e(.*?)_lr', s).group(1) for s in checkpoint_file]
            # attempt to parse to int
            e_list_ints = []
            for ee in e_list_strs:
                if ee.isdigit():
                    e_list_ints.append(int(ee))
                else:
                    raise ValueError(
                        'Something went wrong while extracting epoch list in your checkpoints {} with prefix {}'.format(
                            checkpoint_file, prefix))

            checkpoint_file = [checkpoint_file[argmax(e_list_ints)]]
            prefix = checkpoint_file[0].split("_checkpoint.")[0]
        if len(checkpoint_file) == 0:  # this means that the epoch could be higher than the one in the checkpoint files
            target_epoch = re.search(r'_e(.*?)_ds', prefix).group(1)
            prefix_flexible = prefix.replace("".join(["_e", target_epoch, "_ds"]), r'_e(.*?)_ds')
            checkpoint_file = [f for f in files if bool(re.search(prefix_flexible, f)) and f.endswith("checkpoint.npy")]
            if len(checkpoint_file) > 1:
                e_list_strs = [re.search(r'_e(.*?)_ds', s).group(1) for s in checkpoint_file]
                # attempt to parse to int
                e_list_ints = []
                for ee in e_list_strs:
                    if ee.isdigit():
                        e_list_ints.append(int(ee))
                    else:
                        raise ValueError(
                            'Something went wrong while extracting epoch list in your checkpoints {} with prefix {}'.format(
                                checkpoint_file, prefix))
                checkpoint_file = [checkpoint_file[argmax(e_list_ints)]]
                prefix = checkpoint_file[0].split("_checkpoint.")[0]
            elif len(checkpoint_file) == 1:
                prefix = checkpoint_file[0].split("_checkpoint.")[0]
            else:
                print('No checkpoint file was found for given prefix: ', prefix)

        # presets the epochs
        if len(checkpoint_file) > 0:
            d1 = np.load(os.path.join(root, checkpoint_file[0]), allow_pickle=True)
            self._epoch = d1.item().get('e')
            best_metric = d1.item().get('best_metric')
            monitor_metric = d1.item().get('monitor_metric')
            assert monitor_metric == eval_logging.monitor_metric, \
                "Cannot load checkpoint if monitor metric is not the same. " \
                "Expected {} found {}".format(eval_logging.monitor_metric, monitor_metric)
            print('loaded checkpoint: ', self._epoch, 'best metric: ', best_metric, "monitor_metric: ", monitor_metric)
        else:
            best_metric = None
        # presets the best metric
        eval_logging.best_metric = best_metric

        # loads the loss_log_per_iteration and metrics logs
        lossall_file = list(filter(lambda x: x.endswith('lossall.npy') and x.startswith(prefix), files))
        measurements_file = list(filter(lambda x: x.endswith('measurements.npy') and x.startswith(prefix), files))
        assert len(lossall_file) in [0, 1], 'something failed while searching for the lossall file. ' \
                                            'Found: %s' % str(lossall_file)
        assert len(measurements_file) in [0, 1], 'something failed while searching for the measurements file. ' \
                                                 'Found: %s' % str(measurements_file)
        if len(lossall_file) == 0 or len(measurements_file) == 0:
            print("loss files and measurements were not found")
        else:
            print('Loading checkpoint')
            loss_all = np.load(os.path.join(root, lossall_file[0]))
            eval_logging.update_loss_log(loss_all.tolist())
            measurements = np.load(os.path.join(root, measurements_file[0]), allow_pickle=True)
            measurements = measurements.item()  # convert to dictionary
            eval_logging.reset_measurements(measurements)

    def save_checkpoint(self, training_path, lr, e, EPOCHS, fig_dir, eval_logging, upload=False):
        check_point = {'lr': lr, 'e': e, 'E': EPOCHS, 'best_metric': eval_logging.best_metric,
                       'monitor_metric': eval_logging.monitor_metric}
        print('Saved checkpoint ', e, '/', EPOCHS)
        np.save(training_path.checkpoint_path, check_point)
        loss_per_iter = eval_logging.get_loss_per_iter()
        np.save(training_path.lossall_path, loss_per_iter)
        measurements = eval_logging.get_measurements()
        np.save(training_path.measurements_path, measurements)
        if not len(loss_per_iter) == 0:
            fig = plt.figure(figsize=(15, 10))

            plt.subplot(3, 1, 1)
            plt.plot(loss_per_iter)
            plt.xlabel('iterations')
            plt.ylabel('loss')
            plt.title('Loss history per iteration')

            plt.subplot(3, 1, 2)
            plt.plot(measurements["train_loss"], label="train loss")
            plt.plot(measurements["val_loss"], label="val loss")
            plt.xlabel('epochs')
            plt.ylabel('loss')
            plt.title('Loss history avg per epoch')
            plt.legend()

            plt.subplot(3, 1, 3)
            for target, mea in measurements.items():
                if target not in ["train_loss", "val_loss"]:
                    plt.plot(measurements[target], label=target)
            plt.xlabel('epochs')
            plt.ylabel('metrics')
            plt.title('Evaluation metrics')
            plt.legend()
            savefigs(fig_name=training_path.history_plot_path, fig_dir=fig_dir, fig=fig)

        if upload:
            print('Uploading training')
            upload_training(prefix_model=training_path.prefix_model, prefix=training_path.prefix, EPOCHS=EPOCHS, lr=lr)


class KTrainer(Trainer):
    def __init__(self, model, dataset, **kwargs):
        '''
        Keras trainer
        :param model:
        :param dataset:
        :param kwargs:
        '''
        super(KTrainer, self).__init__(model=model, dataset=dataset, **kwargs)
        self._model_compiled = False

    def update_lr(self, lr):
        if not self.lr == lr or self.optimizer is None:
            self.optimizer = keras.optimizers.SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)
            self.lr = lr

    def save_model(self, path):
        path = path.replace(".pth", ".dat")
        self.model.save(filename=path)

    def load_model(self, model, path):
        self.model = model
        path = path.replace(".pth", ".dat")

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
        dim = 2  # TODO:Hard-Coded
        Y = np.transpose(Y, axes=[0, dim + 1] + list(range(1, dim + 1)))
        B = self.dataset._batch_size
        history = self.model.fit(x=X, y=Y, epochs=1, batch_size=B, verbose=False)
        return history.history['loss'][0]

    def train_epoch(self, lr=0.01, progress_bar=True):
        if not self._model_compiled:
            self.update_lr(lr=lr)
            self.model.compile(optimizer=self.optimizer,
                               metrics=[])  # compile the network (supports keras compile parameters)
            self._model_compiled = True

        return super(KTrainer, self).train_epoch(lr=lr, progress_bar=progress_bar)

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

    def save_checkpoint(self, prefix, prefix_model, lr, e, EPOCHS, fig_dir, eval_logging, upload=False):
        super(KTrainer, self).save_checkpoint(prefix=prefix, prefix_model=prefix_model,
                                              lr=lr, e=e,
                                              EPOCHS=EPOCHS, fig_dir=fig_dir, eval_logging=eval_logging, upload=False)
        if upload:
            print('Uploading training')
            upload_training(prefix_model=prefix_model, prefix=prefix, EPOCHS=EPOCHS, lr=lr, h5format=True)
    # def compile(self):
    #     self.update_lr(self.lr)
    #     self.model.compile(optimizer=self.optimizer,
    #                        metrics=[])  # compile the network (supports keras compile parameters)
