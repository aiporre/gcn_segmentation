import os
from .dataset import Datasets, Dataset, GraphDataset
from .mnist import get_pattern
import torch
from torch_geometric.data import (Dataset, Data)
import torch_geometric.transforms as T
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from lib.graph import grid_tensor


class GMNIST(Datasets):
    def __init__(self, data_dir='data/GMNIST', batch_size=32, test_rate=0.2, validation=False):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.test_rate = test_rate
        self.validation = validation

        train_dataset = _GMNIST(self.data_dir, True, transform=T.Cartesian())
        test_dataset = _GMNIST(self.data_dir, False, transform=T.Cartesian())

        train = GraphDataset(train_dataset, batch_size=self.batch_size, shuffle=True)
        test = GraphDataset(test_dataset, batch_size=self.batch_size, shuffle=False)

        super(GMNIST, self).__init__(train=train, test=test, val=test)



class _GMNIST(Dataset):

    def __init__(self,
                 root,
                 train=True,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None):
        self.offset = 0 if train else 8000
        self.train = train
        super(_GMNIST, self).__init__(root, transform, pre_transform,
                                               pre_filter)

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        if self.train:
            return ['data_{}.pt'.format(i) for i in range(8000)]
        else:
            return ['data_{}.pt'.format(i) for i in range(8000,10000)]

    def download(self):
        pass

    def __len__(self):
        return len(self.processed_file_names)

    def process(self):
        i = self.offset
        mnist = input_data.read_data_sets(
            self.raw_dir, one_hot=True, validation_size=0)
        images = mnist.train.images[0:8000] if self.train else mnist.test.images[8000:10000]

        samples = images.shape[0]
        patterns = np.zeros_like(images)
        patterns_list = [get_pattern() for _ in range(100)]
        for i in range(samples):
            a = patterns_list[np.random.randint(0, len(patterns_list))]
            image = images[i, :, :].reshape(a.shape)
            a[image > 0.3] = 0
            patterns[i, 0, :, :] = a
        images = images + patterns

        masks = (images > 0.1).astype(np.float)

        for image, mask in zip(images, masks):
            # Read data from `raw_path`.
            grid = grid_tensor((28, 28), connectivity=4)
            grid.x = torch.tensor(image.reshape(28 * 28)).float()
            grid.y = torch.tensor([mask.reshape(28 * 28)]).float()
            data = grid

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            torch.save(data, os.path.join(self.processed_dir, 'data_{}.pt'.format(i)))
            i += 1

    def get(self, idx):
        idx += self.offset
        data = torch.load(os.path.join(self.processed_dir, 'data_{}.pt'.format(idx)))
        return data
