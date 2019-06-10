import numpy as np
from .dataset import Datasets, Dataset
from .download import maybe_download_and_extract
import os
URL = 'https://transfer.sh/GeGJ8/m2nist.zip'
class M2NIST(Datasets):
    def __init__(self, data_dir='data/M2NIST', batch_size=32, test_rate = 0.2, validation=False):
        maybe_download_and_extract(URL,data_dir=data_dir)

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.test_rate = test_rate
        self.validation = validation

        print('reading data')
        segmented = np.load(os.path.join(self.data_dir,'segmented.npy'))
        self._N_samples, self._HEIGHT, self._WIDTH, self._N_CLASSES = segmented.shape
        self._combined = np.load("./data/M2NIST/combined.npy").reshape((-1, self._HEIGHT, self._WIDTH, 1))/255
        self._mask = 1-segmented[:, :, :, -1]

        images, labels = self._load_train()
        train = Dataset(images, labels)
        images, labels = self._load_test()
        test = Dataset(images, labels)
        validation = Dataset(images, labels)
        super(M2NIST, self).__init__(train=train, test=test, val=validation)
    @property
    def classes(self):
        return [
            'background','foreground'
        ]

    @property
    def width(self):
        return 32

    @property
    def height(self):
        return 32

    @property
    def num_channels(self):
        return 3

    def _load_train(self):
        #TODO: shuffle data before split
        train_index= np.ceil(self._N_samples*(1-self.test_rate)).astype(int)
        images = self._combined[0:train_index].transpose(0,3,1,2)
        labels = self._mask[0:train_index]
        return images, labels

    def _load_test(self):
        train_index= np.ceil(self._N_samples*(1-self.test_rate)).astype(int)
        images = self._combined[train_index:self._N_samples].transpose(0,3,1,2)
        labels = self._mask[train_index:self._N_samples]
        return images, labels