import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

from .dataset import Datasets, Dataset


class MNIST(Datasets):
    def __init__(self,  data_dir='data/M2NIST', val_size=5000):
        mnist = input_data.read_data_sets(
            data_dir, one_hot=True, validation_size=val_size)

        images = self._preprocess_images(mnist.train.images[0:8000])
        labels = self._preprocess_labels(mnist.train.images[0:8000])
        train = Dataset(images, labels)

        images = self._preprocess_images(mnist.validation.images[0:2000])
        labels = self._preprocess_labels(mnist.validation.images[0:2000])
        val = Dataset(images, labels)

        images = self._preprocess_images(mnist.test.images[8000:10000])
        labels = self._preprocess_labels(mnist.test.images[8000:10000])
        test = Dataset(images, labels)

        super(MNIST, self).__init__(train, val, test)

    @property
    def classes(self):
        return ['foreground', 'background']

    @property
    def width(self):
        return 28

    @property
    def height(self):
        return 28

    @property
    def num_channels(self):
        return 1

    def _preprocess_images(self, images):
        return np.reshape(images, (-1, self.height, self.width,
                                   self.num_channels)).transpose(0,3,1,2)

    def _preprocess_labels(self, images):
        threshold = 0.1
        images = np.reshape(images, (-1, self.height, self.width,
                            self.num_channels))
        masks = (images > threshold).astype(np.float)
        return masks.squeeze()
