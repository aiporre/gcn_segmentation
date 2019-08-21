import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import math
import random
import numpy as np

from .dataset import Datasets, Dataset


class MNIST(Datasets):
    def __init__(self,  data_dir='data/M2NIST', val_size=5000, background=False):
        mnist = input_data.read_data_sets(
            data_dir, one_hot=True, validation_size=val_size)

        images = self._preprocess_images(mnist.train.images[0:8000], background=background)
        labels = self._preprocess_labels(mnist.train.images[0:8000])
        train = Dataset(images, labels)

        images = self._preprocess_images(mnist.validation.images[0:2000], background=background)
        labels = self._preprocess_labels(mnist.validation.images[0:2000])
        val = Dataset(images, labels)

        images = self._preprocess_images(mnist.test.images[8000:10000], background=background)
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

    def _preprocess_images(self, images, background=True):
        images = np.reshape(images, (-1, self.height, self.width,
                            self.num_channels)).transpose(0, 3, 1, 2)
        if background:
            samples = images.shape[0]
            patterns = np.zeros_like(images)
            np.random.seed(0)
            patterns_list = [get_pattern() for _ in range(100)]
            for i in range(samples):
                a = patterns_list[np.random.randint(0,len(patterns_list))].copy()
                image = images[i,0,:,:].reshape(a.shape)
                a[image>0.3] = 0
                patterns[i,0,:,:] = a

        return images +patterns

    def _preprocess_labels(self, images):
        threshold = 0.1
        images = np.reshape(images, (-1, self.height, self.width,
                            self.num_channels))
        masks = (images > threshold).astype(np.float)
        return masks.squeeze()
def get_pattern():


    imgx = 28;
    imgy = 28

    pixels = np.zeros((imgx, imgy))

    f = random.random()*40+10  # frequency
    p = random.random()*math.pi  # phase
    n = random.randint(10, 20)  # of rotations
    # print(f, p, n)

    for ky in range(imgy):
        y = float(ky)/(imgy-1)*4*math.pi-2*math.pi
        for kx in range(imgx):
            x = float(kx)/(imgx-1)*4*math.pi-2*math.pi
            z = 0.0
            for i in range(n):
                r = math.hypot(x, y)
                a = math.atan2(y, x)+i*math.pi*2.0/n
                z += math.cos(r*math.sin(a)*f+p)
            c = int(round(255*z/n))
            pixels[kx, ky] = c/255  # grayscale
    return pixels


# def plot_example(dataset):
#     import matplotlib.pyplot as plt
#
#     images = dataset.train._images
#     mnist = input_data.read_data_sets(
#         'data/M2NIST', one_hot=True, validation_size=5000)
#     images_org = mnist.train.images[0:8000].reshape((-1, 28, 28, 1)).transpose(0, 3, 1, 2)
#     labels = dataset.train._labels
#     plt.subplot(131)
#     plt.imshow(images_org[1, :, :, :].squeeze(), cmap='gray')
#     plt.axis('off')
#
#     plt.subplot(132)
#     plt.imshow(images[1, :, :, :].squeeze(),cmap='gray')
#     plt.axis('off')
#
#     plt.subplot(133)
#     plt.imshow(labels[1, :, :].squeeze(), cmap='gray')
#     plt.axis('off')
#
#     plt.show()



