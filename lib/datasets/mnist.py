import gzip

import numpy as np
from lib.datasets.download import maybe_download_and_extract
import math
import random
import numpy as np

from .dataset import Datasets, Dataset
DEFAULT_SOURCE_URL="http://yann.lecun.com/exdb/mnist/"

def _read32(bytestream):
  dt = np.dtype(np.uint32).newbyteorder('>')
  return np.frombuffer(bytestream.read(4), dtype=dt)[0]

def dense_to_one_hot(labels_dense, num_classes):
  """Convert class labels from scalars to one-hot vectors."""
  num_labels = labels_dense.shape[0]
  index_offset = np.arange(num_labels) * num_classes
  labels_one_hot = np.zeros((num_labels, num_classes))
  labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
  return labels_one_hot


def extract_labels(f, one_hot=False, num_classes=10):
  """Extract the labels into a 1D uint8 numpy array [index].
  Args:
    f: A file object that can be passed into a gzip reader.
    one_hot: Does one hot encoding for the result.
    num_classes: Number of classes for the one hot encoding.
  Returns:
    labels: a 1D uint8 numpy array.
  Raises:
    ValueError: If the bystream doesn't start with 2049.
  """
  print('Extracting', f.name)
  with gzip.GzipFile(fileobj=f) as bytestream:
    magic = _read32(bytestream)
    if magic != 2049:
      raise ValueError('Invalid magic number %d in MNIST label file: %s' %
                       (magic, f.name))
    num_items = _read32(bytestream)
    buf = bytestream.read(num_items)
    labels = np.frombuffer(buf, dtype=np.uint8)
    if one_hot:
      return dense_to_one_hot(labels, num_classes)
    return labels

def extract_images(f):
  """Extract the images into a 4D uint8 numpy array [index, y, x, depth].
  Args:
    f: A file object that can be passed into a gzip reader.
  Returns:
    data: A 4D uint8 numpy array [index, y, x, depth].
  Raises:
    ValueError: If the bytestream does not start with 2051.
  """
  print('Extracting', f.name)
  with gzip.GzipFile(fileobj=f) as bytestream:
    magic = _read32(bytestream)
    if magic != 2051:
      raise ValueError('Invalid magic number %d in MNIST image file: %s' %
                       (magic, f.name))
    num_images = _read32(bytestream)
    rows = _read32(bytestream)
    cols = _read32(bytestream)
    buf = bytestream.read(rows * cols * num_images)
    data = np.frombuffer(buf, dtype=np.uint8)
    data = data.reshape(num_images, rows, cols, 1)
    return data


def read_data_sets(train_dir,
                   one_hot=False,
                   dtype=np.float32,
                   reshape=True,
                   validation_size=5000,
                   seed=None,
                   source_url=DEFAULT_SOURCE_URL):

  if not source_url:  # empty string check
    source_url = DEFAULT_SOURCE_URL

  TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'
  TRAIN_LABELS = 'train-labels-idx1-ubyte.gz'
  TEST_IMAGES = 't10k-images-idx3-ubyte.gz'
  TEST_LABELS = 't10k-labels-idx1-ubyte.gz'

  local_file = maybe_download_and_extract(source_url + TRAIN_IMAGES, train_dir)
  with open(local_file, 'rb') as f:
      train_images = extract_images(f)
  local_file = maybe_download_and_extract(source_url  + TRAIN_LABELS, train_dir)

  with open(local_file, 'rb') as f:
    train_labels = extract_labels(f, one_hot=one_hot)


  local_file = maybe_download_and_extract(source_url  + TEST_IMAGES, train_dir)
  with open(local_file, 'rb') as f:
    test_images = extract_images(f)

  local_file = maybe_download_and_extract(source_url  + TEST_LABELS, train_dir)
  with open(local_file, 'rb') as f:
    test_labels = extract_labels(f, one_hot=one_hot)

  if not 0 <= validation_size <= len(train_images):
    raise ValueError('Validation size should be between 0 and {}. Received: {}.'
                     .format(len(train_images), validation_size))

  validation_images = train_images[:validation_size]
  validation_labels = train_labels[:validation_size]
  train_images = train_images[validation_size:]
  train_labels = train_labels[validation_size:]

  options = dict(dtype=dtype, reshape=reshape, seed=seed)

  train = MDataset(train_images, train_labels, **options)
  validation = MDataset(validation_images, validation_labels, **options)
  test = MDataset(test_images, test_labels, **options)

  return MDatasets(train=train, validation=validation, test=test)

class MDataset:
    def __init__(self, images, labels, options):
        self.images = images
        self.labels = labels
        self.options = options

class MDatasets:
    def __init__(self, train, validation, test):
        self.train = train
        self.validation = validation
        self.test = test

class MNIST(Datasets):
    def __init__(self,  data_dir='data/M2NIST', val_size=5000, background=True):
        mnist = read_data_sets(data_dir, one_hot=True, validation_size=val_size)

        images = self._preprocess_images(mnist.train.images[0:80], background=background)
        labels = self._preprocess_labels(mnist.train.images[0:80])
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

    def _preprocess_images(self, images, background):
        images = np.reshape(images, (-1, self.height, self.width,
                            self.num_channels)).transpose(0, 3, 1, 2)
        patterns = np.zeros_like(images)
        if background:
            samples = images.shape[0]
            np.random.seed(0)
            patterns_list = [get_pattern() for _ in range(100)]
            for i in range(samples):
                a = patterns_list[np.random.randint(0,len(patterns_list))].copy()
                image = images[i,0,:,:].reshape(a.shape)
                a[image>0.3] = 0
                patterns[i,0,:,:] = a

        return images + patterns

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



