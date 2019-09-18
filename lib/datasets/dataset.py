import numpy as np
try:
    from torch_geometric.data import DataLoader
except ImportError:
    print('Warning: Error while trying to import torch_geometric.data.Dataloader')

class Datasets(object):
    def __init__(self, train, val, test):
        self.train = train
        self.val = val
        self.test = test

    @property
    def classes(self):
        raise NotImplementedError

    @property
    def width(self):
        raise NotImplementedError

    @property
    def height(self):
        raise NotImplementedError

    @property
    def num_channels(self):
        raise NotImplementedError

    @property
    def num_classes(self):
        return len(self.classes)

    def classnames(self, label):
        idx = np.where(label == 1)[0]
        return [self.classes[i] for i in idx]


class GraphDataset(object):
    def __init__(self,dataset, batch_size=1, shuffle=False ):
        self.shuffle = shuffle
        self._dataset = dataset
        self._batch_size = batch_size
        self._dataloader = DataLoader(dataset, batch_size=self._batch_size, shuffle=self.shuffle)
        self._dataloader_iter = self._dataloader.__iter__()
        self._index_in_epoch = 0


    def enforce_batch(self, batch_size):
        self._batch_size = batch_size
        self._dataloader = DataLoader(self._dataset, batch_size=self._batch_size, shuffle=self.shuffle)
        self._dataloader_iter = self._dataloader.__iter__()


    @property
    def num_batches(self):
        return int(len(self._dataset) / self._batch_size)

    @property
    def num_examples(self):
        return len(self._dataset)

    def __len__(self):
        return self.num_examples

    def __getitem__(self, idx):
        if not isinstance(idx, int):
            raise TypeError('dataset indices must be integers, not '+ str(type(idx)))
        if idx > self.__len__() or idx < -self.__len__():
            raise IndexError('dataset index out of range')
        if idx < 0:
            idx = self.__len__()+idx
        return self._dataset[idx]

    def __iter__(self):
        i = 0
        while i < self.__len__():
            yield self._dataset[i]
            i +=1

    def batches(self):
        for _ in range(self.num_batches):
            yield self.next_batch(self._batch_size, shuffle=self.shuffle)

    def next_batch(self, batch_size, shuffle=True):
        if not batch_size == self._batch_size:
            self.shuffle =shuffle
            self.enforce_batch(batch_size)
            self._index_in_epoch = 0

        # restarting dataloader iterable when
        start = self._index_in_epoch
        if start + self._batch_size > self.num_examples:
            self._dataloader_iter = self._dataloader.__iter__()
            self._index_in_epoch = 0
        # new sample
        try:
            data = self._dataloader_iter.__next__()
        except StopIteration:
            self._dataloader_iter = self._dataloader.__iter__()
            data = self._dataloader_iter.__next__()
        images, labels = data, data.y
        self._index_in_epoch += self._batch_size

        return images, labels

class Dataset(object):
    def __init__(self, images, labels, batch_size=1, shuffle=True):
        self.epochs_completed=0
        self._images = images
        self._labels = labels
        self._index_in_epoch = 0
        self._shuffle = shuffle
        self._batch_size = batch_size

    def enforce_batch(self, batch_size):
        self._batch_size = batch_size

    def get_images(self):
        return self._images

    def get_labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._labels.shape[0]

    @property
    def num_batches(self):
        K = self.num_examples % self._batch_size
        K1 = int(self.num_examples / self._batch_size)
        K1 = K1 if(K== 0) else K1+1
        return K1

    def _random_shuffle_examples(self):
        perm = np.arange(self.num_examples)
        np.random.shuffle(perm)
        self._images = self._images[perm]
        self._labels = self._labels[perm]


    def __len__(self):
        return self.num_examples

    def __getitem__(self, idx):
        if not isinstance(idx, int):
            raise TypeError('dataset indices must be integers, not ', type(idx))
        if idx > self.__len__() or idx < -self.__len__():
            raise IndexError('dataset index out of range')
        if idx < 0:
            idx = self.__len__()+idx
        return self._images[idx], self._labels[idx]

    def __iter__(self):
        i = 0
        while i< self.__len__():
            yield self._images[i], self._labels[i]
            i += 1
    def batches(self):
        for _ in range(self.num_batches):
            yield self.next_batch(batch_size=self._batch_size,shuffle=self._shuffle)

    def next_batch(self, batch_size, shuffle=True):
        if not batch_size == self._batch_size:
            self.enforce_batch(batch_size)

        start = self._index_in_epoch

        # Shuffle for the first epoch.
        if self.epochs_completed == 0 and start == 0 and shuffle:
            self._random_shuffle_examples()

        if start + batch_size > self.num_examples:
            # Finished epoch.
            self.epochs_completed += 1

            # Get the rest examples in this epoch.
            rest_num_examples = self.num_examples - start
            images_rest = self._images[start:self.num_examples]
            labels_rest = self._labels[start:self.num_examples]

            # Shuffle the examples.
            if shuffle:
                self._random_shuffle_examples()

            # Start next epoch.
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            images_new = self._images[start:end]
            labels_new = self._labels[start:end]

            labels = np.concatenate((labels_rest, labels_new), axis=0)
            images = np.concatenate((images_rest, images_new), axis=0)
        else:
            # Just slice the examples.
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            images = self._images[start:end]
            labels = self._labels[start:end]

        return images, labels
