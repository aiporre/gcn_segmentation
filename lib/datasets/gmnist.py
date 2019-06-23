import os
from .dataset import Datasets, Dataset, GraphDataset
import torch
from torch_geometric.data import (InMemoryDataset, Data, download_url,
                                  extract_tar, DataLoader)
import torch_geometric.transforms as T
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



class _GMNIST(InMemoryDataset):

    url = 'http://ls7-www.cs.uni-dortmund.de/cvpr_geometric_dl/' \
          'mnist_superpixels.tar.gz'

    def __init__(self,
                 root,
                 train=True,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None):
        super(GMNIST, self).__init__(root, transform, pre_transform,
                                               pre_filter)
        path = self.processed_paths[0] if train else self.processed_paths[1]
        self.data, self.slices = torch.load(path)

    @property
    def raw_file_names(self):
        return ['training.pt', 'test.pt']

    @property
    def processed_file_names(self):
        return ['training.pt', 'test.pt']

    def download(self):
        path = download_url(self.url, self.raw_dir)
        extract_tar(path, self.raw_dir, mode='r')
        os.unlink(path)

    def process(self):
        for raw_path, path in zip(self.raw_paths, self.processed_paths):
            x, edge_index, edge_slice, pos, y = torch.load(raw_path)
            edge_index, y = edge_index.to(torch.long), y.to(torch.long)
            m, n = y.size(0), 75
            x, pos = x.view(m * n, 1), pos.view(m * n, 2)
            node_slice = torch.arange(0, (m + 1) * n, step=n, dtype=torch.long)
            graph_slice = torch.arange(m + 1, dtype=torch.long)
            self.data = Data(x=x, edge_index=edge_index, y=y, pos=pos)
            self.slices = {
                'x': node_slice,
                'edge_index': edge_slice,
                'y': graph_slice,
                'pos': node_slice
            }

            if self.pre_filter is not None:
                data_list = [self.get(idx) for idx in range(len(self))]
                data_list = [d for d in data_list if self.pre_filter(d)]
                self.data, self.slices = self.collate(data_list)

            if self.pre_transform is not None:
                data_list = [self.get(idx) for idx in range(len(self))]
                data_list = [self.pre_transform(data) for data in data_list]
                self.data, self.slices = self.collate(data_list)

            torch.save((self.data, self.slices), path)

# if __name__=='__main__':
# dataset = GMNIST('./')
