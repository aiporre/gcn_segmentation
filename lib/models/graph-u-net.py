import os.path as osp

import torch
import torch.nn.functional as F
from torch_geometric.datasets import MNISTSuperpixels
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
from torch_geometric.nn import SplineConv, voxel_grid, max_pool, max_pool_x

class Net(torch.nn.Module):
    def __init__(self,num_features=1, num_classes=2):
        super(Net, self).__init__()
        self.conv1 = SplineConv(num_features, 32, dim=2, kernel_size=5)
        self.conv2 = SplineConv(32, 64, dim=2, kernel_size=5)
        self.conv3 = SplineConv(64, 64, dim=2, kernel_size=5)
        self.fc1 = torch.nn.Linear(4 * 64, 128)
        self.fc2 = torch.nn.Linear(128, num_classes)
        self.transform = T.Cartesian(cat=False)

    def forward(self, data):
        data.x = F.elu(self.conv1(data.x, data.edge_index, data.edge_attr))
        cluster = voxel_grid(data.pos, data.batch, size=5, start=0, end=28)
        data = max_pool(cluster, data, transform=self.transform)

        data.x = F.elu(self.conv2(data.x, data.edge_index, data.edge_attr))
        cluster = voxel_grid(data.pos, data.batch, size=7, start=0, end=28)
        data = max_pool(cluster, data, transform=self.transform)

        data.x = F.elu(self.conv3(data.x, data.edge_index, data.edge_attr))
        cluster = voxel_grid(data.pos, data.batch, size=14, start=0, end=27.99)
        data = max_pool(cluster, data, transform=self.transform)

        # x = max_pool_x(cluster, data.x, data.batch, size=4)

        # x = x.view(-1, self.fc1.weight.size(1))
        # x = F.elu(self.fc1(x))
        # x = F.dropout(x, training=self.training)
        # x = self.fc2(x)
        return data # F.log_softmax(x, dim=1)