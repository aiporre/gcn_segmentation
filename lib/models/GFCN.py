import torch
import torch_geometric.transforms as T
import torch.nn.functional as F
from torch_geometric.utils import normalized_cut
from torch_geometric.data import Data, Batch
from torch_geometric.nn import graclus, max_pool
from torch_geometric.nn import SplineConv



def normalized_cut_2d(edge_index, pos):
    row, col = edge_index
    edge_attr = torch.norm(pos[row] - pos[col], p=2, dim=1)
    return normalized_cut(edge_index, edge_attr, num_nodes=pos.size(0))

def cluster_grid(grid):
    data = grid
    # data.x = conv1(data.x, data.edge_index, data.edge_attr)
    weight = normalized_cut_2d(data.edge_index, data.pos)
    cluster = graclus(data.edge_index, weight, data.x.size(0))
    data.edge_attr = None
    data.batch = None
    data = max_pool(cluster, data, transform=T.Cartesian(cat=False))
    return data, cluster

def consecutive_cluster(src):
    unique, inv = torch.unique(src, sorted=True, return_inverse=True)
    perm = torch.arange(inv.size(0), dtype=inv.dtype, device=inv.device)
    perm = inv.new_empty(unique.size(0)).scatter_(0, inv, perm)
    return inv, perm


def recover_grid(source, pos, edge_index, cluster, batch=None, transform=None):
    device = cluster.device
    cluster, perm = consecutive_cluster(cluster)
    weights = torch.ones((1, len(cluster))).to(device)
    Q = torch.zeros((source.num_nodes, cluster.shape[0])).to(device).scatter_(0, cluster.unsqueeze(0), weights)

    if source.x.dim() == 1:
        x = source.x.unsqueeze(0).mm(Q).squeeze()
    else:
        # the max dimension is 2
        x = Q.transpose(0, 1).mm(source.x)
    if batch is not None:
        data = Batch(x=x, edge_index=edge_index, pos=pos, batch=batch)
    else:
        data = Data(x=x, edge_index=edge_index, pos=pos)

    if transform is not None:
        data = transform(data)
    return data


class GFCN(torch.nn.Module):
    def __init__(self):
        super(GFCN, self).__init__()
        self.conv1 = SplineConv(1, 32, dim=2, kernel_size=5)
        self.conv2 = SplineConv(32, 64, dim=2, kernel_size=5)
        self.conv3 = SplineConv(64, 32, dim=2, kernel_size=5)
        self.conv4 = SplineConv(32, 1, dim=2, kernel_size=5)

    def forward(self, data):
        data.x = F.elu(self.conv1(data.x, data.edge_index, data.edge_attr))
        weight = normalized_cut_2d(data.edge_index, data.pos)
        cluster1 = graclus(data.edge_index, weight, data.x.size(0))
        pos1 = data.pos
        edge_index1 = data.edge_index
        data = max_pool(cluster1, data, transform=T.Cartesian(cat=False))

        data.x = F.elu(self.conv2(data.x, data.edge_index, data.edge_attr))
        weight = normalized_cut_2d(data.edge_index, data.pos)
        pos2 = data.pos
        edge_index2 = data.edge_index
        cluster2 = graclus(data.edge_index, weight, data.x.size(0))
        data = max_pool(cluster2, data, transform=T.Cartesian(cat=False))

        # upsample
        data = recover_grid(data, pos2, edge_index2, cluster2, transform=T.Cartesian(cat=False))
        data.x = F.elu(self.conv3(data.x, data.edge_index, data.edge_attr))

        data = recover_grid(data, pos1, edge_index1, cluster1, transform=T.Cartesian(cat=False))
        data.x = F.elu(self.conv4(data.x, data.edge_index, data.edge_attr))

        x, batch = data.x, torch.zeros(data.num_nodes)

        return F.sigmoid(x)