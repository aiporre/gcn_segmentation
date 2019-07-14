import torch
import torch_geometric.transforms as T
import torch.nn.functional as F
from torch_geometric.utils import normalized_cut
from torch_geometric.data import Data, Batch
from torch_geometric.nn import graclus, max_pool
from torch_geometric.nn import SplineConv
from lib.utils import print_debug



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
    #     weights = weights.to(device)
    # print('======> cluster:', cluster.size())

    if batch is not None:
        data = Batch(x=source.x[cluster], edge_index=edge_index, pos=pos, batch=batch)
    else:
        data = Data(x=source.x[cluster], edge_index=edge_index, pos=pos)
    #     print('reconstructed data.x.shape' , data.x.shape)

    if transform is not None:
        data = transform(data)
    return data


def consecutive_cluster(src):
    unique, inv = torch.unique(src, sorted=True, return_inverse=True)
    perm = torch.arange(inv.size(0), dtype=inv.dtype, device=inv.device)
    perm = inv.new_empty(unique.size(0)).scatter_(0, inv, perm)
    return inv, perm


def bweights(source, cluster):
    cluster, perm = consecutive_cluster(cluster)
    cluster_codes, inversion = cluster.unique(return_inverse=True)

    if source.x.dim() == 1:
        centroids = torch.stack([source.x[cluster == i].mean() for i in cluster_codes]).requires_grad_(False)
#             cluster_count = torch.stack([source.x[cluster==i].count() for i in cluster_codes])
    else:
        #   the max dim is 2
        centroids = torch.stack([source.x[cluster == i].mean(dim=0) for i in cluster_codes]).requires_grad_(False)
#         cluster_count = torch.stack([source.x[cluster==i].count() for i in cluster_codes])

# alternative 1
#     aux = torch.empty(cluster.size(0)).scatter_(0,perm,centroids)
#     weights = data.x/aux
# alternative 2
#     weights = source.x.clone()
#     for i in range(len(weights)):
#         weights[i] = weights[i]/centroids[cluster[i]]
# alternative 3
    weights = source.x/centroids[inversion]

    weights[weights != weights] = 0
    return weights

def recover_grid_barycentric(source, weights, pos, edge_index, cluster, batch=None, transform=None):
    device = cluster.device
    cluster, perm = consecutive_cluster(cluster)
    print_debug('======> weights:', weights.size())
    print_debug('======> cluster:', cluster.size())
    weights = weights.unsqueeze(0) if weights.dim() == 1 else weights
    Q = torch.zeros((source.num_nodes, cluster.shape[0])).to(device).scatter_(0, cluster.unsqueeze(0), weights)

    if source.x.dim() == 1:
        x = source.x.unsqueeze(0).mm(Q).squeeze()
    else:
        # the max dimension is 2
        x = Q.transpose(0, 1).mm(source.x)
        print('x.shape: ', x.shape)
    if batch is not None:
        data = Batch(x=x, edge_index=edge_index, pos=pos, batch=batch)
    else:
        data = Data(x=x, edge_index=edge_index, pos=pos)

    if transform is not None:
        data = transform(data)
    return data

class GFCNA(torch.nn.Module):
    def __init__(self):
        super(GFCNA, self).__init__()
        self.conv1a = SplineConv(1, 32, dim=2, kernel_size=5)
        self.conv1b = SplineConv(32, 32, dim=2, kernel_size=5)

        self.conv2a = SplineConv(32, 64, dim=2, kernel_size=3)
        self.conv2b = SplineConv(64, 64, dim=2, kernel_size=3)

        self.conv3a = SplineConv(64, 64, dim=2, kernel_size=3)
        self.conv3b = SplineConv(64, 64, dim=2, kernel_size=3)

        self.score_fr = SplineConv(64, 1, dim=2, kernel_size=3)
        self.score_pool1 = SplineConv(32, 1, dim=2, kernel_size=3)



    def forward(self, data):
        # (1/32,V_0)
        data.x = F.elu(self.conv1a(data.x, data.edge_index, data.edge_attr))
        data.x = F.elu(self.conv1b(data.x, data.edge_index, data.edge_attr))
        weight = normalized_cut_2d(data.edge_index, data.pos)
        cluster1 = graclus(data.edge_index, weight, data.x.size(0))
        pos1 = data.pos
        edge_index1 = data.edge_index
        batch1 = data.batch if hasattr(data,'batch') else None
        # weights1, centroids1 = bweights(data, cluster1)
        data = max_pool(cluster1, data, transform=T.Cartesian(cat=False))
        pool1 = data.clone()

        # (32/64,V_1)
        data.x = F.elu(self.conv2a(data.x, data.edge_index, data.edge_attr))
        data.x = F.elu(self.conv2b(data.x, data.edge_index, data.edge_attr))
        weight = normalized_cut_2d(data.edge_index, data.pos)
        cluster2 = graclus(data.edge_index, weight, data.x.size(0))
        pos2 = data.pos
        edge_index2 = data.edge_index
        batch2 = data.batch if hasattr(data,'batch') else None
        # weights2, centroids2 = bweights(data, cluster2)
        data = max_pool(cluster2, data, transform=T.Cartesian(cat=False))


        # 64/64
        data.x = F.elu(self.conv3a(data.x, data.edge_index, data.edge_attr))
        data.x = F.elu(self.conv3b(data.x, data.edge_index, data.edge_attr))

        # upsample
        # data = recover_grid_barycentric(data, weights=weights2, pos=pos2, edge_index=edge_index2, cluster=cluster2,
        #                                  batch=batch2, transform=None)
        data.x = F.elu(self.score_fr(data.x, data.edge_index, data.edge_attr))
        data = recover_grid(data, pos2, edge_index2, cluster2, batch=batch2, transform=T.Cartesian(cat=False))
        pool1.x = F.elu(self.score_pool1(pool1.x, pool1.edge_index, pool1.edge_attr))

        # data = recover_grid_barycentric(data, weights=weights1, pos=pos1, edge_index=edge_index1, cluster=cluster1,
        #                                  batch=batch1, transform=None)
        data.x = data.x+pool1.x
        data = recover_grid(data, pos1, edge_index1, cluster1, batch=batch1, transform=T.Cartesian(cat=False))

        # TODO handle contract on trainer and  evaluator

        x, batch = data.x, torch.zeros(data.num_nodes)

        return F.sigmoid(x)

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
        batch1 = data.batch if hasattr(data,'batch') else None
        weights1 = bweights(data, cluster1)
        data = max_pool(cluster1, data, transform=T.Cartesian(cat=False))

        data.x = F.elu(self.conv2(data.x, data.edge_index, data.edge_attr))
        weight = normalized_cut_2d(data.edge_index, data.pos)
        cluster2 = graclus(data.edge_index, weight, data.x.size(0))
        pos2 = data.pos
        edge_index2 = data.edge_index
        batch2 = data.batch if hasattr(data,'batch') else None
        weights2 = bweights(data, cluster2)
        data = max_pool(cluster2, data, transform=T.Cartesian(cat=False))

        # upsample
        print('x tensor s nana:' , torch.isnan(data.x).any())
        data = recover_grid_barycentric(data, weights=weights2, pos=pos2, edge_index=edge_index2, cluster=cluster2,
                                         batch=batch2, transform=T.Cartesian(cat=False))
        # data = recover_grid(data, pos2, edge_index2, cluster2, batch=batch2, transform=T.Cartesian(cat=False))
        print('x tensor s nana:' , torch.isnan(data.x).any())
        data.x = F.elu(self.conv3(data.x, data.edge_index, data.edge_attr))

        print('x tensor s nana:' , torch.isnan(data.x).any())
        data = recover_grid_barycentric(data, weights=weights1, pos=pos1, edge_index=edge_index1, cluster=cluster1,
                                         batch=batch1, transform=T.Cartesian(cat=False))
        # data = recover_grid(data, pos1, edge_index1, cluster1, batch=batch1, transform=T.Cartesian(cat=False))
        data.x = F.elu(self.conv4(data.x, data.edge_index, data.edge_attr))
        print('x tensor s nana:' , torch.isnan(data.x).any())

        # TODO handle contract on trainer and  evaluator

        x = data.x
        print('x is infinite', torch.isinf(x).any())
        print('weights1 is nana:', torch.isnan(weights1).any())
        print('x tensor s nana:' , torch.isnan(x).any())
        print('weights2 is nan:', torch.isnan(weights2).any())

        return F.sigmoid(x)
