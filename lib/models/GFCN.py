import torch
import torch_geometric.transforms as T
import torch.nn.functional as F
from torch_geometric.utils import normalized_cut, scatter_
from torch_geometric.data import Data, Batch
from torch_geometric.nn import graclus, max_pool, avg_pool, fps, radius, knn_interpolate

from torch_geometric.nn import SplineConv
from lib.utils import print_debug
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d as BN


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
    cluster, _ = consecutive_cluster(cluster)
    #     weights = weights.to(device)
    # print('======> cluster:', cluster.size())
    source.x = source.x[cluster]
    source.edge_index = edge_index
    source.pos = pos
    source.batch = batch
    if batch is not None:
        source.batch = batch
        # data = Batch(x=source.x[cluster], edge_index=edge_index, pos=pos, batch=batch)
    #else:
        #data = Data(x=source.x[cluster], edge_index=edge_index, pos=pos)
    #     print('reconstructed data.x.shape' , data.x.shape)

    if transform is not None:
        source = transform(source)
    return source


def consecutive_cluster(src):
    unique, inv = torch.unique(src, sorted=True, return_inverse=True)
    perm = torch.arange(inv.size(0), dtype=inv.dtype, device=inv.device)
    perm = inv.new_empty(unique.size(0)).scatter_(0, inv, perm)
    return inv, perm


def pweights(x, cluster):
    ''' Computes the percentage weights in the simplex formed by the cluster '''
    with torch.no_grad():
        cluster, perm = consecutive_cluster(cluster)
        g = scatter_('add', x, cluster)
        w = x/g[cluster]
        w[w != w] = 0
        return w



def bweights(source, cluster):
    cluster, perm = consecutive_cluster(cluster)
    cluster_codes, inversion = cluster.unique(return_inverse=True)

    if source.x.dim() == 1:
        centroids = torch.stack([source.x[cluster == i].mean() for i in cluster_codes]).requires_grad_(False)
    else:
        centroids = torch.stack([source.x[cluster == i].mean(dim=0) for i in cluster_codes]).requires_grad_(False)

    weights = source.x/centroids[inversion]

    weights[weights != weights] = 0
    return weights

def recover_grid_barycentric(source, weights, pos, edge_index, cluster, batch=None, transform=None):

    cluster, perm = consecutive_cluster(cluster)
    source.x = source.x.squeeze()
    source.x = source.x[cluster]*weights
    source.edge_index = edge_index
    source.pos = pos

    if batch is not None:
        source.batch = batch
    else:
        source.batch = batch
    if transform is not None:
        source = transform(source)
    return source

class GFCNB(torch.nn.Module):
    def __init__(self):
        super(GFCNB, self).__init__()
        self.conv1a = SplineConv(1, 32, dim=2, kernel_size=5)
        self.conv1b = SplineConv(32, 64, dim=2, kernel_size=5)
        self.conv1c = SplineConv(64, 64, dim=2, kernel_size=5)
        self.bn1 = torch.nn.BatchNorm1d(64)

        self.conv2a = SplineConv(64, 128, dim=2, kernel_size=3)
        self.conv2b = SplineConv(128, 128, dim=2, kernel_size=3)
        self.conv2c = SplineConv(128, 256, dim=2, kernel_size=3)

        self.conv3a = SplineConv(256, 256, dim=2, kernel_size=3)
        self.conv3b = SplineConv(256, 128, dim=2, kernel_size=3)

        self.conv4a = SplineConv(128, 64, dim=2, kernel_size=5)
        self.conv4b = SplineConv(64, 32, dim=2, kernel_size=5)

        self.convout = SplineConv(32, 1, dim=2, kernel_size=5)

    def forward(self, data):
        data.x = F.elu(self.conv1a(data.x, data.edge_index, data.edge_attr))
        data.x = F.elu(self.conv1b(data.x, data.edge_index, data.edge_attr))
        data.x = F.elu(self.conv1c(data.x, data.edge_index, data.edge_attr))
        data.x = self.bn1(data.x)
        weight = normalized_cut_2d(data.edge_index, data.pos)
        cluster1 = graclus(data.edge_index, weight, data.x.size(0))
        pos1 = data.pos
        edge_index1 = data.edge_index
        batch1 = data.batch if hasattr(data,'batch') else None
        # weights1 = bweights(data, cluster1)
        data = max_pool(cluster1, data, transform=T.Cartesian(cat=False))

        data.x = F.elu(self.conv2a(data.x, data.edge_index, data.edge_attr))
        data.x = F.elu(self.conv2b(data.x, data.edge_index, data.edge_attr))
        data.x = F.elu(self.conv2c(data.x, data.edge_index, data.edge_attr))
        weight = normalized_cut_2d(data.edge_index, data.pos)
        cluster2 = graclus(data.edge_index, weight, data.x.size(0))
        pos2 = data.pos
        edge_index2 = data.edge_index
        batch2 = data.batch if hasattr(data,'batch') else None
        # weights2 = bweights(data, cluster2)
        data = max_pool(cluster2, data, transform=T.Cartesian(cat=False))

        # upsample
        # data = recover_grid_barycentric(data, weights=weights2, pos=pos2, edge_index=edge_index2, cluster=cluster2,
        #                                  batch=batch2, transform=T.Cartesian(cat=False))
        data.x = F.elu(self.conv3a(data.x, data.edge_index, data.edge_attr))
        data.x = F.elu(self.conv3b(data.x, data.edge_index, data.edge_attr))

        data = recover_grid(data, pos2, edge_index2, cluster2, batch=batch2, transform=T.Cartesian(cat=False))

        # data = recover_grid_barycentric(data, weights=weights1, pos=pos1, edge_index=edge_index1, cluster=cluster1,
        #                                  batch=batch1, transform=T.Cartesian(cat=False))
        data.x = F.elu(self.conv4a(data.x, data.edge_index, data.edge_attr))
        data.x = F.elu(self.conv4b(data.x, data.edge_index, data.edge_attr))
        data = recover_grid(data, pos1, edge_index1, cluster1, batch=batch1, transform=T.Cartesian(cat=False))

        # TODO handle contract on trainer and  evaluator
        data.x = F.elu(self.convout(data.x, data.edge_index, data.edge_attr))

        x = data.x

        # return F.sigmoid(x)
        return x

class GFCNA(torch.nn.Module):
    def __init__(self):
        super(GFCNA, self).__init__()
        self.conv1a = SplineConv(1, 32, dim=2, kernel_size=5)
        self.conv1b = SplineConv(32, 32, dim=2, kernel_size=5)
        self.bn1 = torch.nn.BatchNorm1d(32)

        self.conv2a = SplineConv(32, 64, dim=2, kernel_size=3)
        self.conv2b = SplineConv(64, 64, dim=2, kernel_size=3)
        self.bn2 = torch.nn.BatchNorm1d(64)

        self.conv3a = SplineConv(64, 128, dim=2, kernel_size=3)
        self.conv3b = SplineConv(128, 128, dim=2, kernel_size=1)
        self.bn3 = torch.nn.BatchNorm1d(128)

        self.score_fr = SplineConv(128, 1, dim=2, kernel_size=1)
        self.score_pool2 = SplineConv(64, 1, dim=2, kernel_size=3)



    def forward(self, data):
        # (1/32,V_0/V_1)
        data.x = F.elu(self.conv1a(data.x, data.edge_index, data.edge_attr))
        data.x = F.elu(self.conv1b(data.x, data.edge_index, data.edge_attr))
        data.x = self.bn1(data.x)
        weight = normalized_cut_2d(data.edge_index, data.pos)
        cluster1 = graclus(data.edge_index, weight, data.x.size(0))
        pos1 = data.pos
        edge_index1 = data.edge_index
        batch1 = data.batch if hasattr(data,'batch') else None
        # weights1, centroids1 = bweights(data, cluster1)
        data = max_pool(cluster1, data, transform=T.Cartesian(cat=False))

        # (32/64,V_1/V_2)
        data.x = F.elu(self.conv2a(data.x, data.edge_index, data.edge_attr))
        data.x = F.elu(self.conv2b(data.x, data.edge_index, data.edge_attr))
        data.x = self.bn2(data.x)
        weight = normalized_cut_2d(data.edge_index, data.pos)
        cluster2 = graclus(data.edge_index, weight, data.x.size(0))
        pos2 = data.pos
        edge_index2 = data.edge_index
        batch2 = data.batch if hasattr(data,'batch') else None
        # weights2, centroids2 = bweights(data, cluster2)
        data = max_pool(cluster2, data, transform=T.Cartesian(cat=False))
        pool2 = data.clone()

        # 64/64,V_2/V_3
        data.x = F.elu(self.conv3a(data.x, data.edge_index, data.edge_attr))
        data.x = F.elu(self.conv3b(data.x, data.edge_index, data.edge_attr))
        data.x = self.bn3(data.x)
        weight = normalized_cut_2d(data.edge_index, data.pos)
        cluster3 = graclus(data.edge_index, weight, data.x.size(0))
        pos3 = data.pos
        edge_index3 = data.edge_index
        batch3 = data.batch if hasattr(data,'batch') else None
        # weights2, centroids2 = bweights(data, cluster2)
        data = max_pool(cluster3, data, transform=T.Cartesian(cat=False))




        # upsample
        # data = recover_grid_barycentric(data, weights=weights2, pos=pos2, edge_index=edge_index2, cluster=cluster2,
        #                                  batch=batch2, transform=None)
        data.x = F.elu(self.score_fr(data.x, data.edge_index, data.edge_attr))
        data = recover_grid(data, pos3, edge_index3, cluster3, batch=batch3, transform=T.Cartesian(cat=False))


        pool2.x = F.elu(self.score_pool2(pool2.x, pool2.edge_index, pool2.edge_attr))

        # data = recover_grid_barycentric(data, weights=weights1, pos=pos1, edge_index=edge_index1, cluster=cluster1,
        #                                  batch=batch1, transform=None)
        data.x = data.x+pool2.x
        data = recover_grid(data, pos2, edge_index2, cluster2, batch=batch2, transform=T.Cartesian(cat=False))
        data = recover_grid(data, pos1, edge_index1, cluster1, batch=batch1, transform=T.Cartesian(cat=False))

        # TODO handle contract on trainer and  evaluator

        x = data.x

        return x

class GFCNC(torch.nn.Module):
    def __init__(self):
        super(GFCNC, self).__init__()
        self.conv1a = SplineConv(1, 32, dim=2, kernel_size=5)
        self.conv1b = SplineConv(32, 32, dim=2, kernel_size=5)
        self.bn1 = torch.nn.BatchNorm1d(32)

        self.conv2a = SplineConv(32, 64, dim=2, kernel_size=3)
        self.conv2b = SplineConv(64, 64, dim=2, kernel_size=3)
        self.bn2 = torch.nn.BatchNorm1d(64)

        self.conv3a = SplineConv(64, 128, dim=2, kernel_size=3)
        self.conv3b = SplineConv(128, 128, dim=2, kernel_size=1)
        self.bn3 = torch.nn.BatchNorm1d(128)

        self.conv4a = SplineConv(128, 256, dim=2, kernel_size=1)
        self.conv4b = SplineConv(256, 256, dim=2, kernel_size=1)
        self.bn4 = torch.nn.BatchNorm1d(256)

        self.score_fr = SplineConv(256, 1, dim=2, kernel_size=1)
        self.score_pool2 = SplineConv(64, 1, dim=2, kernel_size=3)
        self.score_pool3 = SplineConv(128, 1, dim=2, kernel_size=3)



    def forward(self, data):
        # (1/32,V_0/V_1)
        # aux = data.x.clone()
        data.x = F.elu(self.conv1a(data.x, data.edge_index, data.edge_attr))
        data.x = F.elu(self.conv1b(data.x, data.edge_index, data.edge_attr))
        data.x = self.bn1(data.x)
        weight = normalized_cut_2d(data.edge_index, data.pos)
        cluster1 = graclus(data.edge_index, weight, data.x.size(0))
        pos1 = data.pos
        edge_index1 = data.edge_index
        batch1 = data.batch if hasattr(data,'batch') else None
        # weights1, centroids1 = bweights(data, cluster1)
        # weights1 = pweights(aux, cluster1)
        data = max_pool(cluster1, data, transform=T.Cartesian(cat=False))

        # (32/64,V_1/V_2)
        data.x = F.elu(self.conv2a(data.x, data.edge_index, data.edge_attr))
        data.x = F.elu(self.conv2b(data.x, data.edge_index, data.edge_attr))
        data.x = self.bn2(data.x)
        weight = normalized_cut_2d(data.edge_index, data.pos)
        cluster2 = graclus(data.edge_index, weight, data.x.size(0))
        pos2 = data.pos
        edge_index2 = data.edge_index
        batch2 = data.batch if hasattr(data,'batch') else None
        # weights2, centroids2 = bweights(data, cluster2)
        data = max_pool(cluster2, data, transform=T.Cartesian(cat=False))
        pool2 = data.clone()

        # 64/128,V_2/V_3
        data.x = F.elu(self.conv3a(data.x, data.edge_index, data.edge_attr))
        data.x = F.elu(self.conv3b(data.x, data.edge_index, data.edge_attr))
        data.x = self.bn3(data.x)
        weight = normalized_cut_2d(data.edge_index, data.pos)
        cluster3 = graclus(data.edge_index, weight, data.x.size(0))
        pos3 = data.pos
        edge_index3 = data.edge_index
        batch3 = data.batch if hasattr(data,'batch') else None
        # weights2, centroids2 = bweights(data, cluster2)
        data = max_pool(cluster3, data, transform=T.Cartesian(cat=False))
        pool3 = data.clone()


        # 128/256,V_3/V_4
        data.x = F.elu(self.conv4a(data.x, data.edge_index, data.edge_attr))
        data.x = F.elu(self.conv4b(data.x, data.edge_index, data.edge_attr))
        data.x = self.bn4(data.x)
        weight = normalized_cut_2d(data.edge_index, data.pos)
        cluster4 = graclus(data.edge_index, weight, data.x.size(0))
        pos4 = data.pos
        edge_index4 = data.edge_index
        batch4 = data.batch if hasattr(data, 'batch') else None
        # weights2, centroids2 = bweights(data, cluster2)
        data = max_pool(cluster4, data, transform=T.Cartesian(cat=False))

        # LAYERS:
        # 256/1, V4/V3
        # data = recover_grid_barycentric(data, weights=weights2, pos=pos2, edge_index=edge_index2, cluster=cluster2,
        #                                  batch=batch2, transform=None)
        # compute score of latent space (V4.128)=>(V4.1)
        data.x = F.elu(self.score_fr(data.x, data.edge_index, data.edge_attr))
        # upsample V4=>V3
        data = recover_grid(data, pos4, edge_index4, cluster4, batch=batch4, transform=T.Cartesian(cat=False))

        # compute score of pool3  (V3.128)=>(V3,1)
        pool3.x = F.elu(self.score_pool3(pool3.x, pool3.edge_index, pool3.edge_attr))

        data.x = data.x+pool3.x
        # upsample V3=>V2
        data = recover_grid(data, pos3, edge_index3, cluster3, batch=batch3, transform=T.Cartesian(cat=False))
        # compute score of pool2 (V2.64)=>(V2.1)
        pool2.x = F.elu(self.score_pool2(pool2.x, pool2.edge_index, pool2.edge_attr))
        data.x = data.x+pool2.x
        # upsample V2=>V1
        data = recover_grid(data, pos2, edge_index2, cluster2, batch=batch2, transform=T.Cartesian(cat=False))
        data = recover_grid(data, pos1, edge_index1, cluster1, batch=batch1, transform=T.Cartesian(cat=False))
        # data = recover_grid_barycentric(data, weights=weights1, pos=pos1, edge_index=edge_index1, cluster=cluster1, batch=batch1, transform=None)

        # TODO handle contract on trainer and  evaluator
        return data.x

#### MODEL
class down(torch.nn.Module):
    def __init__(self, ratio, r, in_channels, out_channels, dim, kernel_size):
        super(down, self).__init__()
        self.ratio = ratio
        self.r = r
        self.conv = SplineConv(in_channels, out_channels, dim=dim, kernel_size=kernel_size)

    def forward(self, data):
        x = F.elu(self.conv(data.x, data.edge_index, data.edge_attr))
        idx = fps(data.pos, data.batch, ratio=self.ratio)
        row, col = radius(data.pos, data.pos[idx], self.r, data.batch, data.batch[idx],
                          max_num_neighbors=64)
        edge_index = torch.stack([col, row], dim=0).to(data.edge_index.device)
        pos, batch = data.pos[idx], data.batch[idx]

        data.x = x[idx]
        data.edge_index = edge_index
        data.pos = pos
        data.batch = batch
        data.edge_attr = data.edge_attr[idx]

        #         data = t(data)
        return data


class up(torch.nn.Module):
    def __init__(self, k, nn):
        super(up, self).__init__()
        self.k = k
        self.nn = nn

    def forward(self, x, pos, batch, x_skip, pos_skip, batch_skip):
        x = knn_interpolate(x, pos, pos_skip, batch, batch_skip, k=self.k)

        if x_skip is not None:
            print('x.size()', x.size(), 'x_skip.size()', x_skip.size())
            x = torch.cat([x, x_skip], dim=1).to(x.device)
        x = self.nn(x)
        return x, pos_skip, batch_skip


def MLP(channels, batch_norm=True):
    return Seq(*[
        Seq(Lin(channels[i - 1], channels[i]), ReLU(), BN(channels[i]))
        for i in range(1, len(channels))
    ])


class GFCND(torch.nn.Module):
    def __init__(self):
        super(GFCND, self).__init__()
        print('model GFCN-A')
        self.down1 = down(0.5, 0.3, 1, 32, dim=2, kernel_size=5)
        #         self.conv2 = SplineConv(32, 64, dim=2, kernel_size=5)
        #         self.conv3 = SplineConv(64, 32, dim=2, kernel_size=5)
        self.up1 = up(3, MLP([1 + 32, 32, 32, 1]))

    def forward(self, data):
        ## LAYER 1 (1,V0)->(32,V1)
        input_state = (data.x.clone().unsqueeze(-1), data.pos, data.batch)
        data = self.down1(data)
        down1_state = (data.x.clone(), data.pos, data.batch)
        x, _, _ = self.up1(*down1_state, *input_state)
        return F.sigmoid(x)


class GFCN(torch.nn.Module):
    def __init__(self):
        super(GFCN, self).__init__()
        print('model GFCN-org..')
        self.conv1 = SplineConv(1, 32, dim=2, kernel_size=5)
        self.conv2 = SplineConv(32, 64, dim=2, kernel_size=5)
        self.conv3 = SplineConv(64, 32, dim=2, kernel_size=5)
        self.conv4 = SplineConv(32, 1, dim=2, kernel_size=5)

    def forward(self, data):

        ## LAYER 1 (1,V0)->(32,V1)
        # pre-pool1
        pos1 = data.pos
        edge_index1 = data.edge_index
        x_pre = data.x.clone().detach()
        batch1 = data.batch if hasattr(data, 'batch') else None
        # convolution
        data.x = F.elu(self.conv1(data.x, data.edge_index, data.edge_attr))
        # clustering
        weight = normalized_cut_2d(data.edge_index, data.pos)
        cluster1 = graclus(data.edge_index, weight, data.x.size(0))
        weights1 = pweights(x_pre,cluster1)
        #pooling
        data = max_pool(cluster1, data, transform=T.Cartesian(cat=False))

        ## LAYER 2 (32,V1)->(64,V2)
        # pre-pool2
        pos2 = data.pos
        edge_index2 = data.edge_index
        batch2 = data.batch if hasattr(data, 'batch') else None
        x_pre = data.x.clone().detach()
        # convolution
        data.x = F.elu(self.conv2(data.x, data.edge_index, data.edge_attr))
        # clustering
        weight = normalized_cut_2d(data.edge_index, data.pos)
        cluster2 = graclus(data.edge_index, weight, data.x.size(0))
        weights2 = pweights(x_pre, cluster2)
        # pooling
        data = max_pool(cluster2, data, transform=T.Cartesian(cat=False))

        # LAYER 3  (64,V2)->(32,V1)
        data.x = F.elu(self.conv3(data.x, data.edge_index, data.edge_attr)) # (32,V2)

        data = recover_grid_barycentric(data, weights=weights2, pos=pos2, edge_index=edge_index2, cluster=cluster2,
                                         batch=batch2, transform=T.Cartesian(cat=False))

        # LAYER 4  (32,V1)->(1,V0)
        data.x = F.elu(self.conv4(data.x, data.edge_index, data.edge_attr))
        data = recover_grid_barycentric(data, weights=weights1, pos=pos1, edge_index=edge_index1, cluster=cluster1,
                                        batch=batch1, transform=T.Cartesian(cat=False))

        # return F.sigmoid(data.x)
        return x
