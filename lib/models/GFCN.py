import torch
import torch_geometric.transforms as T
import torch.nn.functional as F
from torch_geometric.utils import normalized_cut, scatter_
# from torch_scatter import scatter as scatter_
from torch_geometric.data import Data, Batch
from torch_geometric.nn import graclus, max_pool, avg_pool, fps, radius, knn_interpolate, TopKPooling

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


def recover_grid(source: Data, pos, edge_index, cluster, batch=None, transform=None):
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


# def consecutive_cluster(src):
#     unique, inv = torch.unique(src, sorted=True, return_inverse=True)
#     perm = torch.arange(inv.size(0), dtype=inv.dtype, device=inv.device)
#     perm = inv.new_empty(unique.size(0)).scatter_(0, inv, perm)
#     return inv, perm


def pweights(x, cluster):
    ''' Computes the percentage weights in the simplex formed by the cluster '''
    with torch.no_grad():
        cluster, perm = consecutive_cluster(cluster)
        y = torch.ones_like(x)
        g = scatter_('add', x, cluster)
        h = scatter_('add', y, cluster)
        w = h[cluster]*x/(g[cluster]+0.001)
        w[w != w] = 0
        if w.dim() == 1:
            w = w.unsqueeze(-1)

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

    if batch is not None:
        data = Batch(x=source.x[cluster]*weights, edge_index=edge_index, pos=pos, batch=batch)
    else:
        data = Data(x=source.x[cluster]*weights, edge_index=edge_index, pos=pos)

    if transform is not None:
        data = transform(data)
    return data


class GFCNB(torch.nn.Module):
    ''' GFCN equivalent to the FCN32s'''
    def __init__(self, input_channels=1, postnorm_activation=True):
        super(GFCNB, self).__init__()
        self.postnorm_activation = postnorm_activation
        self.conv1a = SplineConv(input_channels, 32, dim=2, kernel_size=5)
        self.conv1b = SplineConv(32, 32, dim=2, kernel_size=5)
        if postnorm_activation:
            self.bn1 = torch.nn.BatchNorm1d(32)
        else:
            self.bn1_1 = torch.nn.BatchNorm1d(32)
            self.bn1_2 = torch.nn.BatchNorm1d(32)

        self.conv2a = SplineConv(32, 64, dim=2, kernel_size=3)
        self.conv2b = SplineConv(64, 64, dim=2, kernel_size=3)
        if postnorm_activation:
            self.bn2= torch.nn.BatchNorm1d(64)
        else:
            self.bn2_1 = torch.nn.BatchNorm1d(64)
            self.bn2_2 = torch.nn.BatchNorm1d(64)

        self.conv3a = SplineConv(64, 128, dim=2, kernel_size=3)
        self.conv3b = SplineConv(128, 128, dim=2, kernel_size=3)
        if postnorm_activation:
            self.bn3= torch.nn.BatchNorm1d(128)
        else:
            self.bn3_1 = torch.nn.BatchNorm1d(128)
            self.bn3_2 = torch.nn.BatchNorm1d(128)

        self.score_fr = SplineConv(128, 32, dim=2, kernel_size=5)

        self.convout = SplineConv(32, 1, dim=2, kernel_size=5)

    def forward(self, data):
        # (V0.1)->(V1,32)
        if self.postnorm_activation:
            data.x = F.elu(self.conv1a(data.x, data.edge_index, data.edge_attr))
            data.x = F.elu(self.conv1b(data.x, data.edge_index, data.edge_attr))
            data.x = self.bn1(data.x)
        else:
            data.x = F.elu(self.bn1_1(self.conv1a(data.x, data.edge_index, data.edge_attr)))
            data.x = F.elu(self.bn1_2(self.conv1b(data.x, data.edge_index, data.edge_attr)))
        weight = normalized_cut_2d(data.edge_index, data.pos)
        cluster1 = graclus(data.edge_index, weight, data.x.size(0))
        pos1 = data.pos
        edge_index1 = data.edge_index
        batch1 = data.batch if hasattr(data,'batch') else None
        # weights1 = bweights(data, cluster1)
        data = max_pool(cluster1, data, transform=T.Cartesian(cat=False))

        # (V1,32)=>(V2,64)
        if self.postnorm_activation:
            data.x = F.elu(self.conv2a(data.x, data.edge_index, data.edge_attr))
            data.x = F.elu(self.conv2b(data.x, data.edge_index, data.edge_attr))
            data.x = self.bn2(data.x)
        else:
            data.x = F.elu(self.bn2_1(self.conv2a(data.x, data.edge_index, data.edge_attr)))
            data.x = F.elu(self.bn2_2(self.conv2b(data.x, data.edge_index, data.edge_attr)))
        weight = normalized_cut_2d(data.edge_index, data.pos)
        cluster2 = graclus(data.edge_index, weight, data.x.size(0))
        pos2 = data.pos
        edge_index2 = data.edge_index
        batch2 = data.batch if hasattr(data,'batch') else None
        # weights2 = bweights(data, cluster2)
        data = max_pool(cluster2, data, transform=T.Cartesian(cat=False))

        # (V2,64)=>(V3.128)
        if self.postnorm_activation:
            data.x = F.elu(self.conv3a(data.x, data.edge_index, data.edge_attr))
            data.x = F.elu(self.conv3b(data.x, data.edge_index, data.edge_attr))
            data.x = self.bn3(data.x)
        else:
            data.x = F.elu(self.bn3_1(self.conv3a(data.x, data.edge_index, data.edge_attr)))
            data.x = F.elu(self.bn3_2(self.conv3b(data.x, data.edge_index, data.edge_attr)))
        weight = normalized_cut_2d(data.edge_index, data.pos)
        cluster3 = graclus(data.edge_index, weight, data.x.size(0))
        pos3 = data.pos
        edge_index3 = data.edge_index
        batch3 = data.batch if hasattr(data,'batch') else None
        # weights2, centroids2 = bweights(data, cluster2)
        data = max_pool(cluster3, data, transform=T.Cartesian(cat=False))

        # Upsampling path (V3.128)=>(V0.1)
        # data = recover_grid_barycentric(data, weights=weights2, pos=pos2, edge_index=edge_index2, cluster=cluster2,
        #                                  batch=batch2, transform=T.Cartesian(cat=False))
        data.x = F.elu(self.score_fr(data.x, data.edge_index, data.edge_attr))
        data = recover_grid(data, pos3, edge_index3, cluster3, batch=batch3, transform=T.Cartesian(cat=False))

        # data = recover_grid_barycentric(data, weights=weights1, pos=pos1, edge_index=edge_index1, cluster=cluster1,
        #                                  batch=batch1, transform=T.Cartesian(cat=False))
        data = recover_grid(data, pos2, edge_index2, cluster2, batch=batch2, transform=T.Cartesian(cat=False))
        data = recover_grid(data, pos1, edge_index1, cluster1, batch=batch1, transform=T.Cartesian(cat=False))

        # TODO handle contract on trainer and  evaluator
        data.x = self.convout(data.x, data.edge_index, data.edge_attr)
        return data

class GFCNA(torch.nn.Module):
    ''' GFCN equivalent to the FCN16s'''
    def __init__(self, input_channels=1, postnorm_activation=True):
        super(GFCNA, self).__init__()
        self.postnorm_activation = postnorm_activation
        self.conv1a = SplineConv(input_channels, 32, dim=2, kernel_size=5)
        self.conv1b = SplineConv(32, 32, dim=2, kernel_size=5)
        if postnorm_activation:
            self.bn1 = torch.nn.BatchNorm1d(32)
        else:
            self.bn1_1 = torch.nn.BatchNorm1d(32)
            self.bn1_2 = torch.nn.BatchNorm1d(32)

        self.conv2a = SplineConv(32, 64, dim=2, kernel_size=3)
        self.conv2b = SplineConv(64, 64, dim=2, kernel_size=3)
        if postnorm_activation:
            self.bn2= torch.nn.BatchNorm1d(64)
        else:
            self.bn2_1 = torch.nn.BatchNorm1d(64)
            self.bn2_2 = torch.nn.BatchNorm1d(64)

        self.conv3a = SplineConv(64, 128, dim=2, kernel_size=3)
        self.conv3b = SplineConv(128, 128, dim=2, kernel_size=1)
        if postnorm_activation:
            self.bn3= torch.nn.BatchNorm1d(128)
        else:
            self.bn3_1 = torch.nn.BatchNorm1d(128)
            self.bn3_2 = torch.nn.BatchNorm1d(128)

        self.score_fr = SplineConv(128, 32, dim=2, kernel_size=1)
        self.score_pool2 = SplineConv(64, 32, dim=2, kernel_size=3)

        self.convout = SplineConv(32, 1, dim=2, kernel_size=5)



    def forward(self, data):
        # (V0.1)=> (V1,32)
        if self.postnorm_activation:
            data.x = F.elu(self.conv1a(data.x, data.edge_index, data.edge_attr))
            data.x = F.elu(self.conv1b(data.x, data.edge_index, data.edge_attr))
            data.x = self.bn1(data.x)
        else:
            data.x = F.elu(self.bn1_1(self.conv1a(data.x, data.edge_index, data.edge_attr)))
            data.x = F.elu(self.bn1_2(self.conv1b(data.x, data.edge_index, data.edge_attr)))
        weight = normalized_cut_2d(data.edge_index, data.pos)
        cluster1 = graclus(data.edge_index, weight, data.x.size(0))
        pos1 = data.pos
        edge_index1 = data.edge_index
        batch1 = data.batch if hasattr(data,'batch') else None
        # weights1, centroids1 = bweights(data, cluster1)
        data = max_pool(cluster1, data, transform=T.Cartesian(cat=False))

        # (V1.32)=> (V2.64)
        if self.postnorm_activation:
            data.x = F.elu(self.conv2a(data.x, data.edge_index, data.edge_attr))
            data.x = F.elu(self.conv2b(data.x, data.edge_index, data.edge_attr))
            data.x = self.bn2(data.x)
        else:
            data.x = F.elu(self.bn2_1(self.conv2a(data.x, data.edge_index, data.edge_attr)))
            data.x = F.elu(self.bn2_2(self.conv2b(data.x, data.edge_index, data.edge_attr)))
        weight = normalized_cut_2d(data.edge_index, data.pos)
        cluster2 = graclus(data.edge_index, weight, data.x.size(0))
        pos2 = data.pos
        edge_index2 = data.edge_index
        batch2 = data.batch if hasattr(data,'batch') else None
        # weights2, centroids2 = bweights(data, cluster2)
        data = max_pool(cluster2, data, transform=T.Cartesian(cat=False))
        pool2 = data.clone()

        # (V2.64)=>(V3.128)
        if self.postnorm_activation:
            data.x = F.elu(self.conv3a(data.x, data.edge_index, data.edge_attr))
            data.x = F.elu(self.conv3b(data.x, data.edge_index, data.edge_attr))
            data.x = self.bn3(data.x)
        else:
            data.x = F.elu(self.bn3_1(self.conv3a(data.x, data.edge_index, data.edge_attr)))
            data.x = F.elu(self.bn3_2(self.conv3b(data.x, data.edge_index, data.edge_attr)))
        weight = normalized_cut_2d(data.edge_index, data.pos)
        cluster3 = graclus(data.edge_index, weight, data.x.size(0))
        pos3 = data.pos
        edge_index3 = data.edge_index
        batch3 = data.batch if hasattr(data,'batch') else None
        # weights2, centroids2 = bweights(data, cluster2)
        data = max_pool(cluster3, data, transform=T.Cartesian(cat=False))

        # Upsampling path (V3.128)=>(V0.1)
        # data = recover_grid_barycentric(data, weights=weights2, pos=pos2, edge_index=edge_index2, cluster=cluster2,
        #                                  batch=batch2, transform=None)
        # (V3.128)=>(V2.32)
        data.x = F.elu(self.score_fr(data.x, data.edge_index, data.edge_attr))
        data = recover_grid(data, pos3, edge_index3, cluster3, batch=batch3, transform=T.Cartesian(cat=False))

        # (V2.64)=>(V2.32)
        pool2.x = F.elu(self.score_pool2(pool2.x, pool2.edge_index, pool2.edge_attr))
        data.x = data.x+pool2.x
        # data = recover_grid_barycentric(data, weights=weights1, pos=pos1, edge_index=edge_index1, cluster=cluster1,
        #                                  batch=batch1, transform=None)
        # (V2.32)=>(V0.32)
        data = recover_grid(data, pos2, edge_index2, cluster2, batch=batch2, transform=T.Cartesian(cat=False))
        data = recover_grid(data, pos1, edge_index1, cluster1, batch=batch1, transform=T.Cartesian(cat=False))

        # (V0.32)=>(V0.1)
        # data.x = F.elu(self.convout(data.x, data.edge_index, data.edge_attr))
        data.x = self.convout(data.x, data.edge_index, data.edge_attr)

        # x = data.x

        return data

class GFCNC(torch.nn.Module):
    ''' model G-FCN 8s equivalent'''
    def __init__(self, input_channels=1, postnorm_activation=True, pweights=False):
        super(GFCNC, self).__init__()
        self.postnorm_activation = postnorm_activation
        self.weight_upool = pweights

        self.conv1a = SplineConv(input_channels, 32, dim=2, kernel_size=5)
        self.conv1b = SplineConv(32, 32, dim=2, kernel_size=5)
        if postnorm_activation:
            self.bn1 = torch.nn.BatchNorm1d(32)
        else:
            self.bn1_1 = torch.nn.BatchNorm1d(32)
            self.bn1_2 = torch.nn.BatchNorm1d(32)

        self.conv2a = SplineConv(32, 64, dim=2, kernel_size=3)
        self.conv2b = SplineConv(64, 64, dim=2, kernel_size=3)
        if postnorm_activation:
            self.bn2= torch.nn.BatchNorm1d(64)
        else:
            self.bn2_1 = torch.nn.BatchNorm1d(64)
            self.bn2_2 = torch.nn.BatchNorm1d(64)

        self.conv3a = SplineConv(64, 128, dim=2, kernel_size=3)
        self.conv3b = SplineConv(128, 128, dim=2, kernel_size=1)
        if postnorm_activation:
            self.bn3= torch.nn.BatchNorm1d(128)
        else:
            self.bn3_1 = torch.nn.BatchNorm1d(128)
            self.bn3_2 = torch.nn.BatchNorm1d(128)

        self.conv4a = SplineConv(128, 256, dim=2, kernel_size=1)
        self.conv4b = SplineConv(256, 256, dim=2, kernel_size=1)
        if postnorm_activation:
            self.bn4= torch.nn.BatchNorm1d(256)
        else:
            self.bn4_1 = torch.nn.BatchNorm1d(256)
            self.bn4_2 = torch.nn.BatchNorm1d(256)

        self.score_fr = SplineConv(256, 32, dim=2, kernel_size=1)
        self.score_pool2 = SplineConv(64, 32, dim=2, kernel_size=3)
        self.score_pool3 = SplineConv(128, 32, dim=2, kernel_size=3)

        self.convout = SplineConv(32, 1, dim=2, kernel_size=5)


    def forward(self, data):
        # define weights as None:
        weights1, weights2, weights3, weights4 = None, None, None, None
        # (V0.1)=>(V1.32)
        if self.postnorm_activation:
            data.x = F.elu(self.conv1a(data.x, data.edge_index, data.edge_attr))
            data.x = F.elu(self.conv1b(data.x, data.edge_index, data.edge_attr))
            data.x = self.bn1(data.x)
        else:
            data.x = F.elu(self.bn1_1(self.conv1a(data.x, data.edge_index, data.edge_attr)))
            data.x = F.elu(self.bn1_2(self.conv1b(data.x, data.edge_index, data.edge_attr)))

        weight = normalized_cut_2d(data.edge_index, data.pos)
        cluster1 = graclus(data.edge_index, weight, data.x.size(0))
        pos1 = data.pos
        edge_index1 = data.edge_index
        batch1 = data.batch if hasattr(data,'batch') else None
        if self.weight_upool:
            weights1 = pweights(data, cluster1)
        data = max_pool(cluster1, data, transform=T.Cartesian(cat=False))

        # (V1.32)=>(V2.64)
        if self.postnorm_activation:
            data.x = F.elu(self.conv2a(data.x, data.edge_index, data.edge_attr))
            data.x = F.elu(self.conv2b(data.x, data.edge_index, data.edge_attr))
            data.x = self.bn2(data.x)
        else:
            data.x = F.elu(self.bn2_1(self.conv2a(data.x, data.edge_index, data.edge_attr)))
            data.x = F.elu(self.bn2_2(self.conv2b(data.x, data.edge_index, data.edge_attr)))
        weight = normalized_cut_2d(data.edge_index, data.pos)
        cluster2 = graclus(data.edge_index, weight, data.x.size(0))
        pos2 = data.pos
        edge_index2 = data.edge_index
        batch2 = data.batch if hasattr(data,'batch') else None
        # weights2, centroids2 = bweights(data, cluster2)
        if self.weight_upool:
            weights2 = pweights(data, cluster2)
        data = max_pool(cluster2, data, transform=T.Cartesian(cat=False))
        pool2 = data.clone()

        # (V2.64)=>(V3.128)
        if self.postnorm_activation:
            data.x = F.elu(self.conv3a(data.x, data.edge_index, data.edge_attr))
            data.x = F.elu(self.conv3b(data.x, data.edge_index, data.edge_attr))
            data.x = self.bn3(data.x)
        else:
            data.x = F.elu(self.bn3_1(self.conv3a(data.x, data.edge_index, data.edge_attr)))
            data.x = F.elu(self.bn3_2(self.conv3b(data.x, data.edge_index, data.edge_attr)))

        weight = normalized_cut_2d(data.edge_index, data.pos)
        cluster3 = graclus(data.edge_index, weight, data.x.size(0))
        pos3 = data.pos
        edge_index3 = data.edge_index
        batch3 = data.batch if hasattr(data,'batch') else None
        # weights2, centroids2 = bweights(data, cluster2)
        if self.weight_upool:
            weights2 = pweights(data, cluster3)
        data = max_pool(cluster3, data, transform=T.Cartesian(cat=False))
        pool3 = data.clone()


        # (V3.128)=>(V4.256)
        if self.postnorm_activation:
            data.x = F.elu(self.conv4a(data.x, data.edge_index, data.edge_attr))
            data.x = F.elu(self.conv4b(data.x, data.edge_index, data.edge_attr))
            data.x = self.bn4(data.x)
        else:
            data.x = F.elu(self.bn4_1(self.conv4a(data.x, data.edge_index, data.edge_attr)))
            data.x = F.elu(self.bn4_2(self.conv4b(data.x, data.edge_index, data.edge_attr)))

        weight = normalized_cut_2d(data.edge_index, data.pos)
        cluster4 = graclus(data.edge_index, weight, data.x.size(0))
        pos4 = data.pos
        edge_index4 = data.edge_index
        batch4 = data.batch if hasattr(data, 'batch') else None
        # weights2, centroids2 = bweights(data, cluster2)
        if self.weight_upool:
            weights4 = pweights(data, cluster4)
        data = max_pool(cluster4, data, transform=T.Cartesian(cat=False))

        # LAYERS:
        # Transform V4.256 to V0.1
        # data = recover_grid_barycentric(data, weights=weights2, pos=pos2, edge_index=edge_index2, cluster=cluster2,
        #                                  batch=batch2, transform=None)
        # compute score of latent space (V4.128)=>(V4.32)
        data.x = F.elu(self.score_fr(data.x, data.edge_index, data.edge_attr))
        # upsample V4=>V3
        if self.weight_upool:
            data = recover_grid_barycentric(data, weights4, pos4, edge_index4, cluster4, batch=batch4, transform=T.Cartesian(cat=False))
        else:
            data = recover_grid(data, pos4, edge_index4, cluster4, batch=batch4, transform=T.Cartesian(cat=False))

        # compute score of pool3  (V3.128)=>(V3,32)
        pool3.x = F.elu(self.score_pool3(pool3.x, pool3.edge_index, pool3.edge_attr))
        data.x = data.x+pool3.x

        # upsample V3=>V2
        if self.weight_upool:
            data = recover_grid_barycentric(data, weights3, pos3, edge_index3, cluster3, batch=batch3, transform=T.Cartesian(cat=False))
        else:
            data = recover_grid(data, pos3, edge_index3, cluster3, batch=batch3, transform=T.Cartesian(cat=False))

        # compute score of pool2 (V2.64)=>(V2.32)
        pool2.x = F.elu(self.score_pool2(pool2.x, pool2.edge_index, pool2.edge_attr))
        data.x = data.x+pool2.x

        # upsample (V2.32)=>(V1.32)=>(V0.32)
        if self.weight_upool:
            data = recover_grid_barycentric(data, weights2, pos2, edge_index2, cluster2, batch=batch2, transform=T.Cartesian(cat=False))
            data = recover_grid_barycentric(data, weights1, pos1, edge_index1, cluster1, batch=batch1, transform=T.Cartesian(cat=False))
        else:
            data = recover_grid(data, pos2, edge_index2, cluster2, batch=batch2, transform=T.Cartesian(cat=False))
            data = recover_grid(data, pos1, edge_index1, cluster1, batch=batch1, transform=T.Cartesian(cat=False))
        # data = recover_grid_barycentric(data, weights=weights1, pos=pos1, edge_index=edge_index1, cluster=cluster1, batch=batch1, transform=None)

        # TODO handle contract on trainer and  evaluator
        # restore original channels (V0.32)=>(V0.1)
        # data.x = F.elu(self.convout(data.x, data.edge_index, data.edge_attr))
        data.x = self.convout(data.x, data.edge_index, data.edge_attr)
        return data

#### MODEL
class down(torch.nn.Module):
    def __init__(self, ratio, r, in_channels, out_channels, dim, kernel_size):
        super(Downsampling, self).__init__()
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


#### MODEL TOP-K + knn-interpolate
class Downsampling(torch.nn.Module):
    def __init__(self, k_range, ratio, in_channels, out_channels, dim, kernel_size,batch_norm=True):
        super(Downsampling, self).__init__()
        self.pool = TopKPooling(k_range, ratio=ratio)
        hidden_channels = int(out_channels/2)
        self.conva = SplineConv(in_channels, out_channels, dim=dim, kernel_size=kernel_size)
        self.convb = SplineConv(out_channels, out_channels, dim=dim, kernel_size=kernel_size)
        self.batch_norm = batch_norm
        if self.batch_norm:
            self.bn = torch.nn.BatchNorm1d(out_channels)

        # self.convc = SplineConv(out_channels, out_channels, dim=dim, kernel_size=kernel_size)

    def forward(self, data):
        data.x = F.elu(self.conva(data.x, data.edge_index, data.edge_attr))
        data.x = F.elu(self.convb(data.x, data.edge_index, data.edge_attr))
        # data.x = F.elu(self.convc(data.x, data.edge_index, data.edge_attr))
        if self.batch_norm:
            data.x = self.bn(data.x)
        # Backsampling
        backsampling = data.clone()
        backsampling.x = data.x.clone().unsqueeze(-1) if data.x.dim() == 1 else data.x.clone()
        # Pooling projection with TOP-K operator
        values = self.pool(data.x, data.edge_index, edge_attr=data.edge_attr, batch=data.batch)
        # x,edge_index,edge_attr,perm,score
        data.x = values[0]
        data.edge_index = values[1]
        data.pos = data.pos[values[4]]
        data.batch = values[3]
        data.edge_attr = values[2]

        return data, backsampling


class Upsampling(torch.nn.Module):
    def __init__(self, k, in_channels, out_channels, dim, kernel_size, conv_layer=True):
        super(Upsampling, self).__init__()
        self.k = k
        self.conv_layer = conv_layer
        if self.conv_layer:
            self.conva = SplineConv(in_channels, out_channels, dim=dim, kernel_size=kernel_size)
            # self.convb = SplineConv(in_channels, out_channels, dim=dim, kernel_size=kernel_size)

    def forward(self, data, backsampling):
        if self.conv_layer:
            data.x = F.elu(self.conva(data.x, data.edge_index, data.edge_attr))
            # data.x = F.elu(self.convb(data.x, data.edge_index, data.edge_attr))

        data.x = knn_interpolate(data.x, data.pos, backsampling.pos, data.batch, backsampling.batch, k=self.k)
        data.pos = backsampling.pos
        data.edge_index = backsampling.edge_index
        data.edge_attr = backsampling.edge_attr
        data.batch = backsampling.batch
        return data


class GFCND(torch.nn.Module):
    ''' GFCN equivalent to the FCN32s with topk'''


    def __init__(self, input_channels=1):
        super(GFCND, self).__init__()
        self.down1 = Downsampling(k_range=32, ratio=0.5, in_channels=input_channels, out_channels=32, dim=2, kernel_size=5,batch_norm=False)
        self.down2 = Downsampling(k_range=64, ratio=0.5, in_channels=32, out_channels=64, dim=2, kernel_size=3)
        self.down3 = Downsampling(k_range=128, ratio=0.5, in_channels=64, out_channels=128, dim=2, kernel_size=3)
        self.up1 = Upsampling(k=3, in_channels=128, out_channels=64, dim=2, kernel_size=3)
        self.score_fs = SplineConv(64, 32, dim=2, kernel_size=3)

        self.up2 = Upsampling(k=3, in_channels=32, out_channels=32, dim=2, kernel_size=5,conv_layer=False)
        self.up3 = Upsampling(k=3, in_channels=32, out_channels=32, dim=2, kernel_size=5,conv_layer=False)

        self.score_pool2 = SplineConv(64, 32, dim=2, kernel_size=3)
        self.convout = SplineConv(32, 1, dim=2, kernel_size=5)

    def forward(self, data):
        # V0,1 -> V1,32
        data, backsampling_1 = self.down1(data)
        # V1,32 -> V2,64
        data, backsampling_2 = self.down2(data)
        pool2 = data.clone()
        # V2,64 -> V3,128
        data, backsampling_3 = self.down3(data)
        # V3,128 -> V2,32 // score FR
        data = self.up1(data, backsampling_3)
        data.x = F.elu(self.score_pool2(data.x, data.edge_index, data.edge_attr))

        # V2,64 -> V2,32 //score pool2
        pool2.x = F.elu(self.score_pool2(pool2.x, pool2.edge_index, pool2.edge_attr))
        # addition
        data.x = data.x+pool2.x
        # V1,128 -> V0,32
        data = self.up2(data, backsampling_2)
        data = self.up3(data, backsampling_1)
        # convout
        # V0,32 -> V0,1
        # data.x = F.elu(self.convout(data.x, data.edge_index, data.edge_attr))
        data.x = self.convout(data.x, data.edge_index, data.edge_attr)
        x = data.x
        # return F.sigmoid(x)
        return data

class GFCN(torch.nn.Module):
    ''' GFCN16s with barycentric upsampling'''
    def __init__(self, input_channels=1):
        super(GFCN, self).__init__()
        self.conv1a = SplineConv(input_channels, 32, dim=2, kernel_size=5)
        self.conv1b = SplineConv(32, 32, dim=2, kernel_size=5)
        # self.bn1 = torch.nn.BatchNorm1d(32)

        self.conv2a = SplineConv(32, 64, dim=2, kernel_size=3)
        self.conv2b = SplineConv(64, 64, dim=2, kernel_size=3)
        self.bn2 = torch.nn.BatchNorm1d(64)

        self.conv3a = SplineConv(64, 128, dim=2, kernel_size=3)
        self.conv3b = SplineConv(128, 128, dim=2, kernel_size=1)
        self.bn3 = torch.nn.BatchNorm1d(128)

        self.score_fr1 = SplineConv(128, 64, dim=2, kernel_size=1)
        self.score_fr2 = SplineConv(64, 32, dim=2, kernel_size=1)
        self.score_pool2 = SplineConv(64, 32, dim=2, kernel_size=3)

        self.convout = SplineConv(32, 1, dim=2, kernel_size=5)

    def forward(self, data):
        # (1/32,V_0/V_1)
        # pre-pool1
        pos1 = data.pos
        edge_index1 = data.edge_index
        x_pre = data.x.clone().detach()
        batch1 = data.batch if hasattr(data, 'batch') else None
        # convolution
        data.x = F.elu(self.conv1a(data.x, data.edge_index, data.edge_attr))
        data.x = F.elu(self.conv1b(data.x, data.edge_index, data.edge_attr))
        # clustering
        weight = normalized_cut_2d(data.edge_index, data.pos)
        cluster1 = graclus(data.edge_index, weight, data.x.size(0))
        weights1 = pweights(x_pre, cluster1)
        # pooling
        data = avg_pool(cluster1, data, transform=T.Cartesian(cat=False))

        # (32/64,V_1/V_2)

        # pre-pool2
        pos2 = data.pos
        edge_index2 = data.edge_index
        x_pre = data.x.clone().detach()
        batch2 = data.batch if hasattr(data, 'batch') else None
        # convolution
        data.x = F.elu(self.conv2a(data.x, data.edge_index, data.edge_attr))
        data.x = F.elu(self.conv2b(data.x, data.edge_index, data.edge_attr))
        data.x = self.bn2(data.x)
        # clustering
        weight = normalized_cut_2d(data.edge_index, data.pos)
        cluster2 = graclus(data.edge_index, weight, data.x.size(0))
        weights2 = pweights(x_pre, cluster2)
        # pooling
        data = avg_pool(cluster2, data, transform=T.Cartesian(cat=False))
        pool2 = data.clone()

        # 64/128,V_2/V_3
        # pre-pool1
        pos3 = data.pos
        edge_index3 = data.edge_index
        x_pre = data.x.clone().detach()
        batch3 = data.batch if hasattr(data, 'batch') else None
        # convolution
        data.x = F.elu(self.conv3a(data.x, data.edge_index, data.edge_attr))
        data.x = F.elu(self.conv3b(data.x, data.edge_index, data.edge_attr))
        data.x = self.bn3(data.x)
        # clustering
        weight = normalized_cut_2d(data.edge_index, data.pos)
        cluster3 = graclus(data.edge_index, weight, data.x.size(0))
        weights3 = pweights(x_pre, cluster3)
        # pooling
        data = avg_pool(cluster3, data, transform=T.Cartesian(cat=False))

        # upsample
        # data = recover_grid_barycentric(data, weights=weights2, pos=pos2, edge_index=edge_index2, cluster=cluster2,
        #                                  batch=batch2, transform=None)
        data.x = F.elu(self.score_fr1(data.x, data.edge_index, data.edge_attr))
        data = recover_grid_barycentric(data, weights=weights3, pos=pos3, edge_index=edge_index3, cluster=cluster3,
                                        batch=batch3, transform=T.Cartesian(cat=False))
        data.x = F.elu(self.score_fr2(data.x, data.edge_index, data.edge_attr))

        pool2.x = F.elu(self.score_pool2(pool2.x, pool2.edge_index, pool2.edge_attr))

        # data = recover_grid_barycentric(data, weights=weights1, pos=pos1, edge_index=edge_index1, cluster=cluster1,
        #                                  batch=batch1, transform=None)
        data.x = data.x+pool2.x
        data = recover_grid_barycentric(data, weights=weights2, pos=pos2, edge_index=edge_index2, cluster=cluster2,
                                        batch=batch2, transform=T.Cartesian(cat=False))

        data = recover_grid_barycentric(data, weights=weights1, pos=pos1, edge_index=edge_index1, cluster=cluster1,
                                        batch=batch1, transform=T.Cartesian(cat=False))

        #
        # data.x = F.elu(self.convout(data.x, data.edge_index, data.edge_attr))
        data.x = self.convout(data.x, data.edge_index, data.edge_attr)

        # x = data.x

        return data
