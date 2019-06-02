import os.path as osp
import sys, os
# sys.path.insert(0, '..')
import time


import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import MNISTSuperpixels
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
from torch_geometric.utils import normalized_cut
from torch_geometric.nn import (NNConv, graclus, max_pool, max_pool_x,
                                global_mean_pool)
# __file__ = osp.abspath('')
# auxiliary functions
from skimage import segmentation as segment, draw
from skimage.measure import regionprops
from skimage.future import graph
from skimage.segmentation import slic, quickshift

#####
# aux libs
from skimage import color
from itertools import product

from lib.segmentation import extract_features_fixed
# from lib.segmentation import slic_fixed
from lib.segmentation import quickshift_fixed
from lib.pipeline import preprocess_pipeline_fixed
from tqdm import tqdm


class PatchGen(object):
    def __init__(self,patch_size=21):
#         assert(patch_size%2==1)
        self.patch_size = patch_size
                 
    def transform(self,im,padding=2):
        p = self.patch_size
        padding = p//2
        im_pad = np.pad(im,((padding,padding),(padding,padding),(0,0)),mode='constant')
        m,n = im.shape[0], im.shape[1]
        patches = []
        for i,j in product(range(n),range(m)):
            patch = im_pad[i:i+p,j:j+p,:]
            patches.append(patch)
#             patch = im_pad[i:i+p,j:j+p,:]
#             patches.append(np.expand_dims(patch,axis=-1))
        return patches
def graph_maker(image, labels):
    def _weight_mean_color(graph, src, dst, n):
        """Callback to handle merging nodes by recomputing mean color.

        The method expects that the mean color of `dst` is already computed.

        Parameters
        ----------
        graph : RAG
            The graph under consideration.
        src, dst : int
            The vertices in `graph` to be merged.
        n : int
            A neighbor of `src` or `dst` or both.

        Returns
        -------
        data : dict
            A dictionary with the `"weight"` attribute set as the absolute
            difference of the mean color between node `dst` and `n`.
        """

        diff = graph.node[dst]['mean color'] - graph.node[n]['mean color']
        diff = np.linalg.norm(diff)
        return {'weight': diff}


    def merge_mean_color(graph, src, dst):
        """Callback called before merging two nodes of a mean color distance graph.

        This method computes the mean color of `dst`.

        Parameters
        ----------
        graph : RAG
            The graph under consideration.
        src, dst : int
            The vertices in `graph` to be merged.
        """
        graph.node[dst]['total color'] += graph.node[src]['total color']
        graph.node[dst]['pixel count'] += graph.node[src]['pixel count']
        graph.node[dst]['mean color'] = (graph.node[dst]['total color'] /
                                         graph.node[dst]['pixel count'])

    g = graph.rag_mean_color(image,labels)
    
    offset = 1
    # create a map array
    map_array = np.arange(labels.max() + 1)
    for n, d in g.nodes(data=True):
        for label in d['labels']:
            map_array[label] = offset
        offset += 1
        
    # compute centroids to the nodes
    g_labels = map_array[labels]
    regions = regionprops(g_labels)
    for (n, data), region in zip(g.nodes(data=True), regions):
        data['centroid'] = region['centroid']
    if g.number_of_nodes()<75:
        print('warning! the number of nodes in the graph is less than 75')
    while g.number_of_nodes()>75:    
        edges=sorted(g.edges(data=True), key=lambda t: t[2].get('weight', 1))
        src, dst = edges[0][0], edges[0][1]
        merge_mean_color(g, src, dst)
        g.merge_nodes(src, dst, weight_func=_weight_mean_color)
    return g


####################
print('reading data')
segmented = np.load("./data/M2NIST/segmented.npy")
_, HEIGHT, WIDTH, N_CLASSES = segmented.shape
mask = 1-segmented[:,:,:,-1]
combined = np.load("./data/M2NIST/combined.npy").reshape((-1, HEIGHT, WIDTH, 1))/255
#####################
padgen = PatchGen()
segmentation_algorithm = lambda x: slic(x, n_segments=75, max_iter=50)
#####################
def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return [rho, phi]

def create_dataset(filename,indices)
    N = combined.shape[0]
    x = []
    y = []
    pos = []
    edge_index = []
    edge_slice = [0]
    print('creating first part from ' + indices[0] + ' to '+indices[-1])
    for i in tqdm(indices):
        w = padgen.transform(combined[i])
        m = padgen.transform(np.expand_dims(mask[i],axis=2))
        s = [segmentation_algorithm(ww.squeeze()) for ww in w]
        g = [graph_maker(ww.squeeze(), ss) for ww, ss in zip(w,s)]
        for mm, gg in zip(m,g):
            node_pos = list(map(lambda n: n[1]['centroid'], gg.nodes(data=True)))
            node_pos = list(map(lambda x: cart2pol(x[0],x[1]), node_pos))
            node_values = list(map(lambda n: n[1]['mean color'][0], gg.nodes(data=True)))
            x.append(node_values)
            pos.append(node_pos)
            edge_index +=[[e[0],e[1]] for e in gg.edges]
            last_slice = edge_slice[-1]
            edge_slice += [last_slice+gg.number_of_edges()]
            y.append(mm[mm.shape[0]//2+1,mm.shape[1]//2+1,0])
    data = (torch.tensor(x),
            torch.tensor(edge_index).t(),
            torch.tensor(edge_slice),
            torch.tensor(pos),
            torch.tensor(y))
    torch.save(data, './data/M2NIST/'+ filename +'.pt')
    print('created part: ', filename)

