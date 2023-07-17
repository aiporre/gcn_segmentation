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

from random import shuffle

from skimage.future.graph import rag


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
        for i,j in product(range(m),range(n)):
            patch = im_pad[i:i+p,j:j+p,:]
            patches.append(patch)
#             patch = im_pad[i:i+p,j:j+p,:]
#             patches.append(np.expand_dims(patch,axis=-1))
        return patches
    def get_central_pixels(self,im):
        im = im.squeeze()
        return im.flatten()


def graph_maker(image, labels):
    def _weight_mean_color(graph, src, dst, n):
        diff = graph.node[dst]['mean color'] - graph.node[n]['mean color']
        diff = np.linalg.norm(diff)
        return {'weight': diff}


    def merge_mean_color(graph, src, dst):
        total_mass = graph.node[dst]['pixel count'] + graph.node[src]['pixel count']
        m_dst = graph.node[dst]['pixel count']/total_mass
        m_src = graph.node[src]['pixel count']/total_mass
        graph.node[dst]['centroid'] = (m_dst*graph.node[dst]['centroid'][0]+m_src*graph.node[src]['centroid'][0],
                                       m_dst * graph.node[dst]['centroid'][1] + m_src * graph.node[src]['centroid'][1])
        graph.node[dst]['total color'] += graph.node[src]['total color']
        graph.node[dst]['pixel count'] += graph.node[src]['pixel count']
        graph.node[dst]['mean color'] = (graph.node[dst]['total color'] /
                                         graph.node[dst]['pixel count'])
    def remove_from_list(src,edges_list):
        return [e for e in edges_list if not e[0]==src and not e[1]==src]

    def translate_graph(nodes_data, edges_data):
        N = len(nodes_data) # number of nodes
        a = {n[0]: i for i, n in enumerate(nodes_data)}
        g2 = rag.RAG()
        # generating the edges with the new numeration
        for e in edges_data:
            g2.add_edge(a[e[0]],a[e[1]],weight=e[2]['weight'])

        # reassinging the values to each node
        for n, values in nodes_data:
            n2 = a[n]
            for k,v in values.items():
                g2.node[n2][k]=v
        return g2

    # start_time = time.time()
    g = graph.rag_mean_color(image,labels)
    # c_time = time.time()
    # print('rag mean calculation',c_time-start_time)
    # start_time = c_time
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
    # c_time = time.time()
    # print('compute centroids',c_time - start_time)
    # start_time = c_time
    if g.number_of_nodes()<75:
        print('warning: number of nodes is less than 75, ', g.number_of_nodes() )
    if g.number_of_nodes()>75:
        indices = list(range(g.number_of_edges()))
        shuffle(indices)
        edges_data = list(g.edges(data=True))
        edges = [edges_data[index] for index in indices]
        edges = sorted(edges, key=lambda t: t[2].get('weight', 1))
        # history_sources = []
        # c_time = time.time()
        # # print('initialization sorting and getting edges data',c_time - start_time)
        # start_time = c_time
        for i in range(75, g.number_of_nodes()):
            src, dst = edges[0][0], edges[0][1]
            merge_mean_color(g, src, dst)
            g.merge_nodes(src, dst, weight_func=_weight_mean_color)
            edges = remove_from_list(src, edges)
            # history_sources.append(src)
        # c_time = time.time()
        # print('removed excess of nodes',c_time - start_time)
        # start_time = c_time
    nodes_data = g.nodes(data=True)
    edges_data = list(g.edges(data=True))
    g_out = translate_graph(nodes_data,edges_data)
    # c_time = time.time()
    # print('reassing node numbers', c_time - start_time)
    return g_out


####################
print('reading data')
segmented = np.load("./data/M2NIST/segmented.npy")
_, HEIGHT, WIDTH, N_CLASSES = segmented.shape
mask = 1-segmented[:,:,:,-1]
combined = np.load("./data/M2NIST/combined.npy").reshape((-1, HEIGHT, WIDTH, 1))/255
#####################
padgen = PatchGen()
segmentation_algorithm = lambda x: slic(x, n_segments=75, compactness=5, max_iter=25, sigma=0)
#####################
def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return [rho, phi]

def create_dataset(filename,indices):
    N = combined.shape[0]
    x = []
    y = []
    pos = []
    edge_index = []
    edge_slice = [0]
    print('creating first part from ' , indices[0] , ' to ', indices[-1],'indices')
    for i in tqdm(indices):
        print('image',i,'/',1+indices[-1])
        print('Generating patches')
        w = padgen.transform(combined[i])
        print('Done')
        print('Computing central pixels')
        m = padgen.get_central_pixels(mask[i])
        print('Done')
        print('Segmentating of each patch')
        s = [segmentation_algorithm(ww.squeeze()) for ww in tqdm(w)]
        print('Done')
        print('generating graphs')
        g = [graph_maker(ww.squeeze(), ss) for ww, ss in tqdm(list(zip(w,s)))]
        print('Done')
        print('Collecting data')
        for mm, gg in zip(m,g):
            node_pos = list(map(lambda n: n[1]['centroid'], gg.nodes(data=True)))
#             node_pos = list(map(lambda x: cart2pol(x[0],x[1]), node_pos))
            node_values = list(map(lambda n: n[1]['mean color'][0], gg.nodes(data=True)))
            x.append(node_values)
            pos.append(node_pos)
            edge_index +=[[e[0],e[1]] for e in gg.edges]
            last_slice = edge_slice[-1]
            edge_slice += [last_slice+gg.number_of_edges()]
            y.append(mm)
        print('Done')
    data = (torch.tensor(x),
        torch.tensor(edge_index).t(),
        torch.tensor(edge_slice),
        torch.tensor(pos),
        torch.tensor(y))
    torch.save(data, './data/M2NIST/raw/'+ filename +'.pt')
    print('created part: ', filename)

if __name__=='__main__':
    create_dataset('data1', range(0, 3))
    create_dataset('data2', range(3, 5))

