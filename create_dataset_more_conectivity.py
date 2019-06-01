import os.path as osp
import sys, os
# sys.path.insert(0, '..')
import time


import numpy as np
import matplotlib.pyplot as plt
import torch
from skimage.measure import regionprops
from skimage.future import graph
from skimage.segmentation import slic, quickshift

#####
# aux libs
from skimage import color
from itertools import product
from tqdm import tqdm
from random import shuffle
from skimage.future.graph import rag
import matplotlib.cm as cm

def skip_diag_strided(A):
    m = A.shape[0]
    strided = np.lib.stride_tricks.as_strided
    s0,s1 = A.strides
    return strided(A.ravel()[1:], shape=(m-1,m), strides=(s0+s1,s1)).reshape(m,-1)

def plot_graph(g, image=None):
    nodes_data = dict(g.nodes(data=True))
    pos_x, pos_y = np.zeros(g.number_of_nodes()), np.zeros(g.number_of_nodes())
    for k, d in nodes_data.items():
        pos_x[k] = d['centroid'][0]
        pos_y[k] = d['centroid'][1]

    np.zeros(g.number_of_nodes())
    coo_matrix = np.array([[e[0], e[1]] for e in g.edges]).T
    print(coo_matrix)
    plt.figure(figsize=(10, 10))
    if image is not None:
        xmin, xmax, ymin, ymax = -0.5, image.shape[0] - 0.5, -0.5, image.shape[1] - 0.5
    else:
        xmin, xmax, ymin, ymax = -0.5, max(pos_x) - 0.5, -0.5, max(pos_y) - 0.5
    if image is not None:
        image = image.copy() / image.max()
        image = color.gray2rgb(image.squeeze())
        plt.imshow(image.transpose((1, 0, 2)))

    for i in range(g.number_of_edges()):
        ii, jj = coo_matrix[0, i], coo_matrix[1, i]
        plt.plot([pos_x[ii], pos_x[jj]], [pos_y[ii], pos_y[jj]], 'g-', alpha=1.0)
    print(list(nodes_data.values())[0])
    values = [d['mean color'][0] for d in nodes_data.values()]
    colors = [cm.viridis(color) for color in values]

    for xx, yy, cc in zip(pos_x, pos_y, colors):
        plt.plot(xx, yy, 'o', color=cc)
    plt.axis('scaled')
    # plt.colorbar()
    plt.xlim([xmin, xmax])
    plt.ylim([ymin, ymax])


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


def graph_maker(image, labels, num_nodes = 69, threshold = 5):
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

    def translate_graph(nodes_data, edges_data, threshold = threshold):
        N = len(nodes_data) # number of nodes
        a = {n[0]: i for i, n in enumerate(nodes_data)}
        a_inv = { i:n[0] for i, n in enumerate(nodes_data)}

        g2 = rag.RAG()

        centroids = [np.array(n[1]['centroid']) for n in nodes_data]
        A = np.array(centroids)
        B = np.repeat(A, N, axis=1)
        X, Y = B[:, 0:N], B[:,N:2*N]
        D = np.sqrt((X - X.T) ** 2 + (Y - Y.T) ** 2)
        # checks that the nodes will be connected with at least one other node to preserve the final number of nodes
        # in the resulting graph
        index_unconnected_nodes = np.invert((D<threshold).sum(axis=1)>1)
        # sum the number problematic nodes indices has to be zero
        if not index_unconnected_nodes.sum()==0:
            D2 = skip_diag_strided(D)
            D_0 = D2.min(axis=1)[index_unconnected_nodes]
            threshold = 2*D_0.max()
            print('warning: Max distance is less than theshold, reassining theshold to 0.7min_dist=',threshold)
        D = np.triu(D, k=1)
        D_1 = D>0
        D_th = D<threshold
        D_th = D_th*D_1
        a1, a2 = np.nonzero(D_th)
        # generating the edges with the new numeration
        for i,j in zip(a1,a2):
            distance = D[i,j]
            g2.add_edge(i,j,weight=distance)

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
    if g.number_of_nodes()<num_nodes:
        print('warning: number of nodes is less than ', num_nodes, ' != ', g.number_of_nodes())
    edges_data = list(g.edges(data=True))
    if g.number_of_nodes()>num_nodes:
        indices = list(range(g.number_of_edges()))
        shuffle(indices)
        edges = [edges_data[index] for index in indices]
        edges = sorted(edges, key=lambda t: t[2].get('weight', 1))
        # history_sources = []
        # c_time = time.time()
        # # print('initialization sorting and getting edges data',c_time - start_time)
        # start_time = c_time
        for i in range(num_nodes, g.number_of_nodes()):
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
    g_out = translate_graph(nodes_data,edges_data,threshold=threshold)
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
segmentation_algorithm = lambda x: slic(x, n_segments=80, max_iter=100)
#####################
def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return [rho, phi]



def create_one_sample(i, x = [], y = [], pos = [], edge_index = [], edge_slice = [0], threshold_dist=5, num_nodes=75):
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
    g = [graph_maker(ww.squeeze(), ss, threshold=threshold_dist, num_nodes= num_nodes ) for ww, ss in tqdm(list(zip(w,s)))]
    print('Done')
    print('Collecting data')
    for mm, gg in zip(m,g):
        node_pos = np.zeros((gg.number_of_nodes(),2))
        node_values = np.zeros(gg.number_of_nodes())

        for k,d in gg.nodes(data=True):
            node_pos[k,0] = d['centroid'][0]
            node_pos[k,1] = d['centroid'][1]
            node_values[k] = d['mean color'][0]
        node_pos = [(node_pos[i,0],node_pos[i,1])for i in range(gg.number_of_nodes())]
        x.append(node_values.tolist())
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
        torch.tensor(y),
        w,
        s,
        g)
    return data




def create_dataset(filename,indices, threshold_dis=10, num_nodes=75):
    N = combined.shape[0]
    x = []
    y = []
    pos = []
    edge_index = []
    edge_slice = [0]
    print('creating first part from ' , indices[0] , ' to ', indices[-1],'indices')
    for i in tqdm(indices):
        create_one_sample(i,x=x,y=y,pos=pos,edge_index=edge_index,edge_slice=edge_slice,threshold_dist=threshold_dis,num_nodes=num_nodes)
    data = (torch.tensor(x),
        torch.tensor(edge_index).t(),
        torch.tensor(edge_slice),
        torch.tensor(pos),
        torch.tensor(y))
    torch.save(data, './data/M2NIST/raw/'+ filename +'.pt')
    print('created part: ', filename)

if __name__=='__main__':
    create_dataset('data1', range(0, 1))
    create_dataset('data2', range(1, 2))
    # create_one_sample(0,threshold_dist=3.5)

