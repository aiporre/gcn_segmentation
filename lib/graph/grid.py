import numpy as np
import scipy.sparse as sp

from .distortion import filter_adj
try:
    from torch_sparse import coalesce
    from torch_geometric.data import Data
    from torch_geometric.utils.grid import grid_pos
except Exception as e:
    print('Warning: torch geometric not installed error importing. ' + str(e))
import torch
from itertools import product



def grid_adj(shape, connectivity=4, dtype=np.float32):
    """Return adjacency matrix of a regular grid."""
    # TODO: has a bug for some numbers filter of nodes is not correct. attemp to create nodes and edges
    # only inside.
    assert connectivity == 4 or connectivity == 8

    h, w = shape

    if connectivity == 4:
        filt = [-w - 2, -1, 1, w + 2]
    else:
        filt = [-w - 3, -w - 2, -w - 1, -1, 1, w + 1, w + 2, w + 3]

    # Build basic rows and cols with +1 padding on all sides.
    n = (h + 1) * (w + 2) - 1
    rows = np.arange(w + 3, n).repeat(connectivity)
    rows = np.reshape(rows, (-1, connectivity))
    cols = rows + filt
    rows = rows.flatten()
    cols = cols.flatten()

    data = np.ones_like(rows, dtype=np.uint8)
    n = (h + 2) * (w + 2)
    adj = sp.coo_matrix((data, (rows, cols)), (n, n))

    # Compute filter nodes.
    rows = np.arange(w + 2, h * (w + 3), w + 2).repeat(w)
    rows = np.reshape(rows, (-1, w))
    cols = np.arange(1, w + 1)
    nodes = (rows + cols).flatten()

    return filter_adj(adj, nodes)


def grid_points(shape, dtype=np.float32):
    """Return the grid points of a given shape with distance `1`."""

    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    y = y.flatten()
    x = x.flatten()

    z = np.empty((shape[0] * shape[1], 2), dtype)
    z[:, 0] = y
    z[:, 1] = x
    return z


def grid_mass(shape, dtype=np.float32):
    """Return the mass of grid points of a given shape with distance `1`."""
    return np.ones(shape[0] * shape[1], dtype)

def grid_index(shape, connectivity=4, device=None):
    h, w = shape
    w = w
    if connectivity == 4:
        kernel = [-w - 1, -1, w - 1, -w, 0, w, -w + 1, 1, w + 1]
    else:
        kernel = [-w - 2, -w - 1, -2 , -1, w - 2, w - 1, -w, 0, w, -w + 1, -w + 2, 1 , 2, w + 1, w + 2]
    kernel = torch.tensor(kernel, device=device)

    row = torch.arange(h * w, dtype=torch.long, device=device)
    row = row.view(-1, 1).repeat(1, kernel.size(0))
    col = row + kernel.view(1, -1)
    row, col = row.view(h, -1), col.view(h, -1)
    index = torch.arange(3, row.size(1) - 3, dtype=torch.long, device=device)
    row, col = row[:, index].view(-1), col[:, index].view(-1)

    mask = (col >= 0) & (col < h * w)
    row, col = row[mask], col[mask]

    edge_index = torch.stack([row, col], dim=0)
    edge_index, _ = coalesce(edge_index, None, h * w, h * w)

    return edge_index

def grid_tensor(shape, connectivity=4, dtype=np.float32, device=None):
    #adj_matrix = grid_adj(shape=shape, connectivity=connectivity, dtype=dtype)
    #cols = adj_matrix.col
    #rows = adj_matrix.row
    #edge_index = torch.tensor([cols,
    #                           rows], dtype=torch.long)
    edge_index = grid_index(shape, connectivity=connectivity, device=device)
    x = torch.ones((shape[0]*shape[1],1), dtype=torch.float, device=device)

    #pos = torch.tensor([(i,j) for i,j in  product(range(shape[0]),range(shape[1]))], dtype=torch.float)
    pos = grid_pos(shape[0], shape[1], dtype=torch.float, device=device)

    data = Data(x=x, edge_index=edge_index, pos=pos, y=torch.Tensor(1))

    return data