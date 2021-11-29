import numpy as np
import torch

from lib.graph import grid_tensor


def reshape_square(x, channels=None):
    # transform to numpy
    if isinstance(x, torch.Tensor):
        x = x.cpu().detach().numpy()
    N = x.size
    if channels is None:
        # assumes that there is just one one channel
        dimension = x.size[0]
        dimension = np.sqrt(dimension).astype(int)
        assert dimension*dimension == N, "This input cannot be transformed to a square dimension are improper %s" \
                                                 % str(x.size)
        x = x.reshape((dimension, dimension))
    else:
        dimension = x.size[0] // channels
        dimension = np.sqrt(dimension).astype(int)
        assert dimension * dimension * channels == N, "This input cannot be transformed to a square dimension are " \
                                                      "improper %s" % str(x.size)
        x = x.reshape((dimension, dimension, channels))
    return x

class Crop(object):
    def __init__(self,x,y,w,h,graph_mode = True):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.graph_mode = graph_mode
    def __call__(self, sample):
        im, lb = sample
        # cropping
        im = im[self.y:self.y+self.h,self.x:self.x+self.w]
        lb = lb[self.y:self.y+self.h, self.x:self.x+self.w]
        if self.graph_mode:
            grid = grid_tensor((self.w, self.h), connectivity=4)
            grid.x = torch.tensor(im.reshape(self.w*self.h)).float()
            grid.y = torch.tensor([lb.reshape(self.w*self.h)]).float()
            return grid
        else:
            return (im,lb)

class CropVessel12(object):
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h

    def __call__(self, dataset):
        # cropping
        return dataset[:,self.y:self.y + self.h, self.x:self.x + self.w]
