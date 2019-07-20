import torch

from lib.graph import grid_tensor


class Crop(object):
    def __init__(self,x,y,w,h):
        self.x = x
        self.y = y
        self.w = h
        self.h = h
    def __call__(self, sample):
        im, lb = sample
        # cropping
        im = im[self.y:self.y+self.h,self.x:self.x+self.w]
        lb = lb[self.y:self.y+self.h, self.x:self.x+self.w]
        grid = grid_tensor((self.w, self.h), connectivity=4)
        grid.x = torch.tensor(im.reshape(self.w*self.h)).float()
        grid.y = torch.tensor([lb.reshape(self.w*self.h)]).float()
        return grid