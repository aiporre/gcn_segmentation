import torch
from torch import sigmoid


class DCS(object):
    def __init__(self, pre_sigmoid=False):
        self.pre_sigmoid = pre_sigmoid
    def __call__(self, inputs, targets):

        """
        This definition generalize to real valued pred and target vector.
        This should be differentiable.
        pred: tensor with first dimension as batch
        target: tensor with first dimension as batch
        """

        smooth = 1.
        # inputs.dims() = (B, H,W,1)---> (B, H*W)
        # have to use contiguous since they may from a torch.view op
        iflat = inputs.flatten(start_dim=1) if not self.pre_sigmoid else sigmoid(inputs.flatten(start_dim=1))
        tflat = targets.flatten(start_dim=1)
        intersection = (iflat*tflat).sum(axis=1)

        A_sum = torch.sum(iflat*iflat, axis=1)
        B_sum = torch.sum(tflat*tflat, axis=1)

        return torch.mean(1 - ((2. * intersection + smooth) / (A_sum + B_sum + smooth)))