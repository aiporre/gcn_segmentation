import torch
from torch import sigmoid
from torch.autograd import Function

from lib.utils import print_debug


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


class DiceCoeff(Function):
    """Dice coeff for individual examples"""

    def forward(self, inputs, targets):
        self.save_for_backward(inputs, targets)
        eps = 0.0001
        try:
            self.inter = torch.dot(inputs.view(-1), targets.view(-1))
        except RuntimeError as e:
            message = 'inputs'+str(inputs.size())+'targets'+str(targets.size())
            print_debug(message)
            print_debug('Error calculation in intersection', exception=e)
            raise e

        self.union = torch.sum(inputs)+torch.sum(targets)+eps

        t = (2*self.inter.float()+eps)/self.union.float()
        return t

    # This function has only a single output, so it gets only one gradient
    def backward(self, grad_output):

        input, target = self.saved_tensors
        grad_input = grad_target = None

        if self.needs_input_grad[0]:
            grad_input = grad_output*2*(target*self.union-self.inter) \
                         /(self.union*self.union)
        if self.needs_input_grad[1]:
            grad_target = None

        return grad_input, grad_target