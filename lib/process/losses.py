import torch
from torch import sigmoid
from torch.nn.functional import binary_cross_entropy
from torch.autograd import Function
from torch_geometric.data import Data

from .progress_bar import printProgressBar
from lib.utils import print_debug

def estimatePositiveWeight(dataset, progress_bar=True):
    positive_count = 0
    negative_count = 0
    L = dataset.num_batches
    prefix_bar = 'Estimating positive weight: '
    if progress_bar:
        printProgressBar(0, L, prefix=prefix_bar, suffix='Complete', length=50)
    i = 0
    for d in dataset:
        if isinstance(d, Data):
            labels = d.y
        else:
            labels = d[1]
        positive_count += (labels == 1.0).sum().cpu().item()
        negative_count += (labels == 0.0).sum().cpu().item()
        if progress_bar:
            printProgressBar(i, L, prefix=prefix_bar, suffix='Complete', length=50)
        else:
            print(prefix_bar, i + 1, ' out of ', L, '(percentage {}%)'.format(100.0 * (i + 1) / L))
        i += 1
    if positive_count == 0 or negative_count == 0:
        positive_weight = 1
    else:
        positive_weight = positive_count/negative_count if negative_count !=0 else 1
    print('Estimated positive weight is :', positive_weight)
    return positive_weight

# TODO: make the implementation of the losses functions: Dice, DICE-L1, Focal-Loss and generalized-Dice

class DCS(object):
    """
    DCS loss function smooth where the values are calculated in batches as:
        $$
        \mathcal{L} = \frac{(M*P + 1)}{M^2 + P^2 + 1}
        $$
    where $M$ and $P$ are the mask and prediction values. The 1 is for numerical stability.
    """

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

class FocalLoss:
    def __init__(self, pre_sigmoid=False,  alpha=0.25, gamma=2.0):
        self.alpha = alpha
        self.gamma = gamma
        self.pre_sigmoid = pre_sigmoid

    def __call__(self, inputs, targets):
        # inputs.dims() = (B, H,W,1)---> (B, H*W)
        # have to use contiguous since they may from a torch.view op
        iflat = inputs.flatten(start_dim=1) if not self.pre_sigmoid else sigmoid(inputs.flatten(start_dim=1))
        tflat = targets.flatten(start_dim=1)
        bce = binary_cross_entropy(iflat, tflat, reduction="none")
        probs =  torch.exp(-bce)
        focal_loss = self.alpha * (1 - probs) ** self.gamma * bce
        return focal_loss.mean()


class DiceLoss:
    def __init__(self, pre_sigmoid=False, epsilon=1E-10):
        self.epsilon = epsilon
        self.pre_sigmoid = pre_sigmoid

    def __call__(self, inputs, targets):
        # inputs.dims() = (B, H,W,1)---> (B, H*W)
        # have to use contiguous since they may from a torch.view op
        iflat = inputs.flatten(start_dim=1) if not self.pre_sigmoid else sigmoid(inputs.flatten(start_dim=1))
        tflat = targets.flatten(start_dim=1)

        A_sum = torch.sum(iflat * tflat, axis=1) + self.epsilon
        B_sum = torch.sum(iflat + tflat, axis=1) + self.epsilon
        C_sum = torch.sum((1 - iflat) * (1 - tflat), axis=1) + self.epsilon
        D_sum = torch.sum(2 - iflat - tflat, axis=1) + self.epsilon

        return torch.mean(1 - (A_sum / B_sum) - (C_sum / D_sum))

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